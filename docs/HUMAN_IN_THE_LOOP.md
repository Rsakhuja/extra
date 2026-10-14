# Human-in-the-Loop, Auto Mode, and Persistent Checkpointing

This document describes how the Extra engine gates every tool call through a
centralized, **deterministic** approval layer, how it interrupts and resumes
runs, and how it selects a checkpointer. It applies uniformly to **all** tool
providers — local plugin tools and tools discovered dynamically from MCP servers
— because the mechanism lives in the engine's tool-execution path, not inside any
one provider.

> Read with [ARCHITECTURE.md](ARCHITECTURE.md), [YAML_SPEC.md](YAML_SPEC.md)
> (`auto`), and [MCP_AND_TOOLS.md](MCP_AND_TOOLS.md).

## 1. Overview

By default **every** tool call an agent requests requires explicit human
approval before the tool runs. Approval is **not** requested when either:

1. the tool was already approved **for this session** (a prior
   `ALLOW_FOR_SESSION` decision), or
2. the agent has `auto: true` in its YAML config.

There is **intentionally no LLM-based risk classification** in this version. No
tools are sent to a model at startup, no risk levels are computed, and no
per-tool configuration decides whether approval is needed. The rule is simple and
deterministic:

```text
if agent.auto:                 approval NOT required  (execute now)
elif tool allowed for session: approval NOT required  (execute now)
else:                          approval REQUIRED       (ask the human)
```

The human's decision is one of three typed values:

```text
ALLOW_ONCE          run this one invocation; ask again next time
ALLOW_FOR_SESSION   run it and stop asking for this tool this session
DENY                do not run it; store nothing
```

## 2. Tool-execution flow

```text
Agent / LangGraph node
        │  requests a concrete tool call
        ▼
ToolInvoker.invoke               (the single choke point for local + MCP tools)
        │
        ▼
ApprovalCoordinator.resolve()    (centralized; no tool-specific logic)
        │   ├── ApprovalPolicy → requires approval?  (auto → session → ask)
        │   ├── if not required → execute
        │   └── if required → ApprovalProvider.request_decision()  (interrupt)
        ▼
   execute=True  ──► idempotency guard ─► provider (LocalTool / MCP tools/call)
   execute=False ──► structured "[denied]" tool result (no execution)
        │
        ▼
ToolUsageTracker                 (records the outcome; never decides anything)
```

There is **no** path that reaches a provider without passing through
`ApprovalCoordinator.resolve()`. The engine uses a custom tool loop
(`run_tool_loop`), not a bypassing LangGraph `ToolNode`, and both local and MCP
tools are invoked from the same `ToolInvoker.invoke`.

Approval and tool-usage tracking are separate subsystems that share identifiers
(`run_id`, `tool_call_id`, agent, tool) but no responsibilities: approval decides
*whether* an invocation may run, tracking records *what happened to it*. See
[MCP_AND_TOOLS.md](MCP_AND_TOOLS.md) for the tracking side; a suspended
invocation is recorded only once it is decided, and a resumed one keeps the
approval's `tool_call_id`, so approve/resume never produces a second record.

Components (all small, single-responsibility):

| Component | Responsibility |
| --- | --- |
| `ApprovalDecision` / `parse_decision` | Typed decision + the single free-text parsing boundary |
| `ApprovalPolicy` / `DefaultApprovalPolicy` | Pure "does this need approval?" rule |
| `SessionApprovalRepository` | Port for looking up, granting, revoking, and clearing session permissions |
| `InMemorySessionApprovalRepository` | Current process-lifetime adapter |
| `ApprovalProvider` | Requests a decision from a human (frontend-agnostic) |
| `ApprovalCoordinator` | Wires policy + session store + provider together |
| `RunRepository` | Abstract run-persistence contract; owns idempotent registration semantics |
| `ApprovalManager` | Pending-approval lifecycle + atomic resume claim |
| `ToolExecutionManager` | Execution idempotency ledger |
| `ToolInvoker` | Coordinates one tool call: identity, limits, gate, execution, recording |
| `CheckpointProviderFactory` | Selects the checkpointer once at startup |

The policy, session repository, and provider are all injected behind interfaces
(Dependency Inversion) — nothing is hardcoded into the coordinator, nodes, or
providers.

### Where a future risk-classification policy would plug in

`ApprovalPolicy` is the extension point. A smarter policy — for example one that
auto-executes provably read-only tools or escalates destructive arguments — can
be supplied to `ApprovalCoordinator` without changing the coordinator, the
session store, the provider, or the runtime tool-execution boundary. The current
`DefaultApprovalPolicy` deliberately performs no classification.

## 3. Decisions and their scope

The three decisions are parsed from user-facing text at exactly one boundary
(`parse_decision`), which accepts values such as `approve`, `allow`, `allow once`,
`allow for this session`, `deny`, and `reject`. Everywhere else the code uses the
typed `ApprovalDecision` enum — no raw-string comparisons.

- **`ALLOW_ONCE`** — the current invocation executes; nothing is persisted, so the
  next invocation of the same tool is asked again. It applies only to the exact
  pending invocation (identified by `tool_call_id`).
- **`ALLOW_FOR_SESSION`** — the current invocation executes and a session
  permission is stored, keyed by **tool identity** (not by argument values), so
  every later call of that tool in the same session runs without asking.
- **`DENY`** — the tool does not run, nothing is stored, and a structured
  `[denied]` result is returned to the model. A denial is a normal outcome, not a
  system failure: the agent can explain it, try another action, or finish.

### Session-permission key and lifecycle

A session permission is keyed by the exact normalized tuple:

```text
system_namespace
+ organization_id
+ user_id
+ session_id
+ agent_id
+ provider_namespaced_tool_identity
```

- `system_namespace` is the validated system name, preventing two systems in the
  same process from sharing a permission.
- `organization_id` and `user_id` come from the server-resolved `RunContext`
  identity when available. Missing optional identity values normalize to an empty
  string so lookup semantics stay deterministic. A permission recorded with no
  authenticated user does not match a later authenticated user.
- `session_id` is the **conversation id** carried on `RunContext`
  (`conversation_id`, from the `X-Session-Id` request header). A conversation
  spans multiple runs, so `ALLOW_FOR_SESSION` persists across the runs of one
  conversation. It is **never** the `run_id`, agent id, process, thread, or user
  id alone. When there is no session id, no key exists — so a session permission
  can be neither found nor stored, and approval is required every time
  (fail-closed).
- `tool_identity` is a stable `provider:namespace:tool` string (MCP tools are
  namespaced by their server id). Two MCP servers exposing the same short tool
  name therefore never share a permission.

`run_id` and `approval_id` are stored as audit provenance, not as lookup-key
fields. `tool_call_id` is never part of this permission. Raw tool arguments are
neither part of the key nor stored: approving `send_email` for a session allows
future `send_email` calls in that scope regardless of their arguments.

### Repository selection and lifetime

The application composition layer creates one async-safe in-memory repository
for the application lifespan and injects that same instance into the long-lived
engine. The engine does not read configuration or know which adapter it receives.
Grants survive later runs and engine calls that share the repository, and they
disappear when the process restarts. Direct engine construction retains an
in-memory default for backwards compatibility and tests.

There is intentionally no database table, migration, or ORM adapter for session
approvals yet. The repository port is the extension point for adding a persistent
adapter later without changing `ApprovalCoordinator` or `LangGraphEngine`.

`revoke(key)` deactivates one exact grant and `clear_session(scope)` deactivates
all tools for one exact `(system, organization, user, session)` scope. The current
adapter supports `expires_at`; expired grants fail lookup. No arbitrary TTL or
automatic logout/deletion cleanup is currently applied because the platform does
not yet expose one reliable session-end lifecycle hook.

## 4. `auto`

`auto` is an optional agent-level flag (default `false`; YAML key `auto`, with
`auto_mode` accepted as an alias — see [YAML_SPEC.md](YAML_SPEC.md)).

```text
auto: false / missing   →  every tool call requires approval (unless session-allowed)
auto: true              →  every tool call executes immediately; provider never invoked
```

`auto` is evaluated **per agent** inside the coordinator, so enabling it for one
agent never enables automatic execution for another agent in the same run. In
auto mode no approval is requested and no session permission is stored.

## 5. Interrupt and resume

When a call requires approval, the `ApprovalProvider` for this runtime
(`InterruptApprovalProvider`):

1. Persists a sanitized `ApprovalRecord` (masked arguments only) and moves the run
   to `PENDING_APPROVAL`.
2. Calls LangGraph `interrupt(payload)`, which persists a **checkpoint** under the
   run's `thread_id` and returns control to the caller.
3. `Engine.run()` detects the interrupt and returns a `RunResult` with
   `status="pending_approval"` and a sanitized `pending_approval` payload. **No
   tool ran; for MCP, no `tools/call` was sent.**

To resume, `Engine.resume(run_id, approval_id, decision)` or the streamed
`Engine.resume_stream(...)` capability:

> **Compatibility note:** Session-bound approvals now fail closed. Callers of
> `resume(...)` and the approval HTTP endpoints must propagate the same session
> identity used when the approval was created (for the engine API, via
> `caller_session_id` / `X-Session-ID`). Omitting or changing it returns 403.

> **Upgrade note:** Runs suspended before checkpointed model/tool steps existed
> cannot be resumed; they fail and their approval is rejected. Decide or cancel
> pending approvals before upgrading. See
> [`ADR 0005`](adr/0005-checkpointed-hitl-continuation.md).

1. **Atomically claims** the approval (`PENDING → RESUMING`) — exactly one caller
   wins (see §7).
2. Moves the run through `RESUMING → RUNNING` and resumes the **same** LangGraph
   thread from its checkpoint via
   `Command(resume={"decision": ..., "approval_id": ..., "tool_call_id": ...})`.
   The graph is not restarted, the agent is not re-selected, and intent/planning
   are not re-run.
3. The node re-executes, restoring its checkpointed model responses and completed
   tool outcomes instead of regenerating them. The gate checks that the resume's
   `approval_id` and stable `tool_call_id` match the invocation it is about to
   run; a missing or different binding aborts the run as `FAILED` and rejects
   the approval. Otherwise the coordinator interprets the typed decision, and:
   - `ALLOW_ONCE` / `ALLOW_FOR_SESSION` → the original tool call executes and its
     result flows back to the agent (and `ALLOW_FOR_SESSION` records a session
     permission);
   - `DENY` → a structured `[denied]` result is returned and **no** tool/MCP call
     is made.

The claimed decision takes precedence over auto mode and session grants for that
one pending invocation. A grant added by another run, or an auto-mode change
after rebuilding the engine, cannot bypass a pending denial. A request-local
binding validates the first unfinished tool step before policy evaluation and
is consumed only by the matching approval interrupt. Later calls use the normal
policy; completed steps are restored without consuming this new decision.

An unrecognized or missing resume decision **fails closed to `DENY`**, so an
ambiguous resume never triggers a side effect.

A LangGraph node re-executes from its start on resume, but each model response
and each completed leaf-tool outcome is a checkpointed LangGraph task. Replay
returns those saved values, so the model cannot propose a different call than
the one reviewed, and earlier tools, decisions, and session grants are not
repeated. Restored steps must match the node, its input, and the invocation
identity; otherwise the run fails with `CheckpointReplayMismatch`. Runs suspended
before this format existed fail closed on resume. Duplicate execution of a call
that started but did not finish is still prevented by the idempotency ledger
(§7).

Because the reviewed call must execute exactly as proposed, the checkpoint holds
the model's raw tool arguments and the tool results; only the `ApprovalRecord`
and the approval payload are masked. Treat the checkpoint store as sensitive.

Identifiers are kept strictly separate:

```text
run_id        business-level run id
thread_id     LangGraph checkpoint id (equals run_id here)
approval_id   one pending approval
tool_call_id  the agent's requested tool call
execution_id  one actual execution attempt (idempotency key)
```

Run states normally follow
`RUNNING → PENDING_APPROVAL → RESUMING → RUNNING → COMPLETED | FAILED`, with
validated transitions (no `COMPLETED → RESUMING`, no `REJECTED → APPROVED`). An
explicit user cancellation also permits active `RUNNING → CANCELLED` and
pending `PENDING_APPROVAL → CANCELLED` transitions.

`CANCELLED` is terminal. For an executing run, it records that a streaming
consumer disconnected or stopped iteration, so the engine cancelled the graph
instead of continuing model calls. Losing a stream after the run reaches
`PENDING_APPROVAL` does not itself cancel the run: its durable checkpoint remains
valid for a later human decision. The owner can instead explicitly cancel that
pending run; its approval is atomically rejected and the checkpoint becomes
unreachable through resume.

## 6. Checkpointer selection

`CheckpointProviderFactory` chooses the checkpointer **once** at startup; nothing
else in the engine branches on the backend:

```text
checkpoint connection string provided  → persistent, shared checkpointer (PostgreSQL)
no connection string                   → in-memory checkpointer (+ startup warning)
```

After construction, interrupt / checkpoint / resume are identical for both. The
graph is always compiled with a checkpointer, so the HITL path is the same in
local development and production.

### In-memory limitations

The in-memory checkpointer is process-local. On selection the engine logs a
warning that it: is not shared between replicas/Pods, is lost on restart, and
cannot resume a run on another instance. It is intended for local development and
tests, never for multi-replica production. Session approvals currently have the
same process-local lifetime.

### Cross-Pod resume

With a shared persistent checkpointer, the run state belongs to the `thread_id`,
not the Pod. Pod A can start a run and reach an interrupt; the approval can later
be delivered to Pod C, which loads the same `thread_id` and resumes. No sticky
sessions are required. Session approvals, the pending-approval repository, and
the execution ledger are currently process-local, so fully cross-Pod HITL remains
future work behind their existing repository contracts.

## 7. Concurrency and idempotency

**Run registration.** `RunRepository` is an explicit abstract base class,
with `InMemoryRunRepository` as its current process-local implementation. The
contract lives in `agent_engine.runs.repository`; the adapter lives separately in
`agent_engine.runs.in_memory`, leaving the approval repositories under
`agent_engine.approvals`
focused on approval and execution-ledger persistence.
`RunLifecycle` sends the abstraction one intent:
create the run if its `run_id` is not already registered. The repository returns
whether it created the record; it must never overwrite or reset an existing run.
The current `InMemoryRunRepository` provides that guarantee only
within one Python process. It does **not** make registration distributed-safe.
A future Redis implementation must provide the same contract across pods using
an appropriate atomic Redis operation, and can be selected through composition
without changing `RunLifecycle` or `LangGraphEngine`.

The adapter is intentionally lock-free under its process-local execution model:
one repository instance is accessed from one asyncio event loop and OS thread,
and its check-and-mutate sequences contain no suspension points. Coroutines
therefore cannot interleave inside a repository operation. The adapter must not
be shared across threads or event loops; a deployment requiring that model must
inject an implementation with the appropriate synchronization.

The in-memory adapter bounds terminal-state retention to prevent completed run
records from accumulating for the lifetime of a long-running process. It keeps
`COMPLETED`, `FAILED`, and `CANCELLED` records for 24 hours by default, using a
configurable adapter-local TTL. Status reads check only their requested run;
transitions additionally process a fixed-size batch from a FIFO expiration
queue so untouched terminal records are eventually reclaimed. Cleanup cost is
therefore bounded and never scans or drains all expired runs on a request path.
`RUNNING`, `PENDING_APPROVAL`, and `RESUMING` records never expire, so cleanup
cannot invalidate an active execution or resumable approval. After a terminal
record expires, its `run_id` is unknown and may be registered again; the TTL is
therefore also the in-memory adapter's idempotency and status-query retention
window. This policy does not change the `RunRepository` contract: applications
that need different retention inject another configured or persistent adapter.

`create_if_absent` deliberately performs no eviction. Its existence check and
insert have no `await` between them, preserving one-winner asyncio concurrency
semantics without a lock. Consequently, attempting to create an existing run
cannot delete it or change its original expiration; an expired record becomes
reusable only after targeted status access or bounded transition cleanup has
applied the normal TTL eviction policy.

Run status changes cross the same boundary through `transition_if_allowed`.
Checking whether a transition is legal and applying it are one repository
operation, so completion and cancellation cannot both win against the same
in-flight state. The in-memory adapter provides that guarantee only inside its
process; a shared adapter must implement the operation atomically across pods.

**Each pending approval is uniquely identified** by its `tool_call_id`, so a
decision can never be matched to a different pending invocation. Concurrent
invocations are resolved independently; a decision for one never approves another.

**Resume deduplication.** Before resuming, `ApprovalManager.claim()` performs an
atomic `PENDING → RESUMING` compare-and-set. Only one concurrent request wins;
others get a stable `ApprovalAlreadyProcessed`. In-memory this is a locked
transition; the shared-DB implementation maps it to a conditional
`UPDATE ... WHERE status = 'pending'`.

**Cancellation race.** Pending-run cancellation performs the competing atomic
`PENDING → REJECTED` approval transition. Exactly one of cancellation or resume
can win. If resume has already claimed the approval, cancellation returns a
conflict and does not imply that an external side effect can be rolled back.

**Session-repository safety.** The in-memory adapter guards lookup, grant,
revocation, expiry removal, and session cleanup with one async lock. Repeated
grants replace the value for the same immutable key and cannot create duplicates.

### Execution cache versus session permission

These are independent mechanisms:

| Mechanism | Key | Purpose | Effect on a later call |
| --- | --- | --- | --- |
| Tool execution ledger | `execution_id` derived from `tool_call_id` | Prevent replay of the same requested side effect | Same call id returns the recorded result without executing again |
| Session approval repository | system/org/user/session/agent/tool identity | Authorize future calls to the tool | A **new** call id skips the prompt but executes normally |

Consequently, `ALLOW_FOR_SESSION` never returns a prior tool result. A later
model request has a new `tool_call_id`, passes the permission lookup, and reaches
the provider again. The execution ledger is consulted afterward only to protect
against replay of that particular call id.

**Execution idempotency.** Each execution attempt has a stable `execution_id`
derived from the `tool_call_id`. Before running a tool the engine checks the
`ToolExecutionRepository`; a completed prior attempt short-circuits and returns
the recorded result instead of causing a second side effect. This is the primary
protection against duplicate external effects (email, payments, deletes) on
retry or graph re-entry.

Delivery semantics are **at-least-once minus detected duplicates**. If an
external system offers idempotency keys, propagate the `execution_id`; if not,
the ledger minimizes and detects duplicates but the engine does not claim
exactly-once for systems that cannot support it.

## 8. API contracts

The stateless engine API (`agent_engine/api/app.py`) exposes:

```text
POST /invoke                                         → run; may return status=pending_approval
GET  /runs/{run_id}                                  → run status + any pending approval
POST /runs/{run_id}/approvals/{approval_id}/decision → resume with a free-text decision
POST /runs/{run_id}/approvals/{approval_id}/approve  → resume with ALLOW_ONCE
POST /runs/{run_id}/approvals/{approval_id}/reject   → resume with DENY
```

The `/decision` endpoint accepts a free-text `decision` (`allow`,
`allow for this session`, `deny`, …), parsed to a typed `ApprovalDecision` at that
single boundary; an unrecognized value returns 400. `/approve` and `/reject` are
convenience routes for the two most common typed decisions. To grant
`ALLOW_FOR_SESSION`, use `/decision` with `"allow for this session"`.
If `/invoke` or `/stream` supplied `X-Session-ID`, every later decision request
for that run must supply the same header. Omitting or changing it fails closed
with 403 before the approval is claimed.

The conversation approval router
(`agent_manager/api/routes/approvals.py`) exposes the same engine capability
inside the conversation ownership boundary:

```text
POST /conversations/{conversation_id}/runs/{run_id}/approvals/{approval_id}/decision
POST /conversations/{conversation_id}/runs/{run_id}/approvals/{approval_id}/decision/stream
POST /conversations/{conversation_id}/runs/{run_id}/approvals/{approval_id}/cancel
```

Its body carries one typed value: `allow_once`, `deny`, or
`allow_for_session`. The route authorizes the conversation owner before resume,
and the engine checks the stored approval session reference before atomically
claiming it. A completed resume is persisted as the assistant turn; if resume
reaches another approval, that new pending request is returned instead.
Assistant persistence uses a deterministic per-run message id. If execution
completed but the response or first database write failed, retrying the same
decision recovers the result from the completed graph checkpoint and does not
duplicate the assistant turn or tool execution.

The widget uses `/decision/stream`. Its first lifecycle event is
`resume_started` with the original `run_id`; later route, token, approval, and
final events come from the same graph thread. Aborting this response closes the
engine-owned resume stream, cancels its LangGraph task, and atomically moves the
same run to `CANCELLED`. Partial assistant output is not persisted as a completed
assistant message.

The `/cancel` route is a separate terminal lifecycle action, not a fourth
approval decision and not an alias for `DENY`. It is allowed only while the
approval remains pending. A successful response returns
`{"run_id": "...", "status": "cancelled"}`; a decision that already won the
atomic race returns 409.

The bundled widget renders these decisions as **Approve**, **Deny**, and
**Approve for this session**, plus a separate **Cancel run** action. Cancel
remains available while a decision request is in flight so the two requests can
race at the authoritative backend. After `resume_started`, the ordinary active
execution **Stop** control replaces pending cancellation. The textarea and Edit
remain available for drafting, while submit stays blocked until the active run
finishes. The last approval option reuses the existing session permission key,
so later calls to the same provider-qualified tool by the same agent in the same
conversation do not prompt again.

A pending-approval payload contains `run_id`, `approval_id`, `agent_id`,
`tool_name`, a human-readable `description` (stating the tool has **not** run
yet), `provider`, optional `server_id`, and **masked** `arguments`. The streaming
and non-streaming forms expose the same safe identity fields. Approval endpoints
validate that the run and approval exist, the approval belongs to the run, the
caller is authorized, the approval is still pending, and the run is resumable.
Errors map to stable status codes (404 / 403 / 409 / 400) without leaking
internals. Secrets, tokens, and unmasked arguments are never persisted or
returned.

Token usage is accumulated on the run across every execution leg: the initial
leg that reaches an interrupt and each later resume. A completed conversation
turn therefore persists the full reported usage and remains subject to the same
conversation-budget accounting as a run that never paused.

## 9. Security notes

Approval records and the approval request store only masked, non-secret data and
an `auth_ref` reference — never access tokens, MCP session tokens, or raw
authorization headers. The sanitizer masks common credential-bearing keys
(`password`, `token`, `secret`, `authorization`, `api_key`, `access_token`,
`refresh_token`, `cookie`, …) recursively through nested structures, and returns a
copy so the arguments handed to the tool are never mutated. Valid credentials are
resolved again at resume time. Structured logs use identifiers and safe metadata
(`run_id`, `approval_id`, `tool_call_id`, `execution_id`, `decision`, …), never
argument bodies.

## 10. Known limitations / decisions

- **No risk classification (by design).** This version asks for approval on every
  tool by default. A future `ApprovalPolicy` implementation can add risk-based
  auto-execution at the extension point in §2 without reworking the workflow.
- **Lifecycle cleanup is explicit.** Expiry, `revoke`, and `clear_session` are
  implemented, but there is no automatic cleanup on logout or conversation
  deletion until a reliable platform session-end hook exists.
- **In-memory session approvals.** Permissions are process-local and lost on
  restart. Cross-Pod behavior requires a future persistent implementation of the
  existing `SessionApprovalRepository` port.
- **Single-node re-execution.** With the current one-node-per-graph-node design,
  resuming replays the node from its start (LangGraph semantics). Preparation is
  deterministic and the idempotency ledger prevents duplicate side effects; a
  future optimization is to split the agent turn into separate LLM/tools nodes so
  only the tools node replays.
- **Shared-DB repositories.** The in-memory `Run`/`Approval`/`ToolExecution`
  repositories back local development. Multi-Pod resume requires a shared
  persistent checkpointer (PostgreSQL), shared DB-backed repositories, and a
  shared session store that implement the same contracts.
