# Runtime Lifecycle

This document defines the lifecycle of the runtime and is binding. The single
most important rule in this repository lives here:

> **`RuntimeEngine` is created once at application startup.
> `ExecutionContext` is created per request. Never create a runtime per
> request.**

---

## Two lifecycles

### Startup (once per process)

```
load config.yml
  → validate
  → compile → CompiledAgentGraph (immutable)
  → construct RuntimeEngine(compiled_graph, providers, plugin registry, MCP manager, ...)
```

The `RuntimeEngine` and the `CompiledAgentGraph` are built **once** and shared,
read-only, across all requests. Expensive setup (parsing the graph, preparing
provider clients, loading and parsing prompt templates, opening MCP connections)
happens here — not on the request path.

The engine is driven through an **async context manager**. `build(spec)`
performs the one-time setup — connecting to MCP servers (one
`MultiServerMCPClient` per configured server via `langchain-mcp-adapters`),
discovering their tools, loading tool/resolver plugins, and compiling the graph.
Unreachable MCP servers are logged as a warning and skipped, so local tools keep
working. `close()` (called on context exit) releases those resources.

```python
async with LangGraphEngine(base_dir) as engine:
    await engine.build(spec)
    result = await engine.run(message)
```

Keeping `build`, runs, and `close` inside one `async with` keeps MCP clients and
their async resources (e.g. AnyIO task groups) on a single long-lived loop.

### Per request (many)

```
request arrives
  → create per-request state (message, trace)
  → filter protected nodes through access plugin (per orchestrator)
  → execute root as a supervisor agent (children exposed as tools)
  → resolve prompt variables through resolver plugins
  → render prompts (per request)
  → orchestrators synthesise; leaf agents execute their tools
  → call configured tool plugins/MCP servers as needed
  → finalize response + trace
```

Everything that varies between requests lives on the `ExecutionContext`.

---

## Conversation history composition

`agent_engine` remains stateless between completed runs. Its `run` and `stream`
ports accept optional typed prior user/assistant messages, convert them to
provider-native messages in order, and append the latest user message. The
engine never stores that history on `RuntimeEngine`.

`agent_manager` owns conversation history through its repository. Before a
turn, `ConversationService` loads prior messages for the session and appends the
new user message. It passes the prior turns through the engine's structured
history argument. For streaming runs, it persists a final assistant response
before exposing that terminal event to the HTTP consumer. Disconnecting before
a terminal event instead closes the engine stream, cancels the graph producer,
and retains only the user message plus authoritative cancelled-run metadata.
During a model tool loop, the tool-call message and matching tool result remain
structured and ordered for the provider. A run paused for tool
approval stores its continuation in the LangGraph checkpoint keyed by `run_id`;
session-wide approval grants live in a separate approval repository keyed by
session and tool identity. Model responses and completed leaf-tool outcomes are
checkpointed as individual steps, so resume restores the reviewed tool request
and reuses earlier results instead of regenerating calls or repeating side
effects. This also applies to nested agents and sequential approvals. Resumed
decisions carry the approval id and stable invocation id, which covers the run,
agent path, provider/server, tool name, and raw arguments without the provider's
transient call id. The tool approval gate rejects missing or mismatched bindings
before execution; the displayed and persisted approval arguments stay masked.
The claimed decision also overrides any auto-mode or session-grant change while
the invocation waited. Its binding is request-local and consumed only at the
matching interrupt; later calls return to normal policy evaluation.
Completing a replay with that binding still unconsumed is a mismatch: for
example, a rebuilt graph cannot silently remove the reviewed tool and report its
approval as applied.
Restored steps also validate the node/input or invocation identity, so a changed
route cannot reuse another agent's positional checkpoint. A mismatch aborts the
run, rejects its claimed approval, and removes any newly created pending approval.
The graph state carries an internal continuation version. Resuming a suspended
run created before these checkpointed substeps existed fails closed before model
or tool execution, because replaying that older format could repeat side effects.
Execution limits still account for restored calls before their cached outcomes
are returned; checkpointing does not reset the run's allowed tool budget.
Restored model steps emit no new answer deltas or token usage. The existing tool
execution ledger still protects calls whose step started but did not finish.
Normal stream completion does not cancel a suspended
approval. An explicit owner cancellation atomically rejects the pending
approval, transitions the run to terminal `CANCELLED`, and prevents later
resume.

After a decision claims an approval, the same run transitions
`PENDING_APPROVAL → RESUMING → RUNNING`. Streamed resume and initial execution
both use the engine's owned graph-task channel. Closing either consumer cancels
and awaits that task, records available partial token usage, and atomically
transitions the run to `CANCELLED`; only an observed finalized event is
persisted as a successful assistant response.

Starting a new session therefore selects both an empty conversation history and
an empty approval scope without putting either kind of request state on the
long-lived engine.

---

## Streaming Runs

`Engine.run(message)` remains the non-streaming API and returns a completed
`RunResult`.

`Engine.stream(message)` is the streaming API. It executes the same compiled
graph and yields platform-level `RunStreamEvent` values such as
`answer_delta`, `route`, and `final`. Callers consume these events instead of
LangChain-specific chunks. The final event preserves the completed answer,
route, system name, and runtime-generated tool usage summary.

**Only the root orchestrator streams its answer to the user.** Inner agents and
nested orchestrators run silently — their tokens are not emitted as
`answer_delta`, so the stream reflects the final synthesised answer, not the
intermediate reasoning of children.

Streaming uses the same async-context lifecycle:

```python
async with LangGraphEngine(base_dir) as engine:
    await engine.build(spec)
    async for event in engine.stream(message):
        ...
```

`agentctl run --stream` uses this flow and writes `answer_delta` content to
stdout as chunks arrive. It does not print the completed answer again after the
final event.

The engine holds a terminal event until its consumer accepts it and lifecycle
finalization succeeds. This prevents a completed producer with a queued final
answer from racing a disconnect into conversation history that has neither an
assistant response nor a cancelled run. On cancellation, token usage already
reported by the provider is persisted before the run transitions to
`CANCELLED`; usage that a provider reports only at normal completion is not
available after an early abort.

---

## RuntimeEngine

- **Long-lived**, created at startup, one per process.
- **Holds shared, read-only collaborators**:
  - the `CompiledAgentGraph`
  - the prompt template loader (raw-template cache)
  - the prompt renderer
  - the resolver plugin registry
  - the access plugin adapter
  - the tool registry
  - the MCP manager
  - the LLM provider registry
  - observability/tracing infrastructure
  - runtime configuration
- **Holds NO request state.** No "current user", no "current request", no
  per-request mutable fields.
- **Thread/async-safe by construction**: because it carries no request state,
  many requests can use it concurrently.

What does **not** belong on `RuntimeEngine`:

- The current `user_id`, `tenant_id`, `organization_id`, identity, or permissions.
- The current request message.
- Rendered prompts for a specific request.
- The current trace path or tool results.
- Any mutable counter/buffer scoped to a single request.

## ExecutionContext

- **Per request**, created fresh, never reused.
- **Carries everything request-scoped**:
  - `request_id`
  - `message`
  - request headers and request data
  - optional identity/context derived by plugins
  - resolved context values
  - the selected node path
  - rendered prompt values
  - temporary tool results
  - trace events
  - errors
- **Passed explicitly** down through routing, context resolution, prompt
  rendering, plugin calls, and tool execution.
- **Discarded** when the request completes.

> **Simple rule:** `RuntimeEngine` = *how the system works*;
> `ExecutionContext` = *what is happening in this specific request*. Never store
> per-request state on `RuntimeEngine`.

---

## Why this matters

- **Correctness:** sharing request state on a long-lived engine causes data to
  leak between concurrent requests (e.g. one tenant seeing another's context).
- **Performance:** recompiling the graph or reconstructing the engine per
  request wastes the expensive work that should happen once.
- **Concurrency:** a stateless engine + per-request context is safe to run
  concurrently without locks around request data.

---

## Validation checklist (for any runtime change)

- [ ] `RuntimeEngine` construction happens only at startup paths (API/CLI boot),
      never inside a request handler.
- [ ] No request-scoped field is added to `RuntimeEngine` or
      `CompiledAgentGraph`.
- [ ] All request-scoped data lives on `ExecutionContext`.
- [ ] The compiled graph is treated as immutable at runtime.
- [ ] Tests cover concurrent requests not leaking state.
- [ ] `make check` passes.
