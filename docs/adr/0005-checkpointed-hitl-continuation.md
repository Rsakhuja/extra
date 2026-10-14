# ADR 0005 — Checkpoint model and tool steps so resume runs the reviewed call

- **Status:** Proposed
- **Date:** 2026-10-14

## Context

A run suspended for tool approval resumed by re-executing its LangGraph node
from the start. The node invoked the model again, and the resume command carried
only `{"decision": ...}`. A non-deterministic model could therefore propose a
different tool, agent, or arguments than the human reviewed, and the positional
decision would still apply to it. Earlier tool calls in the same node were
protected from repeated side effects only by the idempotency ledger, and
restored turns reported their token usage a second time.

## Decision

The engine replays a suspended node from saved steps and binds each decision to
one invocation.

- `run_tool_loop` runs every model response as a LangGraph `@task`, and
  `ToolInvoker` runs each leaf tool's gate and execution as a `@task`. On resume
  LangGraph returns the saved values, so the reviewed tool request is never
  regenerated and completed outcomes, denials, and session grants are not
  repeated. Execution limits are registered before a step is restored, so the
  run's tool budget is unchanged.
- Restored steps validate their identity. A model step carries the node path,
  its non-system input, and its turn. A tool step carries the stable
  `tool_call_id`. A mismatch, for example after route filtering changes the
  replay order, raises `CheckpointReplayMismatch`.
- Resume sends `{"decision", "approval_id", "tool_call_id"}`. The stable id
  covers the run, agent path, provider/server, tool name, and raw arguments, but
  not the provider's transient call id. The approval gate raises
  `ApprovalInvocationMismatch` before execution when the binding is missing or
  different. Masked arguments are never used for binding, because distinct
  secrets share the same redacted display.
- Either mismatch fails the run, rejects the claimed approval, and cancels any
  pending approval created during the failed replay. Orchestrators re-raise both
  errors from nested children instead of turning them into tool text.
- Each resume request carries its claimed binding in a context variable. The
  first unfinished tool step validates that binding before policy evaluation;
  its existing `ToolInvocation.approval_id` marks the claimed decision as
  authoritative over auto mode or session grants. The matching interrupt consumes
  the binding once, so later calls still follow normal policy. Separate requests
  never share this state, and completed steps do not consume a new decision.
- Approval cleanup is best effort. Repository failures are chained or logged
  by type and cannot replace the original replay/consent mismatch.
- Graph state carries `continuation_version`. Resuming a checkpoint without the
  current version fails closed before any model or tool step runs, because an
  older continuation has no saved steps and replaying it could repeat side
  effects.
- Outside a LangGraph runtime (`in_graph_run()` is false) the loop and invoker
  run directly, without tasks.

## Contract changes

- The runtime dependency floor is `langgraph>=1.0,<2.0`, for functional tasks
  inside `StateGraph` nodes and `langgraph.runtime.get_runtime`.
- `GraphState` adds the internal `continuation_version` key.
- The internal resume payload adds `approval_id` and `tool_call_id`.
- New errors: `ApprovalInvocationMismatch` (an `ApprovalError`) and
  `CheckpointReplayMismatch` (a `RuntimeError`).

There is no YAML or HTTP request-shape change.

## Consequences

- Runs that are suspended when this version is deployed cannot be resumed; they
  fail and their approval is rejected. Decide or cancel pending approvals before
  upgrading.
- The checkpoint stores the model's raw tool arguments and the tool results for
  every step. It must be treated as a sensitive store. Only the `ApprovalRecord`
  and the approval payload are masked.
- Checkpoints grow with each model turn and tool result in a suspended run.
- Restored model steps emit no new answer deltas or token usage, so a resumed run
  no longer double-counts tokens.
- The idempotency ledger still protects a call whose step started but did not
  finish.
