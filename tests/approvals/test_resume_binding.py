"""Approval replay must preserve the reviewed action and completed side effects."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall as LCToolCall

from agent_engine.approvals.approval_manager import ApprovalManager
from agent_engine.approvals.approval_provider import ApprovalRequest
from agent_engine.approvals.errors import (
    ApprovalAlreadyProcessed,
    ApprovalError,
    ApprovalInvocationMismatch,
)
from agent_engine.approvals.in_memory_approval_repository import InMemoryApprovalRepository
from agent_engine.approvals.invocation import ToolInvocation
from agent_engine.approvals.models import ApprovalStatus
from agent_engine.core.execution import ExecutionPolicy
from agent_engine.core.spec import AgentSpec
from agent_engine.engine.langgraph.approval_provider import InterruptApprovalProvider
from agent_engine.engine.langgraph.engine import LangGraphEngine
from agent_engine.runs.in_memory import InMemoryRunRepository
from agent_engine.runtime.hooks import RunContext, current_run_context
from agent_engine.tool_usage.in_memory import InMemoryToolUsageRepository
from agent_engine.tool_usage.models import ToolInvocationKind, stable_tool_call_id
from tests.approvals.test_engine_hitl import FakeChatModel, _multi_tool_spec
from tests.engine.test_engine_flow import orchestrator, system
from tests.engine.test_execution_limits import ToolThenAnswerModel


class DriftingModel(FakeChatModel):
    """Change a tool request if the model is asked to regenerate it on resume."""

    def __init__(self, before: str, after: str, *, change_arguments: bool = False) -> None:
        super().__init__()
        self.before = before
        self.after = after
        self.change_arguments = change_arguments
        self.proposals = 0

    def bind_tools(self, tools: list[Any]) -> DriftingModel:
        return self

    def _respond(self, messages: list[Any]) -> AIMessage:
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="done")
        self.proposals += 1
        return AIMessage(
            content="",
            tool_calls=[
                LCToolCall(
                    name=self.before if self.proposals == 1 else self.after,
                    args={
                        "message": "changed"
                        if self.change_arguments and self.proposals > 1
                        else "reviewed"
                    },
                    id=f"transient-{self.proposals}",
                )
            ],
        )


class SequentialDriftingModel(DriftingModel):
    def _respond(self, messages: list[Any]) -> AIMessage:
        completed = sum(isinstance(message, ToolMessage) for message in messages)
        if completed == 2:
            return AIMessage(content="done")
        self.proposals += 1
        name = self.before if completed == 0 else self.after
        return AIMessage(
            content="",
            tool_calls=[
                LCToolCall(
                    name=name,
                    args={"message": f"reviewed-{self.proposals}"},
                    id=f"transient-{self.proposals}",
                )
            ],
        )


class SiblingModel(DriftingModel):
    def _respond(self, messages: list[Any]) -> AIMessage:
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="done")
        self.proposals += 1
        return AIMessage(
            content="",
            tool_calls=[
                LCToolCall(
                    name="create_ticket",
                    args={"message": f"ticket-{self.proposals}"},
                    id=f"ticket-{self.proposals}",
                ),
                LCToolCall(
                    name="send_email", args={"message": "reviewed"}, id=f"email-{self.proposals}"
                ),
            ],
        )


def write_tool(base_dir: Path, name: str, output: Path) -> None:
    path = base_dir / "plugins" / "tools" / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"def {name}(message: str) -> str:\n"
        f"    with open({str(output)!r}, 'a') as output:\n"
        f"        output.write({name!r} + ':' + message + '\\n')\n"
        "    return message\n",
        encoding="utf-8",
    )


async def pending_from_stream(events: AsyncIterator[Any]) -> Any:
    collected = [event async for event in events]
    assert collected[-1].type == "pending_approval"
    return collected[-1]


@pytest.mark.parametrize("initial_stream", [False, True])
@pytest.mark.parametrize("resumed_stream", [False, True])
@pytest.mark.parametrize("change", ["arguments", "tool", "agent"])
async def test_resume_executes_only_the_reviewed_invocation(
    tmp_path: Path, initial_stream: bool, resumed_stream: bool, change: str
) -> None:
    output = tmp_path / "executed.log"
    write_tool(tmp_path, "send_email", output)
    write_tool(tmp_path, "archive_email", output)
    model = DriftingModel(
        "send_email",
        "archive_email" if change == "tool" else "send_email",
        change_arguments=change == "arguments",
    )
    spec = _multi_tool_spec("send_email", "archive_email")
    root = DriftingModel("writer", "other")
    if change == "agent":
        writer = spec.graph
        other = replace(writer, node=replace(writer.node, id="other", name="other"))
        spec = system(orchestrator("root", [writer, other]))

    builds = 0

    def factory(*_: Any, **__: Any) -> BaseChatModel:
        # GraphBuilder builds the root before its children.
        nonlocal builds
        builds += 1
        if change == "agent" and builds == 1:
            return cast(BaseChatModel, root)
        return cast(BaseChatModel, model)

    approvals = InMemoryApprovalRepository()
    usage = InMemoryToolUsageRepository()
    manager = ApprovalManager(run_repository=InMemoryRunRepository(), approval_repository=approvals)
    async with LangGraphEngine(
        tmp_path, model_factory=factory, approval_manager=manager, tool_usage_repository=usage
    ) as engine:
        await engine.build(spec)
        if initial_stream:
            pending = await pending_from_stream(
                engine.stream("review", context=RunContext(run_id="r"))
            )
        else:
            first = await engine.run("review", context=RunContext(run_id="r"))
            pending = first.pending_approval
        assert pending is not None
        assert pending.tool_name == "send_email"
        assert pending.agent_id == "writer"
        assert pending.arguments == {"message": "reviewed"}
        assert not output.exists()

        if resumed_stream:
            events = [
                event
                async for event in engine.resume_stream("r", pending.approval_id, "allow once")
            ]
            assert events[-1].type == "final"
        else:
            result = await engine.resume("r", pending.approval_id, "allow once")
            assert result.status == "completed"
        stored = [
            record
            for record in await usage.list_for_run("r")
            if record.call.kind == ToolInvocationKind.TOOL
        ]
        approval = await approvals.get(pending.approval_id)

    assert output.read_text().splitlines() == ["send_email:reviewed"]
    assert approval is not None
    assert len(stored) == 1
    assert stored[0].call.tool_call_id == approval.tool_call_id
    assert stored[0].call.agent_id == approval.agent_id == "writer"
    assert model.proposals == 1


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("decision", ["allow once", "allow for session", "deny"])
async def test_sequential_approvals_do_not_replay_or_rebind_earlier_steps(
    tmp_path: Path, nested: bool, decision: str
) -> None:
    output = tmp_path / "executed.log"
    for name in ("send_email", "archive_email"):
        write_tool(tmp_path, name, output)
    model = SequentialDriftingModel("send_email", "archive_email")
    spec = _multi_tool_spec("send_email", "archive_email")
    if nested:
        spec = system(orchestrator("root", [orchestrator("inner", [spec.graph])]))

    builds = 0

    def factory(*_: Any, **__: Any) -> BaseChatModel:
        nonlocal builds
        builds += 1
        return cast(BaseChatModel, model if builds == (3 if nested else 1) else FakeChatModel())

    async with LangGraphEngine(tmp_path, model_factory=factory) as engine:
        await engine.build(spec)
        context = RunContext(run_id="sequential", conversation_id="session", user_id="owner")
        first = await pending_from_stream(engine.stream("review", context=context))
        second = await pending_from_stream(
            engine.resume_stream(
                "sequential",
                first.approval_id,
                decision,
                caller_user_id="owner",
                caller_session_id="session",
            )
        )
        assert second.tool_name == "archive_email"
        assert second.arguments == {"message": "reviewed-2"}
        events = [
            event
            async for event in engine.resume_stream(
                "sequential",
                second.approval_id,
                "allow once",
                caller_user_id="owner",
                caller_session_id="session",
            )
        ]
        assert events[-1].type == "final"

    expected = [] if decision == "deny" else ["send_email:reviewed-1"]
    assert output.read_text().splitlines() == [*expected, "archive_email:reviewed-2"]
    assert model.proposals == 2


@pytest.mark.parametrize("resumed_stream", [False, True])
async def test_completed_session_allowed_sibling_is_not_reexecuted(
    tmp_path: Path, resumed_stream: bool
) -> None:
    output = tmp_path / "executed.log"
    for name in ("create_ticket", "send_email"):
        write_tool(tmp_path, name, output)
    model = SiblingModel("create_ticket", "send_email")
    async with LangGraphEngine(
        tmp_path, model_factory=lambda *a, **k: cast(BaseChatModel, model)
    ) as engine:
        await engine.build(_multi_tool_spec("create_ticket", "send_email"))
        first = await engine.run(
            "review", context=RunContext(run_id="siblings", conversation_id="s", user_id="u")
        )
        assert first.pending_approval is not None
        second = await engine.resume(
            "siblings",
            first.pending_approval.approval_id,
            "allow for session",
            caller_user_id="u",
            caller_session_id="s",
        )
        assert second.pending_approval is not None
        assert second.pending_approval.tool_name == "send_email"
        assert output.read_text().splitlines() == ["create_ticket:ticket-1"]
        if resumed_stream:
            events = [
                event
                async for event in engine.resume_stream(
                    "siblings",
                    second.pending_approval.approval_id,
                    "allow once",
                    caller_user_id="u",
                    caller_session_id="s",
                )
            ]
            assert events[-1].type == "final"
        else:
            result = await engine.resume(
                "siblings",
                second.pending_approval.approval_id,
                "allow once",
                caller_user_id="u",
                caller_session_id="s",
            )
            assert result.status == "completed"
    assert output.read_text().splitlines() == ["create_ticket:ticket-1", "send_email:reviewed"]


@pytest.mark.parametrize(
    "change", ["arguments", "secret", "agent", "provider", "server", "tool", "unbound"]
)
async def test_execution_gate_rejects_a_decision_for_a_different_invocation(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    manager = ApprovalManager(
        run_repository=InMemoryRunRepository(), approval_repository=InMemoryApprovalRepository()
    )
    provider = InterruptApprovalProvider(manager)
    original = ToolInvocation(
        tool_call_id="unused",
        agent_id="writer",
        tool_name="send_email",
        arguments={"message": "reviewed", "token": "first-secret"},
    )
    changes: dict[str, dict[str, Any]] = {
        "arguments": {"arguments": {"message": "changed", "token": "first-secret"}},
        "secret": {"arguments": {"message": "reviewed", "token": "changed-secret"}},
        "agent": {"agent_id": "other"},
        "provider": {"provider": "mcp"},
        "server": {"server_id": "other"},
        "tool": {"tool_name": "archive_email"},
        "unbound": {},
    }

    def identified(inv: ToolInvocation) -> ToolInvocation:
        return replace(
            inv,
            tool_call_id=stable_tool_call_id(
                "r", inv.agent_id, inv.provider, inv.server_id, inv.tool_name, inv.arguments
            ),
        )

    original = identified(original)
    changed = identified(replace(original, **changes[change]))
    await manager.create_pending(
        run_id="r",
        thread_id="r",
        approval_id="approved",
        agent_id="writer",
        tool_name="send_email",
        tool_call_id=original.tool_call_id,
        provider="local",
        description="review",
        arguments={"message": "reviewed", "token": "***"},
    )
    await manager.claim(run_id="r", approval_id="approved")
    payload = (
        {"decision": "allow_once"}
        if change == "unbound"
        else {
            "decision": "allow_once",
            "approval_id": "approved",
            "tool_call_id": original.tool_call_id,
        }
    )
    monkeypatch.setattr(
        "agent_engine.engine.langgraph.approval_provider.interrupt", lambda _: payload
    )
    token = current_run_context.set(RunContext(run_id="r"))
    try:
        with pytest.raises(ApprovalError, match="invocation"):
            await provider.request_decision(
                ApprovalRequest(changed, "review", {"message": "reviewed", "token": "***"})
            )
        assert await manager.get_pending("r") is None
    finally:
        current_run_context.reset(token)


@pytest.mark.parametrize("limit", ["max_tool_calls", "max_tool_calls_per_agent"])
@pytest.mark.parametrize("resumed_stream", [False, True])
async def test_restored_tool_calls_still_count_toward_run_limits(
    tmp_path: Path, limit: str, resumed_stream: bool
) -> None:
    output = tmp_path / "executed.log"
    write_tool(tmp_path, "send_email", output)
    model = ToolThenAnswerModel(3, lambda i: {"message": f"message-{i}"})
    policy = (
        ExecutionPolicy(max_tool_calls=2)
        if limit == "max_tool_calls"
        else ExecutionPolicy(max_tool_calls_per_agent=2)
    )
    spec = replace(_multi_tool_spec("send_email"), execution=policy)
    async with LangGraphEngine(
        tmp_path, model_factory=lambda *a, **k: cast(BaseChatModel, model)
    ) as engine:
        await engine.build(spec)
        first = await engine.run("review", context=RunContext(run_id="limited"))
        assert first.pending_approval is not None
        second = await engine.resume("limited", first.pending_approval.approval_id, "allow once")
        assert second.pending_approval is not None
        if resumed_stream:
            events = [
                event
                async for event in engine.resume_stream(
                    "limited", second.pending_approval.approval_id, "allow once"
                )
            ]
            assert events[-1].type == "final"
        else:
            result = await engine.resume(
                "limited", second.pending_approval.approval_id, "allow once"
            )
            assert result.status == "completed"
        assert await engine.get_pending_approval("limited") is None
    assert output.read_text().splitlines() == ["send_email:message-0", "send_email:message-1"]


async def test_replay_does_not_restore_another_agents_step_when_a_child_is_filtered(
    tmp_path: Path,
) -> None:
    output = tmp_path / "executed.log"
    hidden = tmp_path / "hide_first"
    write_tool(tmp_path, "send_email", output)
    access = tmp_path / "plugins/access.py"
    access.write_text(
        "from pathlib import Path\n"
        "class AccessResolver:\n"
        "    def can_access(self, ctx: dict, node_id: str) -> bool:\n"
        f"        return node_id != 'first' or not Path({str(hidden)!r}).exists()\n",
        encoding="utf-8",
    )
    writer = _multi_tool_spec("send_email").graph
    assert isinstance(writer.node, AgentSpec)
    first = replace(
        writer, node=replace(writer.node, id="first", name="first", protected=True, auto_mode=True)
    )
    spec = system(orchestrator("root", [first, writer]))

    class ParentModel(SiblingModel):
        def _respond(self, messages: list[Any]) -> AIMessage:
            response = super()._respond(messages)
            if response.tool_calls:
                response.tool_calls[0]["name"] = "first"
                response.tool_calls[1]["name"] = "writer"
            return response

    models = [
        ParentModel("first", "writer"),
        DriftingModel("send_email", "send_email"),
        DriftingModel("send_email", "send_email"),
    ]
    async with LangGraphEngine(
        tmp_path, model_factory=lambda *a, **k: cast(BaseChatModel, models.pop(0))
    ) as engine:
        await engine.build(spec)
        pending = await engine.run("review", context=RunContext(run_id="filtered"))
        assert pending.pending_approval is not None
        assert output.read_text().splitlines() == ["send_email:reviewed"]
        hidden.touch()
        with pytest.raises(RuntimeError, match="checkpoint"):
            await engine.resume("filtered", pending.pending_approval.approval_id, "allow once")
        assert await engine.get_run_status("filtered") == "failed"
    assert output.read_text().splitlines() == ["send_email:reviewed"]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("resumed_stream", [False, True])
async def test_binding_failure_rejects_the_approval_and_aborts_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool, resumed_stream: bool
) -> None:
    from agent_engine.engine.langgraph import approval_provider

    real_interrupt = approval_provider.interrupt

    def corrupt_resume(payload: dict[str, Any]) -> dict[str, Any]:
        resume = real_interrupt(payload)
        return {**resume, "tool_call_id": "different-invocation"}

    monkeypatch.setattr(approval_provider, "interrupt", corrupt_resume)
    output = tmp_path / "executed.log"
    write_tool(tmp_path, "send_email", output)
    spec = _multi_tool_spec("send_email")
    if nested:
        spec = system(orchestrator("root", [spec.graph]))
    manager = ApprovalManager(
        run_repository=InMemoryRunRepository(), approval_repository=InMemoryApprovalRepository()
    )
    async with LangGraphEngine(
        tmp_path,
        model_factory=lambda *a, **k: cast(BaseChatModel, FakeChatModel()),
        approval_manager=manager,
    ) as engine:
        await engine.build(spec)
        first = await engine.run("review", context=RunContext(run_id="bad-binding"))
        assert first.pending_approval is not None
        approval_id = first.pending_approval.approval_id
        with pytest.raises(ApprovalError, match="invocation"):
            if resumed_stream:
                _ = [
                    event
                    async for event in engine.resume_stream(
                        "bad-binding", approval_id, "allow once"
                    )
                ]
            else:
                await engine.resume("bad-binding", approval_id, "allow once")
        assert await engine.get_run_status("bad-binding") == "failed"
        assert await engine.get_pending_approval("bad-binding") is None
        approval = await manager.get_approval("bad-binding", approval_id)
        assert approval.status == ApprovalStatus.REJECTED
    assert not output.exists()


@pytest.mark.parametrize("resumed_stream", [False, True])
async def test_removed_pending_tool_cannot_complete_without_consuming_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resumed_stream: bool
) -> None:
    from agent_engine.engine.langgraph.checkpointing import CheckpointProviderFactory

    handle = CheckpointProviderFactory().create(None)
    monkeypatch.setattr(CheckpointProviderFactory, "create", lambda *a, **k: handle)
    output = tmp_path / "executed.log"
    write_tool(tmp_path, "send_email", output)
    spec = _multi_tool_spec("send_email")
    manager = ApprovalManager(
        run_repository=InMemoryRunRepository(), approval_repository=InMemoryApprovalRepository()
    )

    async with LangGraphEngine(
        tmp_path,
        model_factory=lambda *a, **k: cast(BaseChatModel, FakeChatModel()),
        approval_manager=manager,
    ) as first_engine:
        await first_engine.build(spec)
        first = await first_engine.run("review", context=RunContext(run_id="removed"))
        assert first.pending_approval is not None

    assert isinstance(spec.graph.node, AgentSpec)
    without_tool = replace(spec, graph=replace(spec.graph, node=replace(spec.graph.node, tools=())))
    async with LangGraphEngine(
        tmp_path,
        model_factory=lambda *a, **k: cast(BaseChatModel, FakeChatModel()),
        approval_manager=manager,
    ) as second_engine:
        await second_engine.build(without_tool)
        approval_id = first.pending_approval.approval_id
        with pytest.raises(ApprovalInvocationMismatch):
            if resumed_stream:
                _ = [
                    event
                    async for event in second_engine.resume_stream(
                        "removed", approval_id, "allow once"
                    )
                ]
            else:
                await second_engine.resume("removed", approval_id, "allow once")
        assert await second_engine.get_run_status("removed") == "failed"
        record = await manager.get_approval("removed", approval_id)
        assert record.status == ApprovalStatus.REJECTED
    assert not output.exists()


@pytest.mark.parametrize("resumed_stream", [False, True])
async def test_legacy_continuation_is_rejected_before_any_replayed_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resumed_stream: bool
) -> None:
    from agent_engine.approvals.in_memory_session_approval_repository import (
        InMemorySessionApprovalRepository,
    )
    from agent_engine.approvals.invocation import SessionApprovalKey
    from agent_engine.engine.langgraph import engine as engine_module
    from agent_engine.engine.langgraph.execution import model_loop
    from agent_engine.engine.langgraph.tools import tool_invoker

    original_input = engine_module._initial_state

    def legacy_input(*args: Any, **kwargs: Any) -> dict[str, Any]:
        state = original_input(*args, **kwargs)
        state.pop("continuation_version", None)
        return state

    monkeypatch.setattr(engine_module, "_initial_state", legacy_input)
    output = tmp_path / "executed.log"
    for name in ("create_ticket", "send_email"):
        write_tool(tmp_path, name, output)
    sessions = InMemorySessionApprovalRepository()
    await sessions.allow(
        SessionApprovalKey(
            "session",
            "writer",
            "local:local:create_ticket",
            system_namespace="hitl",
            user_id="owner",
        )
    )
    model = SiblingModel("create_ticket", "send_email")
    manager = ApprovalManager(
        run_repository=InMemoryRunRepository(), approval_repository=InMemoryApprovalRepository()
    )
    async with LangGraphEngine(
        tmp_path,
        model_factory=lambda *a, **k: cast(BaseChatModel, model),
        approval_manager=manager,
        session_approval_repository=sessions,
    ) as engine:
        await engine.build(_multi_tool_spec("create_ticket", "send_email"))
        # Old continuations have neither a version nor checkpointed substeps.
        with monkeypatch.context() as legacy:
            legacy.setattr(model_loop, "in_graph_run", lambda: False)
            legacy.setattr(tool_invoker, "in_graph_run", lambda: False)
            first = await engine.run(
                "review",
                context=RunContext(run_id="legacy", conversation_id="session", user_id="owner"),
            )
        assert first.pending_approval is not None
        assert first.pending_approval.tool_name == "send_email"
        assert output.read_text().splitlines() == ["create_ticket:ticket-1"]
        with pytest.raises(RuntimeError, match="checkpoint"):
            if resumed_stream:
                _ = [
                    event
                    async for event in engine.resume_stream(
                        "legacy",
                        first.pending_approval.approval_id,
                        "allow once",
                        caller_user_id="owner",
                        caller_session_id="session",
                    )
                ]
            else:
                await engine.resume(
                    "legacy",
                    first.pending_approval.approval_id,
                    "allow once",
                    caller_user_id="owner",
                    caller_session_id="session",
                )
        assert await engine.get_run_status("legacy") == "failed"
        approval = await manager.get_approval("legacy", first.pending_approval.approval_id)
        assert approval.status == ApprovalStatus.REJECTED
    assert model.proposals == 1
    assert output.read_text().splitlines() == ["create_ticket:ticket-1"]


@pytest.mark.parametrize("failure_kind", ["approval", "backend"])
async def test_gate_cleanup_failure_still_reports_the_binding_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    manager = ApprovalManager(
        run_repository=InMemoryRunRepository(), approval_repository=InMemoryApprovalRepository()
    )
    reviewed = ToolInvocation(
        tool_call_id="reviewed-call", agent_id="writer", tool_name="send_email", arguments={}
    )
    await manager.create_pending(
        run_id="r",
        thread_id="r",
        approval_id="approved",
        agent_id="writer",
        tool_name="send_email",
        tool_call_id=reviewed.tool_call_id,
        provider="local",
        description="review",
        arguments={},
    )
    await manager.claim(run_id="r", approval_id="approved")

    async def lost_race(**kwargs: Any) -> None:
        if failure_kind == "backend":
            raise RuntimeError("approval storage unavailable")
        raise ApprovalAlreadyProcessed(kwargs["approval_id"], "rejected")

    monkeypatch.setattr(manager, "cancel_pending", lost_race)
    monkeypatch.setattr(
        "agent_engine.engine.langgraph.approval_provider.interrupt",
        lambda _: {
            "decision": "allow_once",
            "approval_id": "approved",
            "tool_call_id": reviewed.tool_call_id,
        },
    )
    token = current_run_context.set(RunContext(run_id="r"))
    try:
        with pytest.raises(ApprovalInvocationMismatch) as raised:
            await InterruptApprovalProvider(manager).request_decision(
                ApprovalRequest(replace(reviewed, tool_call_id="changed-call"), "review", {})
            )
    finally:
        current_run_context.reset(token)
    expected = ApprovalAlreadyProcessed if failure_kind == "approval" else RuntimeError
    assert isinstance(raised.value.__cause__, expected)


@pytest.mark.parametrize("resumed_stream", [False, True])
@pytest.mark.parametrize("failure_kind", ["approval", "backend"])
async def test_engine_cleanup_failure_still_reports_the_binding_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resumed_stream: bool, failure_kind: str
) -> None:
    from agent_engine.engine.langgraph import approval_provider

    real_interrupt = approval_provider.interrupt

    def corrupt_resume(payload: dict[str, Any]) -> dict[str, Any]:
        return {**real_interrupt(payload), "tool_call_id": "different-invocation"}

    monkeypatch.setattr(approval_provider, "interrupt", corrupt_resume)
    output = tmp_path / "executed.log"
    write_tool(tmp_path, "send_email", output)
    manager = ApprovalManager(
        run_repository=InMemoryRunRepository(), approval_repository=InMemoryApprovalRepository()
    )

    async def lost_race(approval_id: str, *, approved: bool) -> None:
        if failure_kind == "backend":
            raise RuntimeError("approval storage unavailable")
        raise ApprovalAlreadyProcessed(approval_id, "rejected")

    async with LangGraphEngine(
        tmp_path,
        model_factory=lambda *a, **k: cast(BaseChatModel, FakeChatModel()),
        approval_manager=manager,
    ) as engine:
        await engine.build(_multi_tool_spec("send_email"))
        first = await engine.run("review", context=RunContext(run_id="cleanup"))
        assert first.pending_approval is not None
        monkeypatch.setattr(manager, "finalize", lost_race)
        with pytest.raises(ApprovalInvocationMismatch):
            if resumed_stream:
                _ = [
                    event
                    async for event in engine.resume_stream(
                        "cleanup", first.pending_approval.approval_id, "allow once"
                    )
                ]
            else:
                await engine.resume("cleanup", first.pending_approval.approval_id, "allow once")
        assert await engine.get_run_status("cleanup") == "failed"
    assert not output.exists()
