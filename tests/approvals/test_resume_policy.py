"""Claimed decisions remain authoritative when approval policy changes."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel

from agent_engine.approvals.approval_manager import ApprovalManager
from agent_engine.approvals.errors import ApprovalInvocationMismatch
from agent_engine.approvals.in_memory_approval_repository import InMemoryApprovalRepository
from agent_engine.approvals.models import ApprovalStatus
from agent_engine.core.spec import AgentSpec
from agent_engine.engine.langgraph.engine import LangGraphEngine
from agent_engine.runs.in_memory import InMemoryRunRepository
from agent_engine.runtime.hooks import RunContext
from tests.approvals.test_engine_hitl import FakeChatModel, _multi_tool_spec
from tests.approvals.test_resume_binding import write_tool
from tests.engine.test_engine_flow import orchestrator, system


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("resumed_stream", [False, True])
async def test_pending_denial_is_not_bypassed_by_a_new_session_grant(
    tmp_path: Path, nested: bool, resumed_stream: bool
) -> None:
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
        first = await engine.run(
            "first", context=RunContext(run_id="denied", conversation_id="s", user_id="u")
        )
        second = await engine.run(
            "second", context=RunContext(run_id="granted", conversation_id="s", user_id="u")
        )
        assert first.pending_approval is not None
        assert second.pending_approval is not None
        await engine.resume(
            "granted",
            second.pending_approval.approval_id,
            "allow for session",
            caller_user_id="u",
            caller_session_id="s",
        )
        assert output.read_text().splitlines() == ["send_email:go"]
        if resumed_stream:
            events = [
                event
                async for event in engine.resume_stream(
                    "denied",
                    first.pending_approval.approval_id,
                    "deny",
                    caller_user_id="u",
                    caller_session_id="s",
                )
            ]
            assert events[-1].type == "final"
        else:
            result = await engine.resume(
                "denied",
                first.pending_approval.approval_id,
                "deny",
                caller_user_id="u",
                caller_session_id="s",
            )
            assert result.status == "completed"
        record = await manager.get_approval("denied", first.pending_approval.approval_id)
        assert record.status == ApprovalStatus.REJECTED
        assert output.read_text().splitlines() == ["send_email:go"]
        # The denial applies to its one pending call, not the other run's grant.
        later = await engine.run(
            "later", context=RunContext(run_id="later", conversation_id="s", user_id="u")
        )
        assert later.status == "completed"
    assert output.read_text().splitlines() == ["send_email:go", "send_email:go"]


@pytest.mark.parametrize("decision", ["deny", "allow once"])
@pytest.mark.parametrize("resumed_stream", [False, True])
async def test_pending_decision_survives_an_auto_mode_change_on_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: str, resumed_stream: bool
) -> None:
    from agent_engine.engine.langgraph.checkpointing import CheckpointProviderFactory
    from tests.approvals.test_engine_hitl import ChainedApprovalModel

    handle = CheckpointProviderFactory().create(None)
    monkeypatch.setattr(CheckpointProviderFactory, "create", lambda *a, **k: handle)
    output = tmp_path / "executed.log"
    for name in ("send_email", "archive_email"):
        write_tool(tmp_path, name, output)
    spec = _multi_tool_spec("send_email", "archive_email")
    manager = ApprovalManager(
        run_repository=InMemoryRunRepository(), approval_repository=InMemoryApprovalRepository()
    )

    async with LangGraphEngine(
        tmp_path,
        model_factory=lambda *a, **k: cast(BaseChatModel, ChainedApprovalModel()),
        approval_manager=manager,
    ) as first_engine:
        await first_engine.build(spec)
        pending = await first_engine.run("review", context=RunContext(run_id="rebuilt"))
        assert pending.pending_approval is not None
        assert not output.exists()

    assert isinstance(spec.graph.node, AgentSpec)
    automatic = replace(
        spec, graph=replace(spec.graph, node=replace(spec.graph.node, auto_mode=True))
    )
    async with LangGraphEngine(
        tmp_path,
        model_factory=lambda *a, **k: cast(BaseChatModel, ChainedApprovalModel()),
        approval_manager=manager,
    ) as second_engine:
        await second_engine.build(automatic)
        approval_id = pending.pending_approval.approval_id
        if resumed_stream:
            events = [
                event
                async for event in second_engine.resume_stream("rebuilt", approval_id, decision)
            ]
            assert events[-1].type == "final"
        else:
            resumed = await second_engine.resume("rebuilt", approval_id, decision)
            assert resumed.status == "completed"
        record = await manager.get_approval("rebuilt", approval_id)
        assert record.status == (
            ApprovalStatus.REJECTED if decision == "deny" else ApprovalStatus.APPROVED
        )

    expected = [] if decision == "deny" else ["send_email:send_email"]
    # Subsequent calls still use the new policy once the pending decision is consumed.
    assert output.read_text().splitlines() == [*expected, "archive_email:archive_email"]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("resumed_stream", [False, True])
async def test_concurrent_resumes_keep_their_claimed_decisions_separate(
    tmp_path: Path, nested: bool, resumed_stream: bool
) -> None:
    output = tmp_path / "executed.log"
    write_tool(tmp_path, "send_email", output)
    spec = _multi_tool_spec("send_email")
    if nested:
        spec = system(orchestrator("root", [spec.graph]))
    async with LangGraphEngine(
        tmp_path, model_factory=lambda *a, **k: cast(BaseChatModel, FakeChatModel())
    ) as engine:
        await engine.build(spec)
        approvals = []
        for run_id in ("allowed", "denied"):
            pending = await engine.run(
                "review", context=RunContext(run_id=run_id, conversation_id="s", user_id="u")
            )
            assert pending.pending_approval is not None
            approvals.append(pending.pending_approval.approval_id)

        async def resume(run_id: str, approval_id: str, decision: str) -> None:
            if resumed_stream:
                events = [
                    event
                    async for event in engine.resume_stream(
                        run_id, approval_id, decision, caller_user_id="u", caller_session_id="s"
                    )
                ]
                assert events[-1].type == "final"
            else:
                result = await engine.resume(
                    run_id, approval_id, decision, caller_user_id="u", caller_session_id="s"
                )
                assert result.status == "completed"

        await asyncio.gather(
            resume("allowed", approvals[0], "allow for session"),
            resume("denied", approvals[1], "deny"),
        )
    assert output.read_text().splitlines() == ["send_email:go"]


@pytest.mark.parametrize("resumed_stream", [False, True])
async def test_rebuilt_provider_cannot_bypass_binding_with_auto_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resumed_stream: bool
) -> None:
    from langchain_core.tools import StructuredTool

    from agent_engine.core.spec import MCPSpec
    from agent_engine.engine.langgraph.checkpointing import CheckpointProviderFactory
    from agent_engine.engine.langgraph.tools.mcp_connector import MCPConnector

    handle = CheckpointProviderFactory().create(None)
    monkeypatch.setattr(CheckpointProviderFactory, "create", lambda *a, **k: handle)
    output = tmp_path / "executed.log"
    write_tool(tmp_path, "send_email", output)

    def remote_send(message: str) -> str:
        output.write_text("mcp:" + message)
        return message

    remote = StructuredTool.from_function(remote_send, name="send_email", description="Remote send")

    async def connect(self: MCPConnector, specs: Any) -> dict[str, list[Any]]:
        return {"replacement": [remote]} if specs else {}

    monkeypatch.setattr(MCPConnector, "connect", connect)
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
        first = await first_engine.run("review", context=RunContext(run_id="replaced"))
        assert first.pending_approval is not None

    assert isinstance(spec.graph.node, AgentSpec)
    replaced_spec = replace(
        spec,
        graph=replace(
            spec.graph,
            node=replace(
                spec.graph.node,
                tools=(),
                mcps=(MCPSpec(id="replacement", url="http://mcp.invalid/mcp"),),
                auto_mode=True,
            ),
        ),
    )
    async with LangGraphEngine(
        tmp_path,
        model_factory=lambda *a, **k: cast(BaseChatModel, FakeChatModel()),
        approval_manager=manager,
    ) as second_engine:
        await second_engine.build(replaced_spec)
        approval_id = first.pending_approval.approval_id
        with pytest.raises(ApprovalInvocationMismatch):
            if resumed_stream:
                _ = [
                    event
                    async for event in second_engine.resume_stream(
                        "replaced", approval_id, "allow once"
                    )
                ]
            else:
                await second_engine.resume("replaced", approval_id, "allow once")
        assert not output.exists()
        assert await second_engine.get_run_status("replaced") == "failed"
        record = await manager.get_approval("replaced", approval_id)
        assert record.status == ApprovalStatus.REJECTED
