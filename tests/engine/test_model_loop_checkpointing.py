"""Checkpointing is enabled only inside LangGraph, not every Runnable."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableLambda

from agent_engine.engine.langgraph.execution.model_context import ModelContext
from agent_engine.engine.langgraph.execution.model_loop import run_tool_loop
from tests.approvals.test_engine_hitl import FakeChatModel


async def test_model_loop_inside_an_ordinary_runnable_needs_no_graph() -> None:
    async def unexpected_tool(call: dict[str, Any]) -> str:
        raise AssertionError("model did not request a tool")

    async def run(_: object) -> str:
        result = await run_tool_loop(
            FakeChatModel(), ModelContext("system", [], "hello"), "writer", unexpected_tool
        )
        return str(result.content)

    assert await RunnableLambda(run).ainvoke({}) == "done"
