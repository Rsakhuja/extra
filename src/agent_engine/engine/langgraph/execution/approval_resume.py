"""A claimed human decision carried only for one resume request."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

from agent_engine.approvals.errors import ApprovalInvocationMismatch


@dataclass
class ApprovalResume:
    """Bind the first unfinished tool step, then release later calls to policy.

    Completed steps are restored without entering their gate. This request-local
    state is shared by nested graph tasks, never stored on the engine or graph.
    """

    approval_id: str
    tool_call_id: str
    consumed: bool = False

    def for_invocation(self, tool_call_id: str) -> str | None:
        if self.consumed:
            return None
        if tool_call_id != self.tool_call_id:
            raise ApprovalInvocationMismatch(self.approval_id)
        return self.approval_id

    def consume(self, approval_id: str, tool_call_id: str) -> None:
        if not self.consumed:
            if approval_id != self.approval_id or tool_call_id != self.tool_call_id:
                raise ApprovalInvocationMismatch(self.approval_id)
            self.consumed = True


current_approval_resume: ContextVar[ApprovalResume | None] = ContextVar(
    "current_approval_resume", default=None
)
