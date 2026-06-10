"""Lifecycle hooks for agent runs.

Mirrors the nanobot hook pattern: AgentHook provides before/after callbacks
for each LLM iteration, tool execution, and streaming. SDKCaptureHook
records tool usage and messages for RunResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from toolsclaw.provider import LLMResponse, ToolCallRequest


@dataclass
class AgentHookContext:
    """Mutable per-iteration state exposed to hooks."""

    iteration: int
    messages: list[dict[str, Any]]
    response: LLMResponse | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)
    final_content: str | None = None


class AgentHook:
    """Base class for agent lifecycle hooks.

    Override any method to inject behavior at that point in the agent loop.
    """

    async def before_iteration(self, context: AgentHookContext) -> None:
        """Called before each LLM request."""
        pass

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        """Called after LLM responds with tool calls, before execution."""
        pass

    async def after_iteration(self, context: AgentHookContext) -> None:
        """Called after tool results are collected (or LLM returns final answer)."""
        pass

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        """Post-process the final response text. Return the modified content."""
        return content


class SDKCaptureHook(AgentHook):
    """Records tool usage and messages for RunResult.

    Used internally by ToolsClaw.run() to capture what happened during
    the agent loop without exposing the full runner internals.
    """

    def __init__(self) -> None:
        self.tools_used: list[str] = []
        self.messages: list[dict[str, Any]] = []

    async def after_iteration(self, context: AgentHookContext) -> None:
        for tc in context.tool_calls:
            self.tools_used.append(tc.name)
        self.messages = list(context.messages)


class CompositeHook(AgentHook):
    """Fan-out hook that delegates to an ordered list of hooks."""

    __slots__ = ("_hooks",)

    def __init__(self, hooks: list[AgentHook]) -> None:
        super().__init__()
        self._hooks = list(hooks)

    async def before_iteration(self, context: AgentHookContext) -> None:
        for h in self._hooks:
            await h.before_iteration(context)

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        for h in self._hooks:
            await h.before_execute_tools(context)

    async def after_iteration(self, context: AgentHookContext) -> None:
        for h in self._hooks:
            await h.after_iteration(context)

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        for h in self._hooks:
            content = h.finalize_content(context, content)
        return content
