"""Tool base class and registry."""

from __future__ import annotations

import abc
from typing import Any


class Tool(abc.ABC):
    """Abstract base class for agent tools."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Function-call name exposed to the LLM."""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable description shown to the LLM."""

    @property
    @abc.abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for the tool's input parameters."""

    @abc.abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Run the tool and return a string result."""

    def to_schema(self) -> dict[str, Any]:
        """Produce an OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Container for registered tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._definitions: list[dict[str, Any]] | None = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._definitions = None  # invalidate cache

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_definitions(self) -> list[dict[str, Any]]:
        if self._definitions is None:
            self._definitions = [t.to_schema() for t in self._tools.values()]
        return self._definitions

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name. Returns result or error string."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            return await tool.execute(**arguments)
        except Exception as e:
            return f"Error executing {name}: {e}"
