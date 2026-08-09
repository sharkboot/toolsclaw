"""High-level programmatic interface to toolsclaw.

Mirrors the nanobot SDK pattern: create an agent with workspace and skills_dir,
run a message, get back a RunResult with content + tools_used + messages.

Usage::

    from toolsclaw import ToolsClaw

    agent = ToolsClaw.from_config(
        workspace="./my-project",
        skills_dir="./my-skills",
    )
    result = await agent.run("列出所有 Python 文件")
    print(result.content)
    print(result.tools_used)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from toolsclaw.config import Config, ProviderConfig, load_config
from toolsclaw.hook import AgentHook, CompositeHook, SDKCaptureHook
from toolsclaw.runner import AgentRunner


@dataclass
class RunResult:
    """Result of a single agent run."""

    content: str
    """Final text response from the agent."""

    tools_used: list[str]
    """Names of tools that were called during this run."""

    messages: list[dict[str, Any]]
    """Full message history (system + user + assistant + tool messages)."""

    prompt_tokens: int = 0
    """Total prompt tokens consumed across all LLM calls."""

    completion_tokens: int = 0
    """Total completion tokens consumed across all LLM calls."""

    total_tokens: int = 0
    """Total tokens consumed across all LLM calls."""

    iterations: int = 0
    """Number of LLM round-trips."""

    compression_triggered: bool = False
    """Whether memory compression was triggered during this run."""

    compression_count: int = 0
    """How many times compression was applied (cumulative across runs)."""

    memories_loaded: int = 0
    """How many persistent memories were loaded into context."""

    memories_saved: int = 0
    """How many persistent memories were auto-saved during this run."""


class ToolsClaw:
    """Programmatic facade for running the toolsclaw agent.

    Usage::

        # From config file
        agent = ToolsClaw.from_config()
        result = await agent.run("Hello")

        # With custom workspace and skills
        agent = ToolsClaw.from_config(
            workspace="./my-project",
            skills_dir="./my-skills",
        )
        result = await agent.run("Do something")

        # Multiple independent agents
        agent_a = ToolsClaw.from_config(workspace="./project-a")
        agent_b = ToolsClaw.from_config(workspace="./project-b")
    """

    def __init__(self, runner: AgentRunner) -> None:
        self._runner = runner

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        workspace: str | Path | None = None,
        skills_dir: str | Path | None = None,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        max_iterations: int | None = None,
        memory_compression: bool | None = None,
        memory_strategy: str | None = None,
        memory_threshold: int | None = None,
        persistent_memory: bool | None = None,
        persistent_memory_dir: str | None = None,
        auto_save_memories: bool | None = None,
    ) -> ToolsClaw:
        """Create a ToolsClaw instance from a config file.

        Args:
            config_path: Path to ``config.json``.  Defaults to
                ``~/.toolsclaw/config.json``.
            workspace: Override the workspace directory from config.
                All file tools and exec are sandboxed to this directory.
            skills_dir: Extra skills directory to load in addition to
                workspace/skills and builtin skills. Skills in this
                directory take priority over builtin skills.
            model: Model name (e.g. ``"mimo-v2.5-pro"``). Overrides config.
            api_key: API key. Overrides config.
            api_base: API base URL. Overrides config.
            max_iterations: Maximum number of LLM round-trips. Overrides config.
            memory_compression: Enable/disable memory compression.
                Overrides config.
            memory_strategy: Compression strategy (truncate, drop,
                summarize, hybrid). Overrides config.
            memory_threshold: Token threshold to trigger compression.
                Overrides config.
            persistent_memory: Enable/disable persistent memory.
                Overrides config.
            persistent_memory_dir: Custom memory directory.
                Overrides config.
            auto_save_memories: Enable/disable auto-saving memories.
                Overrides config.
        """
        config = load_config(Path(config_path) if config_path else None)

        if workspace is not None:
            config.workspace = str(Path(workspace).expanduser().resolve())

        # apply model / provider overrides
        if model is not None:
            config.model = model
        if api_key is not None or api_base is not None:
            existing = config.get_provider_config()
            config.providers["default"] = ProviderConfig(
                api_key=api_key or existing.api_key,
                api_base=api_base or existing.api_base,
            )
            config.provider = "default"

        if max_iterations is not None:
            config.max_iterations = max_iterations

        # memory compression overrides
        if memory_compression is not None:
            config.memory.enabled = memory_compression
        if memory_strategy is not None:
            config.memory.strategy = memory_strategy
        if memory_threshold is not None:
            config.memory.threshold_tokens = memory_threshold

        # persistent memory overrides
        if persistent_memory is not None:
            config.persistent_memory.enabled = persistent_memory
        if persistent_memory_dir is not None:
            config.persistent_memory.memory_dir = persistent_memory_dir
        if auto_save_memories is not None:
            config.persistent_memory.auto_save = auto_save_memories

        runner = AgentRunner(
            config,
            skills_dir=skills_dir,
        )
        return cls(runner)

    @classmethod
    def from_runner(cls, runner: AgentRunner) -> ToolsClaw:
        """Create a ToolsClaw instance from an existing AgentRunner.

        Useful when you need full control over runner construction
        (custom tools, custom config, etc.).
        """
        return cls(runner)

    async def run(
        self,
        message: str,
        *,
        hooks: list[AgentHook] | None = None,
    ) -> RunResult:
        """Run the agent once and return the result.

        Args:
            message: The user message to process.
            hooks: Optional lifecycle hooks for this run.
                These are composed with any hook already on the runner.

        Returns:
            RunResult with content, tools_used, and messages.
        """
        capture = SDKCaptureHook()

        # compose hooks: capture + user hooks + existing runner hook
        all_hooks: list[AgentHook] = [capture]
        if hooks:
            all_hooks.extend(hooks)
        if self._runner._hook:
            all_hooks.append(self._runner._hook)

        composite = CompositeHook(all_hooks) if len(all_hooks) > 1 else all_hooks[0]

        # temporarily replace the runner's hook
        prev_hook = self._runner._hook
        self._runner._hook = composite
        try:
            content = await self._runner.run(message)
        finally:
            self._runner._hook = prev_hook

        usage = self._runner._last_run_usage

        # extract compression info from the hook chain
        compression_triggered = False
        compression_count = 0
        memories_loaded = 0
        memories_saved = 0
        for h in all_hooks:
            if hasattr(h, "compression_count"):
                compression_count = h.compression_count
                if h.compression_count > 0:
                    compression_triggered = True
            if hasattr(h, "memories_loaded"):
                memories_loaded = getattr(h, "memories_loaded", 0)
            if hasattr(h, "memories_saved"):
                memories_saved = getattr(h, "memories_saved", 0)

        return RunResult(
            content=content,
            tools_used=capture.tools_used,
            messages=capture.messages,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            iterations=usage.iterations,
            compression_triggered=compression_triggered,
            compression_count=compression_count,
            memories_loaded=memories_loaded,
            memories_saved=memories_saved,
        )

    async def stream_run(
        self,
        message: str,
        *,
        hooks: list[AgentHook] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the agent's response incrementally.

        Args:
            message: The user message to process.
            hooks: Optional lifecycle hooks for this run.

        Yields:
            String chunks of the assistant's response content in real-time.

        Usage::

            async for chunk in agent.run("Hello", stream=True):
                print(chunk, end="", flush=True)
        """
        capture = SDKCaptureHook()

        all_hooks: list[AgentHook] = [capture]
        if hooks:
            all_hooks.extend(hooks)
        if self._runner._hook:
            all_hooks.append(self._runner._hook)

        composite = CompositeHook(all_hooks) if len(all_hooks) > 1 else all_hooks[0]

        prev_hook = self._runner._hook
        self._runner._hook = composite
        try:
            async for chunk in await self._runner.run(message, stream=True):  # type: ignore[arg-type]
                yield chunk
        finally:
            self._runner._hook = prev_hook
