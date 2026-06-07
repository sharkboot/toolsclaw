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
from typing import Any

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
        return RunResult(
            content=content,
            tools_used=capture.tools_used,
            messages=capture.messages,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            iterations=usage.iterations,
        )
