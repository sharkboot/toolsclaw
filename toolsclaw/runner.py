"""Agent runner - the core LLM tool execution loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, AsyncIterator

from rich.console import Console

from toolsclaw.config import Config
from toolsclaw.hook import AgentHook, AgentHookContext, CompositeHook
from toolsclaw.provider import LLMProvider
from toolsclaw.skills import (
    Skill,
    build_skills_summary,
    ensure_skill_deps,
    get_always_skills,
    load_skills,
)
from toolsclaw.tool import Tool, ToolRegistry
from toolsclaw.tools import ExecTool, ListDirTool, LoadSkillTool, ReadFileTool, RunScriptTool, WriteFileTool

# resolve the built-in skills directory (sibling of the package)
_BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to tools for file operations and shell execution.
Always use tools when the user asks you to do something that requires reading/writing files or running commands.
Be concise and direct in your responses.
"""

console = Console()


class _TokenAccumulator:
    """Accumulates token usage across multiple LLM calls."""

    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens", "iterations")

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.iterations: int = 0

    def add(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.iterations += 1

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.iterations = 0


class AgentRunner:
    """Runs the tool-calling agent loop.

    Args:
        config: Configuration object.
        skills_dir: Custom skills directory. If None, uses workspace/skills + builtin.
        hook: Optional lifecycle hook for this runner.
    """

    def __init__(
        self,
        config: Config,
        *,
        skills_dir: Path | str | None = None,
        hook: AgentHook | None = None,
    ) -> None:
        self._config = config
        self._workspace = config.get_workspace()
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._provider = LLMProvider(config)
        self._registry = ToolRegistry()
        self._hook = hook
        self._deps_ready = False
        self._last_run_usage = _TokenAccumulator()

        # resolve skills directories
        builtin = _BUILTIN_SKILLS_DIR
        if skills_dir is not None:
            custom = Path(skills_dir).expanduser().resolve()
            self._skills = load_skills(self._workspace, builtin, extra_dirs=[custom])
        else:
            self._skills = load_skills(self._workspace, builtin)

        self._register_tools()

    def _register_tools(self) -> None:
        """Register built-in tools based on config."""
        ws = self._workspace
        ec = self._config.exec

        self._registry.register(ReadFileTool(ws))
        self._registry.register(WriteFileTool(ws))
        self._registry.register(ListDirTool(ws))

        # register load_skill for progressive skill loading
        if self._skills:
            self._registry.register(LoadSkillTool(self._skills))

        if ec.enable:
            self._registry.register(
                ExecTool(
                    workspace=ws,
                    timeout=ec.timeout,
                    deny_patterns=ec.deny_patterns,
                )
            )

        # register run_script if any skill has pip dependencies
        if any(s.available and s.requires.pip for s in self._skills):
            self._registry.register(RunScriptTool(ws, self._skills, timeout=ec.timeout * 2))

    async def _ensure_deps(self) -> None:
        """Lazily ensure skill dependencies are ready (async, runs once)."""
        if self._deps_ready:
            return
        self._deps_ready = True
        await ensure_skill_deps(self._skills)
        # re-register run_script in case skills became available after deps installed
        if any(s.available and s.requires.pip for s in self._skills):
            ec = self._config.exec
            if not self._registry.get("run_script"):
                self._registry.register(RunScriptTool(self._workspace, self._skills, timeout=ec.timeout * 2))

    def _build_system_prompt(self) -> str:
        """Assemble the system prompt with progressive skill loading.

        Progressive disclosure pattern (3 levels):
        1. Always-on skills: full SKILL.md body injected directly
        2. Other skills: name + description + path in summary (agent reads on demand)
        3. Bundled resources: agent reads referenced scripts/assets as needed
        """
        parts = [SYSTEM_PROMPT]

        # Level 1: inject always-on skills with {SKILL_DIR} replaced
        always = get_always_skills(self._skills)
        if always:
            parts.append("# Active Skills\n")
            for s in always:
                skill_dir = str(s.path.parent.resolve())
                content = s.content.replace("{SKILL_DIR}", skill_dir)
                parts.append(content)

        # Level 2: inject skills summary with progressive loading instructions
        summary = build_skills_summary(self._skills)
        if summary:
            parts.append(
                "# Skills\n\n"
                "The following skills extend your capabilities. To use a skill, "
                "call the `load_skill` tool with the skill name. The skill file "
                "contains detailed instructions and scripts you can execute.\n\n"
                "When the user's request matches a skill's description, load it "
                "immediately before proceeding.\n\n"
                "Unavailable skills need dependencies installed first.\n\n"
                f"{summary}"
            )

        # Level 3: inject skill directories for subresource access
        for s in self._skills:
            if s.available:
                skill_dir = str(s.path.parent.resolve())
                parts.append(f"Skill '{s.name}' directory: {skill_dir}")

        parts.append(f"Workspace: {self._workspace}")
        return "\n\n".join(parts)

    async def run(self, user_message: str) -> str:
        """Run a single user message through the agent loop. Returns the final response."""
        await self._ensure_deps()
        self._last_run_usage.reset()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_message},
        ]
        tools = self._registry.get_definitions()

        for i in range(self._config.max_iterations):
            context = AgentHookContext(iteration=i, messages=messages)

            # before_iteration hook
            if self._hook:
                await self._hook.before_iteration(context)

            response = await self._provider.chat(messages, tools)
            u = response.usage
            self._last_run_usage.add(u.prompt_tokens, u.completion_tokens, u.total_tokens)
            context.response = response
            context.tool_calls = list(response.tool_calls)

            # no tool calls → final answer
            if not response.has_tool_calls:
                final = response.content
                if self._hook:
                    await self._hook.after_iteration(context)
                    final = self._hook.finalize_content(context, final) or final
                return final or "(Agent completed without generating a response.)"

            # append assistant message with tool calls
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content or None}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in response.tool_calls
            ]
            messages.append(assistant_msg)

            # before_execute_tools hook
            if self._hook:
                await self._hook.before_execute_tools(context)

            # execute each tool call and append results
            tool_results: list[str] = []
            tool_errors: list[str] = []
            for tc in response.tool_calls:
                args_preview = str(tc.arguments)[:200]
                console.print(f"  [bold blue]CALL {tc.name}[/bold blue]({args_preview})")
                result = await self._registry.execute(tc.name, tc.arguments)
                # use print() with errors='replace' to avoid encoding issues on Windows
                result_preview = (result[:300].replace("\n", " ") if result else "(empty)")
                try:
                    print(f"    -> {result_preview}")
                except UnicodeEncodeError:
                    enc = sys.stdout.encoding or "utf-8"
                    safe = result_preview.encode(enc, errors="replace").decode(enc, errors="replace")
                    print(f"    -> {safe}")
                tool_results.append(result)
                if result.startswith("Error"):
                    tool_errors.append(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            context.tool_results = tool_results
            context.tool_errors = tool_errors

            # after_iteration hook
            if self._hook:
                await self._hook.after_iteration(context)

        return "Error: maximum tool iterations reached."

    async def stream_run(self, user_message: str) -> AsyncIterator[str]:
        """Stream a single user message, yielding incremental response content.

        Yields the assistant's response content in real-time chunks.
        Tool execution is still non-streamed (runs silently between chunks).

        Usage::

            async for chunk in runner.stream_run("Hello"):
                print(chunk, end="", flush=True)
        """
        from typing import AsyncIterator as AI

        await self._ensure_deps()
        self._last_run_usage.reset()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_message},
        ]
        tools = self._registry.get_definitions()

        for i in range(self._config.max_iterations):
            context = AgentHookContext(iteration=i, messages=messages)

            if self._hook:
                await self._hook.before_iteration(context)

            content_chunks: list[str] = []
            async for chunk_resp in self._provider.stream_chat(messages, tools):
                if chunk_resp.content:
                    # only yield NEW content (difference from last chunk)
                    new_content = chunk_resp.content[len("".join(content_chunks)):]
                    if new_content:
                        content_chunks.append(new_content)
                        yield new_content
                if chunk_resp.finish_reason:
                    context.response = chunk_resp

            # finalize usage
            if context.response:
                u = context.response.usage
                self._last_run_usage.add(u.prompt_tokens, u.completion_tokens, u.total_tokens)

            # check if we got tool calls in the last streamed response
            last_response = context.response
            if not last_response or not last_response.has_tool_calls:
                final = last_response.content if last_response else ""
                if self._hook:
                    await self._hook.after_iteration(context)
                    final = self._hook.finalialize_content(context, final) or final
                return

            # build assistant message from streamed tool calls
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": last_response.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in last_response.tool_calls
                ],
            }
            messages.append(assistant_msg)

            if self._hook:
                await self._hook.before_execute_tools(context)

            # execute tools (non-streamed)
            tool_results: list[str] = []
            tool_errors: list[str] = []
            for tc in last_response.tool_calls:
                args_preview = str(tc.arguments)[:200]
                console.print(f"  [bold blue]CALL {tc.name}[/bold blue]({args_preview})")
                result = await self._registry.execute(tc.name, tc.arguments)
                result_preview = (result[:300].replace("\n", " ") if result else "(empty)")
                try:
                    print(f"    -> {result_preview}")
                except UnicodeEncodeError:
                    enc = sys.stdout.encoding or "utf-8"
                    safe = result_preview.encode(enc, errors="replace").decode(enc, errors="replace")
                    print(f"    -> {safe}")
                tool_results.append(result)
                if result.startswith("Error"):
                    tool_errors.append(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            context.tool_results = tool_results
            context.tool_errors = tool_errors

            if self._hook:
                await self._hook.after_iteration(context)

        yield "Error: maximum tool iterations reached."

    async def run_interactive(self) -> None:
        """Run an interactive chat session."""
        await self._ensure_deps()
        console.print("[bold green]toolsclaw[/bold green] - interactive mode (type 'exit' to quit)")
        console.print(f"Workspace: {self._workspace}\n")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
        ]
        tools = self._registry.get_definitions()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Bye![/dim]")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                console.print("[dim]Bye![/dim]")
                break

            messages.append({"role": "user", "content": user_input})

            for i in range(self._config.max_iterations):
                context = AgentHookContext(iteration=i, messages=messages)

                if self._hook:
                    await self._hook.before_iteration(context)

                response = await self._provider.chat(messages, tools)
                context.response = response
                context.tool_calls = list(response.tool_calls)

                if not response.has_tool_calls:
                    final = response.content
                    if self._hook:
                        await self._hook.after_iteration(context)
                        final = self._hook.finalize_content(context, final) or final
                    if final:
                        console.print(f"\n[bold cyan]Assistant:[/bold cyan] {final}\n")
                    break

                # append assistant message with tool calls
                assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content or None}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages.append(assistant_msg)

                if self._hook:
                    await self._hook.before_execute_tools(context)

                tool_results: list[str] = []
                tool_errors: list[str] = []
                for tc in response.tool_calls:
                    console.print(f"  [dim]CALL {tc.name}[/dim]")
                    result = await self._registry.execute(tc.name, tc.arguments)
                    tool_results.append(result)
                    if result.startswith("Error"):
                        tool_errors.append(result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                context.tool_results = tool_results
                context.tool_errors = tool_errors
                if self._hook:
                    await self._hook.after_iteration(context)
            else:
                console.print("[red]Maximum iterations reached.[/red]")
