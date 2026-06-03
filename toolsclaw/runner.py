"""Agent runner — the core LLM ↔ tool execution loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
from toolsclaw.tools import ExecTool, ListDirTool, ReadFileTool, RunScriptTool, WriteFileTool

# resolve the built-in skills directory (sibling of the package)
_BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

MAX_ITERATIONS = 50
SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to tools for file operations and shell execution.
Always use tools when the user asks you to do something that requires reading/writing files or running commands.
Be concise and direct in your responses.
"""

console = Console()


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

        # resolve skills directories
        builtin = _BUILTIN_SKILLS_DIR
        if skills_dir is not None:
            custom = Path(skills_dir).expanduser().resolve()
            self._skills = load_skills(self._workspace, builtin, extra_dirs=[custom])
        else:
            self._skills = load_skills(self._workspace, builtin)

        ensure_skill_deps(self._skills)
        self._register_tools()

    def _register_tools(self) -> None:
        """Register built-in tools based on config."""
        ws = self._workspace
        ec = self._config.exec

        self._registry.register(ReadFileTool(ws))
        self._registry.register(WriteFileTool(ws))
        self._registry.register(ListDirTool(ws))

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

    def _build_system_prompt(self) -> str:
        """Assemble the system prompt with skills."""
        parts = [SYSTEM_PROMPT]

        # inject always-on skills
        always = get_always_skills(self._skills)
        if always:
            parts.append("# Active Skills")
            for s in always:
                parts.append(s.content)

        # inject skills summary
        summary = build_skills_summary(self._skills)
        if summary:
            parts.append(summary)

        parts.append(f"Workspace: {self._workspace}")
        return "\n\n".join(parts)

    async def run(self, user_message: str) -> str:
        """Run a single user message through the agent loop. Returns the final response."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_message},
        ]
        tools = self._registry.get_definitions()

        for i in range(MAX_ITERATIONS):
            context = AgentHookContext(iteration=i, messages=messages)

            # before_iteration hook
            if self._hook:
                await self._hook.before_iteration(context)

            response = await self._provider.chat(messages, tools)
            context.response = response
            context.tool_calls = list(response.tool_calls)

            # no tool calls → final answer
            if not response.has_tool_calls:
                context.final_content = response.content
                if self._hook:
                    await self._hook.after_iteration(context)
                    response.content = self._hook.finalize_content(context, response.content) or response.content
                return response.content

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
            for tc in response.tool_calls:
                console.print(f"  [dim]⚙ {tc.name}[/dim]")
                result = await self._registry.execute(tc.name, tc.arguments)
                tool_results.append(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            context.tool_results = tool_results

            # after_iteration hook
            if self._hook:
                await self._hook.after_iteration(context)

        return "Error: maximum tool iterations reached."

    async def run_interactive(self) -> None:
        """Run an interactive chat session."""
        console.print("[bold green]toolsclaw[/bold green] — interactive mode (type 'exit' to quit)")
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

            for i in range(MAX_ITERATIONS):
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
                for tc in response.tool_calls:
                    console.print(f"  [dim]⚙ {tc.name}[/dim]")
                    result = await self._registry.execute(tc.name, tc.arguments)
                    tool_results.append(result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                context.tool_results = tool_results
                if self._hook:
                    await self._hook.after_iteration(context)
            else:
                console.print("[red]Maximum iterations reached.[/red]")
