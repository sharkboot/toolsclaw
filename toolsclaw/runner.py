"""Agent runner - the core LLM tool execution loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from rich.console import Console

from toolsclaw.config import Config
from toolsclaw.hook import AgentHook, AgentHookContext, CompositeHook, MemoryCompressionHook, PersistentMemoryHook
from toolsclaw.memory import MemoryCompressor
from toolsclaw.persistent_memory import MemoryStore
from toolsclaw.provider import LLMProvider, LLMResponse
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

        # Hook setup: compression + persistent memory
        self._hook = self._setup_hooks(hook)

    def _setup_hooks(self, hook: AgentHook | None) -> AgentHook | None:
        """Set up lifecycle hooks: compression + persistent memory.

        Wraps the provided hook with compression and memory hooks based on
        config.
        """
        hooks: list[AgentHook] = []

        # Persistent memory hook
        pm_cfg = self._config.persistent_memory
        if pm_cfg.enabled:
            if pm_cfg.memory_dir:
                base = Path(pm_cfg.memory_dir)
            else:
                base = self._workspace / ".claude"
            store = MemoryStore(base)
            mem_hook = PersistentMemoryHook(
                store,
                enabled=True,
                max_memories=pm_cfg.max_memories,
                auto_save=pm_cfg.auto_save,
            )
            hooks.append(mem_hook)

        # Memory compression hook
        mc_cfg = self._config.memory
        if mc_cfg.enabled:
            compressor = MemoryCompressor(
                summarize_func=self._make_summarizer(),
            )
            mc_hook = MemoryCompressionHook.from_config(compressor, mc_cfg)
            hooks.append(mc_hook)

        # User-provided hook
        if hook is not None:
            hooks.append(hook)

        if not hooks:
            return None
        if len(hooks) == 1:
            return hooks[0]
        return CompositeHook(hooks)

    def _make_summarizer(self) -> Callable[[str], Awaitable[str]]:
        """Return an async callable that uses the LLM provider to summarize.

        Uses a minimal system prompt to keep the summary concise.
        """

        async def _summarize(prompt: str) -> str:
            msgs = [
                {"role": "system", "content": "You are a concise summarizer. "
                 "Summarize the conversation preserving key facts, decisions, "
                 "file paths, and the original language. Keep it brief."},
                {"role": "user", "content": prompt},
            ]
            resp = await self._provider.chat(msgs, tools=None)
            return resp.content or "(summary unavailable)"

        return _summarize

    @staticmethod
    def _find_memory_hook(hook: AgentHook) -> MemoryCompressionHook | None:
        """Walk a CompositeHook chain to find the MemoryCompressionHook."""
        from toolsclaw.hook import CompositeHook

        if isinstance(hook, MemoryCompressionHook):
            return hook
        if isinstance(hook, CompositeHook):
            for h in hook._hooks:  # type: ignore[attr-defined]
                found = AgentRunner._find_memory_hook(h)
                if found:
                    return found
        return None

    @staticmethod
    def _find_persistent_memory_hook(hook: AgentHook) -> PersistentMemoryHook | None:
        """Walk a CompositeHook chain to find the PersistentMemoryHook."""
        from toolsclaw.hook import CompositeHook

        if isinstance(hook, PersistentMemoryHook):
            return hook
        if isinstance(hook, CompositeHook):
            for h in hook._hooks:  # type: ignore[attr-defined]
                found = AgentRunner._find_persistent_memory_hook(h)
                if found:
                    return found
        return None

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

    async def run(self, user_message: str, *, stream: bool = False) -> str | AsyncIterator[str]:
        """Run a single user message through the agent loop.

        Args:
            user_message: The user message to process.
            stream: If True, returns an async iterator yielding incremental
                    content chunks. If False (default), returns the final string.

        When stream=True, yields content chunks in real-time. Tool execution is still
        non-streamed between chunks.
        """
        if stream:
            return self._run_streaming(user_message)
        return await self._run_sync(user_message)

    async def _run_sync(self, user_message: str) -> str:
        """Non-streaming agent loop. Returns the final response string."""
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

            response = await self._provider.chat(messages, tools)
            u = response.usage
            self._last_run_usage.add(u.prompt_tokens, u.completion_tokens, u.total_tokens)
            context.response = response
            context.tool_calls = list(response.tool_calls)

            if not response.has_tool_calls:
                final = response.content
                if self._hook:
                    await self._hook.after_iteration(context)
                    final = self._hook.finalize_content(context, final) or final
                return final or "(Agent completed without generating a response.)"

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
            for tc in context.tool_calls:
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

        return "Error: maximum tool iterations reached."

    async def _run_streaming(self, user_message: str) -> AsyncIterator[str]:
        """Streaming agent loop. Yields content chunks."""
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

            response: LLMResponse | None = None
            content_parts: list[str] = []

            async for chunk_resp in self._provider.stream_chat(messages, tools):
                response = chunk_resp
                if chunk_resp.content:
                    new = chunk_resp.content[len("".join(content_parts)):]
                    if new:
                        content_parts.append(new)
                        yield new
                if chunk_resp.finish_reason:
                    break

            if response:
                u = response.usage
                self._last_run_usage.add(u.prompt_tokens, u.completion_tokens, u.total_tokens)
                context.response = response
                context.tool_calls = list(response.tool_calls)

            if not response or not response.has_tool_calls:
                if self._hook:
                    await self._hook.after_iteration(context)
                    self._hook.finalize_content(context, response.content if response else "")
                return

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
            for tc in context.tool_calls:
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

                # Log compression if it happened
                if self._hook and hasattr(self._hook, "compression_count"):
                    from toolsclaw.hook import CompositeHook, MemoryCompressionHook
                    mh = self._find_memory_hook(self._hook)
                    if mh and mh.compression_count > 0 and mh.last_compressed_tokens > 0:
                        console.print(
                            f"  [dim]🧠 Memory compressed: "
                            f"{mh.compression_count}x, "
                            f"last saved ~{mh.last_compressed_tokens} tokens[/dim]"
                        )
                        mh.last_compressed_tokens = 0  # only log once per compression

                # Log persistent memory if loaded
                if self._hook and hasattr(self._hook, "memories_loaded"):
                    from toolsclaw.hook import PersistentMemoryHook
                    pm = self._find_persistent_memory_hook(self._hook)
                    if pm and pm.memories_loaded > 0:
                        console.print(
                            f"  [dim]📝 {pm.memories_loaded} memories loaded, "
                            f"{pm.memories_saved} saved[/dim]"
                        )

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
