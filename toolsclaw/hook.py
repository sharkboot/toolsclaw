"""Lifecycle hooks for agent runs.

Mirrors the nanobot hook pattern: AgentHook provides before/after callbacks
for each LLM iteration, tool execution, and streaming. SDKCaptureHook
records tool usage and messages for RunResult.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from toolsclaw.memory import MemoryCompressor
from toolsclaw.persistent_memory import Memory, MemoryStore
from toolsclaw.provider import LLMResponse, ToolCallRequest

logger = logging.getLogger(__name__)


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


class MemoryCompressionHook(AgentHook):
    """Hook that automatically compresses conversation history when it
    exceeds the configured token threshold.

    Triggers on ``before_iteration`` so the LLM always sees a
    context-sized message list.

    Usage::

        compressor = MemoryCompressor(summarize_func=my_summarizer)
        hook = MemoryCompressionHook(compressor, threshold_tokens=80_000)
        runner = AgentRunner(config, hook=hook)
    """

    def __init__(
        self,
        compressor: MemoryCompressor,
        *,
        enabled: bool = True,
        threshold_tokens: int = 80_000,
        strategy: str = "hybrid",
        target_ratio: float = 0.5,
        min_rounds_to_keep: int = 3,
        tool_result_max_chars: int = 300,
    ) -> None:
        super().__init__()
        self._compressor = compressor
        self._enabled = enabled
        self._threshold_tokens = threshold_tokens
        self._strategy = strategy
        self._target_ratio = target_ratio
        self._min_rounds_to_keep = min_rounds_to_keep
        self._tool_result_max_chars = tool_result_max_chars

        self.compression_count: int = 0
        """Number of times compression has been triggered."""

        self.last_compressed_tokens: int = 0
        """Estimated token count before the last compression."""

    async def before_iteration(self, context: AgentHookContext) -> None:
        """Check message size and compress if needed."""
        if not self._enabled:
            return

        if context.iteration == 0:
            # Don't compress on the first iteration (no history yet)
            return

        msgs = context.messages
        if not self._compressor.should_compress(msgs, self._threshold_tokens):
            return

        self.last_compressed_tokens = self._compressor.estimate_tokens(msgs)
        self.compression_count += 1

        compressed = await self._compressor.compress(
            msgs,
            strategy=self._strategy,
            target_ratio=self._target_ratio,
            min_rounds_to_keep=self._min_rounds_to_keep,
            tool_result_max_chars=self._tool_result_max_chars,
        )

        # Replace in-place so the runner's original reference is updated
        context.messages.clear()
        context.messages.extend(compressed)

    @classmethod
    def from_config(
        cls,
        compressor: MemoryCompressor,
        config: Any,  # MemoryCompressionConfig
    ) -> MemoryCompressionHook:
        """Create a hook from a config object."""
        return cls(
            compressor,
            enabled=config.enabled,
            threshold_tokens=config.threshold_tokens,
            strategy=config.strategy,
            target_ratio=config.target_ratio,
            min_rounds_to_keep=config.min_rounds_to_keep,
            tool_result_max_chars=config.tool_result_max_chars,
        )


class PersistentMemoryHook(AgentHook):
    """Hook that loads relevant memories into the system prompt and
    auto-saves important facts after each iteration.

    This hook implements a persistent memory system similar to Claude Code's
    file-based memory: memories are stored as markdown files with frontmatter
    in the workspace's ``.claude/memory/`` directory.

    The hook:

    1. **Before each iteration**: searches for memories relevant to the
       current conversation and injects a memory context block into the
       messages.
    2. **After each iteration**: extracts key facts from the exchange and
       saves them as memories.

    Args:
        store: Initialised MemoryStore instance.
        enabled: Whether this hook is active.
        max_memories: Maximum number of relevant memories to inject.
        auto_save: Whether to auto-save memories after each iteration.
        dedup_threshold: Minimum content length difference fraction (0.0-1.0)
            to consider a new memory worth saving. Higher = less dedup.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        enabled: bool = True,
        max_memories: int = 5,
        auto_save: bool = True,
        dedup_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self._store = store
        self._enabled = enabled
        self._max_memories = max_memories
        self._auto_save = auto_save
        self._dedup_threshold = dedup_threshold

        self.memories_loaded: int = 0
        """Number of times relevant memories were loaded into context."""

        self.memories_saved: int = 0
        """Number of auto-saved memories."""

        self._injected_this_run: bool = False
        """Track whether we've already injected memories this run."""

        self._saved_topics: set[str] = set()
        """Track topics already saved this run to avoid duplicates."""

    # -- public helpers ---------------------------------------------------

    @property
    def store(self) -> MemoryStore:
        return self._store

    # -- AgentHook overrides ----------------------------------------------

    async def before_iteration(self, context: AgentHookContext) -> None:
        """Inject relevant memories into the system prompt."""
        if not self._enabled:
            return

        # Only inject once per run (on first iteration)
        if self._injected_this_run:
            return
        self._injected_this_run = True

        # Build a query from the current user messages
        query = self._build_query(context)
        if not query:
            return

        memories = self._store.search(query, max_results=self._max_memories)
        if not memories:
            return

        # Inject as a system-level memory context block
        context_block = self._format_memory_context(memories)
        if not context_block:
            return

        # Prepend the memory context to the system message
        for i, msg in enumerate(context.messages):
            if msg.get("role") == "system":
                existing = msg.get("content", "")
                if isinstance(existing, str):
                    msg["content"] = existing + "\n\n" + context_block
                break

        self.memories_loaded += len(memories)
        logger.info(
            "PersistentMemoryHook: injected %d memories into context",
            len(memories),
        )

    async def after_iteration(self, context: AgentHookContext) -> None:
        """Auto-save important facts after each iteration."""
        if not self._enabled or not self._auto_save:
            return

        # 1. Save assistant's final response as a project memory
        if context.final_content and len(context.final_content) > 100:
            self._save_response_memory(context)

        # 2. Extract tool usage patterns
        if context.tool_calls:
            self._save_tool_memories(context)

        # 3. Save user query topics as feedback memories
        self._save_user_query_memories(context)

    # -- internal helpers -------------------------------------------------

    def _build_query(self, context: AgentHookContext) -> str:
        """Build a search query from the current conversation context."""
        parts: list[str] = []
        for msg in context.messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    parts.append(content[:500])
        return " ".join(parts[-3:])  # last 3 user messages

    def _format_memory_context(self, memories: list[Memory]) -> str:
        """Format relevant memories as a context block for the system prompt."""
        if not memories:
            return ""

        lines = ["## Relevant Memories\n"]
        for mem in memories:
            tag = f"[{mem.type}]"
            lines.append(f"### {tag} {mem.name}")
            if mem.description:
                lines.append(f"_{mem.description}_")
            lines.append("")
            lines.append(mem.content)
            if mem.links:
                linked = ", ".join(f"[[{l}]]" for l in mem.links)
                lines.append(f"Related: {linked}")
            lines.append("")

        return "\n".join(lines)

    def _save_response_memory(self, context: AgentHookContext) -> None:
        """Save the assistant's response as a memory, deduplicating against
        existing memories with the same topic."""
        content = context.final_content or ""
        if not content:
            return

        # Extract a topic key from the first line or the user's query
        topic = self._extract_topic_key(content, context)
        if not topic or topic in self._saved_topics:
            return

        # Check if a similar memory already exists
        existing = self._store.search(topic, max_results=1)
        if existing:
            prev = existing[0]
            sim = self._content_similarity(content, prev.content)
            if sim > (1.0 - self._dedup_threshold):
                logger.debug(
                    "PersistentMemoryHook: skipping duplicate memory (sim=%.2f): %s",
                    sim, topic,
                )
                return

        self._saved_topics.add(topic)
        safe_name = self._sanitise_name(topic[:40])

        self._auto_save_memory(
            name=safe_name,
            description=f"Assistant response about {topic[:60]}",
            type="reference",
            content=content,
        )

    def _save_tool_memories(self, context: AgentHookContext) -> None:
        """Save tool usage patterns as project memories, grouped by task."""
        tool_names = sorted({tc.name for tc in context.tool_calls})
        if not tool_names:
            return

        # Group tool names into a topic key
        topic_key = "-".join(tool_names[:3])
        if topic_key in self._saved_topics:
            return
        self._saved_topics.add(topic_key)

        # Get user query context for better description
        user_query = ""
        for msg in reversed(context.messages):
            if msg.get("role") == "user":
                user_query = (msg.get("content", "") or "")[:120]
                break

        mem_name = f"tool-pattern-{topic_key}"
        desc = f"Used tools: {', '.join(tool_names)}"
        if user_query:
            desc += f" — {user_query}"

        self._auto_save_memory(
            name=mem_name,
            description=desc,
            type="project",
            content=(
                f"Tool pattern: `{'`, `'.join(tool_names)}`\n\n"
                f"User query: {user_query}\n\n"
                f"Tools called: {len(context.tool_calls)} total"
            ),
        )

    def _save_user_query_memories(self, context: AgentHookContext) -> None:
        """Save user query topics as feedback memories."""
        for msg in context.messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 50:
                continue

            topic = self._extract_topic_key(content, context)
            if not topic or topic in self._saved_topics:
                continue
            self._saved_topics.add(topic)

            safe_name = self._sanitise_name(f"user-query-{topic[:30]}")
            self._auto_save_memory(
                name=safe_name,
                description=f"User asked about: {topic[:80]}",
                type="feedback",
                content=content[:500],
            )

    @staticmethod
    def _extract_topic_key(
        content: str, context: AgentHookContext | None = None,
    ) -> str:
        """Extract a short topic key from content.

        Uses the first substantive line or the first 50 chars of content.
        """
        # Try first non-empty line
        for line in content.split("\n"):
            stripped = line.strip().strip("#* \t")
            if stripped and len(stripped) > 10:
                return stripped[:80]
        # Fallback: first 50 chars
        return content.strip()[:80]

    @staticmethod
    def _content_similarity(a: str, b: str) -> float:
        """Compute a simple content similarity score (0.0-1.0).

        Uses token overlap (intersection / union of word sets).
        """
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @staticmethod
    def _sanitise_name(name: str) -> str:
        """Normalise a string to a valid kebab-case slug."""
        safe = name.strip().lower()
        safe = safe.replace("_", "-").replace(" ", "-")
        safe = re.sub(r"[^a-z0-9-]", "", safe)
        safe = re.sub(r"-{2,}", "-", safe).strip("-")
        return safe[:80] or "unnamed"

    def _auto_save_memory(
        self,
        name: str,
        description: str,
        type: str,
        content: str,
    ) -> None:
        """Save a memory using update_or_create (silently).

        Uses the store's upsert pattern so repeated saves with the same
        name merge content rather than creating duplicates.
        """
        if not name:
            return

        try:
            self._store.update_or_create(
                name=name,
                description=description,
                type=type,
                content=content,
                merge_content=True,
            )
            self.memories_saved += 1
        except (ValueError, OSError):
            logger.debug("PersistentMemoryHook: failed to auto-save %s", name)
