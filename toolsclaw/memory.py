"""Memory compression for conversation history.

Provides configurable strategies to compress LLM conversation histories
when they approach the context window limit. Integrates with the AgentHook
lifecycle for automatic compression.
"""

from __future__ import annotations

import enum
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class CompressionStrategy(str, enum.Enum):
    """Available compression strategies."""

    TRUNCATE = "truncate"
    """Remove oldest rounds, keep a summary placeholder."""

    DROP_TOOL_RESULTS = "drop"
    """Truncate long tool result contents in-place."""

    SUMMARIZE = "summarize"
    """Use an LLM to summarize older rounds into a single message."""

    HYBRID = "hybrid"
    """Combination: drop tool details first, then truncate or summarize if needed."""


# Default max chars per tool result field after compression
_DEFAULT_TOOL_RESULT_MAX_CHARS = 300

# Rough token estimation: average chars per token
_CHARS_PER_TOKEN = 3.0


def _estimate_tokens(text: str) -> int:
    """Rough token count estimation (chars / 3).

    This is a heuristic. For Chinese-heavy content, actual tokens are
    higher, so this is a conservative over-estimate (safer for threshold).
    """
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens consumed by a message list."""
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += _estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += _estimate_tokens(part.get("text", ""))

        # tool_calls field
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                fn = tc.get("function", {})
                total += _estimate_tokens(fn.get("name", ""))
                total += _estimate_tokens(fn.get("arguments", ""))

        # role + tool_call_id overhead (~20 tokens each)
        total += 20

    return total


def _group_into_rounds(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group messages into conversation rounds.

    A round starts at a user message and ends before the next user message.
    System messages are excluded (they should be handled separately).
    """
    rounds: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") == "user" and current:
            # Save previous round, start new one
            rounds.append(current)
            current = [msg]
        else:
            current.append(msg)

    if current:
        rounds.append(current)

    return rounds


def _drop_tool_result_details(
    messages: list[dict[str, Any]],
    max_chars: int = _DEFAULT_TOOL_RESULT_MAX_CHARS,
) -> list[dict[str, Any]]:
    """Truncate long tool result content in-place.

    Returns a new list with truncated content. Non-tool messages are
    returned unchanged.
    """
    result: list[dict[str, Any]] = []
    truncated_count = 0
    saved_chars = 0

    for msg in messages:
        if msg.get("role") != "tool":
            result.append(msg)
            continue

        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > max_chars:
            saved_chars += len(content) - max_chars
            truncated_count += 1
            result.append({
                **msg,
                "content": content[:max_chars]
                + f"\n... (truncated, {len(content)} chars originally)",
            })
        else:
            result.append(msg)

    if truncated_count:
        logger.info(
            "MemoryCompressor: truncated %d tool results, saved ~%d chars",
            truncated_count,
            saved_chars,
        )

    return result


def _build_summary_prompt(rounds: list[list[dict[str, Any]]]) -> str:
    """Build a summarization prompt from the given rounds."""
    lines: list[str] = [
        "Summarize the following conversation concisely. "
        "Preserve key information, decisions, code changes, "
        "file paths, and any important context the assistant needs to know. "
        "Write in the same language as the original conversation.\n"
    ]

    round_idx = 0
    for group in rounds:
        round_idx += 1
        lines.append(f"--- Round {round_idx} ---")
        for msg in group:
            role = msg.get("role", "?")
            content = msg.get("content", "")

            if isinstance(content, str) and content:
                # Truncate very long content for the summary prompt
                text = content[:2000]
                if len(content) > 2000:
                    text += " (...)"
                lines.append(f"[{role}]: {text}")

            tcs = msg.get("tool_calls")
            if isinstance(tcs, list):
                for tc in tcs:
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args = fn.get("arguments", "{}")[:200]
                    lines.append(f"[assistant called tool: {name}({args})]")

    lines.append("\n---\nSummary:")
    return "\n".join(lines)


def _truncate_rounds(
    messages: list[dict[str, Any]],
    target_ratio: float,
    min_rounds_to_keep: int,
) -> list[dict[str, Any]]:
    """Truncate oldest rounds, keeping only recent ones.

    Separates system messages, groups non-system into rounds, then keeps
    the most recent rounds. Older rounds are replaced with a single
    summary placeholder message.
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    rounds = _group_into_rounds(non_system)
    total_rounds = len(rounds)

    if total_rounds <= min_rounds_to_keep:
        return messages  # nothing to truncate

    keep = max(min_rounds_to_keep, int(total_rounds * target_ratio))
    keep = min(keep, total_rounds)  # safety

    compressed = rounds[:-keep]
    kept = rounds[-keep:]

    # Build result
    result = list(system_msgs)

    if compressed:
        result.append({
            "role": "user",
            "content": (
                f"[Previous conversation compressed. "
                f"{len(compressed)} round(s) of history removed, "
                f"{sum(len(g) for g in compressed)} messages. "
                f"Keeping the last {len(kept)} round(s).]"
            ),
        })

    for group in kept:
        result.extend(group)

    logger.info(
        "MemoryCompressor: truncated %d rounds → %d rounds kept",
        len(compressed),
        len(kept),
    )

    return result


class MemoryCompressor:
    """Core memory compressor for conversation history.

    Provides multiple compression strategies and token estimation.

    Args:
        summarize_func: Optional async callable that takes a prompt string
            and returns a summary string. Required for ``summarize`` and
            ``hybrid`` strategies.
    """

    def __init__(
        self,
        summarize_func: Callable[[str], Awaitable[str]] | None = None,
    ) -> None:
        self._summarize_func = summarize_func

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Estimate the total token count of a message list."""
        return estimate_messages_tokens(messages)

    def should_compress(
        self,
        messages: list[dict[str, Any]],
        threshold_tokens: int,
    ) -> bool:
        """Check if messages exceed the compression threshold."""
        return self.estimate_tokens(messages) > threshold_tokens

    async def compress(
        self,
        messages: list[dict[str, Any]],
        strategy: CompressionStrategy | str = CompressionStrategy.HYBRID,
        target_ratio: float = 0.5,
        min_rounds_to_keep: int = 3,
        tool_result_max_chars: int = _DEFAULT_TOOL_RESULT_MAX_CHARS,
    ) -> list[dict[str, Any]]:
        """Compress message history using the given strategy.

        Args:
            messages: The full message list (system + turns).
            strategy: Compression strategy to use.
            target_ratio: Target ratio of original size (0.0-1.0).
            min_rounds_to_keep: Minimum number of conversation rounds to keep.
            tool_result_max_chars: Max chars per tool result field.

        Returns:
            Compressed message list.
        """
        if isinstance(strategy, str):
            strategy = CompressionStrategy(strategy)

        if strategy == CompressionStrategy.DROP_TOOL_RESULTS:
            return _drop_tool_result_details(messages, tool_result_max_chars)

        if strategy == CompressionStrategy.TRUNCATE:
            return _truncate_rounds(messages, target_ratio, min_rounds_to_keep)

        if strategy == CompressionStrategy.SUMMARIZE:
            return await self._summarize_compress(
                messages, target_ratio, min_rounds_to_keep,
            )

        if strategy == CompressionStrategy.HYBRID:
            return await self._hybrid_compress(
                messages, target_ratio, min_rounds_to_keep, tool_result_max_chars,
            )

        return messages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _summarize_compress(
        self,
        messages: list[dict[str, Any]],
        target_ratio: float,
        min_rounds_to_keep: int,
    ) -> list[dict[str, Any]]:
        """Use LLM to summarize older rounds."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        rounds = _group_into_rounds(non_system)
        total_rounds = len(rounds)

        if total_rounds <= min_rounds_to_keep:
            return messages

        keep = max(min_rounds_to_keep, int(total_rounds * target_ratio))
        keep = min(keep, total_rounds)

        to_summarize = rounds[:-keep]
        kept = rounds[-keep:]

        summary_text = self._make_fallback_summary(to_summarize)

        # If we have a summarize function, use it
        if self._summarize_func is not None:
            try:
                prompt = _build_summary_prompt(to_summarize)
                summary_text = await self._summarize_func(prompt)
            except Exception as exc:
                logger.warning(
                    "MemoryCompressor: LLM summarization failed (%s), "
                    "using fallback summary",
                    exc,
                )

        result = list(system_msgs)
        result.append({"role": "user", "content": summary_text})
        for group in kept:
            result.extend(group)

        logger.info(
            "MemoryCompressor: summarized %d rounds → 1 summary + %d rounds kept",
            len(to_summarize),
            len(kept),
        )

        return result

    async def _hybrid_compress(
        self,
        messages: list[dict[str, Any]],
        target_ratio: float,
        min_rounds_to_keep: int,
        tool_result_max_chars: int,
    ) -> list[dict[str, Any]]:
        """Hybrid: drop tool details first, then summarize if still over threshold."""
        result = _drop_tool_result_details(messages, tool_result_max_chars)

        # If still over threshold, summarize older rounds
        if self._summarize_func is not None:
            rounds = _group_into_rounds(
                [m for m in result if m.get("role") != "system"]
            )
            if len(rounds) > min_rounds_to_keep * 2:
                result = await self._summarize_compress(
                    result, target_ratio, min_rounds_to_keep,
                )
        else:
            # Fallback to truncation
            result = _truncate_rounds(result, target_ratio, min_rounds_to_keep)

        return result

    @staticmethod
    def _make_fallback_summary(
        rounds: list[list[dict[str, Any]]],
    ) -> str:
        """Generate a simple text summary without an LLM."""
        total_msgs = sum(len(g) for g in rounds)
        tools_used: set[str] = set()
        user_topics: list[str] = []

        for group in rounds:
            for msg in group:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content:
                        user_topics.append(content[:120])
                tcs = msg.get("tool_calls")
                if isinstance(tcs, list):
                    for tc in tcs:
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            tools_used.add(fn["name"])

        parts = [
            f"[Previous conversation: {len(rounds)} round(s), "
            f"{total_msgs} message(s).",
        ]
        if user_topics:
            topics = "; ".join(
                t.replace("\n", " ").strip()
                for t in user_topics[:5]
            )
            parts.append(f"Topics: {topics}")
        if tools_used:
            parts.append(f"Tools used: {', '.join(sorted(tools_used))}")
        parts.append("]")

        return " ".join(parts)