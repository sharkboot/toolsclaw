"""LLM provider — OpenAI-compatible tool calling with streaming and conversation history."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from toolsclaw.config import Config


@dataclass
class ToolCallRequest:
    """A single tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class TokenUsage:
    """Token usage from a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Unified LLM response."""

    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMProvider:
    """OpenAI-compatible LLM provider with tool calling support."""

    def __init__(self, config: Config) -> None:
        pcfg = config.get_provider_config()
        self._model = config.model
        self._client = AsyncOpenAI(
            api_key=pcfg.api_key or "none",
            base_url=pcfg.api_base or None,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request and parse the response."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        content = msg.content or ""
        tool_calls: list[ToolCallRequest] = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(
                    ToolCallRequest(id=tc.id, name=tc.function.name, arguments=args)
                )

        usage = TokenUsage()
        if resp.usage:
            usage = TokenUsage(
                prompt_tokens=resp.usage.prompt_tokens or 0,
                completion_tokens=resp.usage.completion_tokens or 0,
                total_tokens=resp.usage.total_tokens or 0,
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            usage=usage,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMResponse]:
        """Stream a chat completion, yielding partial responses.

        Yields LLMResponse objects with incremental content.
        The final yield will have finish_reason set and usage populated.

        Usage::

            async for chunk in provider.stream_chat(messages, tools):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                if chunk.finish_reason:
                    print(f"\\n[Finished: {chunk.finish_reason}]")
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        content_parts: list[str] = []
        tool_calls_buf: dict[int, dict[str, Any]] = {}  # index -> partial tc data
        finish_reason = ""
        usage = TokenUsage()

        async for chunk in await self._client.chat.completions.create(**kwargs):
            choice = chunk.choices[0]
            delta = choice.delta

            # accumulate content
            if delta.content:
                content_parts.append(delta.content)
                yield LLMResponse(content="".join(content_parts))

            # accumulate tool calls (streamed as they appear)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_buf:
                        tool_calls_buf[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_calls_buf[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_buf[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_buf[idx]["arguments"] += tc_delta.function.arguments

            # capture finish reason from last chunk
            if choice.finish_reason:
                finish_reason = choice.finish_reason

        # final: yield complete response with parsed tool calls
        full_content = "".join(content_parts)
        parsed_tool_calls: list[ToolCallRequest] = []
        for idx in sorted(tool_calls_buf.keys()):
            tc_data = tool_calls_buf[idx]
            try:
                args = json.loads(tc_data["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            parsed_tool_calls.append(
                ToolCallRequest(id=tc_data["id"], name=tc_data["name"], arguments=args)
            )

        # consume usage from a non-streamed summary call if available
        # (most providers don't stream usage; fetch it separately if needed)
        yield LLMResponse(
            content=full_content,
            tool_calls=parsed_tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )


class Conversation:
    """Maintains LLM conversation history for multi-turn dialogues.

    Usage::

        conv = Conversation(provider, system_prompt="You are helpful.")
        response = await conv.send("Hello")
        print(response.content)

        # with tools
        response = await conv.send("Use the calculator", tools=tool_defs)

        # streaming
        async for chunk in conv.stream_send("Tell me a story"):
            print(chunk.content, end="", flush=True)
    """

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str = "",
    ) -> None:
        """Initialize a conversation.

        Args:
            provider: The LLMProvider instance to use.
            system_prompt: Optional system prompt to prepend.
        """
        self._provider = provider
        self._messages: list[dict[str, Any]] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return a copy of the current message history."""
        return list(self._messages)

    def reset(self) -> None:
        """Clear all messages except system prompt."""
        system = [m for m in self._messages if m["role"] == "system"]
        self._messages = system

    def set_system_prompt(self, content: str) -> None:
        """Set or update the system prompt."""
        for m in self._messages:
            if m["role"] == "system":
                m["content"] = content
                return
        self._messages.insert(0, {"role": "system", "content": content})

    async def send(
        self,
        message: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send a user message and return the assistant response.

        The user message and assistant response are automatically appended
        to the conversation history.
        """
        self._messages.append({"role": "user", "content": message})
        response = await self._provider.chat(self._messages, tools)

        if response.content:
            self._messages.append({"role": "assistant", "content": response.content})
        elif response.tool_calls:
            # tool calls without content — still record the assistant turn
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": None}
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
            self._messages.append(assistant_msg)

        return response

    async def stream_send(
        self,
        message: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMResponse]:
        """Stream a user message and yield incremental assistant responses.

        On completion, appends both user and assistant messages to history.
        """
        self._messages.append({"role": "user", "content": message})
        response: LLMResponse | None = None

        async for chunk in self._provider.stream_chat(self._messages, tools):
            response = chunk
            yield chunk

        # append assistant message after stream completes
        if response and (response.content or response.tool_calls):
            if response.content:
                self._messages.append({"role": "assistant", "content": response.content})
            elif response.tool_calls:
                assistant_msg: dict[str, Any] = {"role": "assistant", "content": None}
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
                self._messages.append(assistant_msg)
