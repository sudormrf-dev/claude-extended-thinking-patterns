"""Streaming extended thinking for responsive UX.

Without streaming, extended thinking calls block for 20-120s before showing
anything. The streaming API lets you surface thinking progress in real-time —
showing a spinner or partial thoughts while the model reasons.

Pattern:
    Open streaming request → yield events as they arrive
    → separate thinking deltas from text deltas → flush both to UI

Usage::

    async for event in stream_thinking(client, prompt="Solve this problem..."):
        if event.is_thinking:
            print(f"[thinking] {event.delta}", end="", flush=True)
        else:
            print(event.delta, end="", flush=True)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class StreamEventType(str, Enum):
    """Type of a streaming event from extended thinking."""

    THINKING_DELTA = "thinking_delta"  # Partial thinking content
    TEXT_DELTA = "text_delta"  # Partial response text
    THINKING_COMPLETE = "thinking_done"  # Thinking block finished
    TEXT_COMPLETE = "text_done"  # Text block finished
    STREAM_COMPLETE = "stream_done"  # Full stream finished
    ERROR = "error"  # Error during streaming


@dataclass
class StreamEvent:
    """A single event from an extended thinking stream.

    Attributes:
        event_type: What kind of event this is.
        delta: Incremental text content (may be empty for non-delta events).
        thinking_tokens_so_far: Running total of thinking tokens seen.
        text_tokens_so_far: Running total of text tokens seen.
        error_message: Set only when event_type is ERROR.
        metadata: Arbitrary extra data from the API event.
    """

    event_type: StreamEventType
    delta: str = ""
    thinking_tokens_so_far: int = 0
    text_tokens_so_far: int = 0
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_thinking(self) -> bool:
        """True for thinking delta events."""
        return self.event_type == StreamEventType.THINKING_DELTA

    @property
    def is_text(self) -> bool:
        """True for text delta events."""
        return self.event_type == StreamEventType.TEXT_DELTA

    @property
    def is_final(self) -> bool:
        """True when the stream has ended."""
        return self.event_type == StreamEventType.STREAM_COMPLETE

    @property
    def is_error(self) -> bool:
        """True when an error occurred."""
        return self.event_type == StreamEventType.ERROR


@dataclass
class StreamResult:
    """Accumulated result after consuming a complete stream.

    Attributes:
        thinking_text: Full thinking block content.
        response_text: Full response text.
        input_tokens: Tokens in the prompt.
        output_tokens: Tokens in the completion.
        stop_reason: Why the model stopped.
    """

    thinking_text: str = ""
    response_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.input_tokens + self.output_tokens


class ThinkingStreamAccumulator:
    """Accumulates streaming events into a final :class:`StreamResult`.

    Tracks thinking and text separately so callers can display progress.

    Example::

        acc = ThinkingStreamAccumulator()
        async for event in stream_thinking(client, prompt):
            acc.feed(event)
            if event.is_thinking:
                update_spinner(f"Thinking... ({acc.thinking_chars} chars)")
            elif event.is_text:
                append_to_output(event.delta)
        result = acc.finalize()
    """

    def __init__(self) -> None:
        self._thinking_parts: list[str] = []
        self._text_parts: list[str] = []
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._stop_reason: str = ""

    def feed(self, event: StreamEvent) -> None:
        """Process a single streaming event.

        Args:
            event: Event to incorporate.
        """
        if event.event_type == StreamEventType.THINKING_DELTA:
            self._thinking_parts.append(event.delta)
        elif event.event_type == StreamEventType.TEXT_DELTA:
            self._text_parts.append(event.delta)
        elif event.event_type == StreamEventType.STREAM_COMPLETE:
            self._input_tokens = event.metadata.get("input_tokens", 0)
            self._output_tokens = event.metadata.get("output_tokens", 0)
            self._stop_reason = event.metadata.get("stop_reason", "end_turn")

    def finalize(self) -> StreamResult:
        """Return the accumulated result.

        Returns:
            :class:`StreamResult` with complete thinking and response text.
        """
        return StreamResult(
            thinking_text="".join(self._thinking_parts),
            response_text="".join(self._text_parts),
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            stop_reason=self._stop_reason,
        )

    @property
    def thinking_chars(self) -> int:
        """Characters accumulated in thinking so far."""
        return sum(len(p) for p in self._thinking_parts)

    @property
    def text_chars(self) -> int:
        """Characters accumulated in response text so far."""
        return sum(len(p) for p in self._text_parts)


async def stream_thinking(
    client: Any,
    prompt: str,
    system: str = "",
    model: str = "claude-opus-4-6",
    thinking_budget: int = 8_000,
    max_tokens: int = 12_000,
) -> AsyncIterator[StreamEvent]:
    """Yield streaming events from an extended thinking API call.

    Args:
        client: Anthropic async client.
        prompt: User prompt text.
        system: Optional system prompt.
        model: Model identifier.
        thinking_budget: Token budget for the thinking block.
        max_tokens: Maximum total response tokens.

    Yields:
        :class:`StreamEvent` instances as the stream progresses.
    """
    messages = [{"role": "user", "content": prompt}]
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
        "messages": messages,
    }
    if system:
        params["system"] = system

    thinking_tokens = 0
    text_tokens = 0

    try:
        async with client.messages.stream(**params) as stream:
            async for raw_event in stream:
                event_type_str: str = getattr(raw_event, "type", "")

                if event_type_str == "content_block_delta":
                    delta = getattr(raw_event, "delta", None)
                    if delta is None:
                        continue
                    delta_type = getattr(delta, "type", "")

                    if delta_type == "thinking_delta":
                        thinking_text: str = getattr(delta, "thinking", "")
                        thinking_tokens += max(1, len(thinking_text) // 4)
                        yield StreamEvent(
                            event_type=StreamEventType.THINKING_DELTA,
                            delta=thinking_text,
                            thinking_tokens_so_far=thinking_tokens,
                            text_tokens_so_far=text_tokens,
                        )

                    elif delta_type == "text_delta":
                        text: str = getattr(delta, "text", "")
                        text_tokens += max(1, len(text) // 4)
                        yield StreamEvent(
                            event_type=StreamEventType.TEXT_DELTA,
                            delta=text,
                            thinking_tokens_so_far=thinking_tokens,
                            text_tokens_so_far=text_tokens,
                        )

                elif event_type_str == "content_block_stop":
                    block_idx = getattr(raw_event, "index", -1)
                    yield StreamEvent(
                        event_type=StreamEventType.THINKING_COMPLETE
                        if block_idx == 0
                        else StreamEventType.TEXT_COMPLETE,
                        thinking_tokens_so_far=thinking_tokens,
                        text_tokens_so_far=text_tokens,
                    )

                elif event_type_str == "message_delta":
                    delta = getattr(raw_event, "delta", None)
                    usage = getattr(raw_event, "usage", None)
                    stop_reason = getattr(delta, "stop_reason", "end_turn") if delta else "end_turn"
                    meta: dict[str, Any] = {
                        "stop_reason": stop_reason,
                        "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
                    }
                    yield StreamEvent(
                        event_type=StreamEventType.STREAM_COMPLETE,
                        thinking_tokens_so_far=thinking_tokens,
                        text_tokens_so_far=text_tokens,
                        metadata=meta,
                    )

    except Exception as exc:
        yield StreamEvent(
            event_type=StreamEventType.ERROR,
            error_message=str(exc),
        )


async def collect_stream(events: AsyncIterator[StreamEvent]) -> StreamResult:
    """Consume an event iterator and return the accumulated result.

    Convenience wrapper around :class:`ThinkingStreamAccumulator`.

    Args:
        events: Async iterator of stream events.

    Returns:
        :class:`StreamResult` with complete thinking and response.
    """
    acc = ThinkingStreamAccumulator()
    async for event in events:
        acc.feed(event)
        if event.is_error:
            break
    return acc.finalize()


def render_thinking_progress(
    thinking_chars: int,
    text_chars: int,
    thinking_budget: int,
) -> str:
    """Render a one-line status string for terminal UX.

    Args:
        thinking_chars: Characters seen in thinking so far.
        text_chars: Characters seen in response so far.
        thinking_budget: Token budget (for percentage display).

    Returns:
        A status string like ``"[thinking 12% | response: 243 chars]"``.
    """
    thinking_tokens_est = thinking_chars // 4
    pct = min(100, int(thinking_tokens_est / max(1, thinking_budget) * 100))
    if text_chars:
        return f"[thinking {pct}% | response: {text_chars} chars]"
    return f"[thinking {pct}%]"


class SimulatedStreamClient:
    """Fake client for testing stream_thinking without API calls.

    Yields thinking and text events from provided text strings with
    configurable per-character delays to simulate network latency.

    Args:
        thinking_text: Fake thinking block content.
        response_text: Fake response text content.
        char_delay: Seconds between characters (default 0 = instant).

    The ``messages`` attribute mimics ``anthropic.AsyncAnthropic().messages``
    so that :func:`stream_thinking` can call ``client.messages.stream(...)``
    without modification.
    """

    def __init__(
        self,
        thinking_text: str = "Let me think about this...",
        response_text: str = "The answer is 42.",
        char_delay: float = 0.0,
    ) -> None:
        self._thinking = thinking_text
        self._response = response_text
        self._delay = char_delay
        # ``messages`` is an instance attribute so the class name avoids N801
        self.messages = _MessagesProxy(thinking_text, response_text, char_delay)


class _MessagesProxy:
    def __init__(self, thinking: str, response: str, delay: float) -> None:
        self._t = thinking
        self._r = response
        self._d = delay

    def stream(self, **kwargs: Any) -> _FakeStreamCM:
        return _FakeStreamCM(self._t, self._r, self._d)


@dataclass
class _FakeThinkingDelta:
    thinking: str
    type: str = "thinking_delta"


@dataclass
class _FakeTextDelta:
    text: str
    type: str = "text_delta"


@dataclass
class _FakeDelta:
    stop_reason: str


@dataclass
class _FakeUsage:
    output_tokens: int


class _FakeEvent:
    def __init__(
        self, event_type: str, delta: Any = None, index: int = 0, usage: Any = None
    ) -> None:
        self.type = event_type
        self.delta = delta
        self.index = index
        self.usage = usage


class _FakeStreamCM:
    def __init__(self, thinking: str, response: str, delay: float) -> None:
        self._t = thinking
        self._r = response
        self._d = delay

    async def __aenter__(self) -> _FakeStreamCM:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def __aiter__(self) -> AsyncIterator[_FakeEvent]:
        for char in self._t:
            if self._d:
                await asyncio.sleep(self._d)
            yield _FakeEvent("content_block_delta", _FakeThinkingDelta(char), index=0)
        yield _FakeEvent("content_block_stop", index=0)

        for char in self._r:
            if self._d:
                await asyncio.sleep(self._d)
            yield _FakeEvent("content_block_delta", _FakeTextDelta(char), index=1)
        yield _FakeEvent("content_block_stop", index=1)

        yield _FakeEvent(
            "message_delta",
            _FakeDelta("end_turn"),
            usage=_FakeUsage(len(self._t) // 4 + len(self._r) // 4),
        )
