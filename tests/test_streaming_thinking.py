"""Tests for streaming_thinking.py."""

from __future__ import annotations

import pytest

from patterns.streaming_thinking import (
    SimulatedStreamClient,
    StreamEvent,
    StreamEventType,
    StreamResult,
    ThinkingStreamAccumulator,
    collect_stream,
    render_thinking_progress,
    stream_thinking,
)


class TestStreamEvent:
    def test_is_thinking(self):
        event = StreamEvent(event_type=StreamEventType.THINKING_DELTA, delta="hmm")
        assert event.is_thinking is True
        assert event.is_text is False

    def test_is_text(self):
        event = StreamEvent(event_type=StreamEventType.TEXT_DELTA, delta="hi")
        assert event.is_text is True
        assert event.is_thinking is False

    def test_is_final(self):
        event = StreamEvent(event_type=StreamEventType.STREAM_COMPLETE)
        assert event.is_final is True

    def test_is_error(self):
        event = StreamEvent(event_type=StreamEventType.ERROR, error_message="boom")
        assert event.is_error is True
        assert event.is_final is False


class TestThinkingStreamAccumulator:
    def test_accumulates_thinking(self):
        acc = ThinkingStreamAccumulator()
        acc.feed(StreamEvent(event_type=StreamEventType.THINKING_DELTA, delta="Thinking"))
        acc.feed(StreamEvent(event_type=StreamEventType.THINKING_DELTA, delta="..."))
        assert acc.thinking_chars == len("Thinking...")

    def test_accumulates_text(self):
        acc = ThinkingStreamAccumulator()
        acc.feed(StreamEvent(event_type=StreamEventType.TEXT_DELTA, delta="Hello "))
        acc.feed(StreamEvent(event_type=StreamEventType.TEXT_DELTA, delta="world"))
        assert acc.text_chars == len("Hello world")

    def test_finalize(self):
        acc = ThinkingStreamAccumulator()
        acc.feed(StreamEvent(event_type=StreamEventType.THINKING_DELTA, delta="Think"))
        acc.feed(StreamEvent(event_type=StreamEventType.TEXT_DELTA, delta="Answer"))
        acc.feed(
            StreamEvent(
                event_type=StreamEventType.STREAM_COMPLETE,
                metadata={"input_tokens": 50, "output_tokens": 30, "stop_reason": "end_turn"},
            )
        )
        result = acc.finalize()
        assert result.thinking_text == "Think"
        assert result.response_text == "Answer"
        assert result.input_tokens == 50
        assert result.output_tokens == 30
        assert result.stop_reason == "end_turn"

    def test_finalize_empty(self):
        acc = ThinkingStreamAccumulator()
        result = acc.finalize()
        assert result.thinking_text == ""
        assert result.response_text == ""
        assert result.total_tokens == 0

    def test_ignores_unknown_event_type(self):
        acc = ThinkingStreamAccumulator()
        acc.feed(StreamEvent(event_type=StreamEventType.THINKING_COMPLETE))
        acc.feed(StreamEvent(event_type=StreamEventType.TEXT_COMPLETE))
        result = acc.finalize()
        assert result.thinking_text == ""


class TestStreamResult:
    def test_total_tokens(self):
        r = StreamResult(input_tokens=100, output_tokens=50)
        assert r.total_tokens == 150


class TestStreamThinking:
    @pytest.mark.asyncio
    async def test_simulated_client_yields_events(self):
        client = SimulatedStreamClient(
            thinking_text="Let me think...",
            response_text="The answer is 42.",
        )
        events = [event async for event in stream_thinking(client, prompt="What is the answer?")]

        thinking_events = [e for e in events if e.is_thinking]
        text_events = [e for e in events if e.is_text]
        final_events = [e for e in events if e.is_final]

        assert len(thinking_events) > 0
        assert len(text_events) > 0
        assert len(final_events) == 1

    @pytest.mark.asyncio
    async def test_collect_stream_accumulates(self):
        client = SimulatedStreamClient(
            thinking_text="deep thought",
            response_text="answer",
        )

        async def gen():
            async for event in stream_thinking(client, prompt="q"):
                yield event

        result = await collect_stream(gen())
        assert result.thinking_text == "deep thought"
        assert result.response_text == "answer"

    @pytest.mark.asyncio
    async def test_error_client_yields_error_event(self):
        error_client = _make_error_client()

        async def gen():
            async for event in stream_thinking(error_client, prompt="q"):
                yield event

        events = [event async for event in gen()]
        assert any(e.is_error for e in events)


class _ErrorStreamCM:
    async def __aenter__(self):
        msg = "Connection failed"
        raise RuntimeError(msg)

    async def __aexit__(self, *args):
        pass

    async def __aiter__(self):
        return
        yield  # make it a generator


class _ErrorMessagesProxy:
    def stream(self, **kwargs):
        return _ErrorStreamCM()


class _ErrorClientImpl:
    def __init__(self) -> None:
        self.messages = _ErrorMessagesProxy()


def _make_error_client() -> _ErrorClientImpl:
    """Return a fake client whose stream raises RuntimeError."""
    return _ErrorClientImpl()


class TestRenderThinkingProgress:
    def test_thinking_only(self):
        s = render_thinking_progress(thinking_chars=400, text_chars=0, thinking_budget=1000)
        assert "thinking" in s
        assert "10%" in s  # 400/4=100 tokens / 1000 = 10%

    def test_with_response(self):
        s = render_thinking_progress(thinking_chars=400, text_chars=50, thinking_budget=1000)
        assert "response" in s
        assert "50 chars" in s

    def test_caps_at_100(self):
        s = render_thinking_progress(thinking_chars=40000, text_chars=0, thinking_budget=100)
        assert "100%" in s

    def test_zero_budget_no_crash(self):
        s = render_thinking_progress(thinking_chars=100, text_chars=0, thinking_budget=0)
        assert isinstance(s, str)
