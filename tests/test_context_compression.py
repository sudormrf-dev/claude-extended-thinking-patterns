"""Tests for context_compression.py."""

from __future__ import annotations

import pytest

from patterns.context_compression import (
    CompressionStats,
    ContextCompressor,
    ConversationContext,
    Turn,
    _default_summarizer,
    _estimate_tokens,
    build_summarizer_prompt,
)


class TestTurn:
    def test_to_api_dict(self):
        turn = Turn(role="user", content="Hello!")
        d = turn.to_api_dict()
        assert d == {"role": "user", "content": "Hello!"}

    def test_token_estimate_auto(self):
        turn = Turn(role="user", content="A" * 35)
        # 35 / 3.5 = 10
        assert turn.token_estimate == 10

    def test_token_estimate_manual(self):
        turn = Turn(role="user", content="x", token_estimate=99)
        assert turn.token_estimate == 99


class TestEstimateTokens:
    def test_short_text(self):
        assert _estimate_tokens("") == 1  # min 1
        assert _estimate_tokens("abc") == 1

    def test_longer_text(self):
        assert _estimate_tokens("a" * 700) == 200


class TestConversationContext:
    def _make_context(self, max_tokens: int = 1000, keep_recent: int = 2) -> ConversationContext:
        async def dummy_summarizer(turns: list[Turn], existing: str) -> str:
            return f"Summary of {len(turns)} turns."

        return ConversationContext(
            max_tokens=max_tokens,
            summary_threshold=0.8,
            keep_recent=keep_recent,
            summarizer=dummy_summarizer,
        )

    def test_add_and_count(self):
        ctx = self._make_context()
        ctx.add_turn("user", "Hi")
        ctx.add_turn("assistant", "Hello!")
        assert ctx.turn_count == 2

    def test_add_thinking_stored(self):
        ctx = self._make_context()
        ctx.add_turn("assistant", "Answer", thinking="Deep thought")
        assert ctx.turn_count == 1

    @pytest.mark.asyncio
    async def test_get_messages_no_compression(self):
        ctx = self._make_context(max_tokens=10_000)
        ctx.add_turn("user", "Hello")
        ctx.add_turn("assistant", "Hi!")
        msgs = await ctx.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_compression_triggered(self):
        # Very small budget to force compression
        ctx = self._make_context(max_tokens=50, keep_recent=1)
        # Add enough content to exceed threshold
        for i in range(5):
            ctx.add_turn("user", "A" * 100)
            ctx.add_turn("assistant", "B" * 100)
        msgs = await ctx.get_messages()
        # Should have summary prefix + recent + assistant ack
        assert ctx.compression_count >= 1
        assert any("Summary" in m["content"] or "compressed" in m["content"].lower() for m in msgs)

    @pytest.mark.asyncio
    async def test_keep_recent_preserved(self):
        ctx = self._make_context(max_tokens=50, keep_recent=2)
        for i in range(10):
            ctx.add_turn("user", f"Message {i} " + "X" * 50)
        # Force compression
        await ctx.get_messages()
        # Should keep at most keep_recent turns after compression
        assert ctx.turn_count <= 2

    @pytest.mark.asyncio
    async def test_summary_prepended(self):
        ctx = self._make_context(max_tokens=50, keep_recent=1)
        for i in range(5):
            ctx.add_turn("user", "X" * 100)
        await ctx.get_messages()
        assert ctx._prepended_summary != ""

    def test_estimated_tokens_sum(self):
        ctx = self._make_context()
        ctx.add_turn("user", "A" * 35)  # 10 tokens
        ctx.add_turn("user", "A" * 35)  # 10 tokens
        # empty _prepended_summary counts as 1 token minimum → 10 + 10 + 1 = 21
        assert ctx.estimated_tokens == 21

    @pytest.mark.asyncio
    async def test_no_compression_when_under_threshold(self):
        ctx = self._make_context(max_tokens=100_000)
        ctx.add_turn("user", "Hello")
        ctx.add_turn("assistant", "Hi!")
        msgs = await ctx.get_messages()
        # No summary prefix injected
        assert all("Summary" not in m["content"] for m in msgs)
        assert ctx.compression_count == 0


class TestContextCompressor:
    @pytest.mark.asyncio
    async def test_session_context_manager(self):
        compressor = ContextCompressor(max_tokens=10_000)
        async with compressor.session() as ctx:
            ctx.add_turn("user", "Hello")
            assert ctx.turn_count == 1

    def test_create_context(self):
        compressor = ContextCompressor()
        ctx = compressor.create_context()
        assert isinstance(ctx, ConversationContext)

    @pytest.mark.asyncio
    async def test_custom_summarizer(self):
        async def my_summarizer(turns: list[Turn], existing: str) -> str:
            return "CUSTOM SUMMARY"

        # keep_recent=1 ensures turns beyond the first get summarized
        compressor = ContextCompressor(max_tokens=50, keep_recent=1, summarizer=my_summarizer)
        async with compressor.session() as ctx:
            for _ in range(5):
                ctx.add_turn("user", "X" * 100)
            await ctx.get_messages()
            assert ctx._prepended_summary == "CUSTOM SUMMARY"


class TestDefaultSummarizer:
    @pytest.mark.asyncio
    async def test_returns_string(self):
        turns = [
            Turn(role="user", content="Hello"),
            Turn(role="assistant", content="Hi! How can I help you today?"),
        ]
        result = await _default_summarizer(turns, "")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_incorporates_existing_summary(self):
        turns = [Turn(role="assistant", content="The answer is yes.")]
        result = await _default_summarizer(turns, "Prior: we discussed X.")
        assert "Prior: we discussed X." in result

    @pytest.mark.asyncio
    async def test_empty_turns(self):
        result = await _default_summarizer([], "")
        assert isinstance(result, str)


class TestBuildSummarizerPrompt:
    def test_includes_turns(self):
        turns = [
            Turn(role="user", content="Hello there"),
            Turn(role="assistant", content="Hi! I'm here."),
        ]
        prompt = build_summarizer_prompt(turns)
        assert "Hello there" in prompt
        assert "Hi! I'm here." in prompt
        assert "USER" in prompt
        assert "ASSISTANT" in prompt

    def test_truncates_long_content(self):
        turns = [Turn(role="user", content="A" * 1000)]
        prompt = build_summarizer_prompt(turns)
        # Content is truncated to 500 in the prompt
        user_section = prompt.split("[USER]:")[1]
        assert len(user_section.strip()) <= 510  # some slack for whitespace


class TestCompressionStats:
    def test_compression_ratio(self):
        stats = CompressionStats(
            turns_before=10,
            turns_after=2,
            tokens_before=1000,
            tokens_after=200,
            summary_tokens=50,
        )
        assert abs(stats.compression_ratio - 0.2) < 0.001

    def test_compression_ratio_zero_before(self):
        stats = CompressionStats(
            turns_before=0,
            turns_after=0,
            tokens_before=0,
            tokens_after=0,
            summary_tokens=0,
        )
        assert stats.compression_ratio == 1.0
