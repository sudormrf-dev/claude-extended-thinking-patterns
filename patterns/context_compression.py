"""Async context compression for long extended thinking sessions.

Extended thinking sessions accumulate history fast — each turn adds both
the thinking block and the response. Without compression, you hit context
limits in 5-10 turns. This module provides sliding-window summarization
and token-aware pruning so long sessions stay within budget.

Pattern:
    Raw history → estimate tokens → prune if near limit
    → summarize oldest turns → prepend summary → continue

Usage::

    compressor = ContextCompressor(max_tokens=100_000, summary_threshold=0.8)
    async with compressor.session() as ctx:
        ctx.add_turn("user", "Hello")
        ctx.add_turn("assistant", "Hi!")
        # ...many turns later...
        messages = await ctx.get_messages()  # automatically compressed
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# Rough tokens-per-char estimate (conservative)
_CHARS_PER_TOKEN: float = 3.5

# System prompt injected when a compressed summary is prepended
_SUMMARY_PREFIX = (
    "The following is a compressed summary of the conversation so far:\n\n"
    "{summary}\n\n"
    "--- Continuing from the most recent turns below ---\n\n"
)


@dataclass
class Turn:
    """A single conversation turn.

    Attributes:
        role: ``user`` or ``assistant``.
        content: Message text.
        thinking: Raw thinking block text (not sent back to API).
        token_estimate: Cached token estimate for this turn.
    """

    role: str
    content: str
    thinking: str = ""
    token_estimate: int = 0

    def __post_init__(self) -> None:
        if not self.token_estimate:
            self.token_estimate = _estimate_tokens(self.content)

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize to API message format."""
        return {"role": self.role, "content": self.content}


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate from character count."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


@dataclass
class CompressionStats:
    """Statistics from a compression pass.

    Attributes:
        turns_before: Number of turns before compression.
        turns_after: Number of turns after compression.
        tokens_before: Estimated tokens before compression.
        tokens_after: Estimated tokens after compression.
        summary_tokens: Tokens in the generated summary.
    """

    turns_before: int
    turns_after: int
    tokens_before: int
    tokens_after: int
    summary_tokens: int

    @property
    def compression_ratio(self) -> float:
        """Fraction of tokens retained (0.0 = perfect, 1.0 = no compression)."""
        if self.tokens_before == 0:
            return 1.0
        return self.tokens_after / self.tokens_before


class ConversationContext:
    """Mutable conversation history with automatic compression.

    Managed by :class:`ContextCompressor` — typically used via the
    ``async with compressor.session()`` context manager.

    Args:
        max_tokens: Hard limit for combined history tokens.
        summary_threshold: Compress when usage exceeds this fraction of max_tokens.
        keep_recent: Number of most-recent turns to always preserve.
        summarizer: Async callable that summarizes a list of turns.
    """

    def __init__(
        self,
        max_tokens: int,
        summary_threshold: float,
        keep_recent: int,
        summarizer: SummarizerFn,
    ) -> None:
        self._max = max_tokens
        self._threshold = summary_threshold
        self._keep = keep_recent
        self._summarizer = summarizer
        self._turns: list[Turn] = []
        self._prepended_summary: str = ""
        self._compression_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str, thinking: str = "") -> None:
        """Append a turn to the conversation.

        Args:
            role: ``"user"`` or ``"assistant"``.
            content: Message text.
            thinking: Optional thinking block (stored but not sent to API).
        """
        self._turns.append(Turn(role=role, content=content, thinking=thinking))

    async def get_messages(self) -> list[dict[str, Any]]:
        """Return the conversation as API-format messages.

        Triggers compression if token budget is near the threshold.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts.
        """
        if self._should_compress():
            await self._compress()
        messages = []
        if self._prepended_summary:
            messages.append(
                {
                    "role": "user",
                    "content": _SUMMARY_PREFIX.format(summary=self._prepended_summary),
                }
            )
            messages.append(
                {"role": "assistant", "content": "Understood. Continuing our conversation."}
            )
        messages.extend(t.to_api_dict() for t in self._turns)
        return messages

    @property
    def estimated_tokens(self) -> int:
        """Rough token estimate for all stored turns."""
        return sum(t.token_estimate for t in self._turns) + _estimate_tokens(
            self._prepended_summary
        )

    @property
    def turn_count(self) -> int:
        """Number of stored turns."""
        return len(self._turns)

    @property
    def compression_count(self) -> int:
        """How many times this context has been compressed."""
        return self._compression_count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _should_compress(self) -> bool:
        if not self._turns:
            return False
        return self.estimated_tokens >= int(self._max * self._threshold)

    async def _compress(self) -> CompressionStats:
        tokens_before = self.estimated_tokens
        turns_before = len(self._turns)

        # Determine split: keep N recent, summarize the rest
        split = max(0, len(self._turns) - self._keep)
        to_summarize = self._turns[:split]
        to_keep = self._turns[split:]

        if not to_summarize:
            return CompressionStats(
                turns_before=turns_before,
                turns_after=turns_before,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                summary_tokens=0,
            )

        # Merge with any existing summary
        existing = (
            f"Previous summary:\n{self._prepended_summary}\n\n" if self._prepended_summary else ""
        )
        new_summary = await self._summarizer(to_summarize, existing)
        self._prepended_summary = new_summary
        self._turns = to_keep
        self._compression_count += 1

        tokens_after = self.estimated_tokens
        return CompressionStats(
            turns_before=turns_before,
            turns_after=len(self._turns),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            summary_tokens=_estimate_tokens(new_summary),
        )


# Type alias for the summarizer function
type SummarizerFn = Any  # (list[Turn], str) -> Awaitable[str]


class ContextCompressor:
    """Factory for managed conversation contexts.

    Args:
        max_tokens: Hard token budget for the history.
        summary_threshold: Compress when above this fraction of max_tokens (0.0-1.0).
        keep_recent: Number of most-recent turns never compressed away.
        summarizer: Async function that takes (turns, existing_summary) and returns
            a string summary. Defaults to a simple bullet-point extractor.

    Example::

        compressor = ContextCompressor(max_tokens=80_000)
        async with compressor.session() as ctx:
            for user_msg, assistant_msg in dialogue:
                ctx.add_turn("user", user_msg)
                ctx.add_turn("assistant", assistant_msg)
            messages = await ctx.get_messages()
    """

    def __init__(
        self,
        max_tokens: int = 80_000,
        summary_threshold: float = 0.8,
        keep_recent: int = 6,
        summarizer: SummarizerFn | None = None,
    ) -> None:
        self._max = max_tokens
        self._threshold = summary_threshold
        self._keep = keep_recent
        self._summarizer: SummarizerFn = summarizer or _default_summarizer

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ConversationContext]:
        """Async context manager yielding a fresh :class:`ConversationContext`.

        Yields:
            A new :class:`ConversationContext` scoped to this session.
        """
        ctx = ConversationContext(
            max_tokens=self._max,
            summary_threshold=self._threshold,
            keep_recent=self._keep,
            summarizer=self._summarizer,
        )
        yield ctx

    def create_context(self) -> ConversationContext:
        """Create a standalone context without the context-manager wrapper.

        Returns:
            A fresh :class:`ConversationContext`.
        """
        return ConversationContext(
            max_tokens=self._max,
            summary_threshold=self._threshold,
            keep_recent=self._keep,
            summarizer=self._summarizer,
        )


async def _default_summarizer(turns: list[Turn], existing_summary: str) -> str:
    """Lightweight summarizer that extracts key points without an LLM call.

    In production, replace with an actual Claude API call for higher quality.
    This fallback uses simple heuristics to produce a compact representation.

    Args:
        turns: Turns to summarize.
        existing_summary: Prior summary to merge in.

    Returns:
        Plain-text summary string.
    """
    await asyncio.sleep(0)  # yield to event loop

    lines: list[str] = []
    if existing_summary:
        lines.append(existing_summary)
        lines.append("")

    # Extract the first sentence of each assistant turn as a key point
    for turn in turns:
        if turn.role == "assistant":
            first_sentence = turn.content.split(".")[0].strip()
            if first_sentence:
                lines.append(f"- {first_sentence[:120]}")

    return "\n".join(lines) if lines else "(No prior context)"


def build_summarizer_prompt(turns: list[Turn]) -> str:
    """Build a prompt to summarize a list of turns via Claude.

    Useful when wiring in a real LLM-based summarizer.

    Args:
        turns: Conversation turns to summarize.

    Returns:
        Prompt text ready to send as a ``user`` message.
    """
    formatted = "\n".join(f"[{t.role.upper()}]: {t.content[:500]}" for t in turns)
    return (
        "Please provide a concise factual summary of this conversation excerpt. "
        "Focus on: decisions made, key facts established, open questions. "
        "Be brief — this summary will replace these turns in context.\n\n"
        f"{formatted}"
    )
