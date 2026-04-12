"""Batch processing for extended thinking via Claude Message Batches API.

Extended thinking on many items in parallel is expensive if done one-by-one:
each call blocks for 20-60s. The Message Batches API accepts up to 10,000
requests, processes them asynchronously, and returns results when complete.

Pattern:
    Build batch requests → submit batch → poll until complete
    → stream results → parse thinking blocks

Usage::

    processor = BatchThinkingProcessor()
    items = [{"id": f"item-{i}", "prompt": f"Analyze: {text}"} for i, text in enumerate(texts)]
    results = await processor.run(items, complexity="high")
    for result in results:
        print(result.item_id, result.response_text)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BatchStatus(str, Enum):
    """Status of a submitted batch job."""

    PENDING = "in_progress"
    COMPLETE = "ended"
    FAILED = "error"


@dataclass
class BatchItem:
    """A single item to process in a batch.

    Attributes:
        item_id: Unique identifier (alphanumeric + hyphens only).
        prompt: The user prompt for this item.
        system: Optional system prompt override.
        thinking_budget: Per-item budget override (0 = use batch default).
    """

    item_id: str
    prompt: str
    system: str = ""
    thinking_budget: int = 0


@dataclass
class BatchResult:
    """Result for a single item in a batch.

    Attributes:
        item_id: Matches the input :attr:`BatchItem.item_id`.
        response_text: Final response text (thinking stripped).
        thinking_text: Raw thinking block content.
        success: False if this item errored.
        error_message: Set when success is False.
        input_tokens: Tokens consumed by the prompt.
        output_tokens: Tokens consumed by the completion.
    """

    item_id: str
    response_text: str
    thinking_text: str = ""
    success: bool = True
    error_message: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed by this item."""
        return self.input_tokens + self.output_tokens


@dataclass
class BatchJobSummary:
    """Summary statistics for a completed batch job.

    Attributes:
        batch_id: API-assigned batch identifier.
        total_items: Total number of items submitted.
        succeeded: Items that completed successfully.
        failed: Items that errored.
        total_input_tokens: Sum of input tokens across all items.
        total_output_tokens: Sum of output tokens across all items.
        elapsed_seconds: Wall-clock time from submit to complete.
    """

    batch_id: str
    total_items: int
    succeeded: int
    failed: int
    total_input_tokens: int
    total_output_tokens: int
    elapsed_seconds: float
    results: list[BatchResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of items that succeeded."""
        if self.total_items == 0:
            return 0.0
        return self.succeeded / self.total_items

    @property
    def cost_estimate_usd(self) -> float:
        """Rough cost estimate based on claude-opus-4-6 pricing.

        Uses public pricing as of 2025-Q1 — verify before production use.
        """
        # $15 / 1M input tokens, $75 / 1M output tokens (opus)
        input_cost = (self.total_input_tokens / 1_000_000) * 15.0
        output_cost = (self.total_output_tokens / 1_000_000) * 75.0
        return round(input_cost + output_cost, 4)


def build_batch_request(
    item: BatchItem,
    model: str,
    default_thinking_budget: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Build a single batch request dict for the Anthropic Batches API.

    Args:
        item: The item to process.
        model: Model identifier (e.g. ``"claude-opus-4-6"``).
        default_thinking_budget: Budget used when item has no override.
        max_tokens: ``max_tokens`` for this request.

    Returns:
        Dict matching the ``BatchCreateParams.requests`` item schema.
    """
    budget = item.thinking_budget if item.thinking_budget > 0 else default_thinking_budget
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled", "budget_tokens": budget},
        "messages": [{"role": "user", "content": item.prompt}],
    }
    if item.system:
        params["system"] = item.system
    return {
        "custom_id": item.item_id,
        "params": params,
    }


def parse_batch_result(raw: dict[str, Any]) -> BatchResult:
    """Parse a raw batch result dict into a :class:`BatchResult`.

    Handles both successful and error responses from the API.

    Args:
        raw: A single result object from the Batches API response stream.

    Returns:
        Parsed :class:`BatchResult`.
    """
    item_id: str = raw.get("custom_id", "unknown")
    result_obj: dict[str, Any] = raw.get("result", {})
    result_type: str = result_obj.get("type", "error")

    if result_type == "error":
        error = result_obj.get("error", {})
        return BatchResult(
            item_id=item_id,
            response_text="",
            success=False,
            error_message=error.get("message", "Unknown error"),
        )

    message = result_obj.get("message", {})
    content_blocks: list[dict[str, Any]] = message.get("content", [])
    usage: dict[str, int] = message.get("usage", {})

    thinking_parts: list[str] = []
    text_parts: list[str] = []

    for block in content_blocks:
        block_type = block.get("type", "")
        if block_type == "thinking":
            thinking_parts.append(block.get("thinking", ""))
        elif block_type == "text":
            text_parts.append(block.get("text", ""))

    return BatchResult(
        item_id=item_id,
        response_text="\n".join(text_parts).strip(),
        thinking_text="\n".join(thinking_parts).strip(),
        success=True,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )


class BatchThinkingProcessor:
    """Async processor that submits extended thinking requests as a batch.

    Args:
        model: Claude model to use.
        thinking_budget: Default per-item thinking token budget.
        max_tokens: ``max_tokens`` per request.
        poll_interval: Seconds between polling batch status.
        max_wait_seconds: Give up after this many seconds (0 = wait forever).

    Example::

        processor = BatchThinkingProcessor(thinking_budget=8_000)
        items = [
            BatchItem(item_id="q1", prompt="Solve: ..."),
            BatchItem(item_id="q2", prompt="Analyze: ..."),
        ]
        summary = await processor.run(items)
        for result in summary.results:
            print(result.response_text)
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        thinking_budget: int = 8_000,
        max_tokens: int = 12_000,
        poll_interval: float = 5.0,
        max_wait_seconds: float = 0,
    ) -> None:
        self._model = model
        self._budget = thinking_budget
        self._max_tokens = max_tokens
        self._poll_interval = poll_interval
        self._max_wait = max_wait_seconds

    async def run(
        self,
        items: list[BatchItem],
        client: Any = None,
    ) -> BatchJobSummary:
        """Submit items as a batch and wait for results.

        Args:
            items: Items to process. Must have unique ``item_id`` values.
            client: Anthropic async client instance. If None, a dry-run
                simulation is returned (useful for testing).

        Returns:
            :class:`BatchJobSummary` with all results attached.
        """
        if client is None:
            return self._simulate(items)

        requests = [
            build_batch_request(item, self._model, self._budget, self._max_tokens) for item in items
        ]

        start = time.monotonic()
        batch = await client.beta.messages.batches.create(requests=requests)
        batch_id: str = batch.id

        # Poll until complete
        while True:
            await asyncio.sleep(self._poll_interval)
            status = await client.beta.messages.batches.retrieve(batch_id)
            if status.processing_status == BatchStatus.COMPLETE:
                break
            if self._max_wait and (time.monotonic() - start) > self._max_wait:
                break

        elapsed = time.monotonic() - start

        # Stream results
        results = [
            parse_batch_result(raw)
            async for raw in await client.beta.messages.batches.results(batch_id)
        ]

        succeeded = sum(1 for r in results if r.success)
        return BatchJobSummary(
            batch_id=batch_id,
            total_items=len(items),
            succeeded=succeeded,
            failed=len(results) - succeeded,
            total_input_tokens=sum(r.input_tokens for r in results),
            total_output_tokens=sum(r.output_tokens for r in results),
            elapsed_seconds=elapsed,
            results=results,
        )

    def _simulate(self, items: list[BatchItem]) -> BatchJobSummary:
        """Return a synthetic batch result for testing without API calls."""
        results = [
            BatchResult(
                item_id=item.item_id,
                response_text=f"[Simulated response for: {item.prompt[:50]}]",
                thinking_text="[Simulated thinking]",
                input_tokens=len(item.prompt) // 4,
                output_tokens=100,
            )
            for item in items
        ]
        return BatchJobSummary(
            batch_id="sim-batch-001",
            total_items=len(items),
            succeeded=len(items),
            failed=0,
            total_input_tokens=sum(r.input_tokens for r in results),
            total_output_tokens=sum(r.output_tokens for r in results),
            elapsed_seconds=0.0,
            results=results,
        )


def chunk_items(items: list[BatchItem], chunk_size: int = 1000) -> list[list[BatchItem]]:
    """Split items into chunks respecting the API limit.

    The Anthropic Batches API accepts up to 10,000 requests per batch.
    This helper splits larger lists into safe chunks.

    Args:
        items: All items to batch.
        chunk_size: Max items per batch (default 1000, API allows up to 10000).

    Returns:
        List of item-list chunks.
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
