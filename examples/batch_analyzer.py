"""Example: batch analysis with extended thinking using Message Batches API.

Demonstrates analyzing a list of items in parallel via the Batches API,
with thinking blocks for deep reasoning on each item.

Run::

    python examples/batch_analyzer.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from patterns.batch_processing import BatchItem, BatchThinkingProcessor, chunk_items

SAMPLE_ITEMS = [
    "Evaluate the trade-offs between microservices and monolithic architecture.",
    "Analyze the CAP theorem implications for a distributed database.",
    "Compare RAFT vs Paxos consensus algorithms for fault tolerance.",
    "Assess the security implications of JWT vs session-based auth.",
    "Review the pros and cons of event sourcing vs CRUD databases.",
]


async def run_batch_analysis(client: Any = None) -> None:
    """Run batch extended thinking analysis on sample items.

    Args:
        client: Anthropic async client. If None, runs in simulation mode.
    """
    processor = BatchThinkingProcessor(
        model="claude-opus-4-6",
        thinking_budget=10_000,
        max_tokens=14_000,
    )

    items = [
        BatchItem(
            item_id=f"item-{i}",
            prompt=text,
            system="You are a senior software architect. Provide deep analysis.",
        )
        for i, text in enumerate(SAMPLE_ITEMS)
    ]

    # Split into chunks if needed (API limit = 10,000 per batch)
    chunks = chunk_items(items, chunk_size=100)

    print(f"Processing {len(items)} items in {len(chunks)} batch(es)\n")

    for chunk_idx, chunk in enumerate(chunks):
        print(f"Submitting batch {chunk_idx + 1}/{len(chunks)} ({len(chunk)} items)...")
        summary = await processor.run(chunk, client=client)

        print(f"\nBatch {chunk_idx + 1} complete:")
        print(f"  Succeeded: {summary.succeeded}/{summary.total_items}")
        print(f"  Failed: {summary.failed}")
        print(f"  Total tokens: {summary.total_input_tokens + summary.total_output_tokens:,}")
        print(f"  Estimated cost: ${summary.cost_estimate_usd:.4f}")
        print(f"  Elapsed: {summary.elapsed_seconds:.1f}s\n")

        for result in summary.results:
            status = "OK" if result.success else "ERR"
            thinking_len = len(result.thinking_text)
            response_preview = result.response_text[:80].replace("\n", " ")
            print(
                f"  [{status}] {result.item_id} | thinking={thinking_len}c | {response_preview!r}"
            )

    print("\nDone!")


if __name__ == "__main__":
    # Pass a real anthropic.AsyncAnthropic() client to use the real API.
    # Without client, runs in simulation mode.
    asyncio.run(run_batch_analysis(client=None))
