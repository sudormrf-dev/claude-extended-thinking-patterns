"""Example: long-running agent with context compression and budget management.

Demonstrates a multi-turn extended thinking session that automatically
compresses history to stay within context limits.

Run::

    python examples/long_running_agent.py
"""

from __future__ import annotations

import asyncio

from patterns.budget_management import TaskComplexity, ThinkingBudgetManager
from patterns.context_compression import ContextCompressor


async def _simulate_response(turn: int, user_msg: str) -> tuple[str, str]:
    """Return fake (thinking, response) text for simulation."""
    await asyncio.sleep(0)
    thinking = f"Turn {turn}: I'm analyzing the request '{user_msg[:40]}'..."
    response = f"Turn {turn} response: I've processed your message and here is my analysis. " * 3
    return thinking, response


async def run_long_session(num_turns: int = 20) -> None:
    """Run a multi-turn session with automatic context compression."""
    budget_manager = ThinkingBudgetManager(
        total_budget=16_000,
        enable_adaptation=True,
    )
    compressor = ContextCompressor(
        max_tokens=40_000,
        summary_threshold=0.7,
        keep_recent=4,
    )

    print(f"Starting {num_turns}-turn extended thinking session\n")

    async with compressor.session() as ctx:
        for turn in range(1, num_turns + 1):
            user_msg = f"Question {turn}: What are the implications of decision #{turn}?"

            # Allocate budget based on complexity
            complexity = TaskComplexity.HIGH if turn % 3 == 0 else TaskComplexity.MEDIUM
            alloc = budget_manager.allocate(complexity)

            # Get compressed history
            messages = await ctx.get_messages()

            # Simulate API call (replace with real anthropic client call)
            thinking, response = await _simulate_response(turn, user_msg)

            # Update history
            ctx.add_turn("user", user_msg)
            ctx.add_turn("assistant", response, thinking=thinking)

            # Record token usage for adaptation
            from patterns.budget_management import BudgetUsage

            budget_manager.record_usage(
                BudgetUsage(
                    input_tokens=len("\n".join(m.get("content", "") for m in messages)) // 4,
                    output_tokens=len(response) // 4 + len(thinking) // 4,
                    thinking_tokens=len(thinking) // 4,
                )
            )

            print(
                f"Turn {turn:2d} | "
                f"budget={alloc.thinking_budget:5d} | "
                f"turns={ctx.turn_count} | "
                f"compressions={ctx.compression_count} | "
                f"est_tokens={ctx.estimated_tokens:5d}"
            )

    summary = budget_manager.budget_summary()
    print(f"\nFinal budget summary: {summary}")


if __name__ == "__main__":
    asyncio.run(run_long_session())
