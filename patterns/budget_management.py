"""Budget token management for Claude extended thinking.

Extended thinking consumes tokens fast. Without a budget strategy you'll hit
max_tokens mid-reasoning and get a truncated, useless response. This module
provides adaptive budget allocation, pre-flight estimation, and graceful
degradation when the budget is exhausted.

Pattern:
    Estimate required thinking budget → set budget_tokens
    → monitor usage → auto-adjust on next turn
    → if budget exceeded: compress context and retry

Usage::

    manager = ThinkingBudgetManager(total_budget=20_000)
    params = manager.allocate(task_complexity="high")
    # → {"thinking": {"type": "enabled", "budget_tokens": 16000}, "max_tokens": 20000}
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskComplexity(str, Enum):
    """Estimated complexity of a task, used to allocate thinking budget."""

    LOW = "low"  # Simple Q&A, factual lookup
    MEDIUM = "medium"  # Multi-step reasoning, code review
    HIGH = "high"  # Math proofs, complex planning, architecture
    EXTREME = "extreme"  # Research-level reasoning, long proofs


# Fraction of total budget allocated to thinking (rest is output)
_THINKING_RATIO: dict[TaskComplexity, float] = {
    TaskComplexity.LOW: 0.3,
    TaskComplexity.MEDIUM: 0.5,
    TaskComplexity.HIGH: 0.75,
    TaskComplexity.EXTREME: 0.85,
}

# Minimum thinking budget (below this extended thinking is useless)
_MIN_THINKING_BUDGET: int = 1024
# Maximum thinking budget per Claude API limits
_MAX_THINKING_BUDGET: int = 100_000


@dataclass
class BudgetAllocation:
    """Result of a budget allocation decision.

    Attributes:
        thinking_budget: Tokens allocated to the thinking block.
        max_tokens: Total max_tokens for the API call (thinking + output).
        complexity: The estimated task complexity used for this allocation.
        thinking_enabled: Whether extended thinking is enabled.
    """

    thinking_budget: int
    max_tokens: int
    complexity: TaskComplexity
    thinking_enabled: bool = True

    def to_api_params(self) -> dict[str, Any]:
        """Serialize to Anthropic API parameters.

        Returns:
            Dict with ``thinking`` and ``max_tokens`` keys.
        """
        if not self.thinking_enabled:
            return {"max_tokens": self.max_tokens}
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            },
            "max_tokens": self.max_tokens,
        }


@dataclass
class BudgetUsage:
    """Recorded token usage from a single extended thinking call.

    Attributes:
        input_tokens: Tokens in the prompt.
        output_tokens: Tokens in the response (includes thinking).
        thinking_tokens: Tokens consumed by the thinking block.
        request_id: Identifier for tracing.
        timestamp: When this usage was recorded.
    """

    input_tokens: int
    output_tokens: int
    thinking_tokens: int = 0
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.input_tokens + self.output_tokens

    @property
    def thinking_fraction(self) -> float:
        """Fraction of output tokens used by thinking (0.0-1.0)."""
        if self.output_tokens == 0:
            return 0.0
        return self.thinking_tokens / self.output_tokens


class ThinkingBudgetManager:
    """Adaptive token budget manager for extended thinking sessions.

    Tracks usage history and adjusts allocations to stay within cost limits
    while maximizing reasoning quality for each task complexity level.

    Args:
        total_budget: Maximum tokens per call (thinking + output combined).
        min_output_tokens: Reserved tokens for the actual response text.
        history_window: Number of recent calls to use for adaptive adjustment.
        enable_adaptation: If True, adjust budgets based on historical usage.

    Example::

        manager = ThinkingBudgetManager(total_budget=16_000)
        params = manager.allocate("high").to_api_params()
        response = client.messages.create(model="claude-opus-4-6", **params, ...)
        manager.record_usage(BudgetUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        ))
    """

    def __init__(
        self,
        total_budget: int = 16_000,
        min_output_tokens: int = 1_000,
        history_window: int = 10,
        enable_adaptation: bool = True,
    ) -> None:
        self._total = total_budget
        self._min_output = min_output_tokens
        self._window = history_window
        self._adapt = enable_adaptation
        self._history: list[BudgetUsage] = []

    def allocate(
        self,
        complexity: TaskComplexity | str = TaskComplexity.MEDIUM,
    ) -> BudgetAllocation:
        """Compute an optimal budget allocation for the given task complexity.

        Applies adaptive adjustment if enough history is available.

        Args:
            complexity: Task complexity level (string or enum).

        Returns:
            :class:`BudgetAllocation` ready for use in an API call.
        """
        if isinstance(complexity, str):
            complexity = TaskComplexity(complexity)

        ratio = _THINKING_RATIO[complexity]
        thinking_budget = int(self._total * ratio)

        # Adaptive: if recent calls used less, shrink to save cost
        if self._adapt and len(self._history) >= 3:
            recent = self._history[-self._window :]
            avg_fraction = sum(u.thinking_fraction for u in recent) / len(recent)
            # Adjust ratio toward actual usage (damped by 0.3 to avoid thrash)
            adjusted_ratio = ratio * 0.7 + avg_fraction * 0.3
            thinking_budget = int(self._total * adjusted_ratio)

        thinking_budget = max(_MIN_THINKING_BUDGET, min(thinking_budget, _MAX_THINKING_BUDGET))
        # Ensure there's room for output
        thinking_budget = min(thinking_budget, self._total - self._min_output)

        return BudgetAllocation(
            thinking_budget=thinking_budget,
            max_tokens=self._total,
            complexity=complexity,
        )

    def allocate_for_tool_use(
        self,
        num_tools: int,
        complexity: TaskComplexity | str = TaskComplexity.MEDIUM,
    ) -> BudgetAllocation:
        """Allocate budget for a call that may invoke tools.

        Tool results add to input tokens on subsequent turns. Reserve extra
        output budget to handle multi-tool chains.

        Args:
            num_tools: Number of tools available in this call.
            complexity: Base task complexity.

        Returns:
            :class:`BudgetAllocation` with adjusted budget for tool calls.
        """
        base = self.allocate(complexity)
        # Tools need more output space (results can be large)
        extra_output = min(num_tools * 500, 4_000)
        adjusted_total = self._total + extra_output
        return BudgetAllocation(
            thinking_budget=base.thinking_budget,
            max_tokens=adjusted_total,
            complexity=base.complexity,
        )

    def record_usage(self, usage: BudgetUsage) -> None:
        """Record actual token usage from a completed call.

        Args:
            usage: Token usage statistics from the API response.
        """
        self._history.append(usage)

    def budget_summary(self) -> dict[str, Any]:
        """Return a summary of budget usage across recorded calls.

        Returns:
            Dict with average tokens, thinking fraction, and call count.
        """
        if not self._history:
            return {"calls": 0, "avg_thinking_fraction": 0.0, "avg_total_tokens": 0}
        avg_frac = sum(u.thinking_fraction for u in self._history) / len(self._history)
        avg_total = sum(u.total_tokens for u in self._history) / len(self._history)
        return {
            "calls": len(self._history),
            "avg_thinking_fraction": round(avg_frac, 3),
            "avg_total_tokens": int(avg_total),
            "total_thinking_tokens": sum(u.thinking_tokens for u in self._history),
        }


def estimate_complexity(prompt: str, tool_count: int = 0) -> TaskComplexity:
    """Heuristic estimate of task complexity from prompt characteristics.

    Uses simple lexical signals — no model call required.

    Args:
        prompt: The user's prompt text.
        tool_count: Number of tools available (more tools = more complexity).

    Returns:
        Estimated :class:`TaskComplexity` level.
    """
    lower = prompt.lower()
    word_count = len(prompt.split())

    # Extreme: proofs, research, many tools
    extreme_signals = ["prove", "formal proof", "research", "exhaustive", "comprehensive analysis"]
    if any(s in lower for s in extreme_signals) or tool_count >= 5:
        return TaskComplexity.EXTREME

    # High: multi-step, architecture, complex code
    high_signals = ["design", "architecture", "implement", "optimize", "algorithm", "step by step"]
    if any(s in lower for s in high_signals) or word_count > 200 or tool_count >= 3:
        return TaskComplexity.HIGH

    # Medium: explanation, analysis, moderate length
    medium_signals = ["explain", "compare", "analyze", "review", "evaluate"]
    if any(s in lower for s in medium_signals) or word_count > 50:
        return TaskComplexity.MEDIUM

    return TaskComplexity.LOW
