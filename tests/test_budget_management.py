"""Tests for budget_management.py."""

from __future__ import annotations

from patterns.budget_management import (
    _MAX_THINKING_BUDGET,
    _MIN_THINKING_BUDGET,
    BudgetAllocation,
    BudgetUsage,
    TaskComplexity,
    ThinkingBudgetManager,
    estimate_complexity,
)


class TestTaskComplexity:
    def test_string_values(self):
        assert TaskComplexity.LOW.value == "low"
        assert TaskComplexity.EXTREME.value == "extreme"

    def test_from_string(self):
        assert TaskComplexity("high") == TaskComplexity.HIGH


class TestBudgetAllocation:
    def test_to_api_params_with_thinking(self):
        alloc = BudgetAllocation(
            thinking_budget=8000,
            max_tokens=12000,
            complexity=TaskComplexity.HIGH,
        )
        params = alloc.to_api_params()
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 8000
        assert params["max_tokens"] == 12000

    def test_to_api_params_without_thinking(self):
        alloc = BudgetAllocation(
            thinking_budget=0,
            max_tokens=4096,
            complexity=TaskComplexity.LOW,
            thinking_enabled=False,
        )
        params = alloc.to_api_params()
        assert "thinking" not in params
        assert params["max_tokens"] == 4096


class TestBudgetUsage:
    def test_total_tokens(self):
        usage = BudgetUsage(input_tokens=500, output_tokens=300, thinking_tokens=200)
        assert usage.total_tokens == 800

    def test_thinking_fraction(self):
        usage = BudgetUsage(input_tokens=100, output_tokens=1000, thinking_tokens=750)
        assert abs(usage.thinking_fraction - 0.75) < 0.001

    def test_thinking_fraction_zero_output(self):
        usage = BudgetUsage(input_tokens=100, output_tokens=0, thinking_tokens=0)
        assert usage.thinking_fraction == 0.0

    def test_timestamp_auto_set(self):
        usage = BudgetUsage(input_tokens=10, output_tokens=20)
        assert usage.timestamp > 0


class TestThinkingBudgetManager:
    def test_allocate_default_medium(self):
        mgr = ThinkingBudgetManager(total_budget=10_000)
        alloc = mgr.allocate()
        assert alloc.complexity == TaskComplexity.MEDIUM
        assert alloc.thinking_budget == 5000  # 50% of 10k

    def test_allocate_high_complexity(self):
        mgr = ThinkingBudgetManager(total_budget=20_000)
        alloc = mgr.allocate("high")
        assert alloc.thinking_budget == 15000  # 75% of 20k

    def test_allocate_extreme_complexity(self):
        mgr = ThinkingBudgetManager(total_budget=10_000)
        alloc = mgr.allocate(TaskComplexity.EXTREME)
        # 85% of 10k = 8500, minus 1000 min output = capped at 9000
        assert alloc.thinking_budget <= 9000

    def test_min_budget_floor(self):
        # total=3000, low ratio=0.3 → 900 raw; floor lifts to 1024 < (3000-1000)=2000
        mgr = ThinkingBudgetManager(total_budget=3000)
        alloc = mgr.allocate("low")
        assert alloc.thinking_budget >= _MIN_THINKING_BUDGET

    def test_max_budget_cap(self):
        mgr = ThinkingBudgetManager(total_budget=200_000)
        alloc = mgr.allocate("extreme")
        assert alloc.thinking_budget <= _MAX_THINKING_BUDGET

    def test_output_reserved(self):
        min_output = 2000
        mgr = ThinkingBudgetManager(total_budget=10_000, min_output_tokens=min_output)
        alloc = mgr.allocate("extreme")
        assert alloc.thinking_budget <= 10_000 - min_output

    def test_allocate_string_complexity(self):
        mgr = ThinkingBudgetManager(total_budget=10_000)
        alloc = mgr.allocate("low")
        assert alloc.complexity == TaskComplexity.LOW

    def test_adaptive_adjustment(self):
        mgr = ThinkingBudgetManager(total_budget=10_000, enable_adaptation=True)
        # Record low thinking usage
        for _ in range(5):
            mgr.record_usage(BudgetUsage(input_tokens=100, output_tokens=1000, thinking_tokens=100))
        base_alloc = ThinkingBudgetManager(total_budget=10_000, enable_adaptation=False).allocate(
            "medium"
        )
        adapted_alloc = mgr.allocate("medium")
        # Adapted budget should be lower because fraction was 0.1 (much below 0.5)
        assert adapted_alloc.thinking_budget < base_alloc.thinking_budget

    def test_adaptation_disabled(self):
        mgr = ThinkingBudgetManager(total_budget=10_000, enable_adaptation=False)
        for _ in range(5):
            mgr.record_usage(BudgetUsage(input_tokens=100, output_tokens=1000, thinking_tokens=100))
        alloc = mgr.allocate("medium")
        # No adaptation → exactly 50% of 10k
        assert alloc.thinking_budget == 5000

    def test_adaptation_needs_3_samples(self):
        mgr = ThinkingBudgetManager(total_budget=10_000)
        for _ in range(2):
            mgr.record_usage(BudgetUsage(input_tokens=100, output_tokens=1000, thinking_tokens=10))
        # Only 2 samples — no adaptation
        alloc = mgr.allocate("medium")
        assert alloc.thinking_budget == 5000

    def test_allocate_for_tool_use(self):
        mgr = ThinkingBudgetManager(total_budget=10_000)
        alloc = mgr.allocate_for_tool_use(num_tools=3, complexity="medium")
        # Extra output space: 3 * 500 = 1500
        assert alloc.max_tokens == 11_500

    def test_budget_summary_empty(self):
        mgr = ThinkingBudgetManager()
        summary = mgr.budget_summary()
        assert summary["calls"] == 0
        assert summary["avg_thinking_fraction"] == 0.0

    def test_budget_summary_populated(self):
        mgr = ThinkingBudgetManager()
        mgr.record_usage(BudgetUsage(input_tokens=100, output_tokens=1000, thinking_tokens=500))
        mgr.record_usage(BudgetUsage(input_tokens=200, output_tokens=2000, thinking_tokens=1000))
        summary = mgr.budget_summary()
        assert summary["calls"] == 2
        assert summary["avg_thinking_fraction"] == 0.5
        assert summary["total_thinking_tokens"] == 1500


class TestEstimateComplexity:
    def test_simple_prompt_is_low(self):
        assert estimate_complexity("What is 2+2?") == TaskComplexity.LOW

    def test_explain_is_medium(self):
        assert estimate_complexity("Explain how recursion works.") == TaskComplexity.MEDIUM

    def test_design_is_high(self):
        assert estimate_complexity("Design a distributed cache system.") == TaskComplexity.HIGH

    def test_prove_is_extreme(self):
        assert (
            estimate_complexity("Prove the Pythagorean theorem formally.") == TaskComplexity.EXTREME
        )

    def test_long_prompt_is_medium_or_higher(self):
        long = "word " * 60
        complexity = estimate_complexity(long)
        assert complexity in {TaskComplexity.MEDIUM, TaskComplexity.HIGH, TaskComplexity.EXTREME}

    def test_many_tools_is_extreme(self):
        assert estimate_complexity("Simple task", tool_count=5) == TaskComplexity.EXTREME

    def test_some_tools_is_high(self):
        assert estimate_complexity("Simple task", tool_count=3) == TaskComplexity.HIGH
