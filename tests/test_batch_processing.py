"""Tests for batch_processing.py."""

from __future__ import annotations

import pytest

from patterns.batch_processing import (
    BatchItem,
    BatchJobSummary,
    BatchResult,
    BatchThinkingProcessor,
    build_batch_request,
    chunk_items,
    parse_batch_result,
)


class TestBatchItem:
    def test_defaults(self):
        item = BatchItem(item_id="x", prompt="hello")
        assert item.system == ""
        assert item.thinking_budget == 0

    def test_custom_fields(self):
        item = BatchItem(
            item_id="q1",
            prompt="Analyze this",
            system="Be analytical",
            thinking_budget=4000,
        )
        assert item.thinking_budget == 4000


class TestBatchResult:
    def test_total_tokens(self):
        r = BatchResult(
            item_id="x",
            response_text="ok",
            input_tokens=100,
            output_tokens=50,
        )
        assert r.total_tokens == 150

    def test_defaults_success(self):
        r = BatchResult(item_id="x", response_text="hello")
        assert r.success is True
        assert r.error_message == ""


class TestBatchJobSummary:
    def _make_summary(self) -> BatchJobSummary:
        return BatchJobSummary(
            batch_id="b1",
            total_items=10,
            succeeded=8,
            failed=2,
            total_input_tokens=5000,
            total_output_tokens=2000,
            elapsed_seconds=30.0,
        )

    def test_success_rate(self):
        s = self._make_summary()
        assert abs(s.success_rate - 0.8) < 0.001

    def test_success_rate_zero_items(self):
        s = BatchJobSummary(
            batch_id="x",
            total_items=0,
            succeeded=0,
            failed=0,
            total_input_tokens=0,
            total_output_tokens=0,
            elapsed_seconds=0,
        )
        assert s.success_rate == 0.0

    def test_cost_estimate_positive(self):
        s = self._make_summary()
        assert s.cost_estimate_usd >= 0.0


class TestBuildBatchRequest:
    def test_structure(self):
        item = BatchItem(item_id="q1", prompt="Hello")
        req = build_batch_request(item, "claude-opus-4-6", 8000, 12000)
        assert req["custom_id"] == "q1"
        assert req["params"]["thinking"]["budget_tokens"] == 8000
        assert req["params"]["max_tokens"] == 12000
        assert req["params"]["messages"][0]["content"] == "Hello"

    def test_item_budget_override(self):
        item = BatchItem(item_id="q1", prompt="Hi", thinking_budget=4000)
        req = build_batch_request(item, "claude-opus-4-6", 8000, 12000)
        assert req["params"]["thinking"]["budget_tokens"] == 4000

    def test_system_prompt_included(self):
        item = BatchItem(item_id="q1", prompt="Hi", system="Be concise.")
        req = build_batch_request(item, "claude-opus-4-6", 8000, 12000)
        assert req["params"]["system"] == "Be concise."

    def test_system_prompt_omitted_when_empty(self):
        item = BatchItem(item_id="q1", prompt="Hi")
        req = build_batch_request(item, "claude-opus-4-6", 8000, 12000)
        assert "system" not in req["params"]


class TestParseBatchResult:
    def test_success_result(self):
        raw = {
            "custom_id": "q1",
            "result": {
                "type": "message",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "I considered this..."},
                        {"type": "text", "text": "The answer is 42."},
                    ],
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
            },
        }
        result = parse_batch_result(raw)
        assert result.item_id == "q1"
        assert result.success is True
        assert result.response_text == "The answer is 42."
        assert "I considered this" in result.thinking_text
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    def test_error_result(self):
        raw = {
            "custom_id": "q2",
            "result": {
                "type": "error",
                "error": {"message": "Rate limit exceeded"},
            },
        }
        result = parse_batch_result(raw)
        assert result.item_id == "q2"
        assert result.success is False
        assert "Rate limit" in result.error_message

    def test_missing_custom_id(self):
        raw = {
            "result": {
                "type": "error",
                "error": {"message": "oops"},
            },
        }
        result = parse_batch_result(raw)
        assert result.item_id == "unknown"

    def test_multiple_text_blocks(self):
        raw = {
            "custom_id": "q3",
            "result": {
                "type": "message",
                "message": {
                    "content": [
                        {"type": "text", "text": "Part 1."},
                        {"type": "text", "text": " Part 2."},
                    ],
                    "usage": {},
                },
            },
        }
        result = parse_batch_result(raw)
        assert "Part 1" in result.response_text
        assert "Part 2" in result.response_text


class TestBatchThinkingProcessor:
    @pytest.mark.asyncio
    async def test_simulation_mode(self):
        processor = BatchThinkingProcessor()
        items = [BatchItem(item_id=f"q{i}", prompt=f"Question {i}") for i in range(3)]
        summary = await processor.run(items, client=None)
        assert summary.total_items == 3
        assert summary.succeeded == 3
        assert summary.failed == 0
        assert len(summary.results) == 3

    @pytest.mark.asyncio
    async def test_simulation_result_content(self):
        processor = BatchThinkingProcessor()
        items = [BatchItem(item_id="test-1", prompt="Analyze this")]
        summary = await processor.run(items)
        result = summary.results[0]
        assert result.item_id == "test-1"
        assert "Simulated" in result.response_text

    @pytest.mark.asyncio
    async def test_empty_items(self):
        processor = BatchThinkingProcessor()
        summary = await processor.run([])
        assert summary.total_items == 0
        assert summary.succeeded == 0


class TestChunkItems:
    def test_even_split(self):
        items = [BatchItem(item_id=f"i{n}", prompt="x") for n in range(6)]
        chunks = chunk_items(items, chunk_size=2)
        assert len(chunks) == 3
        assert all(len(c) == 2 for c in chunks)

    def test_uneven_split(self):
        items = [BatchItem(item_id=f"i{n}", prompt="x") for n in range(7)]
        chunks = chunk_items(items, chunk_size=3)
        assert len(chunks) == 3
        assert len(chunks[-1]) == 1

    def test_single_chunk(self):
        items = [BatchItem(item_id=f"i{n}", prompt="x") for n in range(5)]
        chunks = chunk_items(items, chunk_size=100)
        assert len(chunks) == 1

    def test_empty(self):
        assert chunk_items([], 10) == []
