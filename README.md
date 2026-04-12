# claude-extended-thinking-patterns

Production patterns for Claude extended thinking: adaptive budget management, context compression, batch processing, and streaming UX.

## Patterns

### Budget Management (`patterns/budget_management.py`)

Adaptive token budget allocation for extended thinking sessions.

```python
from patterns.budget_management import ThinkingBudgetManager, estimate_complexity

manager = ThinkingBudgetManager(total_budget=20_000)
complexity = estimate_complexity(prompt, tool_count=2)
params = manager.allocate(complexity).to_api_params()
# → {"thinking": {"type": "enabled", "budget_tokens": 15000}, "max_tokens": 20000}
```

### Context Compression (`patterns/context_compression.py`)

Sliding-window summarization for long multi-turn sessions.

```python
from patterns.context_compression import ContextCompressor

compressor = ContextCompressor(max_tokens=80_000, keep_recent=6)
async with compressor.session() as ctx:
    for user_msg, response in dialogue:
        ctx.add_turn("user", user_msg)
        ctx.add_turn("assistant", response)
    messages = await ctx.get_messages()  # auto-compressed
```

### Batch Processing (`patterns/batch_processing.py`)

Process thousands of items via the Message Batches API.

```python
from patterns.batch_processing import BatchItem, BatchThinkingProcessor

processor = BatchThinkingProcessor(thinking_budget=8_000)
items = [BatchItem(item_id=f"q{i}", prompt=text) for i, text in enumerate(texts)]
summary = await processor.run(items, client=anthropic_client)
for result in summary.results:
    print(result.item_id, result.response_text)
```

### Streaming Thinking (`patterns/streaming_thinking.py`)

Real-time streaming of thinking blocks for responsive UX.

```python
from patterns.streaming_thinking import stream_thinking, collect_stream

async for event in stream_thinking(client, prompt="Solve this..."):
    if event.is_thinking:
        print(f"[thinking] {event.delta}", end="", flush=True)
    elif event.is_text:
        print(event.delta, end="", flush=True)
```

## Installation

```bash
pip install -e ".[dev]"
```

## Testing

```bash
pytest --cov=patterns --cov-report=term-missing
```

## Requirements

- Python 3.12+
- `anthropic` SDK (for real API calls)

## License

MIT
