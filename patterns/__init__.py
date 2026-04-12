"""Extended thinking patterns for Claude API.

Provides adaptive budget management, context compression, batch processing,
and streaming interfaces for extended thinking sessions.
"""

from .batch_processing import (
    BatchItem,
    BatchJobSummary,
    BatchResult,
    BatchStatus,
    BatchThinkingProcessor,
    build_batch_request,
    chunk_items,
    parse_batch_result,
)
from .budget_management import (
    BudgetAllocation,
    BudgetUsage,
    TaskComplexity,
    ThinkingBudgetManager,
    estimate_complexity,
)
from .context_compression import (
    CompressionStats,
    ContextCompressor,
    ConversationContext,
    Turn,
    build_summarizer_prompt,
)
from .streaming_thinking import (
    StreamEvent,
    StreamEventType,
    StreamResult,
    ThinkingStreamAccumulator,
    collect_stream,
    render_thinking_progress,
    stream_thinking,
)

__all__ = [
    "BatchItem",
    "BatchJobSummary",
    "BatchResult",
    "BatchStatus",
    "BatchThinkingProcessor",
    "BudgetAllocation",
    "BudgetUsage",
    "CompressionStats",
    "ContextCompressor",
    "ConversationContext",
    "StreamEvent",
    "StreamEventType",
    "StreamResult",
    "TaskComplexity",
    "ThinkingBudgetManager",
    "ThinkingStreamAccumulator",
    "Turn",
    "build_batch_request",
    "build_summarizer_prompt",
    "chunk_items",
    "collect_stream",
    "estimate_complexity",
    "parse_batch_result",
    "render_thinking_progress",
    "stream_thinking",
]
