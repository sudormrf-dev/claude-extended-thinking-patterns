# claude-extended-thinking-patterns

## Project Overview

Production patterns for Claude extended thinking API: budget management, context compression, batch processing, streaming.

## Development

```bash
pip install -e ".[dev]"
pytest --cov=patterns
ruff check . && ruff format --check .
mypy patterns/
bandit -r patterns/ -ll
```

## Key Design Decisions

- **No external dependencies** — pure Python stdlib only (no pydantic, no anthropic SDK required to run tests)
- **Type aliases**: `SummarizerFn = Any` in context_compression.py to avoid circular typing issues
- **Simulation mode**: all processors work without real API client (pass `client=None`)
- **Python 3.12+**: uses `type` statement for type aliases

## ruff Ignores

- `N818` in `patterns/*` — exception names don't follow Error suffix convention
- `N802` in `patterns/*` — uppercase function constructors intentional
- `T201` in `examples/*` / `benchmarks/*` — print statements intentional
