# Game-Theoretical Routing for MoEs

## Setup

```bash
uv sync                    # install the environment (incl. dev dependencies)
uv run pre-commit install  # install git hooks so code quality checks run on every commit
```

## Development

- `uv run pytest` — run the test suite.
- `uv run ruff check .` / `uv run ruff format .` — lint / format.
- `uv run pre-commit run --all-files` — run all hooks against the whole repo (not just staged files).
- `uv run moe-congestion-routing` — run the package entry point.

Tests are co-located with the code they test (`src/**/<module>_test.py`), not in a separate top-level `tests/` directory, and are excluded from built distributions (`tool.uv.build-backend` in `pyproject.toml`).
