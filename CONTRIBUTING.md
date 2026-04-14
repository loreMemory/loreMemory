# Contributing to Lore Memory

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/loreMemory/loreMemory.git
cd lore-memory
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Code Style

- Use type hints on all public functions.
- Keep the core package zero-dependency (stdlib only).
- Optional features (API, MCP, embeddings) go in optional dependency groups.

## Pull Requests

1. Fork the repo and create a branch from `main`.
2. Add tests for any new functionality.
3. Make sure all tests pass with `pytest`.
4. Keep commits focused — one logical change per commit.
5. Write a clear PR description explaining what changed and why.

## Reporting Issues

Open an issue with:
- What you expected to happen.
- What actually happened.
- Steps to reproduce.
- Python version and OS.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
