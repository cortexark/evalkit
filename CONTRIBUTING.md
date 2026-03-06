# Contributing to evalkit

Thanks for your interest in contributing to evalkit! This guide will help you get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/cortexark/evalkit.git
cd evalkit

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
make test

# Run with coverage
make coverage

# Run specific test file
pytest tests/test_judges.py -v
```

## Code Quality

```bash
# Lint
make lint

# Format
make format

# Type check
make typecheck
```

## Making Changes

1. **Fork** the repo and create a feature branch from `main`
2. **Write tests** for any new functionality
3. **Run the full test suite** before submitting
4. **Follow existing code style** — we use ruff for linting and formatting
5. **Write clear commit messages** describing the change
6. **Submit a PR** with a description of your changes

## Architecture

See [docs/architecture.md](docs/architecture.md) for an overview of the codebase structure. Key decisions are documented in [ADRs](docs/adr/).

## Adding a New Judge

1. Create a new class in `src/evalkit/judges/` extending `BaseJudge`
2. Implement the `evaluate()` method
3. Add tests in `tests/test_judges.py`
4. Update the judges `__init__.py` exports

## Adding a New Rubric

1. Add your rubric definition in `src/evalkit/judges/rubrics.py`
2. Include scoring criteria and examples
3. Add tests validating the rubric structure

## Reporting Issues

Use [GitHub Issues](https://github.com/cortexark/evalkit/issues) with the provided templates for bugs and feature requests.

## Code of Conduct

Be respectful, constructive, and collaborative. We're building tools to make LLM evaluation better for everyone.
