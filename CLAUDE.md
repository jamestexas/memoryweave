# MemoryWeave Development Guide

## Commands
- Install: `uv pip install -e .`
- Install dev dependencies: `uv pip install -g dev`
- Run all tests: `uv run python -m pytest`
- Run single test: `uv run python -m pytest path/to/test_file.py::TestClass::test_function`
- Run unit tests only: `uv run python -m pytest -m unit`
- Run integration tests: `uv run python -m pytest -m integration`
- Type checking: `uv run mypy memoryweave/`
- Linting: `uv run ruff check memoryweave/`
- Format code: `uv run ruff format memoryweave/`
- Run benchmarks: `uv run python -m benchmarks.memory_retrieval_benchmark`

## Code Style
- Python 3.12+ required
- Use pydantic for data validation & models
- Follow PEP 8 naming: snake_case for functions/variables, PascalCase for classes
- Use type hints throughout codebase
- Organize imports: stdlib, third-party, local
- Prefer explicit error handling with custom exceptions
- Use docstrings for public functions and classes

## Refactoring Status
The project is undergoing a major architectural refactoring. Refer to `docs/feature_matrix.md` for current implementation status of features.

## Terminology
- **Memory**: A single unit of stored information with associated embedding and metadata
- **Retrieval Strategy**: Algorithm for selecting relevant memories based on a query
- **Query Adaptation**: Process of adjusting retrieval parameters based on query characteristics
- **Pipeline**: A configurable sequence of components for memory retrieval and processing