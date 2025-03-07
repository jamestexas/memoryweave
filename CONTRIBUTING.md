# Contributing to MemoryWeave

Thank you for your interest in contributing to MemoryWeave! As this project is in active development, we appreciate your patience and understanding as we refine our architecture and APIs.

## Development Environment

We recommend using `uv` for Python environment and package management:

```bash
# Install in development mode
uv pip install -e .

# Install development dependencies
uv pip install -g dev

# Run tests
uv run python -m pytest

# Run linting
uv run ruff check memoryweave/

# Format code
uv run ruff format memoryweave/
```

## Code Style

- Use type hints throughout the codebase
- Follow PEP 8 naming: snake_case for functions/variables, PascalCase for classes
- Write docstrings for public functions and classes
- Organize imports: stdlib, third-party, local

## Pull Requests

1. Fork the repository
1. Create your feature branch (`git checkout -b feature/amazing-feature`)
1. Commit your changes (`git commit -m 'Add some amazing feature'`)
1. Push to the branch (`git push origin feature/amazing-feature`)
1. Open a Pull Request

## Testing

- Add appropriate tests for all new functionality
- Ensure existing tests pass with your changes
- Focus on unit tests for individual components

## Current Focus Areas

The project is currently focused on:

1. Core memory storage and retrieval functionality
1. Query processing and adaptation
1. Retrieval strategy optimization
1. Component-based architecture refinement

## Questions?

If you have questions or need guidance, please open an issue for discussion.
