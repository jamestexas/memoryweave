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
- Run contextual fabric benchmark: `./run_contextual_fabric_benchmark.sh`
- Run baseline comparison: `uv run python run_baseline_comparison.py`
- Run synthetic benchmark: `uv run python run_synthetic_benchmark.py`
- Run semantic benchmark: `uv run python run_semantic_benchmark.py`

## Code Style

- Python 3.12+ required
- Use pydantic for data validation & models
- Follow PEP 8 naming: snake_case for functions/variables, PascalCase for classes
- Use type hints throughout codebase
- Organize imports: stdlib, third-party, local
- Prefer explicit error handling with custom exceptions
- Use docstrings for public functions and classes

## Project Status

The project has undergone a major architectural refactoring from a monolithic design to a component-based architecture. Most features have been implemented in the new architecture, but some work remains to eliminate legacy code.

Current development priorities are tracked in `docs/development_priorities.md`, which should be referenced and updated when implementing new features.

## Documentation Structure

- **readme.md**: Overview, installation, quick start, features
- **migration_guide.md**: Guide for migrating from legacy to component architecture
- **roadmap.md**: Development roadmap and refactoring progress
- **refactoring_summary.md**: Summary of refactoring work and improvements
- **baseline_comparison.md**: Guide for comparing against baseline methods
- **docs/architecture.md**: Detailed architecture documentation
- **docs/benchmark_guide.md**: Guide for running and interpreting benchmarks
- **docs/feature_matrix.md**: Current implementation status of features
- **docs/implementation_constraints.md**: Known limitations and constraints
- **docs/plan_for_improvement.md**: Detailed plan for addressing current issues
- **docs/refactoring_progress.md**: Detailed progress on architectural refactoring
- **docs/next_steps.md**: Next steps for the refactoring process
- **docs/development_priorities.md**: Prioritized development tasks and timelines
- **memoryweave/components/readme.md**: Component architecture documentation
- **memoryweave/components/retrieval_strategies/readme.md**: Retrieval strategies documentation
- **tests/readme.md**: Testing documentation and guidelines

## Key Directories

- **memoryweave/**: Main package source code
  - **core/**: Core memory functionality (partially deprecated)
  - **components/**: New component-based architecture
  - **interfaces/**: Interface definitions for components
  - **factory/**: Factory methods for creating components
  - **adapters/**: Adapters between legacy and new architecture
  - **retrieval/**: Retrieval strategies and components
  - **query/**: Query processing and adaptation
  - **storage/**: Memory and vector storage components
  - **pipeline/**: Pipeline orchestration components
  - **config/**: Configuration utilities
  - **nlp/**: NLP utilities
  - **utils/**: General utilities
  - **deprecated/**: Legacy code being phased out
- **tests/**: Test suite
  - **unit/**: Unit tests for components
  - **integration/**: Integration tests for system
  - **utils/**: Test utilities and fixtures
  - **test_data/**: Test datasets
- **benchmarks/**: Benchmark tools and scripts
- **examples/**: Example usage scenarios
- **datasets/**: Evaluation datasets
- **docs/**: Documentation files

## Terminology

- **Memory**: A single unit of stored information with associated embedding and metadata
- **Retrieval Strategy**: Algorithm for selecting relevant memories based on a query
- **Query Adaptation**: Process of adjusting retrieval parameters based on query characteristics
- **Pipeline**: A configurable sequence of components for memory retrieval and processing
- **Component**: A modular unit with specific functionality
- **Contextual Fabric**: The approach to memory management inspired by biological memory systems
- **ART Clustering**: Adaptive Resonance Theory-inspired clustering for memory organization
- **Activation**: Measure of how recently or frequently a memory has been accessed
- **Two-Stage Retrieval**: Retrieval process with candidate selection and re-ranking phases
- **Memory Decay**: Process of reducing activation levels over time
- **Dynamic Threshold Adjustment**: Automatic adjustment of thresholds based on query performance

## Development Process and Priorities

When implementing new features or fixes:

1. **Check the development priorities**: Review `docs/development_priorities.md` to see where this work fits
1. **Update feature matrix**: Add new entries or modify existing ones in `docs/feature_matrix.md`
1. **Maintain architectural consistency**: Follow component-based approach in the new architecture
1. **Add appropriate tests**: Create unit and integration tests for all changes
1. **Run benchmarks**: Verify there's no performance regression
1. **Update documentation**: Modify relevant documentation to reflect changes

## Current Development Focus

The primary focus areas are:

1. Improving performance at scale for large memory stores
1. Enhancing episodic memory handling and temporal context
1. Fixing query analyzer issues and improving classification
1. Completing the removal of legacy code
1. Adding a persistence layer for long-term memory storage

Refer to `docs/development_priorities.md` for detailed tasks and timelines.

## Contributing

When contributing new features or fixes, follow these guidelines:

1. Check the feature matrix to understand current implementation status
1. Use the component architecture for new features
1. Add proper tests for all new functionality
1. Update documentation to reflect changes
1. Run benchmarks to ensure no performance regression
1. Update the development priorities document if completing tasks

## Documentation Maintenance

When making significant changes:

1. Update relevant documentation files
1. Keep the feature matrix (`docs/feature_matrix.md`) current
1. Mark completed tasks in `docs/development_priorities.md`
1. Add new tasks to the development priorities if identified
