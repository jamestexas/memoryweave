# MemoryWeave Testing Guidelines

This guide provides best practices for writing tests for the MemoryWeave library. Following these guidelines will ensure tests are reliable, maintainable, and actually test the intended behavior.

## Core Testing Principles

1. **Test actual behavior, not implementation details**

   - Tests should verify what components do, not how they do it
   - Avoid testing private methods directly
   - Focus on validating inputs and outputs

1. **Use consistent test fixtures**

   - Use the utilities in `tests/utils/test_fixtures.py`
   - Avoid hardcoded test values like "blue" for favorite color
   - Create deterministic test data that has predictable patterns

1. **Avoid special case handling for tests**

   - Components should behave the same in tests and production
   - Don't add special code paths for test scenarios
   - Tests should validate normal functionality

1. **Make assertions specific and meaningful**

   - Don't just check that some results were returned
   - Verify specific behaviors (e.g., "keyword boosting increases scores for memories with matching keywords")
   - Include precise error messages explaining what failed

## Test Organization

The test suite is organized into:

1. **Unit Tests** (`tests/unit/`): Test individual components in isolation
1. **Integration Tests** (`tests/integration/`): Test multiple components working together
1. **Test Utilities** (`tests/utils/`): Shared test fixtures and helpers

## Using Test Fixtures

The `tests/utils/test_fixtures.py` module provides utilities for creating consistent test data:

```python
from tests.utils.test_fixtures import (
    create_test_embedding,
    create_test_memories,
    create_test_queries,
    verify_retrieval_results,
)

# Create test embeddings
embedding = create_test_embedding("Test query", dim=768)

# Create test memory set
embeddings, texts, metadata = create_test_memories(num_memories=10)

# Create test queries with known relevant results
queries = create_test_queries(num_queries=3)

# Verify retrieval results
success, metrics = verify_retrieval_results(
    results=retrieved_results, expected_indices=[0, 1, 2], require_all=False, check_order=True
)
```

## Writing Good Assertions

Good assertions:

1. **Focus on behavior**: Test what the code is supposed to do
1. **Are specific**: Test exact expected outcomes
1. **Have informative error messages**: Make it clear what failed

Example:

```python
# BAD - just checks something was returned
assert len(results) > 0

# GOOD - verifies specific behavior
expected_indices = [1, 3, 5]  # indices of memories with color references
success, metrics = verify_retrieval_results(results, expected_indices)
assert success, (
    f"Personal query didn't retrieve expected color-related memories. Metrics: {metrics}"
)
assert metrics["precision"] >= 0.5, f"Precision too low: {metrics['precision']}"
```

## Testing Different Configurations

When testing different configurations:

1. Each configuration should have clear behavioral expectations
1. Test the specific differences each configuration introduces
1. Use well-defined test data designed to highlight differences

Example:

```python
# Test that keyword boosting actually boosts keyword matches
results_without_boost = run_retrieval(query, keyword_boost=False)
results_with_boost = run_retrieval(query, keyword_boost=True)

# Verify that keyword-containing memories rank higher with boosting
keyword_indices = [1, 3, 5]  # indices of memories containing query keywords
rank_without_boost = get_average_rank(results_without_boost, keyword_indices)
rank_with_boost = get_average_rank(results_with_boost, keyword_indices)

assert rank_with_boost < rank_without_boost, (
    "Keyword boosting didn't improve ranking of keyword matches"
)
```

## Running Tests

Run all tests:

```bash
uv run python -m pytest
```

Run unit tests only:

```bash
uv run python -m pytest -m unit
```

Run integration tests:

```bash
uv run python -m pytest -m integration
```

Run a specific test:

```bash
uv run python -m pytest path/to/test_file.py::TestClass::test_function
```

## Debugging Test Failures

1. Use the `-v` flag for more verbose output
1. Use `pytest.set_trace()` to debug inside a test
1. Check logs with `--log-cli-level=DEBUG`
