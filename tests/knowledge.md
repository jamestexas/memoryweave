# MemoryWeave Testing Framework

## Test Structure

- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Integration tests for component interactions
- `tests/utils/`: Utility classes for testing (mock models, etc.)

## Mock Models

- `MockEmbeddingModel`: Deterministic embedding generator for reproducible tests
- `MockMemory`: Simplified implementation of ContextualMemory for testing

## Running Tests

```bash
# Run all tests
uv run python tests/run_tests.py

# Run a specific test file
uv run python -m unittest tests/unit/test_memory_manager.py
```

## Refactoring Strategy

1. **Component Interface Alignment**

   - Retrieval strategies need adapter components to implement `process_query`
   - Each component should follow the interface defined in `components/base.py`

1. **Gradual Migration**

   - Start with simpler components (QueryAnalyzer, PersonalAttributeManager)
   - Create wrapper around current retriever that uses new components
   - Verify behavior remains identical with integration tests

1. **Test-Driven Approach**

   - Fix failing tests one by one
   - Use tests to validate component behavior matches original implementation
   - Add new tests for edge cases and specific component features

## Current Issues

- Retrieval strategies don't implement `process_query` method
- Need to create adapter components or modify the memory manager to handle different component interfaces
- Some component tests are failing due to implementation differences

## Next Steps

1. Implement adapter components for retrieval strategies
1. Fix failing unit tests for each component
1. Update integration tests to use the correct component interfaces
1. Gradually migrate functionality from the original retriever to the component-based system
