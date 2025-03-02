# MemoryWeave Tests

This directory contains tests for the MemoryWeave library, organized into:

- `unit/` - Tests for individual components in isolation
- `integration/` - Tests for components working together
- `utils/` - Test utilities and helpers

## Running Tests

Run all tests with:
```bash
uv run pytest
```

Run specific test categories:
```bash
# Unit tests only
uv run pytest -m unit

# Integration tests only
uv run pytest -m integration
```

Run a specific test file:
```bash
uv run pytest tests/unit/path/to/test_file.py
```

Run a specific test:
```bash
uv run pytest tests/unit/path/to/test_file.py::TestClass::test_function
```

## Test Plan

### Core Components Test Status

| Component | Unit Tests | Integration Tests | Status |
|-----------|------------|-------------------|--------|
| QueryTypeAdapter | ✅ | ✅ | Complete |
| SemanticCoherenceProcessor | ✅ | ✅ | Complete |
| TwoStageRetrievalStrategy | ✅ | ✅ | Complete |
| KeywordBoostProcessor | ✅ | ✅ | Complete |
| MemoryDecayComponent | ✅ | ✅ | Complete |
| Retriever | ✅ | ✅ | Complete |

### Benchmark Validation Tests

| Test Case | Description | Status |
|-----------|-------------|--------|
| Configuration Differentiation | Verify different configs produce different metrics | ✅ |
| Memory Usage Validation | Ensure low memory usage during benchmarks | ✅ |
| Performance Tracking | Track execution time across versions | ✅ |

### Test Coverage Summary

- **Unit Tests**: Created tests for core components including QueryTypeAdapter, SemanticCoherenceProcessor, and KeywordBoostProcessor to verify their behavior in isolation
- **Integration Tests**: Added tests to verify components work together properly, especially TwoStageRetrievalStrategy integration
- **Benchmark Validation**: Implemented a mini-benchmark test that verifies different configurations produce measurably different results
- **Performance Testing**: Added memory usage tracking and performance benchmarking to analyze system efficiency

### Remaining Test Needs

1. **Long-running Tests**: End-to-end tests that run complete benchmarks with full datasets (for CI/CD pipelines)
2. **Stress Testing**: Tests that verify system behavior under extreme loads and edge cases
3. **Continuous Integration**: Set up automated test runs to track performance changes over time

### Priority Testing Roadmap

1. **Core Component Verification**
   - Verify components function as expected in isolation
   - Ensure consistent behavior across library versions

2. **Component Integration**
   - Test components work together as expected
   - Validate configuration propagation

3. **Benchmark Validation**
   - Create reduced-size benchmark tests
   - Verify metrics differ between configurations