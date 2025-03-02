# MemoryWeave Tests

This directory contains tests for the MemoryWeave library, organized into:

- `unit/` - Tests for individual components in isolation
- `integration/` - Tests for components working together
- `utils/` - Test utilities and helpers

## Running Tests

Run all tests with:
```bash
uv run python -m pytest
```

Run specific test categories:
```bash
# Unit tests only
uv run python -m pytest -m unit

# Integration tests only
uv run python -m pytest -m integration
```

Run a specific test file:
```bash
uv run python -m pytest tests/unit/path/to/test_file.py
```

## Test Plan

### Core Components Test Status

| Component | Unit Tests | Integration Tests | Status |
|-----------|------------|-------------------|--------|
| QueryTypeAdapter | ✅ | ✅ | Complete |
| SemanticCoherenceProcessor | ✅ | ✅ | Complete |
| TwoStageRetrievalStrategy | ✅ | ✅ | Complete |
| KeywordBoostProcessor | TODO | ✅ | Partial |
| MemoryDecayComponent | ✅ | ✅ | Complete |
| Retriever | ✅ | ✅ | Complete |

### Benchmark Validation Tests

| Test Case | Description | Status |
|-----------|-------------|--------|
| Configuration Differentiation | Verify different configs produce different metrics | ✅ |
| Memory Usage Validation | Ensure low memory usage during benchmarks | Planned |
| Performance Tracking | Track execution time across versions | Planned |

### Test Coverage Summary

- **Unit Tests**: Created tests for core components including QueryTypeAdapter and SemanticCoherenceProcessor to verify their behavior in isolation
- **Integration Tests**: Added tests to verify components work together properly, especially TwoStageRetrievalStrategy integration
- **Benchmark Validation**: Implemented a mini-benchmark test that verifies different configurations produce measurably different results

### Remaining Test Needs

1. **KeywordBoostProcessor Unit Tests**: Create dedicated unit tests for this component
2. **Memory Usage Optimization**: Tests to verify benchmark memory usage stays within acceptable limits 
3. **Long-running Tests**: End-to-end tests that run complete benchmarks with full datasets (for CI/CD pipelines)

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