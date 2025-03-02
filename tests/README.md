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

## Testing Guidelines

### Core Testing Principles

1. **Tests should validate real behavior, not workarounds**
   - Tests must verify actual component functionality, not special test-only code paths
   - Components should not contain special case handling for tests
   - Tests should fail when implementation is incorrect

2. **Strong assertions over weak checks**
   - Use specific value assertions rather than inequality checks (`==` vs `!=`)
   - Verify expected ranges or exact values rather than simple greater/less than
   - Test both positive and negative cases

3. **No test-specific flags or behaviors**
   - Components should not check for test context or test names
   - Use proper dependency injection and mocking instead of hardcoded values
   - Tests should run against the same code that runs in production

4. **Effective test isolation**
   - Tests must not depend on other tests' state or execution order
   - Use proper setup/teardown with fresh instances for each test
   - Mock external dependencies consistently

### Common Anti-Patterns to Avoid

1. ❌ Special case code for tests:
   ```python
   # BAD: Special handling just for tests
   if "test" in config_name or query == "what's my favorite color":
       return {"preferences_color": "blue"}  # Hardcoded test value
   ```

2. ❌ Weak assertions:
   ```python
   # BAD: Only checks if different, not if correct
   assert result_a != result_b
   
   # BAD: Too permissive threshold (0.5 is very low)
   assert precision_score >= 0.5
   ```

3. ❌ Artificial test data:
   ```python
   # BAD: Creating artificial differences for test
   if i % 2 == 0:  # Every other result gets penalty
       result["score"] *= 0.8
   ```

4. ❌ Adding fallbacks for empty test results:
   ```python
   # BAD: Making tests pass with mock data
   if len(results) == 0:
       results = [create_mock_result()]  # Test should fail instead
   ```

### Writing Proper Tests

1. ✅ Test real behavior:
   ```python
   # GOOD: Testing actual expected behavior
   def test_confidence_threshold():
       # Set up retriever with specific threshold
       retriever = Retriever(confidence_threshold=0.7)
       # Get results with scores around the threshold
       results = retriever.retrieve(query, context)
       # Verify only results above threshold are returned
       for result in results:
           assert result.score >= 0.7, f"Result with score {result.score} below threshold"
       # Verify some results were filtered (using proper setup)
       assert len(results) < len(all_potential_results)
   ```

2. ✅ Strong assertions:
   ```python
   # GOOD: Specific behavior validation
   def test_personal_attribute_retrieval():
       manager = PersonalAttributeManager()
       manager.update_attribute("favorite_color", "red")
       
       result = manager.get_attributes("What's my favorite color?")
       assert result == {"preferences_color": "red"}, f"Expected red, got {result}"
   ```

3. ✅ Proper mocking:
   ```python
   # GOOD: Dependency injection with mocks
   def test_query_adaptation():
       mock_analyzer = create_mock_analyzer(returns_query_type="personal")
       adapter = QueryAdapter(analyzer=mock_analyzer)
       
       adapted_params = adapter.adapt_parameters(base_params, query="test")
       assert adapted_params["threshold"] > base_params["threshold"]
       assert 0.65 <= adapted_params["threshold"] <= 0.85
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
4. **Fix Special Case Handling**: Remove workarounds and test-only code paths from components

### Priority Testing Roadmap

1. **Core Component Verification**
   - Verify components function as expected in isolation
   - Ensure consistent behavior across library versions
   - Remove special case handling and hardcoded test values

2. **Component Integration**
   - Test components work together as expected
   - Validate configuration propagation
   - Use proper dependency injection instead of special flags

3. **Benchmark Validation**
   - Create reduced-size benchmark tests
   - Verify metrics differ between configurations through proper implementation differences
   - Remove artificial data manipulation