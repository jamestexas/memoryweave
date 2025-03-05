# Next Steps for MemoryWeave Refactoring

This document outlines the next steps to complete the refactoring of MemoryWeave, focusing on making tests verify real behavior rather than using special case handling.

## 1. Fix Remaining Integration Tests

### 1.1 Test Benchmark Configurations

The `test_benchmark_configurations.py` test needs to be updated to:

1. Use the new test fixtures for creating test data
1. Remove dependencies on configuration names for specific behavior
1. Verify that different configurations produce different results based on actual component behavior
1. Add assertions that validate specific behavior changes, not just numeric differences

```python
# Sample pattern for improved tests
from tests.utils.test_fixtures import (
    create_test_memories,
    create_test_queries,
    verify_retrieval_results,
)

# Create consistent test data
embeddings, texts, metadata = create_test_memories(num_memories=10, embedding_dim=4)
queries = create_test_queries(num_queries=3, embedding_dim=4)

# Test different configurations with specific behavior expectations
# Each configuration should have clear expectations for how it behaves
```

### 1.2 Test Benchmark Performance

The `test_benchmark_performance.py` test needs:

1. Predictable test data that can demonstrate performance differences
1. Standard metrics collection that doesn't depend on special case handling
1. More specific assertions about what's being measured
1. Removal of arbitrary performance thresholds

### 1.3 Test Migrated Pipeline

The `test_migrated_pipeline.py` test needs:

1. Clear separation between component responsibilities
1. Removal of special case logic for migrated components
1. Proper setup of test expectations based on component behavior
1. Validation that relies on behavior rather than numeric coincidence

## 2. Fix Remaining Components

### 2.1 SemanticCoherenceProcessor

This component likely has special case handling:

1. Identify any behavior that depends on test/evaluation mode
1. Remove any hardcoded values or test-specific logic
1. Create proper test fixtures to validate real behavior
1. Ensure consistent behavior across all environments

### 2.2 PersonalAttributeManager

1. Remove hardcoded "blue" preference and other test-specific values
1. Make attribute handling consistent
1. Add proper methods for manipulating attribute structures
1. Update tests to validate behavior rather than specific values

### 2.3 QueryAnalyzer

1. Ensure query analysis doesn't have special case handling
1. Create test fixtures for different query types
1. Validate query adaptation parameters come from consistent logic
1. Remove any dependencies on configuration names

## 3. Create Test Documentation

Create a `tests/README.md` file with guidelines for writing tests:

1. Tests should validate real behavior, not rely on special cases
1. Use test fixtures from `tests/utils/test_fixtures.py`
1. Assertions should validate behavior, not just check for non-zero differences
1. All tests should work in both regular and evaluation modes
1. No hardcoded test values - use patterns and templates

## 4. Verify Behavior Consistency

Final validation ensuring:

1. All benchmarks run correctly with the updated code
1. Benchmark configurations produce meaningful differences
1. Integration tests validate specific behaviors
1. No remaining special case handling for tests

## Timeline

1. **Week 1**: Complete retrieval strategies and test fixtures
1. **Week 2**: Fix remaining integration tests
1. **Week 3**: Update other components with special case handling
1. **Week 4**: Create test documentation and validate consistency
