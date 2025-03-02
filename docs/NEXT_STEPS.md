# Next Steps to Fix Test Workarounds

## Integration Test Failures

After removing special case handling and test workarounds, several integration tests are now failing. Below is a plan to fix each failing test properly instead of relying on workarounds.

### 1. Benchmark Configuration Tests

#### `test_configurations_produce_different_results` in `tests/integration/test_benchmark_configurations.py`

**Current Issue:**
- Test relied on artificial differences between configurations
- Used special config name checks to trigger different behavior
- Added penalties to every other result to ensure differences

**Fix Needed:**
- Create distinct configuration objects with meaningful differences
- Update test to verify specific expected differences between configs
- Replace artificial differences with proper configuration parameters
- Test specific behaviors (e.g., "adding semantic coherence should reduce scores for off-topic results")

#### `test_query_performance_tracking` in `tests/integration/test_benchmark_performance.py`

**Current Issue:**
- Expected special case behavior for performance tracking
- Relied on hardcoded performance metrics

**Fix Needed:**
- Create proper performance tracking with measurable metrics
- Test that performance tracking captures expected metrics
- Implement proper mocks for timing measurements
- Use deterministic data to ensure consistent results

### 2. Two-Stage Retrieval Tests

#### `test_two_stage_includes_post_processing` in `tests/integration/test_two_stage_retrieval.py`

**Current Issue:**
- Test expected specific content based on test-specific embeddings
- Relied on hardcoded flags to enable post-processing

**Fix Needed:**
- Create proper test data with predictable relevance scores
- Mock or stub vector comparisons to ensure consistent results
- Verify post-processing actually modifies scores in expected ways
- Test each post-processor's specific behavior

#### `test_disabled_two_stage_uses_base_strategy` in `tests/integration/test_two_stage_retrieval.py`

**Current Issue:**
- Looks for hardcoded content that's no longer returned
- Relies on special flag behavior that was removed

**Fix Needed:**
- Create proper test fixtures with predictable data
- Mock vector store to return consistent results
- Verify strategy selection logic directly
- Test that disabled flag properly selects base strategy

#### `test_different_configurations_produce_different_results` in `tests/integration/test_two_stage_retrieval.py`

**Current Issue:**
- Uses inequality check instead of verifying specific behaviors
- Expects artificial differences between configurations

**Fix Needed:**
- Create configurations with documented differences
- Test specific expected behaviors for each configuration
- Verify each configuration parameter produces expected changes
- Implement proper test assertions with meaningful error messages

### 3. Migration Pipeline Tests

#### `test_migrator_utility` in `tests/integration/test_migrated_pipeline.py`

**Current Issue:**
- Uses low thresholds (0.5) for recall/precision
- Depends on special case handling that was removed

**Fix Needed:**
- Define proper expected recall/precision values
- Use consistent test data to ensure deterministic results
- Test specific migration logic rather than end-to-end behavior
- Create test fixtures that represent real component configurations

#### Other failures in `tests/integration/test_migrated_pipeline.py`

**Current Issue:**
- Tests crash with errors related to invalid values
- Depends on special case handling for test data

**Fix Needed:**
- Fix data normalization to prevent invalid values
- Create proper test fixtures with valid input data
- Implement better error handling for edge cases
- Test component behavior with controlled inputs

## Implementation Strategy

### Phase 1: Fix Test Data

1. Create predictable test fixtures
   - Implement proper mock embedding functions
   - Generate deterministic test data
   - Ensure consistent vector similarities

2. Improve test utilities
   - Create helper functions for test setup
   - Add assertion utilities for common patterns
   - Implement better test isolation

### Phase 2: Fix Component Interfaces

1. Update component interfaces to be more testable
   - Add explicit parameter validation
   - Improve error messages for invalid inputs
   - Enable dependency injection for testing

2. Document component behaviors
   - Define expected outputs for given inputs
   - Document parameter effects on results
   - Specify valid ranges for configuration values

### Phase 3: Fix Integration Tests

1. Rewrite integration tests with proper assertions
   - Test specific behaviors with meaningful assertions
   - Replace inequalities with expected value ranges
   - Verify each configuration actually works as expected

2. Improve test coverage
   - Add tests for error handling
   - Test edge cases properly
   - Ensure all code paths are covered

## Fixing the Two-Stage Retrieval Strategy

The two-stage retrieval strategy needs particular attention:

1. Clearly document the parameters that affect behavior
   - Define how `first_stage_k` affects results
   - Document how thresholds work in both stages

2. Ensure stage interactions work properly
   - Test that first stage filters as expected
   - Verify second stage properly ranks results

3. Remove special case handling
   - Fix config name checks in strategy code
   - Remove test-specific behavior flags

## Addressing Semantic Coherence Processor

The semantic coherence processor has several issues:

1. Fix the coherence calculation logic
   - Ensure penalties are applied consistently
   - Test with controlled inputs
   - Remove special case handling

2. Improve configurability
   - Document config parameters and effects
   - Test each configuration option separately
   - Ensure coherence values make semantic sense

## Conclusion

Rather than adding more workarounds or special cases to make tests pass, these changes will address the root causes of the test failures. The goal is to have tests that verify actual component behavior rather than artificial scenarios designed just to make tests pass.

The work should focus on one component at a time, starting with the most fundamental ones (vector store, memory store) and working up to higher-level components (retrieval strategies, query adapters).