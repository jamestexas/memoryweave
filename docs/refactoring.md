# MemoryWeave Refactoring Strategy

## Feature Matrix & Refactoring Tracker

This document outlines the current state of retrieval strategies, feature distribution, and prioritized refactoring tasks.

### Phase 1: Test Coverage & Stability

| Component Type | Current Coverage | Target | Priority | Status |
|----------------|-----------------|--------|----------|--------|
| Core Memory Store | ~92% | 95% | High | In progress |
| Retrieval Strategies | ~65% | 80% | **Critical** | ✅ Making good progress |
| Activation System | ~50% | 80% | Medium | In progress |
| API Layer | ~20% | 70% | Low | Pending |

**Recommendation**: Continue improving test coverage for retrieval strategies while addressing core functionality issues:

1. ✅ Document expected behavior
1. ✅ Prevent regressions during refactoring
1. ✅ Identify and fix bugs in current implementation (activation ranking, hybrid detection)

### Retrieval Strategy Feature Distribution

| Feature | Simple Similarity | Contextual Fabric | Chunked Fabric | Hybrid Fabric | HybridBM25Vector | Two-Stage |
|---------|-----------------|-------------------|----------------|---------------|------------------|-----------|
| Vector similarity | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Threshold adjustment | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Result formatting | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Activation boost | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Configurable weights | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Temporal context | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Associative links | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Progressive filtering | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Chunking support | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Keyword matching | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Two-stage retrieval | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| Memory efficiency | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Debug logging | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Activation ranking | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Structured results | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Hybrid support detection | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |

### Refactoring Priority Map

| Task | Complexity | Impact | Priority | Dependencies | Status |
|------|------------|--------|----------|--------------|--------|
| Fix test failures | Low | High | 1️⃣ | None | ✅ Completed |
| Ensure correct activation ranking | Medium | High | 1️⃣ | Test coverage | ✅ Completed |
| Fix hybrid support detection | Low | High | 1️⃣ | Test coverage | ✅ Completed |
| Standardize initialization pattern | Low | High | 2️⃣ | Fix test failures | 🟡 In progress |
| Create unified strategy base class | Medium | High | 3️⃣ | Standardize initialization | ⏳ Pending |
| Extract common scoring logic | Medium | Medium | 4️⃣ | Create base class | ⏳ Pending |
| Reorganize directory structure | Medium | Medium | 5️⃣ | Extract common logic | ⏳ Pending |
| Implement proper strategy hierarchy | High | High | 6️⃣ | All above | ⏳ Pending |
| Extract shared components | Medium | Medium | 7️⃣ | Implement hierarchy | ⏳ Pending |
| Optimize performance bottlenecks | High | Medium | 8️⃣ | Test coverage | ⏳ Pending |
| Implement monorepo structure | High | Low | 9️⃣ | Most refactoring complete | ⏳ Pending |

## Implementation Phases

### Phase 1: Test & Stabilize (Current Focus)

- ✅ Fix all current test failures
- ✅ Fix activation ranking in ContextualFabricStrategy
- ✅ Correct hybrid support detection in HybridFabricStrategy
- ✅ Fix associative context enhancement
- 🟡 Increase test coverage to 80% for retrieval strategies (about 65% complete)
- ✅ Document expected behavior of each strategy
- ✅ Identify and fix bugs in current implementation

### Phase 2: Consolidate & Standardize

- 🟡 Create standard initialization pattern across strategies
- 🟡 Implement proper strategy base class
- ⏳ Extract common utilities (scoring, filtering, etc.)
- 🟡 Standardize parameter handling

### Phase 3: Restructure & Optimize

- ⏳ Implement proper strategy hierarchy
- ⏳ Reorganize directory structure
- ⏳ Eliminate code duplication
- ⏳ Optimize performance bottlenecks

### Phase 4: Package & Deploy

- ⏳ Implement monorepo structure
- ⏳ Create clear package boundaries
- ⏳ Define stable public APIs
- ⏳ Create comprehensive documentation

## Code Duplication Hotspots

| Functionality | Locations | Refactoring Approach | Status |
|---------------|-----------|----------------------|--------|
| Vector similarity calculation | `retrieval/`, `storage/`, `vector_search/` | Extract to common utility | ⏳ Pending |
| Parameter parsing | All strategy implementations | Move to base class | 🟡 In progress |
| Result formatting | All strategy implementations | Create formatter component | ✅ Created MemoryResult model |
| Activation handling | Strategies, `activation.py` | Centralize in activation manager | 🟡 Partially fixed in ContextualFabricStrategy |
| Temporal context | Multiple strategies | Extract to dedicated service | ⏳ Pending |
| Debug logging | Throughout codebase | Standardize logging pattern | 🟡 Improved in ContextualFabricStrategy |
| Multi-signal ranking | Multiple strategies | Create shared scoring functions | 🟡 In progress |
| Keyword extraction | Multiple locations | Extract to shared utility | ⏳ Pending |

## Recent Improvements

### Core Functionality Fixes

1. **Activation Ranking**: Fixed issues with activation scores not properly influencing result ranking in ContextualFabricStrategy

   - Enhanced activation contribution calculation
   - Improved boosting for highly activated memories
   - Added minimum impact for activated memories regardless of similarity

1. **Hybrid Support Detection**: Fixed hybrid capability detection in HybridFabricStrategy

   - Added explicit initialization of supports_hybrid flag
   - Improved detection logic for search capabilities
   - Fixed test issues related to MagicMock behavior

1. **Associative Context Enhancement**: Fixed associative memory linking in HybridFabricStrategy

   - Added support for both direct links and network traversal
   - Improved error handling for memory content access
   - Better handling of link strengths

### Testing Improvements

1. **MagicMock Usage**: Fixed issues with MagicMock behavior in tests

   - Used spec_set to create more precise mocks
   - Correctly simulated memory stores with/without specific capabilities
   - Added better assertion error messages

1. **Debug Logging**: Added comprehensive debug logging

   - Detailed memory result tables for better visibility
   - Component-level logging for diagnostic purposes
   - Better tracing of scoring contributions

## Recommended Next Steps

1. **Complete strategy refactoring** - Continue standardizing interfaces and extracting common functionality
1. **Improve parameter handling** - Create consistent parameter adaptation patterns
1. **Create standard result model** - Expand MemoryResult usage across all strategies
1. **Standardize error handling** - Create consistent error handling and fallback patterns
1. **Address performance hotspots** - Focus on vector operations and batched processing
