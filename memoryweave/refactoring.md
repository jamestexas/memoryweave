# MemoryWeave Refactoring Strategy

## Feature Matrix & Refactoring Tracker

This document outlines the current state of retrieval strategies, feature distribution, and prioritized refactoring tasks.

### Phase 1: Test Coverage & Stability

| Component Type | Current Coverage | Target | Priority | Status |
|----------------|-----------------|--------|----------|--------|
| Core Memory Store | ~90% | 95% | High | In progress |
| Retrieval Strategies | ~30% | 80% | **Critical** | ⚠️ Needs focus |
| Activation System | ~40% | 80% | Medium | Pending |
| API Layer | ~20% | 70% | Low | Pending |

**Recommendation**: Complete test coverage for retrieval strategies before significant refactoring, as this will:

1. Document expected behavior
1. Prevent regressions during refactoring
1. Identify bugs in current implementation

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
| Two-stage retrieval | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Memory efficiency | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |

### Refactoring Priority Map

| Task | Complexity | Impact | Priority | Dependencies |
|------|------------|--------|----------|--------------|
| Fix test failures | Low | High | 1️⃣ | None |
| Standardize initialization pattern | Low | High | 2️⃣ | Fix test failures |
| Create unified strategy base class | Medium | High | 3️⃣ | Standardize initialization |
| Extract common scoring logic | Medium | Medium | 4️⃣ | Create base class |
| Reorganize directory structure | Medium | Medium | 5️⃣ | Extract common logic |
| Implement proper strategy hierarchy | High | High | 6️⃣ | All above |
| Extract shared components | Medium | Medium | 7️⃣ | Implement hierarchy |
| Optimize performance bottlenecks | High | Medium | 8️⃣ | Test coverage |
| Implement monorepo structure | High | Low | 9️⃣ | Most refactoring complete |

## Implementation Phases

### Phase 1: Test & Stabilize (Current Focus)

- ✅ Fix all current test failures
- ✅ Increase test coverage to 80% for retrieval strategies
- ✅ Document expected behavior of each strategy
- ✅ Identify and fix bugs in current implementation

### Phase 2: Consolidate & Standardize

- ⬜ Create standard initialization pattern across strategies
- ⬜ Implement proper strategy base class
- ⬜ Extract common utilities (scoring, filtering, etc.)
- ⬜ Standardize parameter handling

### Phase 3: Restructure & Optimize

- ⬜ Implement proper strategy hierarchy
- ⬜ Reorganize directory structure
- ⬜ Eliminate code duplication
- ⬜ Optimize performance bottlenecks

### Phase 4: Package & Deploy

- ⬜ Implement monorepo structure
- ⬜ Create clear package boundaries
- ⬜ Define stable public APIs
- ⬜ Create comprehensive documentation

## Code Duplication Hotspots

| Functionality | Locations | Refactoring Approach |
|---------------|-----------|----------------------|
| Vector similarity calculation | `retrieval/`, `storage/`, `vector_search/` | Extract to common utility |
| Parameter parsing | All strategy implementations | Move to base class |
| Result formatting | All strategy implementations | Create formatter component |
| Activation handling | Strategies, `activation.py` | Centralize in activation manager |
| Temporal context | Multiple strategies | Extract to dedicated service |
| Debug logging | Throughout codebase | Standardize logging pattern |

## Recommended Next Steps

1. **Focus on test coverage first** - Continue creating tests for retrieval strategies to document behavior and identify issues
1. **Fix initialization inconsistencies** - Ensure all strategies initialize parameters consistently
1. **Create base class prototype** - Start designing a unified base class that can be gradually adopted
1. **Document feature boundaries** - Clearly document which features belong to which strategy
