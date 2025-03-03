# Deprecated Code Removal

This document provides information about the deprecated code removal strategy in the MemoryWeave project.

## Approach to Deprecated Code

We are taking a phased approach to removing deprecated code:

1. **Phase 1: Remove fully deprecated modules**
   - Remove modules that are fully replaced and not referenced
   - Update documentation to reflect changes

2. **Phase 2: Convert legacy core components to stubs**
   - Replace implementation with stub versions that emit warnings
   - Update imports to point to new architecture
   - Maintain backward compatibility through stubs

3. **Phase 3: Complete removal**
   - Remove all remaining deprecated code
   - Update all examples and tests to use new architecture only

## Removed Code

### Phase 1 (Completed)
1. **memoryweave/deprecated/ directory**
   - The entire deprecated directory was removed
   - This directory contained legacy code that was moved to the component architecture

2. **memoryweave/core/contextual_fabric.py**
   - This file was a re-export of ContextualMemory from contextual_memory.py
   - It contained a deprecation warning advising users to import from contextual_memory.py instead

### Phase 2 (In Progress)
1. **memoryweave/core/__init__.py**
   - Updated with stronger deprecation warnings
   - References to be removed in Phase 3

2. **memoryweave/core/memory_encoding.py**
   - Converted to a stub with deprecation warnings
   - Still maintains the original interface for backward compatibility

## Transition Guide

If you were using any of the removed or deprecated code in your application, please follow these guidelines to update your code:

### If you were importing from `memoryweave.core.contextual_fabric`:
```python
# Old import (no longer works)
from memoryweave.core.contextual_fabric import ContextualMemory

# Transitional import (will work but shows deprecation warning)
from memoryweave.core.contextual_memory import ContextualMemory

# Recommended import (use the new architecture)
from memoryweave.components.memory_manager import MemoryManager
```

### If you were using classes from the deprecated directory:
Please refer to the `MIGRATION_GUIDE.md` document for a comprehensive guide on migrating from the old architecture to the new component-based architecture.

## Future Deprecated Code Removal (Phase 3)

The following files are still in use but are marked as deprecated and will be completely removed:

1. **memoryweave/core/__init__.py** - Contains a deprecation warning for the entire core module
2. **memoryweave/core/memory_retriever.py** - Core retrieval functionality, being replaced by component-based retrieval
3. **memoryweave/core/contextual_memory.py** - Core memory management, being replaced by MemoryManager component
4. **memoryweave/core/memory_encoding.py** - Memory encoding functionality, being replaced by component-based adapters
5. **memoryweave/core/category_manager.py** - Category management, being replaced by component-based alternatives
6. **memoryweave/core/core_memory.py** - Core memory storage, being replaced by storage components
7. **memoryweave/core/refactored_retrieval.py** - Transitional retrieval, being replaced by retrieval components

## Migration Strategy

To facilitate migration, we've provided several tools:

1. **Adapter Components**
   - `memoryweave/adapters/memory_adapter.py`
   - `memoryweave/adapters/retrieval_adapter.py`
   - `memoryweave/adapters/pipeline_adapter.py`

2. **Migration Utilities**
   - `memoryweave/adapters/component_migration.py` - Contains the `FeatureMigrator` utility

3. **Example Migration Code**
   - `examples/migration_example.py` - Shows three different migration approaches

Please refer to `MIGRATION_GUIDE.md` for detailed instructions on migrating your code.