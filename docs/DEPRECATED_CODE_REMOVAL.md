# Deprecated Code Removal

This document provides information about the removed deprecated code in the MemoryWeave project.

## Removed Code

### 1. memoryweave/deprecated/ directory
- The entire deprecated directory was removed
- This directory contained legacy code that was moved to the component architecture

### 2. memoryweave/core/contextual_fabric.py
- This file was a re-export of ContextualMemory from contextual_memory.py
- It contained a deprecation warning advising users to import from contextual_memory.py instead

## Transition Guide

If you were using any of the removed code in your application, please follow these guidelines to update your code:

### If you were importing from `memoryweave.core.contextual_fabric`:
```python
# Old import
from memoryweave.core.contextual_fabric import ContextualMemory

# New import
from memoryweave.core.contextual_memory import ContextualMemory
```

### If you were using the deprecated directory:
Please refer to the `MIGRATION_GUIDE.md` document for a comprehensive guide on migrating from the old architecture to the new component-based architecture.

## Future Deprecated Code Removal

The following files are still in use but are marked as deprecated and should be migrated away from:

1. **memoryweave/core/__init__.py** - Contains a deprecation warning for the entire core module
2. **memoryweave/core/memory_retriever.py** - Core retrieval functionality, being replaced by component-based retrieval
3. **memoryweave/core/contextual_memory.py** - Core memory management, being replaced by MemoryManager component

These files will be removed in a future update once all dependent code has been migrated to the new architecture.