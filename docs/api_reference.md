# MemoryWeave API Reference

This document provides a reference for the main classes and methods in the MemoryWeave library.

## Memory Storage

### StandardMemoryStore

Primary storage implementation for memory embeddings and content.

```python
from memoryweave.storage.refactored.memory_store import StandardMemoryStore

store = StandardMemoryStore()
```

**Key Methods:**

- `add(embedding, content, metadata=None)` - Add a memory
- `get(memory_id)` - Retrieve a single memory
- `update(memory_id, content=None, metadata=None)` - Update a memory
- `remove(memory_id)` - Remove a memory
- `retrieve_memories(query_embedding, top_k=5)` - Get similar memories

### HybridMemoryStore

Storage that combines vector similarity with keyword-based search.

```python
from memoryweave.storage.refactored.hybrid_store import HybridMemoryStore

store = HybridMemoryStore()
```

## Memory Encoding

### MemoryEncoder

Encodes different content types into memory embeddings.

```python
from memoryweave.components.memory_encoding import MemoryEncoder

# or
from memoryweave.factory.memory_factory import create_memory_encoder

encoder = create_memory_encoder("sentence-transformers/paraphrase-MiniLM-L6-v2")
```

**Key Methods:**

- `encode_text(text)` - Encode raw text
- `encode_interaction(query, response, metadata=None)` - Encode a conversation interaction
- `encode_concept(concept, definition, examples=None)` - Encode a concept definition
- `process(data, context)` - General processing interface

## Retrieval

### Retriever

Main component for retrieving memories.

```python
from memoryweave.components.retriever import Retriever

retriever = Retriever(memory=store, embedding_model=embedding_model)
retriever.initialize_components()
```

**Key Methods:**

- `retrieve(query, top_k=5, strategy=None)` - Retrieve similar memories
- `configure_two_stage_retrieval(enable, first_stage_k, threshold_factor)` - Configure two-stage retrieval
- `configure_query_type_adaptation(enable, adaptation_strength)` - Configure query adaptation
- `configure_semantic_coherence(enable)` - Enable semantic coherence filtering
- `configure_pipeline(pipeline_config)` - Configure custom pipeline

## Categories

### CategoryManager

Manages dynamic categorization of memories.

```python
from memoryweave.components.category_manager import CategoryManager

category_manager = CategoryManager(memory_store)
```

**Key Methods:**

- `add_to_category(memory_id, embedding)` - Categorize a memory
- `get_category(memory_id)` - Get a memory's category
- `get_category_members(category_id)` - Get memories in a category
- `consolidate_categories(similarity_threshold=None)` - Merge similar categories
- `recategorize(memory_id, embedding)` - Update a memory's category
- `get_statistics()` - Get categorization statistics
- `filter_by_category(results, query_embedding)` - Filter results by category

## Factory Functions

```python
from memoryweave.factory.memory_factory import (
    create_memory_store_and_adapter,
    create_memory_encoder,
)

# Create memory store and adapter
memory_store, memory_adapter = create_memory_store_and_adapter(
    MemoryStoreConfig(store_type="standard")
)

# Create memory encoder
encoder = create_memory_encoder("sentence-transformers/paraphrase-MiniLM-L6-v2")
```
