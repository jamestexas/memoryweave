# MemoryWeave Storage Components

This directory contains the storage components for MemoryWeave, responsible for storing and retrieving memories in different formats.

## Architecture

The storage system follows a layered architecture:

1. **Base Stores** - Handle persistence and core operations
1. **Adapters** - Provide efficient access patterns and ID resolution
1. **Vector Search** - Specialized components for similarity search

## Directories

- `refactored/` - Modern implementation with better ID handling and separation of concerns
- `vector_search/` - Dedicated vector similarity search implementations

## Component Types

### Memory Stores

- `StandardMemoryStore` - Basic memory store for simple use cases
- `ChunkedMemoryStore` - Supports breaking large texts into chunks for better representation
- `HybridMemoryStore` - Memory-efficient approach combining full embeddings with selective chunks

### Adapters

- `MemoryAdapter` - Basic adapter for standard memory stores
- `ChunkedMemoryAdapter` - Adapter for chunked memory stores
- `HybridMemoryAdapter` - Adapter for hybrid memory stores

### Vector Search Providers

- `NumpyVectorSearch` - Simple NumPy-based implementation
- `FaissVectorSearch` - High-performance implementation using FAISS
- `HybridBM25VectorSearch` - Combined lexical and semantic search (future)

## Usage

### Basic Usage

```python
from memoryweave.storage.refactored import StandardMemoryStore, MemoryAdapter
from memoryweave.storage.vector_search import create_vector_search_provider

# Create store and adapter
store = StandardMemoryStore()
adapter = MemoryAdapter(store)

# Add a memory
memory_id = adapter.add(embedding, content, metadata)

# Search for similar memories
results = adapter.search_by_vector(query_vector, limit=10)
```

### Using Factory

```python
from memoryweave.factory.memory_factory import MemoryStoreConfig, create_memory_store_and_adapter

# Configure memory store and adapter
config = MemoryStoreConfig(
    type="hybrid",
    vector_search=VectorSearchConfig(type="faiss", faiss_index_type="IVF100,Flat"),
    adaptive_threshold=800,
    max_chunks_per_memory=3,
)

# Create adapter with appropriate store
adapter = create_memory_store_and_adapter(config)
```
