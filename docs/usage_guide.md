# MemoryWeave Usage Guide

This guide demonstrates how to use MemoryWeave's component-based architecture for memory management with LLMs.

## Basic Components

MemoryWeave is built around these core components:

1. **Memory Storage**: StandardMemoryStore, HybridMemoryStore, ChunkedMemoryStore
1. **Memory Encoding**: MemoryEncoder for converting content to embeddings
1. **Retrieval**: Retriever as the main interface for memory retrieval
1. **Components**: Modular parts that can be combined into retrieval pipelines

## Example: Setting Up Memory Management

```python
from sentence_transformers import SentenceTransformer
from memoryweave.components import MemoryEncoder, Retriever
from memoryweave.storage.refactored.memory_store import StandardMemoryStore

# Create embedding model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# Create memory store
memory_store = StandardMemoryStore()

# Create memory encoder
encoder = MemoryEncoder(embedding_model)
encoder.initialize({"context_window_size": 3, "use_episodic_markers": True})

# Create retriever
retriever = Retriever(memory=memory_store, embedding_model=embedding_model)
retriever.initialize_components()
```

## Example: Adding Memories

```python
# Encode and add simple text memories
texts = [
    "MemoryWeave uses a component-based architecture for memory management.",
    "Components can be composed into retrieval pipelines.",
    "The Contextual Fabric strategy combines semantic, temporal and associative signals.",
    "Category management helps organize memories into semantic clusters.",
]

for text in texts:
    # Create embedding
    embedding = encoder.encode_text(text)

    # Add to memory store
    memory_store.add(embedding, text)

# Add an interaction memory (conversation)
query = "How does memory retrieval work?"
response = "Memory retrieval combines vector similarity with context."
metadata = {"importance": 0.8, "topic": "retrieval"}

interaction_embedding = encoder.encode_interaction(query, response, metadata)
memory_store.add(interaction_embedding, response, metadata=metadata)

# Add a concept memory
concept = "Contextual Fabric"
definition = "A retrieval approach that weaves multiple signals together"
examples = ["Semantic similarity", "Temporal recency", "Associative links"]

concept_embedding = encoder.encode_concept(concept, definition, examples)
memory_store.add(
    concept_embedding,
    definition,
    metadata={"type": "concept", "concept": concept, "examples": examples},
)
```

## Example: Retrieving Memories

```python
# Simple retrieval
query = "How does MemoryWeave organize information?"
results = retriever.retrieve(query, top_k=3)

for i, result in enumerate(results):
    print(f"{i + 1}. Score: {result['relevance_score']:.4f} - {result['content']}")

# Retrieval with different strategies
strategy_results = retriever.retrieve(
    query,
    strategy="contextual_fabric",  # Other options: similarity, category, temporal, hybrid_bm25
    top_k=3,
)

# Configuring retrieval behavior
retriever.configure_two_stage_retrieval(
    enable=True, first_stage_k=20, first_stage_threshold_factor=0.7
)

retriever.configure_query_type_adaptation(enable=True, adaptation_strength=0.8)

retriever.configure_semantic_coherence(enable=True)
```

## Advanced: Creating Custom Pipelines

```python
# Define a custom retrieval pipeline
pipeline_config = [
    {"component": "query_analyzer", "config": {}},
    {"component": "keyword_expander", "config": {"enable_expansion": True}},
    {"component": "category_retrieval", "config": {"confidence_threshold": 0.1}},
    {"component": "semantic_coherence", "config": {"coherence_threshold": 0.2}},
]

retriever.configure_pipeline(pipeline_config)

# Execute the pipeline
results = retriever.retrieve(query, top_k=5)
```

## Advanced: Category Management

```python
from memoryweave.components import CategoryManager

# Create category manager
category_manager = CategoryManager(memory_store)

# Use it for categorization and retrieval enhancement
for memory_id in range(memory_store.count()):
    memory = memory_store.get(memory_id)
    category_id = category_manager.add_to_category(memory_id, memory.embedding)

# Consolidate similar categories
consolidated = category_manager.consolidate_categories(similarity_threshold=0.8)
print(f"Consolidated {len(consolidated)} categories")

# Get category statistics
stats = category_manager.get_statistics()
print(f"Total categories: {stats['total_categories']}")
print(f"Average category size: {stats['average_category_size']}")
```

## Further Resources

For more detailed examples:

- [Memory Encoding Example](examples/memory_encoding_example.py)
- [Retrieval Strategies Example](examples/retrieval_strategies_example.py)
- [Category Management Example](examples/category_management_example.py)

For API details, see the [API Reference](api_reference.md).
