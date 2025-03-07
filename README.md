# MemoryWeave

MemoryWeave is a memory management system for language models that uses a contextual fabric approach inspired by biological memory systems.

> **Note:** This project is in active development. APIs are subject to change, and documentation is intentionally minimal until the architecture stabilizes.

## Overview

MemoryWeave moves beyond traditional knowledge graph approaches to memory management. Rather than using discrete nodes and edges, it focuses on capturing rich contextual signatures of information for improved long-context coherence in language model conversations.

## Key Concepts

MemoryWeave implements several biologically-inspired memory management principles:

- **Contextual Fabric**: Memory traces capture rich contextual signatures rather than isolated facts
- **Activation-Based Retrieval**: Memory retrieval uses dynamic activation patterns similar to biological systems
- **Episodic Structure**: Memories maintain temporal relationships and episodic anchoring
- **Non-Structured Memory**: Works with raw LLM outputs without requiring structured formats
- **Component-Based Architecture**: Modular design for flexible retrieval pipelines

## Installation

```bash
# Using pip
pip install memoryweave

# For development install
git clone https://github.com/yourusername/memoryweave.git
cd memoryweave
pip install -e .
```

## Quick Start

```python
from memoryweave.components import MemoryEncoder, Retriever
from memoryweave.storage.memory_store import StandardMemoryStore
from sentence_transformers import SentenceTransformer

# Create components
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
store = StandardMemoryStore()
encoder = MemoryEncoder(embedding_model)
retriever = Retriever(memory=store, embedding_model=embedding_model)
retriever.initialize_components()

# Add memories
text1 = "MemoryWeave uses a component-based architecture for memory management."
text2 = "Components can be combined into retrieval pipelines with various strategies."
embedding1 = encoder.encode_text(text1)
embedding2 = encoder.encode_text(text2)
memory_id1 = store.add(embedding1, text1)
memory_id2 = store.add(embedding2, text2)

# Retrieve memories
query = "How does MemoryWeave organize its code?"
results = retriever.retrieve(query, top_k=3)
for result in results:
    print(f"Score: {result['relevance_score']:.4f} - {result['content']}")
```

## Core Components

- **Memory Storage**: StandardMemoryStore, HybridMemoryStore, ChunkedMemoryStore
- **Retrieval Strategies**: Various approaches for retrieving relevant memories
- **Query Processing**: Tools for analyzing and adapting queries
- **Memory Encoding**: Converts different content types into memory representations
- **Post-Processing**: Fine-tuning of retrieval results based on various signals

## Project Status

MemoryWeave is undergoing active development with a focus on:

- Refining the core retrieval functionality
- Optimizing performance for large memory sets
- Enhancing the component-based architecture
- Improving retrieval precision and recall

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest
```

## License

[MIT License](LICENSE)
