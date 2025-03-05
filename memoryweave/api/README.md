# MemoryWeave API

This directory contains the high-level API components for MemoryWeave, designed for easy integration with applications.

## Components

### MemoryWeaveAPI

The main API class that provides a unified interface for memory management and retrieval.

```python
from memoryweave.api.memory_weave import MemoryWeaveAPI

# Initialize
api = MemoryWeaveAPI()

# Add a memory
memory_id = api.add_memory("This is a sample memory", {"type": "note"})

# Chat with context
response = api.chat("What did I tell you earlier?")
```

### Specialized APIs

- `ChunkedMemoryWeaveAPI` - Enhanced API with chunking support for large contexts
- `HybridMemoryWeaveAPI` - Memory-efficient API with selective chunking

## Architecture

The API layer:

1. Uses refactored storage components under the hood
2. Handles embedding generation
3. Manages conversation history
4. Orchestrates the retrieval process
5. Formats prompts for the LLM

## Dependencies

- `MemoryAdapter` from storage.refactored
- `LLMProvider` for model interactions
- `RetrievalOrchestrator` for memory search

## Usage Examples

### Standard API

```python
from memoryweave.api.memory_weave import MemoryWeaveAPI

api = MemoryWeaveAPI()
api.add_memory("The capital of France is Paris.")
response = api.chat("What's the capital of France?")
```

### Chunked API for Large Texts

```python
from memoryweave.api.chunked_memory_weave import ChunkedMemoryWeaveAPI

api = ChunkedMemoryWeaveAPI()
api.add_memory(long_document_text)
response = api.chat("Summarize the key points from the document.")
```

### Streaming API

```python
from memoryweave.api.memory_weave import MemoryWeaveAPI

api = MemoryWeaveAPI()
api.add_memory("The Eiffel Tower is in Paris, France.")

# Stream response
async for token in api.chat_stream("Where is the Eiffel Tower?"):
    print(token, end="", flush=True)
```