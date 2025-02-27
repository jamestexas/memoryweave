# MemoryWeave

MemoryWeave is an experimental approach to memory management for language models that uses a "contextual fabric" approach inspired by biological memory systems. Rather than traditional knowledge graph approaches with discrete nodes and edges, MemoryWeave focuses on capturing rich contextual signatures of information for improved long-context coherence in LLM conversations.

> **Note:** This project is in early development and is not yet ready for production use.

## Table of Contents
- [Key Concepts](#key-concepts)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Architecture](#architecture)
- [Examples](#examples)
- [Current Limitations](#current-limitations)
- [Contributing](#contributing)

## Key Concepts
<a id="key-concepts"></a>

MemoryWeave implements several biologically-inspired memory management principles:

- **Contextual Fabric**: Memory traces capture rich contextual signatures rather than isolated facts
- **Activation-Based Retrieval**: Memory retrieval uses dynamic activation patterns similar to biological systems
- **Episodic Structure**: Memories maintain temporal relationships and episodic anchoring
- **Non-Structured Memory**: Works with raw LLM outputs without requiring structured formats
- **ART-Inspired Clustering**: Optional memory categorization based on Adaptive Resonance Theory

<details>
<summary><strong>More about the contextual fabric approach</strong></summary>

Traditional LLM memory systems often rely on vector databases with discrete entries, losing much of the rich contextual information that helps humans navigate memories effectively. MemoryWeave attempts to address this by:

1. **Contextual Encoding**: Memories include surrounding context and metadata
2. **Activation Dynamics**: Recently or frequently accessed memories have higher activation levels
3. **Temporal Organization**: Memories maintain their relationship to other events in time
4. **Associative Retrieval**: Memories can be retrieved through multiple pathways beyond simple similarity
5. **Dynamic Categorization**: Memories self-organize into categories using ART-inspired clustering

This allows for more nuanced and effective memory retrieval during conversations, especially over long contexts or multiple sessions.
</details>

## Installation
<a id="installation"></a>

```bash
# Using uv (recommended)
uv pip install memoryweave

# Using pip
pip install memoryweave
```

## Basic Usage
<a id="basic-usage"></a>

```python
import torch
from transformers import AutoModel, AutoTokenizer
from memoryweave.core import ContextualMemory, MemoryEncoder, ContextualRetriever
from memoryweave.integrations import HuggingFaceAdapter

# Load models
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
llm_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Initialize memory components with ART clustering
memory = ContextualMemory(
    embedding_dim=384,
    use_art_clustering=True,
    vigilance_threshold=0.85
) 
encoder = MemoryEncoder(embedding_model)
retriever = ContextualRetriever(memory, embedding_model)
memory_system = {"memory": memory, "encoder": encoder, "retriever": retriever}

# Create adapter
adapter = HuggingFaceAdapter(
    memory_system=memory_system,
    model=llm_model,
    tokenizer=tokenizer
)

# Generate with memory augmentation
response = adapter.generate(
    user_input="What do you know about neural networks?",
    conversation_history=[]
)
```

<details>
<summary><strong>More comprehensive example</strong></summary>

```python
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# Define a wrapper for embedding model
class EmbeddingModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Mean pooling
        attention_mask = inputs["attention_mask"]
        embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts
        
        return mean_pooled.numpy()[0]

# Load models
emb_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
emb_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = EmbeddingModelWrapper(emb_model, emb_tokenizer)

# Initialize memory system
from memoryweave.core import ContextualMemory, MemoryEncoder, ContextualRetriever
embedding_dim = emb_model.config.hidden_size  # 384 for MiniLM-L6
memory = ContextualMemory(
    embedding_dim=embedding_dim,
    use_art_clustering=True,
    vigilance_threshold=0.85
)
encoder = MemoryEncoder(embedding_model)
retriever = ContextualRetriever(memory=memory, embedding_model=embedding_model)
memory_system = {"memory": memory, "encoder": encoder, "retriever": retriever}

# Initialize LLM
lm_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
lm_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Create adapter
from memoryweave.integrations import HuggingFaceAdapter
adapter = HuggingFaceAdapter(
    memory_system=memory_system,
    model=lm_model,
    tokenizer=lm_tokenizer
)

# Run a conversation with memory
conversation_history = []

# First turn
response1 = adapter.generate(
    user_input="My favorite color is blue.",
    conversation_history=conversation_history,
    generation_kwargs={"max_new_tokens": 100}
)
conversation_history.append({"speaker": "user", "message": "My favorite color is blue.", "response": response1})

# Second turn
response2 = adapter.generate(
    user_input="What color did I say I liked?",
    conversation_history=conversation_history,
    generation_kwargs={"max_new_tokens": 100}
)
```
</details>

## Architecture
<a id="architecture"></a>

MemoryWeave uses a modular architecture with three main components:

1. **ContextualMemory**: Stores embeddings and metadata with activation levels
2. **MemoryEncoder**: Converts different content types into rich memory representations
3. **ContextualRetriever**: Retrieves memories using context-aware strategies

<details>
<summary><strong>Architecture diagram</strong></summary>

```
┌─────────────────────────────────────┐
│            LLM Framework            │
│  (Hugging Face, OpenAI, LangChain)  │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│           Adapter Layer             │
│    (HuggingFaceAdapter, etc.)       │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│         ContextualRetriever         │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│          ContextualMemory           │
│     (with ART-inspired clustering)  │
└─────────────────────────────────────┘
                ▲
                │
┌─────────────────────────────────────┐
│           MemoryEncoder             │
└─────────────────────────────────────┘
```
</details>

## Examples
<a id="examples"></a>

Check out the following examples to get started:

- `test_memory.py`: Basic memory operations
- `test_with_tinyllama.py`: Integration with TinyLlama 1.1B
- `test_with_orca.py`: Integration with Orca Mini 3B
- `test_art_memory.py`: Demonstration of ART-inspired clustering

## Current Limitations
<a id="current-limitations"></a>

This project is in early development and has several limitations:

- Limited testing with large-scale models
- No persistence layer for long-term storage
- Basic interface that will likely change
- Performance not yet optimized for large memory stores

## Contributing
<a id="contributing"></a>

Contributions are welcome! Since this is an early-stage project, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development
<a id="development"></a>

This project uses `uv` for package management:

```bash
# Install in development mode
uv pip install -e .

# Run a script
uv run python script_name.py

# Run tests
uv run python -m pytest
```