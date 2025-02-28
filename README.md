# MemoryWeave

MemoryWeave is an experimental approach to memory management for language models that uses a "contextual fabric" approach inspired by biological memory systems. Rather than traditional knowledge graph approaches with discrete nodes and edges, MemoryWeave focuses on capturing rich contextual signatures of information for improved long-context coherence in LLM conversations.

> **Note:** This project is in early development and is not yet ready for production use.

## Table of Contents
- [Key Concepts](#key-concepts)
- [Architecture](#architecture)
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

## Architecture
<a id="architecture"></a>

MemoryWeave uses a modular architecture with three main components:

1. **ContextualMemory**: Stores embeddings and metadata with activation levels
2. **MemoryEncoder**: Converts different content types into rich memory representations
3. **ContextualRetriever**: Retrieves memories using context-aware strategies

<details>
<summary><strong>Architecture diagram</strong></summary>

```mermaid
flowchart TD
    LLM["**LLM Framework:**<br>Hugging Face, OpenAI, LangChain"] --> Adapter
    Adapter["**Adapter Layer**<br>HuggingFaceAdapter, etc."] --> Retriever
    Retriever[ContextualRetriever] --> Memory
    Memory["**ContextualMemory**<br>with ART-inspired clustering"] --> Retriever
    Encoder[MemoryEncoder] --> Memory
    
    classDef primary fill:#d0e0ff,stroke:#3080ff,stroke-width:2px
    classDef secondary fill:#e0f0e0,stroke:#30a030,stroke-width:2px
    
    class Memory,Retriever,Encoder primary
    class LLM,Adapter secondary
```

```mermaid
flowchart TD
    subgraph MemoryWeave[MemoryWeave System]
        Memory[ContextualMemory]
        Encoder[MemoryEncoder]
        Retriever[ContextualRetriever]
    end
    
    subgraph Integration[Integration Layer]
        Adapter[Adapter\nHuggingFace/OpenAI/LangChain]
    end
    
    subgraph LLM[LLM Framework]
        Model[Language Model]
    end
    
    User[User Input] --> Adapter
    Adapter --> Model
    Adapter --> Retriever
    Retriever --> Memory
    Memory --> Retriever
    Encoder --> Memory
    Model --> Adapter
    Adapter --> Response[Response to User]
    
    classDef primary fill:#d0e0ff,stroke:#3080ff,stroke-width:2px
    classDef secondary fill:#e0f0e0,stroke:#30a030,stroke-width:2px
    classDef external fill:#f0e0d0,stroke:#a07030,stroke-width:2px
    
    class Memory,Retriever,Encoder primary
    class Adapter secondary
    class User,Model,Response external
```
</details>

<details>
<summary><strong>Memory retrieval mechanism</strong></summary>

```mermaid
flowchart TD
    Query[User Query] --> QueryEmbed[Encode Query]
    QueryEmbed --> RetrievalStrategy{Retrieval<br>Strategy}
    
    RetrievalStrategy -->|Similarity| SimilarityRetrieval[Similarity-Based<br>Retrieval]
    RetrievalStrategy -->|Temporal| TemporalRetrieval[Recency-Based<br>Retrieval]
    RetrievalStrategy -->|Hybrid| HybridRetrieval[Hybrid<br>Retrieval]
    
    SimilarityRetrieval --> ConfidenceFilter[Confidence<br>Thresholding]
    TemporalRetrieval --> ActivationBoost[Activation<br>Boosting]
    HybridRetrieval --> KeywordBoost[Keyword<br>Boosting]
    
    ConfidenceFilter --> CoherenceCheck{Semantic<br>Coherence Check}
    ActivationBoost --> AdaptiveK[Adaptive K<br>Selection]
    KeywordBoost --> PersonalAttributes[Personal Attribute<br>Enhancement]
    
    CoherenceCheck -->|Yes| CoherentMemories[Coherent<br>Memories]
    CoherenceCheck -->|No| BestMemory[Best Single<br>Memory]
    
    AdaptiveK --> FinalMemories[Final Retrieved<br>Memories]
    PersonalAttributes --> FinalMemories
    CoherentMemories --> FinalMemories
    BestMemory --> FinalMemories
    
    FinalMemories --> PromptAugmentation[Prompt<br>Augmentation]
    PromptAugmentation --> LLMGeneration[LLM<br>Generation]
    
    classDef primary fill:#d0e0ff,stroke:#3080ff,stroke-width:2px
    classDef secondary fill:#e0f0e0,stroke:#30a030,stroke-width:2px
    classDef decision fill:#ffe0d0,stroke:#ff8030,stroke-width:2px
    
    class Query,QueryEmbed,FinalMemories,PromptAugmentation,LLMGeneration primary
    class SimilarityRetrieval,TemporalRetrieval,HybridRetrieval,ConfidenceFilter,ActivationBoost,KeywordBoost,CoherentMemories,BestMemory,AdaptiveK,PersonalAttributes secondary
    class RetrievalStrategy,CoherenceCheck decision
```

```mermaid
flowchart TD
    subgraph ART[ART-Inspired Clustering]
        Input[New Memory] --> Vigilance{Vigilance<br>Check}
        Vigilance -->|Match| UpdateCategory[Update Existing<br>Category]
        Vigilance -->|No Match| CreateCategory[Create New<br>Category]
        UpdateCategory --> Consolidation{Consolidation<br>Check}
        CreateCategory --> Consolidation
        Consolidation -->|Needed| MergeCategories[Merge Similar<br>Categories]
        Consolidation -->|Not Needed| Done[Done]
        MergeCategories --> Done
    end
    
    subgraph Retrieval[Category-Based Retrieval]
        QueryInput[Query] --> CategoryMatch[Find Matching<br>Categories]
        CategoryMatch --> MemoryRetrieval[Retrieve Memories<br>from Categories]
        MemoryRetrieval --> Ranking[Rank by<br>Relevance]
        Ranking --> TopResults[Return Top<br>Results]
    end
    
    classDef primary fill:#d0e0ff,stroke:#3080ff,stroke-width:2px
    classDef secondary fill:#e0f0e0,stroke:#30a030,stroke-width:2px
    classDef decision fill:#ffe0d0,stroke:#ff8030,stroke-width:2px
    
    class Input,QueryInput,TopResults primary
    class UpdateCategory,CreateCategory,CategoryMatch,MemoryRetrieval,Ranking,MergeCategories secondary
    class Vigilance,Consolidation decision
```
</details>

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

# Run tests
uv run python -m pytest
```

Check the [ROADMAP.md](ROADMAP.md) file for planned future developments.