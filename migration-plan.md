# Memory Weave Migration Plan

## Current Status Analysis

Based on an analysis of the codebase, the migration from core/ to component-based architecture is already well underway. Most of the core interfaces have implementations, and the components are structured according to the planned architecture.

### Already Implemented Components

1. **Memory Layer**

   - `StandardMemoryStore`, `HybridMemoryStore`, `ChunkedMemoryStore` (implements `IMemoryStore`)
   - `SimpleVectorStore`, `ActivationVectorStore`, `ANNVectorStore` (implements `IVectorStore`)
   - `ActivationManager`, `TemporalActivationManager` (implements `IActivationManager`)

1. **Retrieval Layer**

   - `SimilarityRetrievalStrategy`, `HybridRetrievalStrategy`, `TemporalRetrievalStrategy` (implements `IRetrievalStrategy`)

1. **Query Processing Layer**

   - `SimpleQueryAnalyzer` (implements `IQueryAnalyzer`)
   - `QueryTypeAdapter` (implements `IQueryAdapter`)
   - `KeywordExpander` (implements `IQueryExpander`)

1. **Integration Layer (Partial)**

   - `PipelineManager` and `PipelineBuilder` for component pipelines
   - `MemoryAdapter` for consistent access to memory stores

### Missing Components

Based on the provided code and outlined architecture:

1. **Main Retriever Component**

   - Need a central `MemoryRetriever` to orchestrate retrieval components

1. **Memory Manager**

   - Need a central manager to coordinate memory operations

1. **Integration Components**

   - Main API components are needed for app-level integration
   - Factories/config parsers for creating component instances

## Implementation Plan

### 1. Create Memory Manager

Create a comprehensive component to coordinate memory operations.

```python
# memoryweave/components/memory_manager.py
from typing import Any, Dict, List, Optional

from memoryweave.interfaces.memory import (
    IMemoryStore,
    IVectorStore,
    IActivationManager,
    Memory,
    EmbeddingVector,
    MemoryID,
)
from memoryweave.interfaces.pipeline import IPipelineStage


class MemoryManager(IPipelineStage):
    """Manages memory operations across different stores and components."""

    def __init__(
        self,
        memory_store: IMemoryStore,
        vector_store: IVectorStore,
        activation_manager: Optional[IActivationManager] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.memory_store = memory_store
        self.vector_store = vector_store
        self.activation_manager = activation_manager
        self.component_id = "memory_manager"
        self._config = config or {}

    def process(self, input_data: Any) -> Any:
        """Process the input as a pipeline stage."""
        if isinstance(input_data, dict):
            if "operation" in input_data:
                op = input_data["operation"]
                if op == "add_memory":
                    return self.add(
                        input_data.get("embedding"),
                        input_data.get("content"),
                        input_data.get("metadata"),
                    )
                elif op == "get_memory":
                    return self.get(input_data.get("memory_id"))
                elif op == "search":
                    return self.search(
                        input_data.get("query_embedding"),
                        input_data.get("limit", 10),
                        input_data.get("threshold"),
                    )
                # Add other operations as needed

        return input_data  # Pass through if not handled

    def add(
        self, embedding: EmbeddingVector, content: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryID:
        """Add a memory to the system."""
        # Add to memory store
        memory_id = self.memory_store.add(embedding, content, metadata)

        # Add to vector store
        self.vector_store.add(memory_id, embedding)

        # Initialize activation if available
        if self.activation_manager:
            self.activation_manager.update_activation(memory_id, 0.0)

        return memory_id

    def get(self, memory_id: MemoryID) -> Memory:
        """Get a memory by ID."""
        memory = self.memory_store.get(memory_id)

        # Update activation if available
        if self.activation_manager:
            self.activation_manager.update_activation(memory_id, 0.1)

        return memory

    def search(
        self, query_embedding: EmbeddingVector, limit: int = 10, threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar memories."""
        similar_vectors = self.vector_store.search(query_embedding, limit, threshold)

        results = []
        for memory_id, similarity in similar_vectors:
            memory = self.memory_store.get(memory_id)

            # Update activation if available
            if self.activation_manager:
                self.activation_manager.update_activation(memory_id, 0.2)

            result = {
                "memory_id": memory_id,
                "content": memory.content,
                "metadata": memory.metadata,
                "similarity": similarity,
            }
            results.append(result)

        return results

    # Other methods: remove, clear, update_metadata, etc.
```

### 2. Create Memory Retriever Component

Implement a central retriever to orchestrate retrieval operations.

```python
# memoryweave/components/memory_retriever.py
from typing import Any, Dict, List, Optional

from memoryweave.interfaces.memory import EmbeddingVector
from memoryweave.interfaces.retrieval import (
    IRetrievalStrategy,
    Query,
    RetrievalResult,
    RetrievalParameters,
)
from memoryweave.interfaces.query import IQueryAnalyzer, IQueryAdapter, IQueryExpander
from memoryweave.interfaces.pipeline import IPipelineStage


class MemoryRetriever(IPipelineStage):
    """Orchestrates the memory retrieval process."""

    def __init__(
        self,
        retrieval_strategy: IRetrievalStrategy,
        query_analyzer: Optional[IQueryAnalyzer] = None,
        query_adapter: Optional[IQueryAdapter] = None,
        query_expander: Optional[IQueryExpander] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.retrieval_strategy = retrieval_strategy
        self.query_analyzer = query_analyzer
        self.query_adapter = query_adapter
        self.query_expander = query_expander
        self.component_id = "memory_retriever"
        self._config = config or {}

    def process(self, input_data: Any) -> Any:
        """Process the input as a pipeline stage."""
        if isinstance(input_data, str):
            # Simple string query - process as query text
            return self.retrieve_from_text(input_data)
        elif isinstance(input_data, dict):
            if "text" in input_data:
                # Dict with query text
                parameters = input_data.get("parameters", {})
                return self.retrieve_from_text(input_data["text"], parameters)
            elif "embedding" in input_data:
                # Dict with query embedding
                parameters = input_data.get("parameters", {})
                return self.retrieve(input_data["embedding"], parameters)

        return input_data  # Pass through if not handled

    def retrieve_from_text(
        self, query_text: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve memories based on query text."""
        # 1. Analyze query if analyzer is available
        query_type = None
        keywords = []
        entities = []
        if self.query_analyzer:
            query_type = self.query_analyzer.analyze(query_text)
            keywords = self.query_analyzer.extract_keywords(query_text)
            entities = self.query_analyzer.extract_entities(query_text)

        # Get query embedding from parameters or compute it
        query_embedding = None
        if parameters and "query_embedding" in parameters:
            query_embedding = parameters["query_embedding"]

        # Ensure we have an embedding
        if query_embedding is None:
            # Would need an encoder here - for now assume it's in parameters
            raise ValueError("Query embedding is required when no encoder is available")

        # 2. Create Query object
        query = Query(
            text=query_text,
            embedding=query_embedding,
            query_type=query_type,
            extracted_keywords=keywords,
            extracted_entities=entities,
            context=None,  # Could add context if available
        )

        # 3. Expand query if expander is available
        if self.query_expander:
            query = self.query_expander.expand(query)

        # 4. Adapt parameters if adapter is available
        retrieval_params = {}
        if parameters:
            retrieval_params.update(parameters)

        if self.query_adapter:
            adapter_params = self.query_adapter.adapt_parameters(query)
            retrieval_params.update(adapter_params)

        # 5. Retrieve results using the strategy
        return self.retrieval_strategy.retrieve(query.embedding, retrieval_params)

    def retrieve(
        self, query_embedding: EmbeddingVector, parameters: Optional[RetrievalParameters] = None
    ) -> List[RetrievalResult]:
        """Retrieve memories based on query embedding."""
        return self.retrieval_strategy.retrieve(query_embedding, parameters)
```

### 3. Create API Facade for Application Integration

Create a simpler API for application-level integration.

```python
# memoryweave/components/memory_api.py
from typing import Any, Dict, List, Optional

from memoryweave.components.memory_manager import MemoryManager
from memoryweave.components.memory_retriever import MemoryRetriever
from memoryweave.interfaces.memory import EmbeddingVector, MemoryID


class MemoryAPI:
    """Main API for interacting with the memory system."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        memory_retriever: MemoryRetriever,
        encoder: Optional[Any] = None,  # Could use a specific encoder interface
    ):
        self.memory_manager = memory_manager
        self.memory_retriever = memory_retriever
        self.encoder = encoder

    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryID:
        """Add a memory to the system."""
        # Generate embedding if encoder is available
        embedding = None
        if self.encoder:
            embedding = self.encoder.encode_text(content)

        if embedding is None:
            raise ValueError("Encoder is required to add memories by content")

        return self.memory_manager.add(embedding, content, metadata)

    def add_memory_with_embedding(
        self, embedding: EmbeddingVector, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryID:
        """Add a memory with a pre-computed embedding."""
        return self.memory_manager.add(embedding, content, metadata)

    def get_memory(self, memory_id: MemoryID) -> Dict[str, Any]:
        """Get a memory by ID."""
        memory = self.memory_manager.get(memory_id)
        return {"id": memory.id, "content": memory.content, "metadata": memory.metadata}

    def search_by_text(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for memories by text query."""
        # Generate embedding if encoder is available
        embedding = None
        if self.encoder:
            embedding = self.encoder.encode_text(query_text)

        if embedding is None:
            raise ValueError("Encoder is required to search by text")

        # Use the retriever to get results
        parameters = {"max_results": limit}
        results = self.memory_retriever.retrieve_from_text(
            query_text, {"query_embedding": embedding, **parameters}
        )

        # Format results for API response
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "memory_id": result["memory_id"],
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "relevance_score": result["relevance_score"],
                }
            )

        return formatted_results

    def search_by_embedding(
        self, query_embedding: EmbeddingVector, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for memories by embedding."""
        # Use the retriever to get results
        parameters = {"max_results": limit}
        results = self.memory_retriever.retrieve(query_embedding, parameters)

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "memory_id": result["memory_id"],
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "relevance_score": result["relevance_score"],
                }
            )

        return formatted_results
```

### 4. Create Factory for Component Configuration

Create a factory to simplify component creation from configuration.

```python
# memoryweave/components/factory.py
from typing import Any, Dict, Optional, Type

from memoryweave.interfaces.memory import IMemoryStore, IVectorStore, IActivationManager
from memoryweave.interfaces.retrieval import IRetrievalStrategy
from memoryweave.interfaces.query import IQueryAnalyzer, IQueryAdapter, IQueryExpander

from memoryweave.storage.refactored.memory_store import StandardMemoryStore
from memoryweave.storage.refactored.hybrid_store import HybridMemoryStore
from memoryweave.storage.refactored.chunked_store import ChunkedMemoryStore
from memoryweave.storage.vector_store import (
    SimpleVectorStore,
    ActivationVectorStore,
    ANNVectorStore,
)
from memoryweave.storage.activation import ActivationManager, TemporalActivationManager

from memoryweave.retrieval.similarity import SimilarityRetrievalStrategy
from memoryweave.retrieval.hybrid import HybridRetrievalStrategy
from memoryweave.retrieval.temporal import TemporalRetrievalStrategy

from memoryweave.query.analyzer import SimpleQueryAnalyzer
from memoryweave.query.adaptation import QueryTypeAdapter
from memoryweave.query.keyword import KeywordExpander

from memoryweave.components.memory_manager import MemoryManager
from memoryweave.components.memory_retriever import MemoryRetriever
from memoryweave.components.memory_api import MemoryAPI


class ComponentFactory:
    """Factory for creating memory components from configuration."""

    @staticmethod
    def create_memory_store(config: Dict[str, Any]) -> IMemoryStore:
        """Create a memory store from configuration."""
        store_type = config.get("type", "standard")

        if store_type == "standard":
            return StandardMemoryStore()
        elif store_type == "hybrid":
            return HybridMemoryStore()
        elif store_type == "chunked":
            return ChunkedMemoryStore()
        else:
            raise ValueError(f"Unknown memory store type: {store_type}")

    @staticmethod
    def create_vector_store(config: Dict[str, Any]) -> IVectorStore:
        """Create a vector store from configuration."""
        store_type = config.get("type", "simple")

        if store_type == "simple":
            return SimpleVectorStore()
        elif store_type == "activation":
            activation_weight = config.get("activation_weight", 0.2)
            return ActivationVectorStore(activation_weight=activation_weight)
        elif store_type == "ann":
            dimension = config.get("dimension", 768)
            index_type = config.get("index_type", "IVF100,Flat")
            metric = config.get("metric", "cosine")
            nprobe = config.get("nprobe", 10)
            build_threshold = config.get("build_threshold", 50)
            quantize = config.get("quantize", False)

            return ANNVectorStore(
                dimension=dimension,
                index_type=index_type,
                metric=metric,
                nprobe=nprobe,
                build_threshold=build_threshold,
                quantize=quantize,
            )
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")

    @staticmethod
    def create_activation_manager(config: Dict[str, Any]) -> IActivationManager:
        """Create an activation manager from configuration."""
        manager_type = config.get("type", "standard")

        if manager_type == "standard":
            initial_activation = config.get("initial_activation", 0.0)
            max_activation = config.get("max_activation", 10.0)
            min_activation = config.get("min_activation", -10.0)

            return ActivationManager(
                initial_activation=initial_activation,
                max_activation=max_activation,
                min_activation=min_activation,
            )
        elif manager_type == "temporal":
            initial_activation = config.get("initial_activation", 0.0)
            max_activation = config.get("max_activation", 10.0)
            min_activation = config.get("min_activation", -10.0)
            half_life_days = config.get("half_life_days", 7.0)

            return TemporalActivationManager(
                initial_activation=initial_activation,
                max_activation=max_activation,
                min_activation=min_activation,
                half_life_days=half_life_days,
            )
        else:
            raise ValueError(f"Unknown activation manager type: {manager_type}")

    @staticmethod
    def create_retrieval_strategy(
        config: Dict[str, Any],
        memory_store: IMemoryStore,
        vector_store: IVectorStore,
        activation_manager: Optional[IActivationManager] = None,
    ) -> IRetrievalStrategy:
        """Create a retrieval strategy from configuration."""
        strategy_type = config.get("type", "similarity")

        if strategy_type == "similarity":
            return SimilarityRetrievalStrategy(memory_store, vector_store)
        elif strategy_type == "hybrid":
            if activation_manager is None:
                raise ValueError("Activation manager is required for hybrid retrieval strategy")

            return HybridRetrievalStrategy(memory_store, vector_store, activation_manager)
        elif strategy_type == "temporal":
            if activation_manager is None:
                raise ValueError("Activation manager is required for temporal retrieval strategy")

            return TemporalRetrievalStrategy(memory_store, activation_manager)
        else:
            raise ValueError(f"Unknown retrieval strategy type: {strategy_type}")

    @staticmethod
    def create_memory_system(config: Dict[str, Any]) -> MemoryAPI:
        """Create a complete memory system from configuration."""
        # Create memory store
        memory_store = ComponentFactory.create_memory_store(config.get("memory_store", {}))

        # Create vector store
        vector_store = ComponentFactory.create_vector_store(config.get("vector_store", {}))

        # Create activation manager
        activation_manager = None
        if "activation_manager" in config:
            activation_manager = ComponentFactory.create_activation_manager(
                config["activation_manager"]
            )

        # Create memory manager
        memory_manager = MemoryManager(
            memory_store=memory_store,
            vector_store=vector_store,
            activation_manager=activation_manager,
        )

        # Create query analyzer
        query_analyzer = None
        if "query_analyzer" in config:
            query_analyzer = SimpleQueryAnalyzer()

        # Create query adapter
        query_adapter = None
        if "query_adapter" in config:
            query_adapter = QueryTypeAdapter()

        # Create query expander
        query_expander = None
        if "query_expander" in config:
            query_expander = KeywordExpander()

        # Create retrieval strategy
        retrieval_strategy = ComponentFactory.create_retrieval_strategy(
            config.get("retrieval_strategy", {}), memory_store, vector_store, activation_manager
        )

        # Create memory retriever
        memory_retriever = MemoryRetriever(
            retrieval_strategy=retrieval_strategy,
            query_analyzer=query_analyzer,
            query_adapter=query_adapter,
            query_expander=query_expander,
        )

        # Create encoder if specified
        encoder = None
        # TODO: Implement encoder based on config

        # Create and return API
        return MemoryAPI(
            memory_manager=memory_manager, memory_retriever=memory_retriever, encoder=encoder
        )
```

### 5. Remove CoreMemory and Other Deprecated Core Components

Ensure that the core directory is properly deprecated:

1. Document CoreMemory as deprecated (already done)
1. Modify imports throughout codebase to use component implementations
1. Maintain compatibility layer if needed (forwarding calls to new components)

### 6. Create Integration Tests for the New Component-Based Architecture

Create integration tests to validate the component-based implementation:

```python
# tests/integration/test_memory_components.py

# Test basic memory operations
def test_memory_add_and_retrieve():
    # Create system with factory
    config = {
        "memory_store": {"type": "standard"},
        "vector_store": {"type": "simple"},
        "activation_manager": {"type": "standard"},
    }
    memory_api = ComponentFactory.create_memory_system(config)

    # Add memory
    test_content = "This is a test memory"
    memory_id = memory_api.add_memory(test_content)

    # Retrieve memory
    memory = memory_api.get_memory(memory_id)

    assert memory["content"] == test_content


# Test retrieval
def test_memory_retrieval():
    # Create system with factory
    config = {...}  # Similar to above
    memory_api = ComponentFactory.create_memory_system(config)

    # Add memories
    memory_api.add_memory("The quick brown fox jumps over the lazy dog")
    memory_api.add_memory("Brown foxes are known for their agility")

    # Search
    results = memory_api.search_by_text("foxes jumping")

    assert len(results) > 0
    assert "fox" in results[0]["content"]
```

## Completion Milestones

1. Implement MemoryManager component
1. Implement MemoryRetriever component
1. Implement MemoryAPI facade
1. Implement ComponentFactory
1. Remove all core/ imports and references
1. Create integration tests
1. Verify feature parity through test coverage
