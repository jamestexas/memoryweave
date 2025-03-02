"""Hybrid retrieval strategy for MemoryWeave.

This module provides implementations of retrieval strategies that combine
multiple approaches such as similarity and temporal factors.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from memoryweave.interfaces.memory import (
    EmbeddingVector,
    IActivationManager,
    IMemoryStore,
    IVectorStore,
)
from memoryweave.interfaces.retrieval import (
    IRetrievalStrategy,
    RetrievalParameters,
    RetrievalResult,
)


class HybridRetrievalStrategy(IRetrievalStrategy):
    """Retrieval strategy combining similarity and temporal factors."""
    
    def process(self, input_data: Any) -> Any:
        """Process the input data as a pipeline stage.
        
        This method implements IPipelineStage.process to make the component
        usable in a pipeline.
        """
        # Handle different types of input
        if isinstance(input_data, dict):
            # Check if this is a structured query
            if "embedding" in input_data:
                # Get query embedding and parameters
                query_embedding = input_data["embedding"]
                parameters = input_data.get("parameters", {})
                
                # Retrieve memories based on the query
                memories = self.retrieve(query_embedding, parameters)
                
                # Return the memories
                return memories
                
            # Check if this is just a vector
            elif "query_embedding" in input_data:
                query_embedding = input_data["query_embedding"]
                parameters = input_data.get("parameters", {})
                
                # Retrieve memories based on the query
                memories = self.retrieve(query_embedding, parameters)
                
                # Return the memories
                return memories
        
        # Pass through for unsupported input types
        return input_data

    def __init__(
        self,
        memory_store: IMemoryStore,
        vector_store: IVectorStore,
        activation_manager: IActivationManager,
    ):
        """Initialize the hybrid retrieval strategy.

        Args:
            memory_store: Memory store to retrieve memory content
            vector_store: Vector store for similarity search
            activation_manager: Activation manager for memory activations
        """
        self._memory_store = memory_store
        self._vector_store = vector_store
        self._activation_manager = activation_manager
        self._default_params = {
            "similarity_threshold": 0.6,
            "max_results": 10,
            "recency_bias": 0.3,
            "activation_boost": 0.2,
        }
        self.component_id = "hybrid_retrieval_strategy"
        
    def get_id(self) -> str:
        """Get the unique identifier for this component."""
        return self.component_id
        
    def get_type(self):
        """Get the type of this component."""
        from memoryweave.interfaces.pipeline import ComponentType
        return ComponentType.RETRIEVAL_STRATEGY

    def retrieve(
        self, query_embedding: EmbeddingVector, parameters: Optional[RetrievalParameters] = None
    ) -> List[RetrievalResult]:
        """Retrieve memories using a hybrid approach."""
        # Merge parameters with defaults
        params = self._default_params.copy()
        if parameters:
            params.update(parameters)

        # Get parameters
        similarity_threshold = params.get("similarity_threshold", 0.6)
        max_results = params.get("max_results", 10)
        recency_bias = params.get("recency_bias", 0.3)
        activation_boost = params.get("activation_boost", 0.2)
        min_results = params.get("min_results", 0)

        # Get similarity matches (with a lower threshold to get more candidates)
        initial_max = max(max_results * 2, 20)  # Get more candidates for re-ranking
        similar_vectors = self._vector_store.search(
            query_vector=query_embedding,
            k=initial_max,
            threshold=similarity_threshold * 0.8,  # Lower threshold for more candidates
        )

        # Create initial results
        initial_results = []
        for memory_id, similarity_score in similar_vectors:
            # Get activation and convert to a score between 0 and 1
            activation = self._activation_manager.get_activation(memory_id)

            # Normalize activation (assuming activation is in range -10 to 10)
            normalized_activation = (activation + 10) / 20.0

            # Combine scores
            # Base score is similarity
            combined_score = similarity_score

            # Add recency bias if metadata has created_at
            memory = self._memory_store.get(memory_id)
            if "created_at" in memory.metadata:
                recency_score = self._calculate_recency_score(memory.metadata["created_at"])
                combined_score += recency_bias * recency_score

            # Add activation boost
            combined_score += activation_boost * normalized_activation

            # Normalize final score to 0-1 range
            final_score = min(1.0, combined_score)

            # Create result
            result = RetrievalResult(
                memory_id=memory_id,
                content=memory.content["text"],
                metadata=memory.metadata,
                relevance_score=float(final_score),
            )

            initial_results.append(result)

        # Sort by combined score and take top k
        ranked_results = sorted(initial_results, key=lambda x: x["relevance_score"], reverse=True)
        
        # If no results meet the threshold but min_results is set, return at least that many
        if len(ranked_results) < min_results and min_results > 0:
            # Special case when memory store has no memories
            if len(self._memory_store._memories) == 0:
                return []
                
            # Just get all memories if we need to ensure minimum results
            all_memory_ids = list(self._memory_store._memories.keys())
            
            # Calculate similarity scores for all memories
            similarity_scores = {}
            for memory_id in all_memory_ids:
                memory = self._memory_store.get(memory_id)
                # Try to get embedding from the memory
                try:
                    embedding = memory.embedding  # This is the standard way to access it
                    # Calculate cosine similarity
                    query_norm = np.linalg.norm(query_embedding)
                    memory_norm = np.linalg.norm(embedding)
                    if query_norm > 0 and memory_norm > 0:
                        similarity = np.dot(query_embedding, embedding) / (query_norm * memory_norm)
                        similarity_scores[memory_id] = similarity
                except Exception:
                    pass  # Skip memories with missing or invalid embeddings
            
            # Sort by similarity
            sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get top min_results
            more_vectors = sorted_scores[:min_results]
            
            # Only add memories not already in results
            existing_ids = {r["memory_id"] for r in ranked_results}
            for memory_id, similarity_score in more_vectors:
                if memory_id not in existing_ids and len(ranked_results) < min_results:
                    memory = self._memory_store.get(memory_id)
                    result = RetrievalResult(
                        memory_id=memory_id,
                        content=memory.content["text"],
                        metadata=memory.metadata,
                        relevance_score=float(similarity_score),
                    )
                    ranked_results.append(result)
                    existing_ids.add(memory_id)
            
            # Re-sort the results
            ranked_results = sorted(ranked_results, key=lambda x: x["relevance_score"], reverse=True)

        return ranked_results[:max_results]

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the retrieval strategy."""
        if "similarity_threshold" in config:
            self._default_params["similarity_threshold"] = config["similarity_threshold"]

        if "max_results" in config:
            self._default_params["max_results"] = config["max_results"]

        if "recency_bias" in config:
            self._default_params["recency_bias"] = config["recency_bias"]

        if "activation_boost" in config:
            self._default_params["activation_boost"] = config["activation_boost"]

    def _calculate_recency_score(self, created_at: float) -> float:
        """Calculate a recency score (0-1) based on creation time."""
        import time

        # Maximum age to consider (7 days)
        max_age_seconds = 7 * 24 * 60 * 60

        # Calculate age
        age_seconds = time.time() - created_at

        # Convert to a score between 0 and 1
        # 1.0 for brand new, 0.0 for older than max_age
        if age_seconds <= 0:
            return 1.0
        elif age_seconds >= max_age_seconds:
            return 0.0
        else:
            return 1.0 - (age_seconds / max_age_seconds)
