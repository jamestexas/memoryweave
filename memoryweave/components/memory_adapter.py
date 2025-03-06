"""
Memory adapter for integrating core memory with the components architecture.
"""

from typing import Any, Optional

import numpy as np

from memoryweave.components.base import MemoryComponent

# Remove the import from core
# from memoryweave.core.contextual_memory import ContextualMemory


class MemoryAdapter(MemoryComponent):
    """
    Adapter for integrating the core memory system with the components architecture.

    This class wraps the core memory system and exposes it through the component
    interface, allowing it to be used seamlessly within the pipeline architecture.
    """

    def __init__(self, memory: Optional[Any] = None, **memory_kwargs):
        """
        Initialize the memory adapter.

        Args:
            memory: Existing memory instance to adapt
            **memory_kwargs: Arguments to pass to memory constructor if creating a new instance
        """
        self.memory = memory

        # If no memory provided, use the refactored memory store
        if self.memory is None:
            from memoryweave.storage.refactored.adapter import MemoryAdapter as RefactoredAdapter
            from memoryweave.storage.refactored.memory_store import StandardMemoryStore

            memory_store = StandardMemoryStore()
            self.memory = RefactoredAdapter(memory_store)

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        # Apply any configuration updates
        # For now, we don't change any settings after initialization
        pass

    def process(self, data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        Process memory data with context.

        This method handles memory operations like adding or retrieving memories.

        Args:
            data: Data to process
            context: Context for processing

        Returns:
            Updated context with processing results
        """
        operation = data.get("operation")

        if operation == "add_memory":
            # Add a memory to the system
            embedding = data.get("embedding")
            text = data.get("text")
            metadata = data.get("metadata", {})

            memory_id = self.memory.add_memory(embedding, text, metadata)
            return {"memory_id": memory_id}

        elif operation == "retrieve_memories":
            # Retrieve memories from the system
            query_embedding = data.get("query_embedding")
            top_k = data.get("top_k", 5)
            confidence_threshold = data.get("confidence_threshold")

            results = self.memory.retrieve_memories(
                query_embedding=query_embedding,
                top_k=top_k,
                confidence_threshold=confidence_threshold,
            )

            # Convert tuple results to dictionaries for easier handling in pipeline
            formatted_results = []
            for idx, score, metadata in results:
                formatted_results.append({"memory_id": idx, "relevance_score": score, **metadata})

            return {"results": formatted_results}

        elif operation == "get_category_statistics":
            # Get statistics about categories
            stats = self.memory.get_category_statistics()
            return {"category_statistics": stats}

        elif operation == "consolidate_categories":
            # Manually consolidate categories
            threshold = data.get("threshold")
            result = self.memory.consolidate_categories_manually(threshold)
            return {"num_categories": result}

        # Default response if no operation matched
        return {}

    def get_memory(self) -> Any:
        """Get the underlying memory instance."""
        return self.memory

    def search_hybrid(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        threshold: float = 0.0,
        keywords: list[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid search using vector similarity and keywords.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            keywords: Optional keywords to boost results with

        Returns:
            List of memory dicts with relevance scores
        """
        # First perform vector search
        vector_results = self.search_by_vector(
            query_vector=query_vector,
            limit=limit * 2,  # Get more candidates
            threshold=threshold,
        )

        # If no keywords, just return vector results
        if not keywords:
            return vector_results[:limit]

        # Enhance scores for results containing keywords
        for result in vector_results:
            content = str(result.get("content", "")).lower()

            # Count keyword matches
            keyword_matches = sum(1 for kw in keywords if kw.lower() in content)

            if keyword_matches > 0:
                # Apply keyword boost proportional to matches
                boost = min(0.3 * keyword_matches / len(keywords), 0.5)

                # Update score
                original_score = result.get("relevance_score", result.get("score", 0))
                new_score = min(0.99, original_score + boost * (1.0 - original_score))

                # Apply the boost
                result["relevance_score"] = new_score
                result["keyword_boost"] = boost
                result["keyword_matches"] = keyword_matches

        # Sort by boosted scores
        vector_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return vector_results[:limit]
