"""
Vector search baseline retriever implementation.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from memoryweave.baselines.base import BaselineRetriever
from memoryweave.interfaces.retrieval import Query
from memoryweave.storage.memory_store import Memory


class VectorBaselineRetriever(BaselineRetriever):
    """Simple vector search baseline retriever.

    This retriever implements basic vector search using cosine similarity,
    providing a baseline for comparison with more sophisticated
    retrieval approaches.
    """

    name: str = "vector_baseline"

    def __init__(self, use_exact_search: bool = True):
        """Initialize vector baseline retriever.

        Args:
            use_exact_search: Whether to use exact search (True) or approximate (False)
        """
        self.use_exact_search = use_exact_search
        self.memories: List[Memory] = []
        self.embeddings: Optional[np.ndarray] = None
        self.stats = {
            "index_size": 0,
            "dimensions": 0,
            "query_times": [],
            "avg_query_time": 0,
        }

    def index_memories(self, memories: List[Memory]) -> None:
        """Index a list of memories for vector retrieval.

        Args:
            memories: List of memories to index
        """
        start_time = time.time()

        self.memories = memories

        # Extract embeddings from memories
        if memories and len(memories) > 0:
            # Skip memories without embeddings
            valid_memories = []
            valid_embeddings = []

            for memory in memories:
                if memory.embedding is not None:
                    valid_memories.append(memory)
                    valid_embeddings.append(memory.embedding)

            self.memories = valid_memories

            if valid_embeddings:
                self.embeddings = np.vstack(valid_embeddings)
                self.stats["dimensions"] = self.embeddings.shape[1]

        self.stats["index_size"] = len(self.memories)
        self.stats["indexing_time"] = time.time() - start_time

    def retrieve(
        self, query: Query, top_k: int = 10, threshold: float = 0.0, **kwargs
    ) -> Dict[str, Any]:
        """Retrieve memories using vector similarity.

        Args:
            query: The query to search for
            top_k: Maximum number of results to return
            threshold: Minimum similarity score threshold
            **kwargs: Additional parameters (ignored)

        Returns:
            RetrievalResult containing matched memories
        """
        start_time = time.time()

        if len(self.memories) == 0 or self.embeddings is None or query.embedding is None:
            parameters = {"max_results": top_k, "threshold": threshold}

            return {
                "memories": [],
                "scores": [],
                "strategy": self.name,
                "parameters": parameters,
                "metadata": {"query_time": 0.0},
            }

        # Calculate similarity between query and all memories
        query_embedding = np.array(query.embedding).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Find top-k results above threshold
        if threshold > 0:
            mask = similarities >= threshold
            indices = np.where(mask)[0]
            scores = similarities[mask]
        else:
            # Get all scores
            indices = np.arange(len(similarities))
            scores = similarities

        # Sort by score descending
        sorted_indices = np.argsort(-scores)

        # Get top-k results
        top_indices = sorted_indices[:top_k]
        top_scores = scores[sorted_indices[:top_k]]

        # Get corresponding memories
        result_memories = [self.memories[i] for i in indices[top_indices] if i < len(self.memories)]
        result_scores = top_scores.tolist()

        query_time = time.time() - start_time
        self.stats["query_times"].append(query_time)
        self.stats["avg_query_time"] = np.mean(self.stats["query_times"])

        parameters = {"max_results": top_k, "threshold": threshold}

        return {
            "memories": result_memories,
            "scores": result_scores,
            "strategy": self.name,
            "parameters": parameters,
            "metadata": {
                "query_time": query_time,
                "search_type": "exact" if self.use_exact_search else "approximate",
            },
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics for the vector baseline.

        Returns:
            Dictionary of statistics including index size and query times
        """
        return self.stats

    def clear(self) -> None:
        """Clear the indexed memories."""
        self.memories = []
        self.embeddings = None
        self.stats = {
            "index_size": 0,
            "dimensions": 0,
            "query_times": [],
            "avg_query_time": 0,
        }
