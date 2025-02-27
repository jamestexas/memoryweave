"""
Implements context-aware memory retrieval strategies.
"""

from typing import Any, Optional

import numpy as np

from memoryweave.core.contextual_fabric import ContextualMemory


class ContextualRetriever:
    """
    Retrieves memories from the contextual fabric based on the current
    conversation context, using activation patterns and contextual relevance.
    """

    def __init__(
        self,
        memory: ContextualMemory,
        embedding_model: Any,
        retrieval_strategy: str = "hybrid",
        recency_weight: float = 0.3,
        relevance_weight: float = 0.7,
    ):
        """
        Initialize the contextual retriever.

        Args:
            memory: The contextual memory to retrieve from
            embedding_model: Model for encoding queries
            retrieval_strategy: Strategy for retrieval ('similarity', 'temporal', 'hybrid')
            recency_weight: Weight given to recency in hybrid retrieval
            relevance_weight: Weight given to relevance in hybrid retrieval
        """
        self.memory = memory
        self.embedding_model = embedding_model
        self.retrieval_strategy = retrieval_strategy
        self.recency_weight = recency_weight
        self.relevance_weight = relevance_weight

        # Conversation context tracking
        self.conversation_state = {
            "recent_topics": [],
            "user_interests": set(),
            "interaction_count": 0,
        }

    def retrieve_for_context(
        self,
        current_input: str,
        conversation_history: Optional[list[dict]] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve memories relevant to the current conversation context.

        Args:
            current_input: The current user input
            conversation_history: Recent conversation history
            top_k: Number of memories to retrieve

        Returns:
            list of relevant memory entries with metadata
        """
        # Update conversation state
        self._update_conversation_state(current_input, conversation_history)

        # Encode the query context
        query_context = self._build_query_context(current_input, conversation_history)
        query_embedding = self.embedding_model.encode(query_context)

        # Retrieve memories using the specified strategy
        if self.retrieval_strategy == "similarity":
            return self._retrieve_by_similarity(query_embedding, top_k)
        elif self.retrieval_strategy == "temporal":
            return self._retrieve_by_recency(top_k)
        else:  # hybrid approach
            return self._retrieve_hybrid(query_embedding, top_k)

    def _update_conversation_state(
        self, current_input: str, conversation_history: Optional[list[dict]]
    ) -> None:
        """Update internal conversation state tracking."""
        self.conversation_state["interaction_count"] += 1

        # Extract potential topics from current input
        # In a real implementation, this would use NLP to extract entities/topics
        potential_topics = current_input.split()[:5]  # Simplified example
        self.conversation_state["recent_topics"] = potential_topics

        # Update user interests based on conversation
        if conversation_history:
            # This is a simplistic approach; a real implementation would use
            # more sophisticated interest extraction
            for exchange in conversation_history[-3:]:  # Look at last 3 exchanges
                if "user" in exchange.get("speaker", "").lower():
                    words = exchange.get("message", "").split()
                    self.conversation_state["user_interests"].update(words[:3])

    def _build_query_context(
        self, current_input: str, conversation_history: Optional[list[dict]]
    ) -> str:
        """Build a rich query context from current input and conversation history."""
        query_context = f"Current input: {current_input}"

        if conversation_history and len(conversation_history) > 0:
            # Add recent conversation turns
            query_context += "\nRecent conversation:"
            for turn in conversation_history[-3:]:  # Last 3 turns
                speaker = turn.get("speaker", "Unknown")
                message = turn.get("message", "")
                query_context += f"\n{speaker}: {message}"

        # Add any persistent user interests
        if self.conversation_state["user_interests"]:
            interests = list(self.conversation_state["user_interests"])[:5]
            query_context += f"\nUser interests: {', '.join(interests)}"

        return query_context

    def _retrieve_by_similarity(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        """Retrieve memories based purely on contextual similarity."""
        results = self.memory.retrieve_memories(query_embedding, top_k=top_k, activation_boost=True)

        # Format results
        return [
            {"memory_id": idx, "relevance_score": score, **metadata}
            for idx, score, metadata in results
        ]

    def _retrieve_by_recency(self, top_k: int) -> list[dict]:
        """Retrieve memories based on recency and activation."""
        # This is a simplified implementation
        # In a real system, this would be more sophisticated

        # Get memories sorted by temporal markers (most recent first)
        temporal_order = np.argsort(-self.memory.temporal_markers)[:top_k]

        results = []
        for idx in temporal_order:
            results.append({
                "memory_id": int(idx),
                "relevance_score": float(self.memory.activation_levels[idx]),
                **self.memory.memory_metadata[idx],
            })

        return results

    def _retrieve_hybrid(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        """
        Hybrid retrieval combining similarity and recency.
        """
        # Get similarity scores
        similarities = np.dot(self.memory.memory_embeddings, query_embedding)

        # Normalize temporal factors
        max_time = float(self.memory.current_time)
        temporal_factors = self.memory.temporal_markers / max_time if max_time > 0 else 0

        # Combine scores
        combined_scores = (
            self.relevance_weight * similarities + self.recency_weight * temporal_factors
        )

        # Apply activation boost
        combined_scores = combined_scores * self.memory.activation_levels

        # Get top-k indices
        if top_k >= len(combined_scores):
            top_indices = np.argsort(-combined_scores)
        else:
            top_indices = np.argpartition(-combined_scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-combined_scores[top_indices])]

        # Format results
        results = []
        for idx in top_indices:
            score = float(combined_scores[idx])
            if score > 0:  # Only include positively scored results
                results.append({
                    "memory_id": int(idx),
                    "relevance_score": score,
                    "similarity": float(similarities[idx]),
                    "recency": float(temporal_factors[idx]),
                    **self.memory.memory_metadata[idx],
                })

        return results
