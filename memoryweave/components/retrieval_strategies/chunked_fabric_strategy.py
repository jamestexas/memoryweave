"""
Chunked Contextual Fabric Retrieval Strategy for MemoryWeave.

This module extends the ContextualFabricStrategy to support chunked memory retrieval,
enabling more accurate retrieval from large text contexts.
"""

import logging
from typing import Any, Optional

import numpy as np
from rich.logging import RichHandler

from memoryweave.components.activation import ActivationManager
from memoryweave.components.associative_linking import AssociativeMemoryLinker
from memoryweave.components.component_names import ComponentName
from memoryweave.components.retrieval_strategies.contextual_fabric_strategy import (
    ContextualFabricStrategy,
)
from memoryweave.components.temporal_context import TemporalContextBuilder

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(markup=True),  # allow colors in terminal
    ],
)


class ChunkedFabricStrategy(ContextualFabricStrategy):
    """
    An extension of ContextualFabricStrategy that leverages chunked memory representation.

    This strategy enhances the contextual fabric approach by:
    1. Performing chunk-level retrieval for more precise matching
    2. Aggregating results by memory to avoid fragmentation
    3. Leveraging chunk position and relationships
    4. Maintaining compatibility with existing components
    """

    def __init__(
        self,
        memory_store: Optional[Any] = None,
        associative_linker: Optional[AssociativeMemoryLinker] = None,
        temporal_context: Optional[TemporalContextBuilder] = None,
        activation_manager: Optional[ActivationManager] = None,
    ):
        """
        Initialize the chunked fabric strategy.

        Args:
            memory_store: Memory store or adapter supporting chunked retrieval
            associative_linker: Associative memory linker for traversing links
            temporal_context: Temporal context builder for time-based relevance
            activation_manager: Activation manager for memory accessibility
        """
        super().__init__(
            memory_store=memory_store,
            associative_linker=associative_linker,
            temporal_context=temporal_context,
            activation_manager=activation_manager,
        )
        self.component_id = ComponentName.CHUNKED_FABRIC_STRATEGY

        # Chunking specific parameters
        self.chunk_weight_decay = 0.8  # Weight decay for additional chunks from same memory
        self.max_chunks_per_memory = 3  # Maximum number of chunks to consider per memory
        self.combine_chunk_scores = True  # Whether to combine scores from multiple chunks
        self.prioritize_coherent_chunks = True  # Prioritize chunks that form coherent sections

        # Detect if memory store supports chunks
        self.supports_chunks = False
        # Will be set in initialize()

    def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the strategy with configuration.

        Args:
            config: Configuration dictionary with parameters
        """
        # First, initialize base class
        super().initialize(config)

        # Then, set chunking specific parameters
        self.chunk_weight_decay = config.get("chunk_weight_decay", self.chunk_weight_decay)
        self.max_chunks_per_memory = config.get("max_chunks_per_memory", self.max_chunks_per_memory)
        self.combine_chunk_scores = config.get("combine_chunk_scores", self.combine_chunk_scores)
        self.prioritize_coherent_chunks = config.get(
            "prioritize_coherent_chunks", self.prioritize_coherent_chunks
        )

        # Check if memory_store supports chunks
        if self.memory_store is not None:
            # Check if it's a ChunkedMemoryAdapter or supports chunk_embeddings property
            if hasattr(self.memory_store, "chunk_embeddings"):
                self.supports_chunks = True
                if self.debug:
                    self.logger.debug("Chunking support detected in memory store")
            elif hasattr(self.memory_store, "search_chunks"):
                self.supports_chunks = True
                if self.debug:
                    self.logger.debug("Chunking support detected via search_chunks method")
            elif hasattr(self.memory_store, "memory_store") and hasattr(
                self.memory_store.memory_store, "get_chunks"
            ):
                self.supports_chunks = True
                if self.debug:
                    self.logger.debug("Chunking support detected in nested memory store")

    def retrieve(
        self,
        query: str,
        query_embedding=None,
        top_k: int = 10,
        confidence_threshold: float = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Direct access to the chunked retrieval system.

        Args:
            query: Query text
            query_embedding: Optional pre-computed embedding (will compute if not provided)
            top_k: Maximum number of results to return
            confidence_threshold: Optional confidence threshold for filtering results
            **kwargs: Additional parameters for the retrieval strategy

        Returns:
            List of retrieved memories with relevance scores
        """
        # Determine if query needs chunking
        should_chunk_query = len(query) > self.auto_chunk_threshold

        # Handle query embedding differently based on size
        if should_chunk_query and query_embedding is None:
            # Create query chunks
            query_chunks = self.text_chunker.create_chunks(query)

            # Create embeddings for each chunk
            query_embeddings = []
            for chunk in query_chunks:
                chunk_text = chunk["text"]
                embedding = self.embedding_model.encode(
                    chunk_text, show_progress_bar=self.show_progress_bar
                )
                query_embeddings.append(embedding)

            # Use the first chunk as primary embedding, but store chunk info for context
            query_embedding = query_embeddings[0]
            kwargs["query_chunks"] = query_chunks
            kwargs["query_embeddings"] = query_embeddings
        elif query_embedding is None:
            # Compute standard embedding if not provided
            query_embedding = self._compute_embedding(query)
            if query_embedding is None:
                return []

        # Analyze query to get type and keywords
        query_info = self._analyze_query(query)
        _, adapted_params, expanded_keywords, query_type, entities = query_info

        # Override with any provided parameters
        if confidence_threshold is not None:
            if adapted_params is None:
                adapted_params = {}
            adapted_params["confidence_threshold"] = confidence_threshold

        # Add any custom parameters from kwargs
        for key, value in kwargs.items():
            if adapted_params is None:
                adapted_params = {}
            adapted_params[key] = value

        # Use retrieval orchestrator for consistent handling
        memories = self.retrieval_orchestrator.retrieve(
            query_embedding=query_embedding,
            query=query,
            query_type=query_type,
            expanded_keywords=expanded_keywords,
            entities=entities,
            adapted_params=adapted_params,
            top_k=top_k,
        )

        return memories

    def _retrieve_similar_chunks(
        self,
        query_embedding: np.ndarray,
        max_results: int,
        memory_store,
    ) -> list[dict[str, Any]]:
        """
        Enhanced chunk retrieval with semantic weighting.

        Args:
            query_embedding: Query embedding for similarity matching
            max_results: Maximum number of chunk results
            memory_store: Memory store to search

        Returns:
            List of chunk results with scores
        """
        if hasattr(memory_store, "search_chunks"):
            # Use direct search_chunks method but with higher limit to ensure good coverage
            # Get 3x more chunks than needed to ensure we have enough context
            initial_results = memory_store.search_chunks(
                query_embedding, limit=max_results * 3, threshold=None
            )

            # Group chunks by memory_id to establish document context
            memory_chunks = {}
            for result in initial_results:
                memory_id = result.get("memory_id")
                if memory_id not in memory_chunks:
                    memory_chunks[memory_id] = []
                memory_chunks[memory_id].append(result)

            # Select chunks considering document context
            enhanced_results = []
            for _memory_id, chunks in memory_chunks.items():
                # Sort chunks by similarity
                chunks.sort(key=lambda x: x.get("chunk_similarity", 0.0), reverse=True)

                # Take best chunk plus adjacent chunks for context (if available)
                best_chunk = chunks[0]
                best_idx = best_chunk.get("chunk_index", 0)

                # Find adjacent chunks to maintain coherence
                adjacent_chunks = [
                    c for c in chunks if abs(c.get("chunk_index", 0) - best_idx) <= 1
                ]

                # Add best chunk with enhanced score
                best_chunk_copy = dict(best_chunk)
                enhanced_results.append(best_chunk_copy)

                # Add adjacent chunks with slightly reduced scores
                for adj_chunk in adjacent_chunks[1:]:  # Skip the best chunk which is already added
                    adj_copy = dict(adj_chunk)
                    # Slightly reduce score but keep it high enough to be relevant
                    adj_copy["chunk_similarity"] = adj_copy.get("chunk_similarity", 0.0) * 0.9
                    enhanced_results.append(adj_copy)

            # Sort by similarity and limit results
            enhanced_results.sort(key=lambda x: x.get("chunk_similarity", 0.0), reverse=True)
            return enhanced_results[:max_results]

        # Fallback: Manual search using chunk_embeddings property
        if not hasattr(memory_store, "chunk_embeddings"):
            # Cannot perform chunk search
            if self.debug:
                self.logger.debug("Cannot perform chunk search, no chunk embeddings available")
            return []

        chunk_embeddings = memory_store.chunk_embeddings
        if len(chunk_embeddings) == 0:
            return []

        # Normalize query vector
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            query_norm = 1e-10
        normalized_query = query_embedding / query_norm

        # Calculate similarities
        similarities = np.dot(chunk_embeddings, normalized_query)

        # Get top indices
        top_indices = np.argsort(-similarities)[: max_results * 3]  # Get extra for grouping

        # Group by memory_id to establish document context
        memory_chunks = {}

        # Create result objects
        for idx in top_indices:
            # Get chunk info
            chunk_metadata = memory_store.chunk_metadata[idx]
            memory_id = chunk_metadata.get("memory_id", None)
            chunk_index = chunk_metadata.get("chunk_index", 0)

            # Create result object
            result = {
                "memory_id": memory_id,
                "chunk_index": chunk_index,
                "chunk_similarity": float(similarities[idx]),
                "metadata": chunk_metadata,
            }

            # If chunk_text is available, include it
            if "chunk_text" in chunk_metadata:
                result["content"] = chunk_metadata["chunk_text"]

            # Group by memory_id
            if memory_id not in memory_chunks:
                memory_chunks[memory_id] = []
            memory_chunks[memory_id].append(result)

        # Process each memory's chunks to enhance coherence
        enhanced_results = []
        for _memory_id, chunks in memory_chunks.items():
            # Sort chunks by similarity
            chunks.sort(key=lambda x: x.get("chunk_similarity", 0.0), reverse=True)

            # Take best chunk plus adjacent chunks for context
            best_chunk = chunks[0]
            best_idx = best_chunk.get("chunk_index", 0)

            # Find adjacent chunks to maintain coherence
            adjacent_chunks = [c for c in chunks if abs(c.get("chunk_index", 0) - best_idx) <= 1]

            # Add best chunk as-is
            enhanced_results.append(best_chunk)

            # Add adjacent chunks with slightly reduced scores
            for adj_chunk in adjacent_chunks[1:]:  # Skip the best chunk already added
                # Slightly reduce score but keep it relevant
                adj_chunk["chunk_similarity"] = adj_chunk.get("chunk_similarity", 0.0) * 0.9
                enhanced_results.append(adj_chunk)

        # Sort by similarity and limit final results
        enhanced_results.sort(key=lambda x: x.get("chunk_similarity", 0.0), reverse=True)
        return enhanced_results[:max_results]

    def _aggregate_chunks_by_memory(
        self,
        chunk_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Enhanced chunk aggregation with semantic coherence preservation.
        """
        if not chunk_results:
            return []

        # Group by memory ID
        memory_groups = {}
        for result in chunk_results:
            memory_id = result.get("memory_id")
            if memory_id is None:
                continue

            if memory_id not in memory_groups:
                memory_groups[memory_id] = []

            memory_groups[memory_id].append(result)

        # Calculate memory-level scores with improved logic
        memory_results = []
        for memory_id, chunks in memory_groups.items():
            # Sort chunks by similarity (descending)
            sorted_chunks = sorted(
                chunks, key=lambda x: x.get("chunk_similarity", 0.0), reverse=True
            )

            # Limit to max_chunks_per_memory
            top_chunks = sorted_chunks[: self.max_chunks_per_memory]

            # Identify sequential chunks for stronger semantic coherence
            chunk_indices = [chunk.get("chunk_index", 0) for chunk in top_chunks]
            has_sequential_chunks = self._are_chunks_sequential(chunk_indices)

            # Calculate combined score with emphasis on sequential chunks
            if self.combine_chunk_scores:
                # Use a weighted sum with decay and sequential bonus
                weights = [self.chunk_weight_decay**i for i in range(len(top_chunks))]
                total_weight = sum(weights)

                # Basic weighted score
                basic_score = (
                    sum(
                        chunk.get("chunk_similarity", 0.0) * weight
                        for chunk, weight in zip(top_chunks, weights)
                    )
                    / total_weight
                )

                # Apply sequential bonus if applicable (up to 30% boost)
                sequential_bonus = 0.3 if has_sequential_chunks else 0.0
                combined_score = min(1.0, basic_score * (1.0 + sequential_bonus))
            else:
                # Use the best chunk score
                combined_score = top_chunks[0].get("chunk_similarity", 0.0)

            # Create the memory result
            best_chunk = top_chunks[0]
            memory_result = {
                "memory_id": memory_id,
                "similarity_score": combined_score,
                "chunk_count": len(chunks),
                "top_chunks": top_chunks,
                "best_chunk_index": best_chunk.get("chunk_index", 0),
                "has_sequential_chunks": has_sequential_chunks,
            }

            # Add metadata from the best chunk
            if "metadata" in best_chunk:
                for key, value in best_chunk["metadata"].items():
                    if key not in ("memory_id", "chunk_index", "chunk_text"):
                        memory_result[key] = value

            # Build content with improved semantic coherence
            if has_sequential_chunks and len(top_chunks) > 1:
                # Get the sequential chunks in order
                sequential_chunks = self._get_sequential_chunks(top_chunks)
                combined_content = " ".join(chunk.get("content", "") for chunk in sequential_chunks)
                memory_result["content"] = combined_content
            else:
                memory_result["content"] = best_chunk.get("content", "")

            memory_results.append(memory_result)

        # Sort by similarity score
        memory_results.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)

        return memory_results

    def _enhance_with_chunk_context(
        self, memory_results: list[dict[str, Any]], chunk_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Enhance memory results with chunk context information.

        This method adds information about chunk positions and relationships
        to the result objects.

        Args:
            memory_results: List of memory-level results
            chunk_results: List of chunk-level results

        Returns:
            Enhanced results
        """
        if not memory_results or not chunk_results:
            return memory_results

        # Build a map of memory ID to top chunks
        memory_to_chunks = {}
        for result in chunk_results:
            memory_id = result.get("memory_id")
            if memory_id is None:
                continue

            if memory_id not in memory_to_chunks:
                memory_to_chunks[memory_id] = []

            memory_to_chunks[memory_id].append(result)

        # Enhance each memory result
        for result in memory_results:
            memory_id = result.get("memory_id")
            if memory_id is None or memory_id not in memory_to_chunks:
                continue

            chunks = memory_to_chunks[memory_id]

            # Sort chunks by similarity
            chunks.sort(key=lambda x: x.get("chunk_similarity", 0.0), reverse=True)

            # Add chunk information
            result["chunk_indices"] = [chunk.get("chunk_index", 0) for chunk in chunks]
            result["chunk_similarities"] = [chunk.get("chunk_similarity", 0.0) for chunk in chunks]

            # Check if result contains sequential chunks
            if self._are_chunks_sequential(result["chunk_indices"]):
                result["has_sequential_chunks"] = True

                # If the content isn't already a combined sequential chunk content,
                # and we have sequential chunks, combine them
                if "top_chunks" in result and len(result["top_chunks"]) > 1:
                    sequential_chunks = self._get_sequential_chunks(result["top_chunks"])
                    if sequential_chunks and self.prioritize_coherent_chunks:
                        combined_content = " ".join(
                            chunk.get("content", "") for chunk in sequential_chunks
                        )
                        result["content"] = combined_content
            else:
                result["has_sequential_chunks"] = False

        return memory_results

    def _are_chunks_sequential(self, chunk_indices: list[int]) -> bool:
        """
        Check if chunks are sequential.

        Args:
            chunk_indices: List of chunk indices

        Returns:
            True if chunks are sequential
        """
        if not chunk_indices or len(chunk_indices) < 2:
            return True

        # Sort indices
        sorted_indices = sorted(chunk_indices)

        # Check if they form a sequence
        for i in range(1, len(sorted_indices)):
            if sorted_indices[i] != sorted_indices[i - 1] + 1:
                return False

        return True

    def _get_sequential_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Get the longest sequence of sequential chunks.

        Args:
            chunks: List of chunk results

        Returns:
            List of sequential chunks
        """
        if not chunks or len(chunks) < 2:
            return chunks

        # Sort by chunk index
        sorted_chunks = sorted(chunks, key=lambda x: x.get("chunk_index", 0))

        # Find the longest sequence
        current_sequence = [sorted_chunks[0]]
        longest_sequence = current_sequence

        for i in range(1, len(sorted_chunks)):
            current_index = sorted_chunks[i].get("chunk_index", 0)
            prev_index = sorted_chunks[i - 1].get("chunk_index", 0)

            if current_index == prev_index + 1:
                # Continue the sequence
                current_sequence.append(sorted_chunks[i])
            else:
                # Start a new sequence
                if len(current_sequence) > len(longest_sequence):
                    longest_sequence = current_sequence

                current_sequence = [sorted_chunks[i]]

        # Check the last sequence
        if len(current_sequence) > len(longest_sequence):
            longest_sequence = current_sequence

        return longest_sequence
