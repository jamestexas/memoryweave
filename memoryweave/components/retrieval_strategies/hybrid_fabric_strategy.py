"""
Hybrid Fabric Retrieval Strategy for MemoryWeave.

This module implements a memory-efficient retrieval strategy that combines
full embeddings, selective chunks, and keyword filtering for optimal
retrieval performance with minimal memory usage.
"""

import logging
import time
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
from memoryweave.storage.refactored import HybridMemoryStore

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(markup=True),  # allow colors in terminal
    ],
)


class HybridFabricStrategy(ContextualFabricStrategy):
    """
    A memory-efficient retrieval strategy combining multiple approaches.

    This strategy enhances the contextual fabric approach by:
    1. Using both full embeddings and selective chunks
    2. Leveraging keyword filtering for more efficient retrieval
    3. Prioritizing relevant memory access patterns
    4. Optimizing memory usage throughout retrieval
    """

    def __init__(
        self,
        memory_store: Optional[Any] = None,
        associative_linker: Optional[AssociativeMemoryLinker] = None,
        temporal_context: Optional[TemporalContextBuilder] = None,
        activation_manager: Optional[ActivationManager] = None,
    ):
        """
        Initialize the hybrid fabric strategy.

        Args:
            memory_store: Memory store or adapter with hybrid capabilities
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
        self.component_id = ComponentName.CONTEXTUAL_FABRIC_STRATEGY

        # Hybrid-specific parameters
        self.use_keyword_filtering = True
        self.keyword_boost_factor = 0.3
        self.max_chunks_per_memory = 3
        self.prioritize_full_embeddings = True

        # Detect if memory store supports hybrid features
        self.supports_hybrid = False
        # Will be set in initialize()

    def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the strategy with configuration.

        Args:
            config: Configuration dictionary with parameters
        """
        # First, initialize base class
        super().initialize(config)

        # Then, set hybrid specific parameters
        self.use_keyword_filtering = config.get("use_keyword_filtering", self.use_keyword_filtering)
        self.keyword_boost_factor = config.get("keyword_boost_factor", self.keyword_boost_factor)
        self.max_chunks_per_memory = config.get("max_chunks_per_memory", self.max_chunks_per_memory)
        self.prioritize_full_embeddings = config.get(
            "prioritize_full_embeddings", self.prioritize_full_embeddings
        )

        # Check if memory_store supports hybrid features
        if self.memory_store is not None:
            # Check for hybrid memory support via different methods
            if hasattr(self.memory_store, "search_hybrid"):
                self.supports_hybrid = True
                if self.debug:
                    self.logger.debug("Hybrid search support detected in memory store")
            elif hasattr(self.memory_store, "memory_store") and hasattr(
                self.memory_store.memory_store, "search_hybrid"
            ):
                self.supports_hybrid = True
                if self.debug:
                    self.logger.debug("Hybrid search support detected in nested memory store")
            elif hasattr(self.memory_store, "search_chunks"):
                self.supports_hybrid = True
                if self.debug:
                    self.logger.debug("Chunk search support detected - will use for hybrid search")

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories using the hybrid fabric strategy.

        This method efficiently combines multiple retrieval approaches,
        optimizing for both relevance and memory usage.

        Args:
            query_embedding: Query embedding for similarity matching
            top_k: Number of results to return
            context: Context containing query, memory, etc.

        Returns:
            list of retrieved memory dicts with relevance scores
        """
        # If hybrid is not supported, fall back to base implementation
        if not self.supports_hybrid:
            if self.debug:
                self.logger.debug("Hybrid search not supported, using base implementation")
            return super().retrieve(query_embedding, top_k, context)

        # Get memory store from context or instance
        memory_store = context.get("memory_store", self.memory_store)

        # Get query from context
        query = context.get("query", "")

        # Get current time from context or use current time
        context.get("current_time", time.time())

        # Apply parameter adaptation if available
        adapted_params = context.get("adapted_retrieval_params", {})
        confidence_threshold = adapted_params.get("confidence_threshold", self.confidence_threshold)
        self._apply_adapted_params(adapted_params)

        # Extract keywords for filtering if enabled
        keywords = None
        if self.use_keyword_filtering:
            # Get keywords from context if available
            keywords = context.get("important_keywords", None)
            if keywords is None and "query_analyzer" in context:
                # Extract using query analyzer if available
                analyzer = context["query_analyzer"]
                if hasattr(analyzer, "extract_keywords"):
                    keywords = analyzer.extract_keywords(query)

            # Fallback: simple keyword extraction
            if keywords is None:
                keywords = self._extract_simple_keywords(query)

        # Log retrieval details if debug enabled
        if self.debug:
            self.logger.debug(f"HybridFabricStrategy: Retrieving for query: '{query}'")
            self.logger.debug(
                f"HybridFabricStrategy: Using confidence threshold: {confidence_threshold}"
            )
            if keywords:
                self.logger.debug(f"HybridFabricStrategy: Using keywords: {keywords}")

        # Step 1: Get hybrid search results (combines full and chunk embeddings with keywords)
        if hasattr(memory_store, "search_hybrid"):
            # Direct hybrid search if available
            memory_results = memory_store.search_hybrid(
                query_embedding, self.max_candidates, confidence_threshold, keywords
            )
        else:
            # Fallback: Combine full and chunk search manually
            memory_results = self._combined_search(
                query_embedding=query_embedding,
                max_results=self.max_candidates,
                threshold=confidence_threshold,
                keywords=keywords,
                memory_store=memory_store,
            )

        # Step 2: Get associative matches (if linker available)
        associative_results = self._retrieve_associative_results(memory_results)

        # Step 3: Get temporal context (if available)
        temporal_results = self._retrieve_temporal_results(query, context, memory_store)

        # Step 4: Get activation scores (if available)
        activation_results = {}
        if self.activation_manager is not None:
            activations = self.activation_manager.get_activated_memories(threshold=0.1)
            activation_results = dict(activations)

        # Step 5: Combine all sources
        combined_results = self._combine_results(
            similarity_results=memory_results,
            associative_results=associative_results,
            temporal_results=temporal_results,
            activation_results=activation_results,
            memory_store=memory_store,
        )

        # Step 6: Apply threshold and sort
        filtered_results = [
            r for r in combined_results if r["relevance_score"] >= confidence_threshold
        ]
        if len(filtered_results) < self.min_results:
            filtered_results = combined_results[: self.min_results]

        top_k = min(top_k, len(filtered_results))
        results = filtered_results[:top_k]

        # Debug logging
        if self.debug:
            self.logger.debug(f"HybridFabricStrategy: Retrieved {len(results)} results")
            if results:
                self.logger.debug(
                    f"HybridFabricStrategy: Top 3 scores: {[r['relevance_score'] for r in results[:3]]}"
                )

        return results

    def _combined_search(
        self,
        query_embedding: np.ndarray,
        max_results: int,
        threshold: Optional[float],
        keywords: Optional[list[str]],
        memory_store: HybridMemoryStore,
    ) -> list[dict[str, Any]]:
        """
        Combine full embedding search and chunk search with keyword filtering.

        This fallback method is used when direct hybrid search is not available.

        Args:
            query_embedding: Query embedding
            max_results: Maximum number of results to return
            threshold: Minimum similarity threshold
            keywords: Optional list of keywords for filtering
            memory_store: Memory store to search

        Returns:
            Combined search results
        """
        # First get results from full embeddings
        if hasattr(memory_store, "search_by_vector"):
            full_results = memory_store.search_by_vector(
                query_vector=query_embedding,
                limit=max_results,
                threshold=threshold,
            )
        else:
            # Fallback to similarity search
            full_results = self._retrieve_by_similarity(query_embedding, max_results, memory_store)

        # Then get results from chunks if supported
        chunk_results = []
        if hasattr(memory_store, "search_chunks"):
            chunk_results = memory_store.search_chunks(query_embedding, max_results, threshold)

        # Convert chunk results to match full result format
        formatted_chunk_results = []
        for result in chunk_results:
            formatted_result = {
                "memory_id": result.get("memory_id"),
                "content": result.get("content", ""),
                "relevance_score": result.get("chunk_similarity", 0.5),
                "is_hybrid": True,
                "chunk_index": result.get("chunk_index", 0),
            }

            # Add metadata
            if "metadata" in result:
                for key, value in result["metadata"].items():
                    if key not in ("memory_id", "chunk_index"):
                        formatted_result[key] = value

            formatted_chunk_results.append(formatted_result)

        # Combine the results
        combined_results = []
        seen_memory_ids = set()

        # Process full results first (prioritize them)
        for result in full_results:
            memory_id = result.get("memory_id")
            if memory_id in seen_memory_ids:
                continue

            seen_memory_ids.add(memory_id)

            # Apply keyword boosting if keywords are provided
            if keywords and len(keywords) > 0:
                content = str(result.get("content", "")).lower()
                keyword_matches = sum(1 for kw in keywords if kw.lower() in content)
                keyword_boost = min(self.keyword_boost_factor, keyword_matches * 0.05)
                result["relevance_score"] = min(
                    1.0, result.get("relevance_score", 0.5) + keyword_boost
                )
                result["keyword_matches"] = keyword_matches

            combined_results.append(result)

        # Then add unique chunk results
        for result in formatted_chunk_results:
            memory_id = result.get("memory_id")
            if memory_id in seen_memory_ids:
                continue

            seen_memory_ids.add(memory_id)

            # Apply keyword boosting if keywords are provided
            if keywords and len(keywords) > 0:
                content = str(result.get("content", "")).lower()
                keyword_matches = sum(1 for kw in keywords if kw.lower() in content)
                keyword_boost = min(self.keyword_boost_factor, keyword_matches * 0.05)
                result["relevance_score"] = min(
                    1.0, result.get("relevance_score", 0.5) + keyword_boost
                )
                result["keyword_matches"] = keyword_matches

            combined_results.append(result)

        # Sort by relevance score
        combined_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # Return top results
        return combined_results[:max_results]

    def _extract_simple_keywords(self, text: str) -> list[str]:
        """
        Extract simple keywords from text for filtering.

        This lightweight method extracts potential keywords without
        requiring a full NLP pipeline, optimizing for memory efficiency.

        Args:
            text: The text to extract keywords from

        Returns:
            list of potential keywords
        """
        # Simple stopwords list
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "because",
            "as",
            "what",
            "when",
            "where",
            "how",
            "who",
            "which",
            "this",
            "that",
            "these",
            "those",
            "then",
            "just",
            "so",
            "than",
            "such",
            "both",
            "through",
            "about",
            "for",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "can",
            "could",
            "will",
            "would",
            "shall",
            "should",
            "may",
            "might",
            "must",
            "to",
            "in",
            "on",
            "at",
            "by",
            "with",
            "from",
        }

        # Tokenize the text
        tokens = text.lower().split()

        # Filter out stopwords and short words
        potential_keywords = [
            token
            for token in tokens
            if token not in stopwords and len(token) > 3 and token.isalpha()
        ]

        # Limit to top keywords by length (longer words are often more specific)
        potential_keywords.sort(key=len, reverse=True)

        return potential_keywords[:10]  # Limit to 10 keywords

    def _combine_results(
        self,
        similarity_results: list[dict[str, Any]],
        associative_results: dict[str, float],
        temporal_results: dict[str, float],
        activation_results: dict[str, float],
        memory_store: Optional[Any],
    ) -> list[dict[str, Any]]:
        """
        Combine results from different sources with memory-efficient processing.

        This method overrides the base implementation to optimize memory usage
        during result combination and scoring.

        Args:
            similarity_results: Results from direct similarity
            associative_results: Results from associative traversal
            temporal_results: Results from temporal context
            activation_results: Results from activation levels
            memory_store: Memory store for metadata

        Returns:
            Combined and sorted results
        """
        # Create a combined results dictionary with memory-efficient approach
        # Reuse the similarity results as our starting point
        combined_dict = {result["memory_id"]: result for result in similarity_results}

        # Set defaults for missing fields
        for memory_id, result in combined_dict.items():
            result.setdefault("associative_score", 0.0)
            result.setdefault("temporal_score", 0.0)
            result.setdefault("activation_score", 0.0)

            # Ensure relevance_score exists
            if "relevance_score" not in result and "similarity_score" in result:
                result["relevance_score"] = result["similarity_score"]
            elif "relevance_score" not in result:
                result["relevance_score"] = 0.0

        # Process the minimum viable set of memories for each additional source
        memory_limit = 1000  # Set a reasonable limit to avoid excessive memory use

        # Add associative results efficiently
        processed_count = 0
        for memory_id, score in associative_results.items():
            if processed_count >= memory_limit:
                break

            if memory_id in combined_dict:
                # Update existing result
                combined_dict[memory_id]["associative_score"] = score
            else:
                # Only add new memories if they have a significant score
                if score > 0.3 and memory_store is not None:
                    try:
                        # Get memory information efficiently
                        memory = memory_store.get(memory_id)

                        # Create minimal result object
                        result = {
                            "memory_id": memory_id,
                            "similarity_score": 0.0,
                            "associative_score": score,
                            "temporal_score": 0.0,
                            "activation_score": 0.0,
                            "relevance_score": 0.0,  # Will be calculated later
                        }

                        # Add content efficiently
                        if hasattr(memory, "content"):
                            if isinstance(memory.content, dict) and "text" in memory.content:
                                result["content"] = memory.content["text"]
                            else:
                                result["content"] = str(memory.content)

                        # Add selective metadata
                        if hasattr(memory, "metadata") and memory.metadata:
                            # Only copy essential metadata fields
                            for key in ["type", "created_at", "importance"]:
                                if key in memory.metadata:
                                    result[key] = memory.metadata[key]

                        combined_dict[memory_id] = result
                    except Exception as e:
                        if self.debug:
                            self.logger.debug(
                                f"Error processing associative memory {memory_id}: {e}"
                            )

            processed_count += 1

        # Add temporal results efficiently
        processed_count = 0
        for memory_id, score in temporal_results.items():
            if processed_count >= memory_limit:
                break

            if memory_id in combined_dict:
                # Update existing memory
                combined_dict[memory_id]["temporal_score"] = score
            else:
                # Only add if score is significant
                if score > 0.5 and memory_store is not None:
                    try:
                        # Add with minimal processing
                        memory = memory_store.get(memory_id)

                        # Create minimal result object
                        result = {
                            "memory_id": memory_id,
                            "similarity_score": 0.0,
                            "associative_score": 0.0,
                            "temporal_score": score,
                            "activation_score": 0.0,
                            "relevance_score": 0.0,  # Will be calculated later
                        }

                        # Add content efficiently
                        if hasattr(memory, "content"):
                            if isinstance(memory.content, dict) and "text" in memory.content:
                                result["content"] = memory.content["text"]
                            else:
                                result["content"] = str(memory.content)

                        # Add selective metadata
                        if hasattr(memory, "metadata") and memory.metadata:
                            # Only copy essential metadata fields
                            for key in ["type", "created_at", "importance"]:
                                if key in memory.metadata:
                                    result[key] = memory.metadata[key]

                        combined_dict[memory_id] = result
                    except Exception as e:
                        if self.debug:
                            self.logger.debug(f"Error processing temporal memory {memory_id}: {e}")

            processed_count += 1

        # Add activation results efficiently
        processed_count = 0
        for memory_id, score in activation_results.items():
            if processed_count >= memory_limit:
                break

            if memory_id in combined_dict:
                # Update existing memory
                combined_dict[memory_id]["activation_score"] = score

            processed_count += 1

        # Calculate final scores efficiently
        for memory_id, result in combined_dict.items():
            # Extract scores
            similarity = result.get("similarity_score", 0.0)
            associative = result.get("associative_score", 0.0)
            temporal = result.get("temporal_score", 0.0)
            activation = result.get("activation_score", 0.0)

            # Calculate weighted scores
            similarity_contribution = similarity * self.similarity_weight
            associative_contribution = associative * self.associative_weight
            temporal_contribution = temporal * self.temporal_weight
            activation_contribution = activation * self.activation_weight

            # For temporal queries, don't downweight temporal contributions even if similarity is low
            has_temporal_component = temporal > 0.5

            # If similarity is low and this isn't a strong temporal match, reduce other contributions
            if similarity < 0.3 and not has_temporal_component:
                scaling_factor = max(0.1, similarity / 0.3)
                associative_contribution *= scaling_factor
                activation_contribution *= scaling_factor
                # Note: We don't scale temporal_contribution here

            # Calculate combined score
            combined_score = (
                similarity_contribution
                + associative_contribution
                + temporal_contribution
                + activation_contribution
            )

            # Store combined score and contributions
            result["relevance_score"] = combined_score
            result["similarity_contribution"] = similarity_contribution
            result["associative_contribution"] = associative_contribution
            result["temporal_contribution"] = temporal_contribution
            result["activation_contribution"] = activation_contribution

            # Flag if below threshold
            result["below_threshold"] = combined_score < self.confidence_threshold

        # Convert to list and sort by relevance score
        combined_results = list(combined_dict.values())

        # Use efficient sorting rather than creating a new list
        combined_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return combined_results
