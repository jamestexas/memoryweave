# memoryweave/components/retrieval_strategies/contextual_fabric_strategy.py
"""
Contextual Fabric Retrieval Strategy for MemoryWeave.

This module implements a retrieval strategy that leverages the contextual fabric structure
of MemoryWeave, combining direct similarity, associative links, and temporal context
to provide more relevant and contextually-aware memory retrieval.
"""

import logging
from typing import Any, Optional

import numpy as np

from memoryweave.components.activation import ActivationManager
from memoryweave.components.associative_linking import AssociativeMemoryLinker
from memoryweave.components.base import RetrievalStrategy
from memoryweave.components.component_names import ComponentName
from memoryweave.components.temporal_context import TemporalContextBuilder
from memoryweave.interfaces.memory import IMemoryStore, MemoryID


class ContextualFabricStrategy(RetrievalStrategy):
    """
    A retrieval strategy that leverages the contextual fabric structure.

    This strategy combines:
    1. Direct similarity retrieval based on embeddings
    2. Associative link traversal for related memories
    3. Temporal context for time-based relevance
    4. Activation patterns for memory accessibility

    The result is a more contextually-aware retrieval that mimics human
    memory access patterns.
    """

    def __init__(
        self,
        memory_store: Optional[IMemoryStore] = None,
        associative_linker: Optional[AssociativeMemoryLinker] = None,
        temporal_context: Optional[TemporalContextBuilder] = None,
        activation_manager: Optional[ActivationManager] = None,
    ):
        """
        Initialize the contextual fabric strategy.

        Args:
            memory_store: Memory store to retrieve from
            associative_linker: Associative memory linker for traversing links
            temporal_context: Temporal context builder for time-based relevance
            activation_manager: Activation manager for memory accessibility
        """
        self.memory_store = memory_store
        self.associative_linker = associative_linker
        self.temporal_context = temporal_context
        self.activation_manager = activation_manager
        self.component_id = ComponentName.CONTEXTUAL_FABRIC_STRATEGY

        # Retrieval parameters
        self.confidence_threshold = 0.1
        self.similarity_weight = 0.5
        self.associative_weight = 0.3
        self.temporal_weight = 0.1
        self.activation_weight = 0.1
        self.max_associative_hops = 2
        self.activation_boost_factor = 1.5
        self.min_results = 5
        self.max_candidates = 50

        # Debug parameters
        self.debug = False
        self.logger = logging.getLogger(__name__)

    def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the strategy with configuration.

        Args:
            config: Configuration dictionary with parameters:
                - confidence_threshold: Minimum confidence for results (default: 0.1)
                - similarity_weight: Weight for direct similarity (default: 0.5)
                - associative_weight: Weight for associative links (default: 0.3)
                - temporal_weight: Weight for temporal relevance (default: 0.1)
                - activation_weight: Weight for activation level (default: 0.1)
                - max_associative_hops: Maximum hops for associative traversal (default: 2)
                - activation_boost_factor: Boost factor for active memories (default: 1.5)
                - min_results: Minimum number of results to return (default: 5)
                - max_candidates: Maximum number of candidate memories to consider (default: 50)
                - debug: Enable debug logging (default: False)
        """
        self.confidence_threshold = config.get("confidence_threshold", 0.1)
        self.similarity_weight = config.get("similarity_weight", 0.5)
        self.associative_weight = config.get("associative_weight", 0.3)
        self.temporal_weight = config.get("temporal_weight", 0.1)
        self.activation_weight = config.get("activation_weight", 0.1)
        self.max_associative_hops = config.get("max_associative_hops", 2)
        self.activation_boost_factor = config.get("activation_boost_factor", 1.5)
        self.min_results = config.get("min_results", 5)
        self.max_candidates = config.get("max_candidates", 50)
        self.debug = config.get("debug", False)

        # Set components if provided in config
        if "memory_store" in config:
            self.memory_store = config["memory_store"]

        if "associative_linker" in config:
            self.associative_linker = config["associative_linker"]

        if "temporal_context" in config:
            self.temporal_context = config["temporal_context"]

        if "activation_manager" in config:
            self.activation_manager = config["activation_manager"]

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories using the contextual fabric strategy.

        Args:
            query_embedding: Query embedding for similarity matching
            top_k: Number of results to return
            context: Context containing query, memory, etc.

        Returns:
            list of retrieved memory dicts with relevance scores
        """
        # Get memory store from context or instance
        memory_store = context.get("memory_store", self.memory_store)

        # Get query from context
        query = context.get("query", "")

        # Apply parameter adaptation if available from DynamicContextAdapter or QueryAdapter
        adapted_params = context.get("adapted_retrieval_params", {})

        # Set parameters from adaptation
        confidence_threshold = adapted_params.get("confidence_threshold", self.confidence_threshold)

        # Apply additional parameter adaptations if provided
        if "similarity_weight" in adapted_params:
            self.similarity_weight = adapted_params["similarity_weight"]
        if "associative_weight" in adapted_params:
            self.associative_weight = adapted_params["associative_weight"]
        if "temporal_weight" in adapted_params:
            self.temporal_weight = adapted_params["temporal_weight"]
        if "activation_weight" in adapted_params:
            self.activation_weight = adapted_params["activation_weight"]
        if "max_associative_hops" in adapted_params:
            self.max_associative_hops = adapted_params["max_associative_hops"]

        # Apply progressive filtering for large memory stores
        use_progressive_filtering = adapted_params.get("use_progressive_filtering", False)
        use_batched_computation = adapted_params.get("use_batched_computation", False)
        batch_size = adapted_params.get("batch_size", 200)

        # Log retrieval details if debug enabled
        if self.debug:
            self.logger.debug(f"ContextualFabricStrategy: Retrieving for query: '{query}'")
            self.logger.debug(
                f"ContextualFabricStrategy: Using confidence threshold: {confidence_threshold}"
            )

        # Step 1: Get direct similarity matches
        similarity_results = self._retrieve_by_similarity(
            query_embedding=query_embedding,
            max_results=self.max_candidates,
            memory_store=memory_store,
            use_progressive_filtering=use_progressive_filtering,
            use_batched_computation=use_batched_computation,
            batch_size=batch_size,
        )

        # Step 2: Get associative matches (if linker available)
        associative_results = {}
        if self.associative_linker is not None:
            # Get top similarity matches as starting points
            top_similarity_ids = [
                r["memory_id"] for r in similarity_results[: min(5, len(similarity_results))]
            ]

            # Traverse associative network from each starting point
            for memory_id in top_similarity_ids:
                activations = self.associative_linker.traverse_associative_network(
                    start_id=memory_id, max_hops=self.max_associative_hops, min_strength=0.1
                )

                # Add to results (taking maximum activation if memory appears multiple times)
                for assoc_id, strength in activations.items():
                    if assoc_id in associative_results:
                        associative_results[assoc_id] = max(associative_results[assoc_id], strength)
                    else:
                        associative_results[assoc_id] = strength

        # Step 3: Get temporal context (if available)
        temporal_results = {}
        if self.temporal_context is not None:
            # Extract time references from query
            time_info = self.temporal_context.extract_time_references(query)

            # If query has temporal references, get memories from relevant time
            if time_info["has_temporal_reference"] and time_info["relative_time"]:
                target_time = time_info["relative_time"]

                # Get all memories (inefficient but ensures we don't miss anything)
                if memory_store:
                    all_memories = memory_store.get_all()

                    for memory in all_memories:
                        creation_time = memory.metadata.get("created_at", 0)
                        if creation_time > 0:
                            # Calculate temporal proximity
                            time_diff = abs(target_time - creation_time)
                            # Use a Gaussian decay function (1-day scale by default)
                            temporal_scale = 86400
                            temporal_relevance = np.exp(-(time_diff**2) / (2 * temporal_scale**2))

                            if temporal_relevance > 0.2:  # Only keep reasonably close matches
                                temporal_results[memory.id] = temporal_relevance

        # Step 4: Get activation scores (if available)
        activation_results = {}
        if self.activation_manager is not None:
            # Get all activated memories
            activations = self.activation_manager.get_activated_memories(threshold=0.1)
            activation_results = dict(activations)

        # Step 5: Combine all sources
        combined_results = self._combine_results(
            similarity_results=similarity_results,
            associative_results=associative_results,
            temporal_results=temporal_results,
            activation_results=activation_results,
            memory_store=memory_store,
        )

        # Step 6: Apply threshold and sort
        filtered_results = [
            r for r in combined_results if r["relevance_score"] >= confidence_threshold
        ]

        # Apply minimum results guarantee
        if len(filtered_results) < self.min_results:
            # Use top min_results from combined results
            filtered_results = combined_results[: self.min_results]

        # Limit to top_k
        top_k = min(top_k, len(filtered_results))
        results = filtered_results[:top_k]

        # Debug logging
        if self.debug:
            self.logger.debug(f"ContextualFabricStrategy: Retrieved {len(results)} results")
            self.logger.debug(
                f"ContextualFabricStrategy: Top 3 scores: {[r['relevance_score'] for r in results[:3]]}"
            )

            # Log contribution breakdown for top result
            if results:
                top = results[0]
                self.logger.debug("Top result contributions:")
                self.logger.debug(
                    f"- Similarity: {top.get('similarity_score', 0):.3f} * {self.similarity_weight:.1f} = {top.get('similarity_contribution', 0):.3f}"
                )
                self.logger.debug(
                    f"- Associative: {top.get('associative_score', 0):.3f} * {self.associative_weight:.1f} = {top.get('associative_contribution', 0):.3f}"
                )
                self.logger.debug(
                    f"- Temporal: {top.get('temporal_score', 0):.3f} * {self.temporal_weight:.1f} = {top.get('temporal_contribution', 0):.3f}"
                )
                self.logger.debug(
                    f"- Activation: {top.get('activation_score', 0):.3f} * {self.activation_weight:.1f} = {top.get('activation_contribution', 0):.3f}"
                )
                self.logger.debug(f"- Total: {top['relevance_score']:.3f}")

        return results

    def _retrieve_by_similarity(
        self,
        query_embedding: np.ndarray,
        max_results: int,
        memory_store: Optional[IMemoryStore],
        use_progressive_filtering: bool = False,
        use_batched_computation: bool = False,
        batch_size: int = 200,
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories by direct similarity.

        Args:
            query_embedding: Query embedding
            max_results: Maximum number of results to return
            memory_store: Memory store to retrieve from
            use_progressive_filtering: Whether to use progressive filtering
            use_batched_computation: Whether to use batched computation
            batch_size: Size of batches for computation

        Returns:
            list of retrieved memory dicts with similarity scores
        """
        if memory_store is None or not hasattr(memory_store, "memory_embeddings"):
            return []

        memory_size = len(memory_store.memory_embeddings)

        # For large memory stores, use optimized computation
        if memory_size > 500 and (use_progressive_filtering or use_batched_computation):
            return self._optimized_similarity_retrieval(
                query_embedding,
                max_results,
                memory_store,
                use_progressive_filtering,
                use_batched_computation,
                batch_size,
            )

        # Standard computation for smaller stores
        # Compute cosine similarities
        similarities = np.dot(memory_store.memory_embeddings, query_embedding)

        # Apply normalization to prevent score compression and improve discrimination
        # between results
        if len(similarities) > 1:
            # Get statistics for normalization
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)

            # Apply z-score normalization if standard deviation is meaningful
            if std_sim > 1e-5:  # Avoid division by near-zero
                normalized_similarities = (similarities - mean_sim) / std_sim
            else:
                # Fallback to min-max scaling if std is too small
                min_sim = np.min(similarities)
                max_sim = np.max(similarities)
                sim_range = max_sim - min_sim
                if sim_range > 1e-5:
                    normalized_similarities = (similarities - min_sim) / sim_range
                else:
                    normalized_similarities = similarities

            # Apply a non-linear transformation to stretch differences
            # This improves discrimination between close scores
            normalized_similarities = (
                np.sign(normalized_similarities) * np.abs(normalized_similarities) ** 0.5
            )
        else:
            # Just one memory, no normalization needed
            normalized_similarities = similarities

        # Get top indices based on normalized similarities
        top_indices = np.argsort(-normalized_similarities)[:max_results]

        # Format results
        results = []

        for idx in top_indices:
            raw_similarity = float(similarities[idx])
            normalized_score = float(normalized_similarities[idx])

            # Use a more meaningful threshold based on normalized score
            # Z-score > -1.0 means "not unusually dissimilar"
            if normalized_score > -1.0 or raw_similarity > 0.5:
                # Add to results
                results.append({
                    "memory_id": int(idx),
                    "similarity_score": raw_similarity,
                    "normalized_score": normalized_score,
                    **memory_store.memory_metadata[idx],
                })

        return results

    def _optimized_similarity_retrieval(
        self,
        query_embedding: np.ndarray,
        max_results: int,
        memory_store: IMemoryStore,
        use_progressive_filtering: bool,
        use_batched_computation: bool,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        """
        Optimized similarity retrieval for large memory stores.

        Uses two key optimizations:
        1. Progressive filtering: First get a rough set of candidates, then refine
        2. Batched computation: Process large embedding matrices in batches

        Args:
            query_embedding: Query embedding
            max_results: Maximum number of results to return
            memory_store: Memory store to retrieve from
            use_progressive_filtering: Whether to use progressive filtering
            use_batched_computation: Whether to use batched computation
            batch_size: Size of batches for computation

        Returns:
            list of retrieved memory dicts with similarity scores
        """
        memory_embeddings = memory_store.memory_embeddings
        memory_size = len(memory_embeddings)

        # Progressive filtering approach
        if use_progressive_filtering:
            # First pass: get a larger set of candidates using a fast approximation
            # Sample a subset of dimensions to compute a rough similarity
            embedding_dim = query_embedding.shape[0]
            sample_size = min(100, embedding_dim)
            sample_indices = np.random.choice(embedding_dim, sample_size, replace=False)

            # Use sampled dimensions for rough similarity
            sampled_query = query_embedding[sample_indices]
            sampled_memory = memory_embeddings[:, sample_indices]

            # Fast similarity computation on reduced dimensions
            rough_similarities = np.dot(sampled_memory, sampled_query)

            # Get top candidates (3-5x more than needed)
            candidate_count = min(memory_size, max(max_results * 3, 200))
            candidate_indices = np.argsort(-rough_similarities)[:candidate_count]

            # Second pass: compute exact similarity only for the candidates
            candidate_embeddings = memory_embeddings[candidate_indices]
            similarities = np.dot(candidate_embeddings, query_embedding)

            # Convert back to original indices
            top_indices = candidate_indices[np.argsort(-similarities)[:max_results]]

            # Get corresponding similarities
            filtered_similarities = similarities[np.argsort(-similarities)[:max_results]]

        # Batched computation approach
        elif use_batched_computation:
            # Process in batches to reduce memory pressure
            if memory_size <= batch_size:
                # Small enough to compute directly
                similarities = np.dot(memory_embeddings, query_embedding)
            else:
                # Initialize similarity array
                similarities = np.zeros(memory_size)

                # Process in batches
                for i in range(0, memory_size, batch_size):
                    end_idx = min(i + batch_size, memory_size)
                    batch_embeddings = memory_embeddings[i:end_idx]
                    batch_similarities = np.dot(batch_embeddings, query_embedding)
                    similarities[i:end_idx] = batch_similarities

            # Get top indices
            top_indices = np.argsort(-similarities)[:max_results]
            filtered_similarities = similarities[top_indices]

        else:
            # Fallback to standard computation
            similarities = np.dot(memory_embeddings, query_embedding)
            top_indices = np.argsort(-similarities)[:max_results]
            filtered_similarities = similarities[top_indices]

        # Normalize selected similarities
        if len(filtered_similarities) > 1:
            # Get statistics for normalization
            mean_sim = np.mean(filtered_similarities)
            std_sim = np.std(filtered_similarities)

            # Apply normalization if standard deviation is meaningful
            if std_sim > 1e-5:
                normalized_similarities = (filtered_similarities - mean_sim) / std_sim
            else:
                # Fallback to min-max scaling
                min_sim = np.min(filtered_similarities)
                max_sim = np.max(filtered_similarities)
                sim_range = max_sim - min_sim
                if sim_range > 1e-5:
                    normalized_similarities = (filtered_similarities - min_sim) / sim_range
                else:
                    normalized_similarities = filtered_similarities

            # Apply non-linear transformation to stretch differences
            normalized_similarities = (
                np.sign(normalized_similarities) * np.abs(normalized_similarities) ** 0.5
            )
        else:
            normalized_similarities = filtered_similarities

        # Format results
        results = []
        for i, idx in enumerate(top_indices):
            raw_similarity = float(filtered_similarities[i])
            normalized_score = float(normalized_similarities[i])

            # Apply filtering threshold
            if normalized_score > -1.0 or raw_similarity > 0.5:
                # Add to results
                results.append({
                    "memory_id": int(idx),
                    "similarity_score": raw_similarity,
                    "normalized_score": normalized_score,
                    **memory_store.memory_metadata[idx],
                })

        return results

    def _combine_results(
        self,
        similarity_results: list[dict[str, Any]],
        associative_results: dict[MemoryID, float],
        temporal_results: dict[MemoryID, float],
        activation_results: dict[MemoryID, float],
        memory_store: Optional[IMemoryStore],
    ) -> list[dict[str, Any]]:
        """
        Combine results from different sources.

        Args:
            similarity_results: Results from direct similarity
            associative_results: Results from associative traversal
            temporal_results: Results from temporal context
            activation_results: Results from activation levels
            memory_store: Memory store for metadata

        Returns:
            Combined and sorted results
        """
        # Create a combined results dictionary
        combined_dict = {}

        # Estimate memory store size for adaptive weighting
        memory_store_size = 0
        if memory_store is not None and hasattr(memory_store, "memory_embeddings"):
            memory_store_size = len(memory_store.memory_embeddings)

        # Adjust weights based on memory store size to prevent activation dominance
        # in large memory stores
        similarity_weight = self.similarity_weight
        associative_weight = self.associative_weight
        temporal_weight = self.temporal_weight
        activation_weight = self.activation_weight

        # Scale weights for larger memory stores
        if memory_store_size > 100:
            # Reduce activation weight as memory size increases
            # This prevents activation from dominating other factors in large stores
            scaling_factor = min(0.5, 100 / memory_store_size)
            activation_weight = self.activation_weight * scaling_factor

            # Increase semantic similarity weight to compensate
            similarity_weight = min(
                0.8, self.similarity_weight + (self.activation_weight - activation_weight) * 0.7
            )

            # Adjust other weights proportionally
            remaining_weight = 1.0 - (similarity_weight + activation_weight)
            proportion = self.associative_weight / (self.associative_weight + self.temporal_weight)
            associative_weight = remaining_weight * proportion
            temporal_weight = remaining_weight * (1 - proportion)

        # Add similarity results
        for result in similarity_results:
            memory_id = result["memory_id"]

            if memory_id not in combined_dict:
                combined_dict[memory_id] = {
                    "memory_id": memory_id,
                    "similarity_score": result["similarity_score"],
                    "normalized_score": result.get("normalized_score", result["similarity_score"]),
                    "associative_score": 0.0,
                    "temporal_score": 0.0,
                    "activation_score": 0.0,
                    **result,
                }
            else:
                combined_dict[memory_id]["similarity_score"] = result["similarity_score"]
                if "normalized_score" in result:
                    combined_dict[memory_id]["normalized_score"] = result["normalized_score"]

        # Add associative results
        for memory_id, score in associative_results.items():
            if memory_id not in combined_dict and memory_store is not None:
                # Add new memory
                try:
                    memory = memory_store.get(memory_id)
                    combined_dict[memory_id] = {
                        "memory_id": memory_id,
                        "similarity_score": 0.0,
                        "normalized_score": 0.0,
                        "associative_score": score,
                        "temporal_score": 0.0,
                        "activation_score": 0.0,
                    }

                    # Add metadata if available
                    if hasattr(memory, "metadata"):
                        combined_dict[memory_id].update(memory.metadata)

                except (KeyError, AttributeError):
                    # Skip if memory not found or no metadata
                    pass
            elif memory_id in combined_dict:
                # Update existing memory
                combined_dict[memory_id]["associative_score"] = score

        # Add temporal results
        for memory_id, score in temporal_results.items():
            if memory_id not in combined_dict and memory_store is not None:
                # Add new memory
                try:
                    memory = memory_store.get(memory_id)
                    combined_dict[memory_id] = {
                        "memory_id": memory_id,
                        "similarity_score": 0.0,
                        "normalized_score": 0.0,
                        "associative_score": 0.0,
                        "temporal_score": score,
                        "activation_score": 0.0,
                    }

                    # Add metadata if available
                    if hasattr(memory, "metadata"):
                        combined_dict[memory_id].update(memory.metadata)

                except (KeyError, AttributeError):
                    # Skip if memory not found or no metadata
                    pass
            elif memory_id in combined_dict:
                # Update existing memory
                combined_dict[memory_id]["temporal_score"] = score

        # Add activation results
        for memory_id, score in activation_results.items():
            if memory_id not in combined_dict and memory_store is not None:
                # Add new memory
                try:
                    memory = memory_store.get(memory_id)
                    combined_dict[memory_id] = {
                        "memory_id": memory_id,
                        "similarity_score": 0.0,
                        "normalized_score": 0.0,
                        "associative_score": 0.0,
                        "temporal_score": 0.0,
                        "activation_score": score,
                    }

                    # Add metadata if available
                    if hasattr(memory, "metadata"):
                        combined_dict[memory_id].update(memory.metadata)

                except (KeyError, AttributeError):
                    # Skip if memory not found or no metadata
                    pass
            elif memory_id in combined_dict:
                # Update existing memory
                combined_dict[memory_id]["activation_score"] = score

        # Calculate weighted scores
        for memory_id, result in combined_dict.items():
            # Use normalized score if available, otherwise raw similarity
            similarity = result.get("normalized_score", result["similarity_score"])

            # If similarity is very high, reduce influence of activation to avoid
            # returning the same set of "activated" memories for different queries
            if similarity > 0.7:
                # For highly similar results, reduce activation influence
                local_activation_weight = activation_weight * 0.5
            else:
                local_activation_weight = activation_weight

            # Calculate contribution from each source
            similarity_contribution = similarity * similarity_weight
            associative_contribution = result["associative_score"] * associative_weight
            temporal_contribution = result["temporal_score"] * temporal_weight
            activation_contribution = result["activation_score"] * local_activation_weight

            # If similarity is low, reduce other contributions proportionally
            # This ensures semantic relevance is primary
            if similarity < 0.3:
                scaling_factor = max(0.1, similarity / 0.3)
                associative_contribution *= scaling_factor
                activation_contribution *= scaling_factor
                # Keep temporal contribution since it may be unrelated to similarity

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
            result["adjusted_activation_weight"] = local_activation_weight

            # Flag if below threshold
            result["below_threshold"] = combined_score < self.confidence_threshold

        # Convert to list and sort by relevance
        combined_results = list(combined_dict.values())
        combined_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Apply result diversity - avoid returning the same memory topics
        # This helps prevent the system from returning the same results for different queries
        if len(combined_results) > self.min_results:
            diverse_results = []
            seen_topics = set()

            # Always include the top result
            if combined_results:
                diverse_results.append(combined_results[0])

                # Extract topics from the result
                topics = set()
                if "topics" in combined_results[0]:
                    topics.update(combined_results[0]["topics"])
                elif (
                    "metadata" in combined_results[0]
                    and "topics" in combined_results[0]["metadata"]
                ):
                    topics.update(combined_results[0]["metadata"]["topics"])
                seen_topics.update(topics)

            # Process remaining results with diversity
            for result in combined_results[1:]:
                # Extract topics
                topics = set()
                if "topics" in result:
                    topics.update(result["topics"])
                elif "metadata" in result and "topics" in result["metadata"]:
                    topics.update(result["metadata"]["topics"])

                # Check for topic diversity
                if not topics or len(topics.intersection(seen_topics)) < len(topics):
                    # Some new topics or no topics (include it)
                    diverse_results.append(result)
                    seen_topics.update(topics)

                    # If we have enough diverse results, stop
                    if len(diverse_results) >= len(combined_results):
                        break

            # If we don't have enough diverse results, add more from the original list
            remaining = [r for r in combined_results if r not in diverse_results]
            if len(diverse_results) < self.min_results and remaining:
                diverse_results.extend(remaining[: self.min_results - len(diverse_results)])

            # Use diverse results if we have enough
            if len(diverse_results) >= self.min_results:
                return diverse_results

        return combined_results
