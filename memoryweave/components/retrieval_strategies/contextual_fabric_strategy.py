# memoryweave/components/retrieval_strategies/contextual_fabric_strategy.py
"""
Contextual Fabric Retrieval Strategy for MemoryWeave.

This module implements a retrieval strategy that leverages the contextual fabric structure
of MemoryWeave, combining direct similarity, associative links, and temporal context
to provide more relevant and contextually-aware memory retrieval.
"""

import logging
from typing import Any, Dict, List, Optional

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
        activation_manager: Optional[ActivationManager] = None
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

    def initialize(self, config: Dict[str, Any]) -> None:
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
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories using the contextual fabric strategy.
        
        Args:
            query_embedding: Query embedding for similarity matching
            top_k: Number of results to return
            context: Context containing query, memory, etc.
            
        Returns:
            List of retrieved memory dicts with relevance scores
        """
        # Get memory store from context or instance
        memory_store = context.get("memory_store", self.memory_store)

        # Get query from context
        query = context.get("query", "")

        # Apply query type adaptation if available
        adapted_params = context.get("adapted_retrieval_params", {})
        confidence_threshold = adapted_params.get("confidence_threshold", self.confidence_threshold)

        # Log retrieval details if debug enabled
        if self.debug:
            self.logger.debug(f"ContextualFabricStrategy: Retrieving for query: '{query}'")
            self.logger.debug(f"ContextualFabricStrategy: Using confidence threshold: {confidence_threshold}")

        # Step 1: Get direct similarity matches
        similarity_results = self._retrieve_by_similarity(
            query_embedding=query_embedding,
            max_results=self.max_candidates,
            memory_store=memory_store
        )

        # Step 2: Get associative matches (if linker available)
        associative_results = {}
        if self.associative_linker is not None:
            # Get top similarity matches as starting points
            top_similarity_ids = [r["memory_id"] for r in similarity_results[:min(5, len(similarity_results))]]

            # Traverse associative network from each starting point
            for memory_id in top_similarity_ids:
                activations = self.associative_linker.traverse_associative_network(
                    start_id=memory_id,
                    max_hops=self.max_associative_hops,
                    min_strength=0.1
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
                            temporal_relevance = np.exp(-(time_diff ** 2) / (2 * temporal_scale ** 2))

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
            memory_store=memory_store
        )

        # Step 6: Apply threshold and sort
        filtered_results = [
            r for r in combined_results
            if r["relevance_score"] >= confidence_threshold
        ]

        # Apply minimum results guarantee
        if len(filtered_results) < self.min_results:
            # Use top min_results from combined results
            filtered_results = combined_results[:self.min_results]

        # Limit to top_k
        top_k = min(top_k, len(filtered_results))
        results = filtered_results[:top_k]

        # Debug logging
        if self.debug:
            self.logger.debug(f"ContextualFabricStrategy: Retrieved {len(results)} results")
            self.logger.debug(f"ContextualFabricStrategy: Top 3 scores: {[r['relevance_score'] for r in results[:3]]}")

            # Log contribution breakdown for top result
            if results:
                top = results[0]
                self.logger.debug("Top result contributions:")
                self.logger.debug(f"- Similarity: {top.get('similarity_score', 0):.3f} * {self.similarity_weight:.1f} = {top.get('similarity_contribution', 0):.3f}")
                self.logger.debug(f"- Associative: {top.get('associative_score', 0):.3f} * {self.associative_weight:.1f} = {top.get('associative_contribution', 0):.3f}")
                self.logger.debug(f"- Temporal: {top.get('temporal_score', 0):.3f} * {self.temporal_weight:.1f} = {top.get('temporal_contribution', 0):.3f}")
                self.logger.debug(f"- Activation: {top.get('activation_score', 0):.3f} * {self.activation_weight:.1f} = {top.get('activation_contribution', 0):.3f}")
                self.logger.debug(f"- Total: {top['relevance_score']:.3f}")

        return results

    def _retrieve_by_similarity(
        self,
        query_embedding: np.ndarray,
        max_results: int,
        memory_store: Optional[IMemoryStore]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories by direct similarity.
        
        Args:
            query_embedding: Query embedding
            max_results: Maximum number of results to return
            memory_store: Memory store to retrieve from
            
        Returns:
            List of retrieved memory dicts with similarity scores
        """
        if memory_store is None or not hasattr(memory_store, "memory_embeddings"):
            return []

        # Compute cosine similarities
        similarities = np.dot(memory_store.memory_embeddings, query_embedding)

        # Get top indices
        top_indices = np.argsort(-similarities)[:max_results]

        # Format results
        results = []

        for idx in top_indices:
            similarity = float(similarities[idx])

            # Skip near-zero similarities
            if similarity < 0.01:
                continue

            # Add to results
            results.append({
                "memory_id": int(idx),
                "similarity_score": similarity,
                **memory_store.memory_metadata[idx]
            })

        return results

    def _combine_results(
        self,
        similarity_results: List[Dict[str, Any]],
        associative_results: Dict[MemoryID, float],
        temporal_results: Dict[MemoryID, float],
        activation_results: Dict[MemoryID, float],
        memory_store: Optional[IMemoryStore]
    ) -> List[Dict[str, Any]]:
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

        # Add similarity results
        for result in similarity_results:
            memory_id = result["memory_id"]

            if memory_id not in combined_dict:
                combined_dict[memory_id] = {
                    "memory_id": memory_id,
                    "similarity_score": result["similarity_score"],
                    "associative_score": 0.0,
                    "temporal_score": 0.0,
                    "activation_score": 0.0,
                    **result
                }
            else:
                combined_dict[memory_id]["similarity_score"] = result["similarity_score"]

        # Add associative results
        for memory_id, score in associative_results.items():
            if memory_id not in combined_dict and memory_store is not None:
                # Add new memory
                try:
                    memory = memory_store.get(memory_id)
                    combined_dict[memory_id] = {
                        "memory_id": memory_id,
                        "similarity_score": 0.0,
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
            # Calculate contribution from each source
            similarity_contribution = result["similarity_score"] * self.similarity_weight
            associative_contribution = result["associative_score"] * self.associative_weight
            temporal_contribution = result["temporal_score"] * self.temporal_weight
            activation_contribution = result["activation_score"] * self.activation_weight

            # Calculate combined score
            combined_score = (
                similarity_contribution +
                associative_contribution +
                temporal_contribution +
                activation_contribution
            )

            # Store combined score and contributions
            result["relevance_score"] = combined_score
            result["similarity_contribution"] = similarity_contribution
            result["associative_contribution"] = associative_contribution
            result["temporal_contribution"] = temporal_contribution
            result["activation_contribution"] = activation_contribution

            # Flag if below threshold
            result["below_threshold"] = combined_score < self.confidence_threshold

        # Convert to list and sort by relevance
        combined_results = list(combined_dict.values())
        combined_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return combined_results
