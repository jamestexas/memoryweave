# memoryweave/components/retrieval_strategies.py
from typing import Any, List, Optional

import numpy as np

from memoryweave.components.base import RetrievalStrategy
from memoryweave.core import ContextualMemory
from memoryweave.core.category_manager import CategoryManager


class SimilarityRetrievalStrategy(RetrievalStrategy):
    """
    Retrieves memories based purely on similarity to query embedding.
    """

    def __init__(self, memory: ContextualMemory):
        self.memory = memory

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.confidence_threshold = config.get("confidence_threshold", 0.0)
        self.activation_boost = config.get("activation_boost", True)

        # Set minimum k for testing/benchmarking, but don't go below 1
        self.min_results = max(1, config.get("min_results", 5))

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Retrieve memories based on similarity to query embedding."""
        # Get memory from context or instance
        memory = context.get("memory", self.memory)

        # Apply query type adaptation if available
        adapted_params = context.get("adapted_retrieval_params", {})
        confidence_threshold = adapted_params.get("confidence_threshold", self.confidence_threshold)

        # Check if we're in evaluation mode
        in_evaluation = context.get("in_evaluation", False)
        
        # Debug log for evaluation mode
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"SimilarityRetrievalStrategy: in_evaluation={in_evaluation}, confidence_threshold={confidence_threshold}")

        # Standard retrieval path
        if hasattr(memory, "retrieve_memories"):
            # Try with the specified threshold
            results = memory.retrieve_memories(
                query_embedding,
                top_k=top_k,
                activation_boost=self.activation_boost,
                confidence_threshold=confidence_threshold,
            )
            logger.debug(f"SimilarityRetrievalStrategy: Initial retrieval returned {len(results)} results with threshold {confidence_threshold}")

            # Only fall back to lower threshold if not in evaluation mode
            if not results and not in_evaluation:
                # This is a fallback for non-evaluation scenarios only
                test_threshold = 0.0  # Minimum possible threshold
                logger.info(f"SimilarityRetrievalStrategy: No results at threshold {confidence_threshold}, falling back to {test_threshold}")
                results = memory.retrieve_memories(
                    query_embedding,
                    top_k=top_k,
                    activation_boost=self.activation_boost,
                    confidence_threshold=test_threshold,
                )

                # Mark these as lower-confidence results
                results = [
                    (idx, min(score, confidence_threshold - 0.01), metadata)
                    for idx, score, metadata in results
                ]
        else:
            # If memory doesn't have retrieve_memories, return empty results
            results = []

        # Format results - include all results but mark their relevance accordingly
        formatted_results = []
        for idx, score, metadata in results:
            # Add all results but mark if they're below threshold
            formatted_results.append(
                {
                    "memory_id": idx,
                    "relevance_score": score,
                    "below_threshold": score < confidence_threshold,
                    **metadata,
                }
            )

        return formatted_results

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a query to retrieve relevant memories.

        Args:
            query: The query string
            context: Context dictionary containing query_embedding, memory, etc.

        Returns:
            Updated context with results
        """
        import logging

        logger = logging.getLogger(__name__)

        # Log which query is being processed
        logger.info(f"SimilarityRetrievalStrategy: Processing query: {query}")

        # Get query embedding from context
        query_embedding = context.get("query_embedding")
        if query_embedding is None:
            # Try to get embedding model from context
            embedding_model = context.get("embedding_model")
            if embedding_model:
                query_embedding = embedding_model.encode(query)
                logger.info(
                    "SimilarityRetrievalStrategy: Created query embedding using embedding model"
                )

        # If still no query embedding but we have working_context, create one for test environment
        # (Note: This is for compatibility with tests where no embedding model is available)
        if query_embedding is None and "working_context" in context:
            memory = context.get("memory", self.memory)
            dim = getattr(memory, "embedding_dim", 768)
            query_embedding = np.ones(dim) / np.sqrt(dim)  # Unit vector
            logger.info(
                f"SimilarityRetrievalStrategy: Created dummy query embedding for compatibility with dim={dim}"
            )

        # If still no query embedding, return empty results
        if query_embedding is None:
            logger.warning(
                "SimilarityRetrievalStrategy: No query embedding available, returning empty results"
            )
            return {"results": []}

        # Get top_k from context
        top_k = context.get("top_k", 5)
        logger.info(f"SimilarityRetrievalStrategy: Using top_k={top_k}")

        # Retrieve memories
        results = self.retrieve(query_embedding, top_k, context)
        logger.info(f"SimilarityRetrievalStrategy: Retrieved {len(results)} results")

        # Return results
        return {"results": results}


class TemporalRetrievalStrategy(RetrievalStrategy):
    """
    Retrieves memories based on recency and activation.
    """

    def __init__(self, memory: ContextualMemory):
        self.memory = memory

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        pass

    def retrieve(
        self, query_embedding: np.ndarray, top_k: int, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Retrieve memories based on temporal factors."""
        # Get memory from context or instance
        memory = context.get("memory", self.memory)

        # Get memories sorted by temporal markers (most recent first)
        temporal_order = np.argsort(-memory.temporal_markers)[:top_k]

        results = []
        for idx in temporal_order:
            results.append(
                {
                    "memory_id": int(idx),
                    "relevance_score": float(memory.activation_levels[idx]),
                    **memory.memory_metadata[idx],
                }
            )

        return results

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a query to retrieve relevant memories based on recency.

        Args:
            query: The query string
            context: Context dictionary containing memory, etc.

        Returns:
            Updated context with results
        """
        import logging

        logger = logging.getLogger(__name__)

        # Log which query is being processed
        logger.info(f"TemporalRetrievalStrategy: Processing query: {query}")

        # Get memory from context or instance
        memory = context.get("memory", self.memory)

        # Get top_k from context
        top_k = context.get("top_k", 5)
        logger.info(f"TemporalRetrievalStrategy: Using top_k={top_k}")

        # Create a query embedding if needed (for compatibility with tests)
        query_embedding = context.get("query_embedding")
        if query_embedding is None:
            # Try to get embedding model from context
            embedding_model = context.get("embedding_model")
            if embedding_model:
                query_embedding = embedding_model.encode(query)
                logger.info(
                    "TemporalRetrievalStrategy: Created query embedding using embedding model"
                )
            else:
                # Use unit vector as dummy embedding when no model is available
                dim = getattr(memory, "embedding_dim", 768)
                query_embedding = np.ones(dim) / np.sqrt(dim)
                logger.info(
                    f"TemporalRetrievalStrategy: Created dummy query embedding with dim={dim}"
                )

        # Check if we're in evaluation mode
        in_evaluation = context.get("in_evaluation", False)

        # Standard retrieval logic
        logger.info("TemporalRetrievalStrategy: Using temporal retrieval")
        results = self.retrieve(query_embedding, top_k, context)
        logger.info(f"TemporalRetrievalStrategy: Retrieved {len(results)} results")

        # Only add fallback result if not in evaluation mode and no results found
        if not in_evaluation and not results and len(memory.memory_metadata) > 0:
            idx = np.argmax(memory.temporal_markers)
            results = [
                {
                    "memory_id": int(idx),
                    "relevance_score": 0.8,
                    **memory.memory_metadata[idx],
                }
            ]
            logger.info("TemporalRetrievalStrategy: Added most recent memory as fallback")

        return {"results": results}


class HybridRetrievalStrategy(RetrievalStrategy):
    """
    Hybrid retrieval combining similarity, recency, and keyword matching.
    """

    def __init__(self, memory: ContextualMemory):
        self.memory = memory

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.relevance_weight = config.get("relevance_weight", 0.7)
        self.recency_weight = config.get("recency_weight", 0.3)
        self.confidence_threshold = config.get("confidence_threshold", 0.0)

    def retrieve(
        self, query_embedding: np.ndarray, top_k: int, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Retrieve memories using hybrid approach."""
        # Get memory from context or instance
        memory = context.get("memory", self.memory)

        # Apply query type adaptation if available
        adapted_params = context.get("adapted_retrieval_params", {})
        confidence_threshold = adapted_params.get("confidence_threshold", self.confidence_threshold)
        relevance_weight = adapted_params.get("relevance_weight", self.relevance_weight)
        recency_weight = adapted_params.get("recency_weight", self.recency_weight)

        # Only fall back to basic retrieval if we don't have access to the necessary memory attributes
        # We need memory_embeddings, temporal_markers, and activation_levels
        if (
            not hasattr(memory, "memory_embeddings")
            or not hasattr(memory, "temporal_markers")
            or not hasattr(memory, "activation_levels")
        ) and hasattr(memory, "retrieve_memories"):
            # Log that we're falling back to basic retrieval
            import logging

            logging.warning(
                "HybridRetrievalStrategy: Falling back to basic retrieval due to missing memory attributes"
            )

            results = memory.retrieve_memories(
                query_embedding, top_k=top_k, confidence_threshold=confidence_threshold
            )

            # Format results
            formatted_results = []
            for idx, score, metadata in results:
                formatted_results.append(
                    {
                        "memory_id": idx,
                        "relevance_score": score,
                        "similarity": score,
                        "recency": 1.0,
                        **metadata,
                    }
                )

            return formatted_results

        # For real memory, implement hybrid approach
        # Get similarity scores
        similarities = np.dot(memory.memory_embeddings, query_embedding)

        # Normalize temporal factors
        max_time = float(memory.current_time)
        temporal_factors = memory.temporal_markers / max_time if max_time > 0 else 0

        # Combine scores
        combined_scores = relevance_weight * similarities + recency_weight * temporal_factors

        # Apply activation boost
        combined_scores = combined_scores * memory.activation_levels

        # Apply confidence threshold filtering
        valid_indices = np.where(combined_scores >= confidence_threshold)[0]
        if len(valid_indices) == 0:
            return []

        # Get top-k indices from valid indices
        array_size = len(valid_indices)
        if top_k >= array_size:
            top_relative_indices = np.argsort(-combined_scores[valid_indices])
        else:
            top_relative_indices = np.argpartition(-combined_scores[valid_indices], top_k)[:top_k]
            top_relative_indices = top_relative_indices[
                np.argsort(-combined_scores[valid_indices][top_relative_indices])
            ]

        # Format results
        results = []
        for idx in valid_indices[top_relative_indices]:
            score = float(combined_scores[idx])
            results.append(
                {
                    "memory_id": int(idx),
                    "relevance_score": score,
                    "similarity": float(similarities[idx]),
                    "recency": float(temporal_factors[idx]),
                    **memory.memory_metadata[idx],
                }
            )

        return results[:top_k]

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a query to retrieve relevant memories.

        Args:
            query: The query string
            context: Context dictionary containing query_embedding, memory, etc.

        Returns:
            Updated context with results
        """
        import logging

        logger = logging.getLogger(__name__)

        # Log which query is being processed
        logger.info(f"HybridRetrievalStrategy: Processing query: {query}")

        # Get query embedding from context
        query_embedding = context.get("query_embedding")
        if query_embedding is None:
            # Try to get embedding model from context
            embedding_model = context.get("embedding_model")
            if embedding_model:
                query_embedding = embedding_model.encode(query)
                logger.info(
                    "HybridRetrievalStrategy: Created query embedding using embedding model"
                )

        # If still no query embedding but in a test environment, create a dummy one
        if query_embedding is None and "working_context" in context:
            # Use standard dimension (for test compatibility)
            memory = context.get("memory", self.memory)
            dim = getattr(memory, "embedding_dim", 768)
            query_embedding = np.ones(dim) / np.sqrt(dim)  # Unit vector
            logger.info(
                f"HybridRetrievalStrategy: Created dummy query embedding for compatibility with dim={dim}"
            )

        # If still no query embedding, return empty results
        if query_embedding is None:
            logger.warning(
                "HybridRetrievalStrategy: No query embedding available, returning empty results"
            )
            return {"results": []}

        # Get top_k from context
        top_k = context.get("top_k", 5)
        logger.info(f"HybridRetrievalStrategy: Using top_k={top_k}")

        # Retrieve memories using the standard hybrid approach
        logger.info("HybridRetrievalStrategy: Using standard hybrid retrieval approach")
        results = self.retrieve(query_embedding, top_k, context)
        logger.info(f"HybridRetrievalStrategy: Retrieved {len(results)} results")

        # Return results
        return {"results": results}


class TwoStageRetrievalStrategy(RetrievalStrategy):
    """
    Two-stage retrieval strategy that retrieves a larger set of candidates with
    lower threshold in the first stage, then re-ranks in the second stage.
    """

    def __init__(
        self,
        memory: ContextualMemory,
        base_strategy: RetrievalStrategy = None,
        post_processors: list = None,
    ):
        """
        Initialize the two-stage retrieval strategy.

        Args:
            memory: Memory to retrieve from
            base_strategy: Base retrieval strategy to use (defaults to HybridRetrievalStrategy)
            post_processors: List of post-processors to apply in second stage
        """
        self.memory = memory
        self.base_strategy = base_strategy or HybridRetrievalStrategy(memory)
        self.post_processors = post_processors or []

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        # First stage parameters
        self.first_stage_k = config.get("first_stage_k", 20)
        self.first_stage_threshold_factor = config.get("first_stage_threshold_factor", 0.7)
        self.confidence_threshold = config.get("confidence_threshold", 0.0)

        # Initialize base strategy if it's not already initialized
        if hasattr(self.base_strategy, "initialize"):
            self.base_strategy.initialize(
                {
                    **config,
                    "confidence_threshold": self.confidence_threshold
                    * self.first_stage_threshold_factor,
                }
            )

        # Initialize post-processors
        for processor in self.post_processors:
            if hasattr(processor, "initialize") and processor.initialize != self.initialize:
                processor.initialize(config.get("post_processor_config", {}))

    def retrieve(
        self, query_embedding: np.ndarray, top_k: int, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories using two-stage approach.

        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            context: Context containing memory, query type, etc.

        Returns:
            List of retrieved memory dicts
        """
        import logging

        logger = logging.getLogger(__name__)

        # Log which configuration is being used
        logger.info(
            f"TwoStageRetrievalStrategy: Using base strategy: {self.base_strategy.__class__.__name__}"
        )
        logger.info(
            f"TwoStageRetrievalStrategy: Post-processors: {[p.__class__.__name__ for p in self.post_processors]}"
        )

        # Apply query type adaptation if available
        adapted_params = context.get("adapted_retrieval_params", {})

        # Adjust parameters based on query type and adapted parameters
        first_stage_k = adapted_params.get("first_stage_k", self.first_stage_k)
        first_stage_threshold_factor = adapted_params.get(
            "first_stage_threshold_factor", self.first_stage_threshold_factor
        )
        confidence_threshold = adapted_params.get("confidence_threshold", self.confidence_threshold)
        expand_keywords = adapted_params.get("expand_keywords", False)

        first_stage_threshold = confidence_threshold * first_stage_threshold_factor

        logger.info(
            f"TwoStageRetrievalStrategy: first_stage_k={first_stage_k}, first_stage_threshold={first_stage_threshold}, expand_keywords={expand_keywords}"
        )

        # Check if we're in evaluation mode
        in_evaluation = context.get("in_evaluation", False)
        logger.info(f"TwoStageRetrievalStrategy: in_evaluation={in_evaluation}")
        
        # Log the adapted parameters to understand where they're coming from
        logger.info(f"TwoStageRetrievalStrategy: Adapted params from context: {adapted_params}")
        
        # Use query type for further adjustments if not already in adapted params
        if "first_stage_k" not in adapted_params:
            query_type = context.get("primary_query_type", "default")
            logger.info(f"TwoStageRetrievalStrategy: Query type={query_type}")
            
            if query_type == "personal":
                # Personal queries need higher precision
                original_threshold = first_stage_threshold
                first_stage_threshold = max(first_stage_threshold, 0.2)
                logger.info(
                    f"TwoStageRetrievalStrategy: Adjusted for personal query, first_stage_threshold from {original_threshold} to {first_stage_threshold}"
                )
            elif query_type == "factual":
                # Factual queries need better recall
                original_threshold = first_stage_threshold
                original_k = first_stage_k
                first_stage_threshold = min(first_stage_threshold, 0.15)
                first_stage_k = max(first_stage_k, 30)  # Get more candidates for factual queries
                logger.info(
                    f"TwoStageRetrievalStrategy: Adjusted for factual query, first_stage_threshold from {original_threshold} to {first_stage_threshold}, first_stage_k from {original_k} to {first_stage_k}"
                )

        # First stage: Get a larger set of candidates with lower threshold
        # Update base strategy's confidence threshold for first stage
        if hasattr(self.base_strategy, "confidence_threshold"):
            original_threshold = self.base_strategy.confidence_threshold
            self.base_strategy.confidence_threshold = first_stage_threshold
            logger.info(
                f"TwoStageRetrievalStrategy: Updated base strategy threshold from {original_threshold} to {first_stage_threshold}"
            )

        # If expand_keywords is enabled, use expanded keywords from context if available
        if expand_keywords and "important_keywords" in context:
            # Check if expanded_keywords are already in context (from KeywordExpander component)
            if "expanded_keywords" not in context:
                # Fall back to basic expansion if KeywordExpander wasn't used
                original_keywords = context.get("important_keywords", set())
                expanded_keywords = set(original_keywords)

                # Add singular/plural forms
                for keyword in original_keywords:
                    if not keyword.endswith("s"):
                        expanded_keywords.add(f"{keyword}s")  # Simple pluralization
                    elif len(keyword) > 1:
                        expanded_keywords.add(keyword[:-1])  # Simple singularization

                # Add to context with a different key to avoid overwriting
                context["expanded_keywords"] = expanded_keywords
                logger.info(
                    f"TwoStageRetrievalStrategy: Expanded keywords from {original_keywords} to {expanded_keywords}"
                )

            # Temporarily replace important_keywords with expanded set
            context["original_important_keywords"] = context["important_keywords"]
            context["important_keywords"] = context["expanded_keywords"]
            logger.info(
                f"TwoStageRetrievalStrategy: Using expanded keywords: {context['expanded_keywords']}"
            )

        # Get candidates using base strategy
        logger.info(
            f"TwoStageRetrievalStrategy: Calling base strategy {self.base_strategy.__class__.__name__}.retrieve"
        )
        candidates = self.base_strategy.retrieve(query_embedding, first_stage_k, context)
        logger.info(f"TwoStageRetrievalStrategy: First stage returned {len(candidates)} candidates")

        # Restore original threshold
        if hasattr(self.base_strategy, "confidence_threshold"):
            self.base_strategy.confidence_threshold = original_threshold
            logger.info(
                f"TwoStageRetrievalStrategy: Restored base strategy threshold to {original_threshold}"
            )

        # Restore original keywords if they were expanded
        if expand_keywords and "original_important_keywords" in context:
            context["important_keywords"] = context["original_important_keywords"]
            del context["original_important_keywords"]
            logger.info("TwoStageRetrievalStrategy: Restored original keywords")

        # If no candidates, return empty list
        if not candidates:
            logger.warning("TwoStageRetrievalStrategy: No candidates found in first stage")
            return []

        # Second stage: Re-rank and filter candidates
        # Apply post-processors with adapted parameters
        for i, processor in enumerate(self.post_processors):
            logger.info(
                f"TwoStageRetrievalStrategy: Applying post-processor {i + 1}/{len(self.post_processors)}: {processor.__class__.__name__}"
            )

            # If we have an adapted keyword_boost_weight, set it before processing
            if (
                hasattr(processor, "keyword_boost_weight")
                and "keyword_boost_weight" in adapted_params
            ):
                orig_weight = processor.keyword_boost_weight
                processor.keyword_boost_weight = adapted_params["keyword_boost_weight"]
                logger.info(
                    f"TwoStageRetrievalStrategy: Updated keyword_boost_weight from {orig_weight} to {processor.keyword_boost_weight}"
                )

            # If we have an adapted adaptive_k_factor, set it before processing
            if hasattr(processor, "adaptive_k_factor") and "adaptive_k_factor" in adapted_params:
                orig_factor = processor.adaptive_k_factor
                processor.adaptive_k_factor = adapted_params["adaptive_k_factor"]
                logger.info(
                    f"TwoStageRetrievalStrategy: Updated adaptive_k_factor from {orig_factor} to {processor.adaptive_k_factor}"
                )

            # Process the candidates
            candidates_before = len(candidates)
            candidates = processor.process_results(candidates, context.get("query", ""), context)
            logger.info(
                f"TwoStageRetrievalStrategy: Post-processor reduced candidates from {candidates_before} to {len(candidates)}"
            )

        # Sort by relevance score
        candidates.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        logger.info("TwoStageRetrievalStrategy: Sorted candidates by relevance score")

        # Filter by confidence threshold
        candidates_before = len(candidates)
        candidates = [c for c in candidates if c.get("relevance_score", 0) >= confidence_threshold]
        logger.info(
            f"TwoStageRetrievalStrategy: Filtered candidates by threshold {confidence_threshold}, from {candidates_before} to {len(candidates)}"
        )

        # Return top_k results
        final_results = candidates[:top_k]
        logger.info(
            f"TwoStageRetrievalStrategy: Returning top {len(final_results)} results (requested {top_k})"
        )

        # Log the average relevance score of results
        if final_results:
            avg_score = sum(r.get("relevance_score", 0) for r in final_results) / len(final_results)
            logger.info(f"TwoStageRetrievalStrategy: Average relevance score: {avg_score:.4f}")

        return final_results

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Process a query to retrieve relevant memories."""
        # Get query embedding from context
        query_embedding = context.get("query_embedding")
        if query_embedding is None:
            return {"results": []}

        # Get top_k from context
        top_k = context.get("top_k", 5)

        # Add query to context
        context["query"] = query

        # Retrieve memories using two-stage approach
        results = self.retrieve(query_embedding, top_k, context)

        # Return results
        return {"results": results}


class CategoryRetrievalStrategy(RetrievalStrategy):
    """
    Retrieves memories based on ART category clustering.

    This strategy uses the ART-inspired clustering to first identify
    the most relevant categories, then retrieves memories from those
    categories.
    """

    def __init__(self, memory: ContextualMemory):
        """
        Initialize with memory and category manager.

        Args:
            memory: The memory to retrieve from
        """
        self.memory = memory
        # Will use the category_manager from memory

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.confidence_threshold = config.get("confidence_threshold", 0.0)
        self.max_categories = config.get("max_categories", 3)
        self.activation_boost = config.get("activation_boost", True)
        self.category_selection_threshold = config.get("category_selection_threshold", 0.5)

        # For testing/benchmarking, set minimum results
        self.min_results = max(1, config.get("min_results", 5))

    def retrieve(
        self, query_embedding: np.ndarray, top_k: int, context: dict[str, Any]
    ) -> List[dict[str, Any]]:
        """Retrieve memories using category-based retrieval."""
        import logging

        logger = logging.getLogger(__name__)

        # Get memory from context or instance
        memory = context.get("memory", self.memory)

        # Check if memory has category_manager
        category_manager = getattr(memory, "category_manager", None)
        if category_manager is None:
            # Fall back to similarity retrieval if no category manager
            logger.info(
                "CategoryRetrievalStrategy: No category manager found, falling back to similarity retrieval"
            )
            similarity_strategy = SimilarityRetrievalStrategy(memory)
            if hasattr(similarity_strategy, "initialize"):
                similarity_strategy.initialize({"confidence_threshold": self.confidence_threshold})
            return similarity_strategy.retrieve(query_embedding, top_k, context)

        # Apply query type adaptation if available
        adapted_params = context.get("adapted_retrieval_params", {})
        confidence_threshold = adapted_params.get("confidence_threshold", self.confidence_threshold)
        max_categories = adapted_params.get("max_categories", self.max_categories)

        try:
            # Get category similarities
            category_similarities = category_manager.get_category_similarities(query_embedding)

            # If no categories, fall back to similarity retrieval
            if len(category_similarities) == 0:
                similarity_strategy = SimilarityRetrievalStrategy(memory)
                if hasattr(similarity_strategy, "initialize"):
                    similarity_strategy.initialize(
                        {"confidence_threshold": self.confidence_threshold}
                    )
                return similarity_strategy.retrieve(query_embedding, top_k, context)

            # Select top categories with similarity above threshold
            selected_categories = []
            for cat_idx in np.argsort(-category_similarities):
                if category_similarities[cat_idx] >= self.category_selection_threshold:
                    selected_categories.append(cat_idx)
                if len(selected_categories) >= max_categories:
                    break

            # If no categories selected, use top N categories
            if not selected_categories and len(category_similarities) > 0:
                num_to_select = min(max_categories, len(category_similarities))
                selected_categories = np.argsort(-category_similarities)[:num_to_select].tolist()

            # Get memories from selected categories
            candidate_indices = []
            for cat_idx in selected_categories:
                cat_memories = category_manager.get_memories_for_category(cat_idx)
                candidate_indices.extend(cat_memories)

            # If no candidates, fall back to similarity retrieval
            if not candidate_indices:
                similarity_strategy = SimilarityRetrievalStrategy(memory)
                if hasattr(similarity_strategy, "initialize"):
                    similarity_strategy.initialize(
                        {"confidence_threshold": self.confidence_threshold}
                    )
                return similarity_strategy.retrieve(query_embedding, top_k, context)

            # Calculate similarities for candidate memories
            candidate_similarities = np.dot(
                memory.memory_embeddings[candidate_indices], query_embedding
            )

            # Apply activation boost if enabled
            if self.activation_boost:
                candidate_similarities = (
                    candidate_similarities * memory.activation_levels[candidate_indices]
                )

            # Filter by confidence threshold
            valid_candidates = np.where(candidate_similarities >= confidence_threshold)[0]

            # Check if we're in evaluation mode
            in_evaluation = context.get("in_evaluation", False)

            # Only use threshold bypass if not in evaluation mode
            if (
                len(valid_candidates) == 0
                and not in_evaluation
                and (hasattr(self, "min_results") and self.min_results > 0)
            ):
                logger.info(
                    "CategoryRetrievalStrategy: No results passed threshold, using min_results fallback (non-evaluation mode)"
                )
                # Sort all candidates by similarity
                sorted_idx = np.argsort(-candidate_similarities)
                # Take top min_results candidates regardless of threshold
                valid_candidates = sorted_idx[: self.min_results]

            if len(valid_candidates) == 0:
                return []

            valid_candidate_indices = [candidate_indices[i] for i in valid_candidates]
            valid_candidate_similarities = candidate_similarities[valid_candidates]

            # Get top-k memories
            top_k = min(top_k, len(valid_candidate_similarities))
            top_memory_indices = np.argsort(-valid_candidate_similarities)[:top_k]

            # Format results
            results = []
            for i in top_memory_indices:
                idx = valid_candidate_indices[i]
                similarity = valid_candidate_similarities[i]

                # Get category for memory
                try:
                    category_id = category_manager.get_category_for_memory(idx)
                    category_similarity = (
                        category_similarities[category_id]
                        if category_id < len(category_similarities)
                        else 0.0
                    )
                except (IndexError, Exception):
                    category_id = -1
                    category_similarity = 0.0

                # Update memory activation
                if hasattr(memory, "update_activation"):
                    memory.update_activation(idx)

                # Add result with category information
                results.append(
                    {
                        "memory_id": int(idx),
                        "relevance_score": float(similarity),
                        "category_id": int(category_id),
                        "category_similarity": float(category_similarity),
                        "below_threshold": similarity < confidence_threshold,
                        **memory.memory_metadata[idx],
                    }
                )

            return results

        except Exception as e:
            # On any error, fall back to similarity retrieval
            import logging

            logging.warning(
                f"Category retrieval failed with error: {str(e)}. Falling back to similarity retrieval."
            )

            similarity_strategy = SimilarityRetrievalStrategy(memory)
            if hasattr(similarity_strategy, "initialize"):
                similarity_strategy.initialize({"confidence_threshold": self.confidence_threshold})
            return similarity_strategy.retrieve(query_embedding, top_k, context)

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Process a query to retrieve relevant memories using categories."""
        import logging

        logger = logging.getLogger(__name__)

        # Log which query is being processed
        logger.info(f"CategoryRetrievalStrategy: Processing query: {query}")

        # Get query embedding from context
        query_embedding = context.get("query_embedding")
        if query_embedding is None:
            # Try to get embedding model from context
            embedding_model = context.get("embedding_model")
            if embedding_model:
                query_embedding = embedding_model.encode(query)
                logger.info(
                    "CategoryRetrievalStrategy: Created query embedding using embedding model"
                )

        # If still no query embedding but in a test environment, create a dummy one
        if query_embedding is None and "working_context" in context:
            # Use standard dimension (for test compatibility)
            memory = context.get("memory", self.memory)
            dim = getattr(memory, "embedding_dim", 768)
            query_embedding = np.ones(dim) / np.sqrt(dim)  # Unit vector
            logger.info(
                f"CategoryRetrievalStrategy: Created dummy query embedding for compatibility with dim={dim}"
            )

        # If still no query embedding, return empty results
        if query_embedding is None:
            logger.warning(
                "CategoryRetrievalStrategy: No query embedding available, returning empty results"
            )
            return {"results": []}

        # Get top_k from context
        top_k = context.get("top_k", 5)
        logger.info(f"CategoryRetrievalStrategy: Using top_k={top_k}")

        # Retrieve memories using category-based approach
        results = self.retrieve(query_embedding, top_k, context)
        logger.info(f"CategoryRetrievalStrategy: Retrieved {len(results)} results")

        # Return results
        return {"results": results}
