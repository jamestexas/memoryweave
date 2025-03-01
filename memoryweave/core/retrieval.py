"""
DEPRECATED: ContextualRetriever implementation.

This module is deprecated and will be removed in a future version.
Please use the component-based implementation in memoryweave.components instead.

The original implementation is preserved in memoryweave.deprecated.core.retrieval
for reference and backward compatibility during the transition period.
"""

import re
import warnings
from typing import Any, Optional

import numpy as np

from memoryweave.core.contextual_fabric import ContextualMemory
from memoryweave.deprecated.core.retrieval import ContextualRetriever
from memoryweave.utils.nlp_extraction import NLPExtractor

# Emit deprecation warning when imported
warnings.warn(
    "memoryweave.core.retrieval is deprecated. Please use memoryweave.components instead.",
    DeprecationWarning,
    stacklevel=2,
)


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
        keyword_boost_weight: float = 0.5,
        confidence_threshold: float = 0.3,
        semantic_coherence_check: bool = False,
        adaptive_retrieval: bool = False,
        adaptive_k_factor: float = 0.3,  # Added parameter to control adaptive K threshold
        use_two_stage_retrieval: bool = False,  # New parameter for two-stage retrieval
        first_stage_k: int = 20,  # Number of candidates to retrieve in first stage
        query_type_adaptation: bool = False,  # New parameter for query type adaptation
        dynamic_threshold_adjustment: bool = False,  # New parameter for dynamic threshold adjustment
        threshold_adjustment_window: int = 5,  # Window size for dynamic threshold adjustment
        min_confidence_threshold: float = 0.1,  # Minimum confidence threshold
        max_confidence_threshold: float = 0.7,  # Maximum confidence threshold
        memory_decay_enabled: bool = False,  # Whether to apply memory decay
        memory_decay_rate: float = 0.99,  # Rate at which memory activations decay
        memory_decay_interval: int = 10,  # Apply decay every N interactions
        nlp_model_name: str = "en_core_web_sm",  # spaCy model name for NLP extraction
        # New parameters for enhanced retrieval
        personal_query_threshold: float = 0.5,  # Higher threshold for personal queries
        factual_query_threshold: float = 0.2,  # Lower threshold for factual queries
        min_results_guarantee: int = 1,  # Minimum number of results to guarantee
        enable_keyword_expansion: bool = False,  # Whether to expand keywords for factual queries
    ):
        """
        Initialize the contextual retriever.

        Args:
            memory: The contextual memory to retrieve from
            embedding_model: Model for encoding queries
            retrieval_strategy: Strategy for retrieval ('similarity', 'temporal', 'hybrid')
            recency_weight: Weight given to recency in hybrid retrieval
            relevance_weight: Weight given to relevance in hybrid retrieval
            keyword_boost_weight: Weight given to keyword matching in retrieval
            confidence_threshold: Minimum similarity score for retrieved memories
            semantic_coherence_check: Whether to check for semantic coherence among retrieved memories
            adaptive_retrieval: Whether to adaptively determine number of results to return
            adaptive_k_factor: Controls conservativeness of adaptive retrieval
            use_two_stage_retrieval: Whether to use two-stage retrieval pipeline
            first_stage_k: Number of candidates to retrieve in first stage
            query_type_adaptation: Whether to adapt retrieval based on query type
            dynamic_threshold_adjustment: Whether to adjust thresholds dynamically
            threshold_adjustment_window: Window size for dynamic threshold adjustment
            min_confidence_threshold: Minimum confidence threshold
            max_confidence_threshold: Maximum confidence threshold
            memory_decay_enabled: Whether to apply memory decay
            memory_decay_rate: Rate at which memory activations decay
            memory_decay_interval: Apply decay every N interactions
            nlp_model_name: spaCy model name for NLP extraction
            personal_query_threshold: Threshold for personal queries
            factual_query_threshold: Threshold for factual queries
            min_results_guarantee: Minimum number of results to guarantee
            enable_keyword_expansion: Whether to expand keywords for factual queries
        """  # noqa: W505
        self.memory = memory
        self.embedding_model = embedding_model
        self.retrieval_strategy = retrieval_strategy
        self.recency_weight = recency_weight
        self.relevance_weight = relevance_weight
        self.keyword_boost_weight = keyword_boost_weight
        self.confidence_threshold = confidence_threshold
        self.semantic_coherence_check = semantic_coherence_check
        self.adaptive_retrieval = adaptive_retrieval
        self.adaptive_k_factor = (
            adaptive_k_factor  # Controls conservativeness of adaptive retrieval
        )

        # New parameters for enhanced retrieval
        self.use_two_stage_retrieval = use_two_stage_retrieval
        self.first_stage_k = first_stage_k
        self.query_type_adaptation = query_type_adaptation
        self.personal_query_threshold = personal_query_threshold
        self.factual_query_threshold = factual_query_threshold
        self.min_results_guarantee = min_results_guarantee
        self.enable_keyword_expansion = enable_keyword_expansion

        # Dynamic threshold adjustment parameters
        self.dynamic_threshold_adjustment = dynamic_threshold_adjustment
        self.threshold_adjustment_window = threshold_adjustment_window
        self.min_confidence_threshold = min_confidence_threshold
        self.max_confidence_threshold = max_confidence_threshold
        self.recent_retrieval_metrics = []  # Store recent retrieval metrics

        # Memory decay parameters
        self.memory_decay_enabled = memory_decay_enabled
        self.memory_decay_rate = memory_decay_rate
        self.memory_decay_interval = memory_decay_interval
        self.interaction_count = 0

        # Initialize NLP extractor for enhanced extraction
        self.nlp_extractor = NLPExtractor(model_name=nlp_model_name)

        # Conversation context tracking
        self.conversation_state = {
            "recent_topics": [],
            "user_interests": set(),
            "interaction_count": 0,
        }

        # Persistent personal attributes dictionary
        self.personal_attributes = {
            "preferences": {},  # e.g., {"color": "blue", "food": "pizza"}
            "demographics": {},  # e.g., {"location": "Seattle", "occupation": "engineer"}
            "traits": {},  # e.g., {"personality": "introvert", "hobbies": ["hiking", "reading"]}
            "relationships": {},  # e.g., {"family": {"spouse": "Alex", "children": ["Sam", "Jamie"]}}
        }

        # Pre-compile regex patterns for performance
        self._compile_regex_patterns()

        # Patterns for detecting factual queries
        self.factual_query_patterns = [
            re.compile(r"what is|who is|where is|when is|why is|how is", re.IGNORECASE),
            re.compile(r"what are|who are|where are|when are|why are|how are", re.IGNORECASE),
            re.compile(r"tell me about|explain|describe", re.IGNORECASE),
            re.compile(r"capital of|author of|wrote|invented|discovered", re.IGNORECASE),
        ]

    def _is_factual_query(self, query: str) -> bool:
        """
        Determine if a query is factual/general knowledge rather than personal.

        Args:
            query: The query text

        Returns:
            True if the query appears to be factual, False otherwise
        """
        # Check if query matches factual patterns
        for pattern in self.factual_query_patterns:
            if pattern.search(query):
                # Check if it's not a personal query (doesn't contain "my", "I", etc.)
                personal_indicators = ["my", " i ", "i'm", "i've", "i'll", "i'd", "me", "mine"]
                if not any(indicator in query.lower() for indicator in personal_indicators):
                    return True

        return False

    def _adjust_threshold_dynamically(self, current_threshold: float) -> float:
        """
        Dynamically adjust the confidence threshold based on recent retrieval metrics.

        Args:
            current_threshold: Current confidence threshold

        Returns:
            Adjusted confidence threshold
        """
        if len(self.recent_retrieval_metrics) < self.threshold_adjustment_window:
            return current_threshold

        # Calculate average metrics over the window
        avg_retrieved = np.mean([m["retrieved_count"] for m in self.recent_retrieval_metrics])
        avg_similarity = np.mean([m["avg_similarity"] for m in self.recent_retrieval_metrics])

        # Adjust threshold based on metrics
        if avg_retrieved < 2:  # Too few memories being retrieved
            # Lower threshold to retrieve more memories
            new_threshold = max(self.min_confidence_threshold, current_threshold * 0.9)
        elif avg_retrieved > 8:  # Too many memories being retrieved
            # Raise threshold to be more selective
            new_threshold = min(self.max_confidence_threshold, current_threshold * 1.1)
        elif avg_similarity < 0.3:  # Low quality memories being retrieved
            # Raise threshold to improve quality
            new_threshold = min(self.max_confidence_threshold, current_threshold * 1.05)
        else:
            # Keep current threshold
            new_threshold = current_threshold

        return new_threshold

    def _track_retrieval_metrics(
        self, query_embedding: np.ndarray, retrieved_memories: list[dict]
    ) -> None:
        """
        Track retrieval metrics for dynamic threshold adjustment.

        Args:
            query_embedding: Query embedding
            retrieved_memories: Retrieved memories
        """
        # Extract memory IDs
        memory_ids = [
            m.get("memory_id") for m in retrieved_memories if isinstance(m.get("memory_id"), int)
        ]

        # Calculate average similarity
        if memory_ids:
            similarities = np.dot(self.memory.memory_embeddings[memory_ids], query_embedding)
            avg_similarity = float(np.mean(similarities))
        else:
            avg_similarity = 0.0

        # Store metrics
        metrics = {
            "retrieved_count": len(memory_ids),
            "avg_similarity": avg_similarity,
            "threshold_used": self.confidence_threshold,
        }

        # Add to recent metrics
        self.recent_retrieval_metrics.append(metrics)

        # Keep only the most recent metrics
        if len(self.recent_retrieval_metrics) > self.threshold_adjustment_window:
            self.recent_retrieval_metrics.pop(0)

    def _apply_memory_decay(self) -> None:
        """
        Apply exponential decay to memory activations.
        """
        self.interaction_count += 1

        # Apply decay every N interactions
        if self.interaction_count % self.memory_decay_interval == 0:
            # Apply exponential decay to all activations
            self.memory.activation_levels *= self.memory_decay_rate

            # Also decay category activations if using ART clustering
            if self.memory.use_art_clustering:
                self.memory.category_activations *= self.memory_decay_rate

    def retrieve_for_context(
        self,
        current_input: str,
        conversation_history: Optional[list[dict]] = None,
        top_k: int = 5,
        confidence_threshold: float = None,
    ) -> list[dict]:
        """
        Retrieve memories relevant to the current conversation context.

        Args:
            current_input: The current user input
            conversation_history: Recent conversation history
            top_k: Number of memories to retrieve
            confidence_threshold: Minimum similarity score for memory inclusion (overrides default)

        Returns:
            list of relevant memory entries with metadata
        """
        # Update conversation state
        self._update_conversation_state(current_input, conversation_history)

        # Encode the query context
        query_context = self._build_query_context(current_input, conversation_history)
        query_embedding = self.embedding_model.encode(
            query_context
        )  # Extract important keywords for direct reference matching
        important_keywords = self.nlp_extractor.extract_important_keywords(current_input)

        # Extract and update personal attributes if present in the input or response
        if conversation_history:
            for turn in conversation_history[-3:]:  # Look at recent turns
                message = turn.get("message", "")
                response = turn.get("response", "")
                extracted_attributes = self.nlp_extractor.extract_personal_attributes(message)
                self._update_personal_attributes(extracted_attributes)

                extracted_attributes = self.nlp_extractor.extract_personal_attributes(response)
                self._update_personal_attributes(extracted_attributes)

        # Also check current input for personal attributes
        extracted_attributes = self.nlp_extractor.extract_personal_attributes(current_input)
        self._update_personal_attributes(extracted_attributes)

        # If no confidence threshold is provided, use the default
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        # Determine query type and adjust parameters accordingly
        query_type, adjusted_params = self._adapt_to_query_type(current_input, confidence_threshold)
        # Extract and update personal attributes if present in the input or response
        if conversation_history:
            for turn in conversation_history[-3:]:  # Look at recent turns
                message = turn.get("message", "")
                response = turn.get("response", "")
                extracted_attributes = self.nlp_extractor.extract_personal_attributes(message)
                self._update_personal_attributes(extracted_attributes)

                extracted_attributes = self.nlp_extractor.extract_personal_attributes(response)
                self._update_personal_attributes(extracted_attributes)

        # Also check current input for personal attributes
        extracted_attributes = self.nlp_extractor.extract_personal_attributes(current_input)
        self._update_personal_attributes(extracted_attributes)

        # If no confidence threshold is provided, use the default
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        # Determine query type and adjust parameters accordingly
        query_type, adjusted_params = self._adapt_to_query_type(
            current_input, confidence_threshold
        )  # Use two-stage retrieval if enabled
        if self.use_two_stage_retrieval:
            memories = self._two_stage_retrieval(
                query_embedding, query_type, important_keywords, top_k, adjusted_params
            )
        else:
            # Use the original retrieval methods with adjusted parameters
            memories = self._single_stage_retrieval(
                query_embedding, query_type, important_keywords, top_k, adjusted_params
            )

        # Enhance results with personal attributes relevant to the query
        enhanced_memories = self._enhance_with_personal_attributes(memories, current_input)

        # Apply memory decay if enabled
        if self.memory_decay_enabled:
            self._apply_memory_decay()

        # Track retrieval metrics for dynamic threshold adjustment
        if self.dynamic_threshold_adjustment:
            self._track_retrieval_metrics(query_embedding, enhanced_memories)

        # Ensure we have at least min_results_guarantee results
        if len(enhanced_memories) < self.min_results_guarantee:
            # If we don't have enough results, try again with a lower threshold
            if len(memories) < self.min_results_guarantee:
                fallback_threshold = max(0.05, adjusted_params["confidence_threshold"] * 0.5)
                fallback_memories = self._retrieve_by_similarity(
                    query_embedding,
                    self.min_results_guarantee,
                    important_keywords,
                    fallback_threshold,
                )

                # Add any new memories that weren't already retrieved
                existing_ids = {
                    m.get("memory_id")
                    for m in enhanced_memories
                    if isinstance(m.get("memory_id"), int)
                }

                for memory in fallback_memories:
                    if memory["memory_id"] not in existing_ids:
                        enhanced_memories.append(memory)
                        if len(enhanced_memories) >= self.min_results_guarantee:
                            break

        return enhanced_memories

    def _two_stage_retrieval(
        self,
        query_embedding: np.ndarray,
        query_type: str,
        important_keywords: set,
        top_k: int,
        params: dict[str, Any],
    ) -> list[dict]:
        """
        Perform two-stage retrieval.

        Args:
            query_embedding: Query embedding
            query_type: Type of query (personal, factual, etc.)
            important_keywords: Important keywords from the query
            top_k: Number of results to return
            params: Adjusted parameters for this query type

        Returns:
            list of retrieved memories
        """
        # First stage: Retrieve a larger set of candidates with lower threshold
        first_stage_threshold = max(
            0.05, params["confidence_threshold"] * params.get("first_stage_threshold_factor", 0.7)
        )

        # Expand keywords for factual queries if enabled
        expanded_keywords = important_keywords
        if self.enable_keyword_expansion and query_type == "factual":
            expanded_keywords = self._expand_keywords(important_keywords)

        # Retrieve candidates using the appropriate strategy
        if self.retrieval_strategy == "similarity":
            candidates = self._retrieve_by_similarity(
                query_embedding, self.first_stage_k, expanded_keywords, first_stage_threshold
            )
        elif self.retrieval_strategy == "temporal":
            candidates = self._retrieve_by_recency(self.first_stage_k)
        else:  # hybrid approach
            candidates = self._retrieve_hybrid(
                query_embedding, self.first_stage_k, expanded_keywords, first_stage_threshold
            )

        # Second stage: Re-rank candidates
        # Sort by relevance score (already includes keyword boosting)
        candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Apply semantic coherence check if enabled
        if self.semantic_coherence_check and len(candidates) > 1:
            # Convert to proper format for memory's _apply_coherence_check
            candidate_tuples = [
                (
                    c["memory_id"],
                    c["relevance_score"],
                    {k: v for k, v in c.items() if k not in ["memory_id", "relevance_score"]},
                )
                for c in candidates
            ]

            coherent_tuples = self.memory._apply_coherence_check(candidate_tuples, query_embedding)

            # Convert back to our format
            coherent_candidates = []
            for memory_id, score, metadata in coherent_tuples:
                coherent_candidates.append(
                    {
                        "memory_id": memory_id,
                        "relevance_score": score,
                        **metadata,
                    }
                )

            candidates = coherent_candidates

        # Apply adaptive k selection if enabled
        if self.adaptive_retrieval and len(candidates) > 1:
            # Use the adjusted adaptive_k_factor
            adaptive_k_factor = params.get("adaptive_k_factor", self.adaptive_k_factor)
            scores = np.array([c["relevance_score"] for c in candidates])
            diffs = np.diff(scores)

            # Find significant drops
            significance_threshold = adaptive_k_factor * scores[0]
            significant_drops = np.where((-diffs) > significance_threshold)[0]

            if len(significant_drops) > 0:
                # Use the first significant drop as the cut point
                cut_idx = significant_drops[0] + 1
                candidates = candidates[:cut_idx]

        # Take top_k from the re-ranked candidates
        return candidates[:top_k]

    def _single_stage_retrieval(
        self,
        query_embedding: np.ndarray,
        query_type: str,
        important_keywords: set,
        top_k: int,
        params: dict[str, Any],
    ) -> list[dict]:
        """
        Perform single-stage retrieval.

        Args:
            query_embedding: Query embedding
            query_type: Type of query (personal, factual, etc.)
            important_keywords: Important keywords from the query
            top_k: Number of results to return
            params: Adjusted parameters for this query type

        Returns:
            list of retrieved memories
        """
        confidence_threshold = params.get("confidence_threshold", self.confidence_threshold)
        adaptive_k_factor = params.get("adaptive_k_factor", self.adaptive_k_factor)

        # Expand keywords for factual queries if enabled
        expanded_keywords = important_keywords
        if self.enable_keyword_expansion and query_type == "factual":
            expanded_keywords = self._expand_keywords(important_keywords)

        # Use the appropriate retrieval strategy
        if self.retrieval_strategy == "similarity":
            memories = self._retrieve_by_similarity(
                query_embedding, top_k, expanded_keywords, confidence_threshold
            )
        elif self.retrieval_strategy == "temporal":
            memories = self._retrieve_by_recency(top_k)
        else:  # hybrid approach
            memories = self._retrieve_hybrid(
                query_embedding,
                top_k,
                expanded_keywords,
                confidence_threshold,
                adaptive_k_factor=adaptive_k_factor,
            )

        return memories

    def _expand_keywords(self, keywords: set[str]) -> set[str]:
        """
        Expand keywords with related terms for better recall.

        Args:
            keywords: Original keywords

        Returns:
            Expanded set of keywords
        """
        expanded = set(keywords)

        # Add singular/plural forms safely
        plural_exceptions = {
            "child": "children",
            "person": "people",
            "man": "men",
            "woman": "women",
        }
        for keyword in list(
            keywords
        ):  # Iterate over a copy to avoid modifying the set while iterating
            if keyword in plural_exceptions:
                expanded.add(plural_exceptions[keyword])  # Handle known exceptions
            elif not keyword.endswith("s"):
                expanded.add(f"{keyword}s")  # Simple pluralization
            elif keyword.endswith("s") and len(keyword) > 1:
                expanded.add(keyword[:-1])  # Simple singularization

        # Add common synonyms for certain terms
        synonym_map = {
            "job": ["occupation", "profession", "career", "work", "employment"],
            "home": ["house", "apartment", "residence", "place"],
            "car": ["vehicle", "automobile", "ride"],
            "color": ["colour", "hue", "shade"],
            "food": ["meal", "dish", "cuisine"],
            "hobby": ["pastime", "activity", "interest"],
        }

        for keyword in list(keywords):  # Safe iteration over a copy
            if keyword in synonym_map:
                expanded.update(synonym_map[keyword])

        return expanded

    def _adapt_to_query_type(self, query: str, base_threshold: float) -> tuple[str, dict[str, Any]]:
        """
        Adapt retrieval parameters based on query type.

        Args:
            query: The query text
            base_threshold: Base confidence threshold

        Returns:
            tuple of (query_type, adjusted_parameters)
        """
        if not self.query_type_adaptation:
            return "default", {
                "confidence_threshold": base_threshold,
                "adaptive_k_factor": self.adaptive_k_factor,
            }

        # Use NLP extractor to identify query type
        query_types = self.nlp_extractor.identify_query_type(query)
        primary_type = max(query_types.items(), key=lambda x: x[1])[0]

        # Adjust parameters based on query type
        if primary_type == "personal":
            # Personal queries need higher thresholds for better precision
            return (
                "personal",
                {
                    "confidence_threshold": self.personal_query_threshold,
                    "adaptive_k_factor": self.adaptive_k_factor
                    * 1.2,  # More conservative for personal queries
                    "first_stage_threshold_factor": 0.8,  # Less aggressive first stage for personal queries
                },
            )
        elif primary_type == "factual":
            # Factual queries need lower thresholds for better recall
            return (
                "factual",
                {
                    "confidence_threshold": self.factual_query_threshold,
                    "adaptive_k_factor": self.adaptive_k_factor
                    * 0.5,  # Less conservative for factual queries
                    "first_stage_threshold_factor": 0.6,  # More aggressive first stage for factual queries
                },
            )
        else:
            # Default to a balanced approach for other query types
            return primary_type, {
                "confidence_threshold": base_threshold,
                "adaptive_k_factor": self.adaptive_k_factor,
                "first_stage_threshold_factor": 0.7,  # Default first stage factor
            }

    def _two_stage_retrieval(
        self,
        query_embedding: np.ndarray,
        query_type: str,
        important_keywords: set[str],
        top_k: int,
        params: dict[str, Any],
    ) -> list[dict]:
        """
        Perform two-stage retrieval.

        Args:
            query_embedding: Query embedding
            query_type: Type of query (personal, factual, etc.)
            important_keywords: Important keywords from the query
            top_k: Number of results to return
            params: Adjusted parameters for this query type

        Returns:
            list of retrieved memories
        """
        # First stage: Retrieve a larger set of candidates with lower threshold
        first_stage_threshold = max(
            0.05, params["confidence_threshold"] * params.get("first_stage_threshold_factor", 0.7)
        )

        # Expand keywords for factual queries if enabled
        expanded_keywords = important_keywords
        if self.enable_keyword_expansion and query_type == "factual":
            expanded_keywords = self._expand_keywords(important_keywords)

        # Retrieve candidates using the appropriate strategy
        if self.retrieval_strategy == "similarity":
            candidates = self._retrieve_by_similarity(
                query_embedding, self.first_stage_k, expanded_keywords, first_stage_threshold
            )
        elif self.retrieval_strategy == "temporal":
            candidates = self._retrieve_by_recency(self.first_stage_k)
        else:  # hybrid approach
            candidates = self._retrieve_hybrid(
                query_embedding, self.first_stage_k, expanded_keywords, first_stage_threshold
            )

        # Second stage: Re-rank candidates
        # Sort by relevance score (already includes keyword boosting)
        candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Apply semantic coherence check if enabled
        if self.semantic_coherence_check and len(candidates) > 1:
            # Convert to proper format for memory's _apply_coherence_check
            candidate_tuples = [
                (
                    c["memory_id"],
                    c["relevance_score"],
                    {k: v for k, v in c.items() if k not in ["memory_id", "relevance_score"]},
                )
                for c in candidates
            ]

            coherent_tuples = self.memory._apply_coherence_check(candidate_tuples, query_embedding)

            # Convert back to our format
            coherent_candidates = []
            for memory_id, score, metadata in coherent_tuples:
                coherent_candidates.append(
                    {
                        "memory_id": memory_id,
                        "relevance_score": score,
                        **metadata,
                    }
                )

            candidates = coherent_candidates

        # Apply adaptive k selection if enabled
        if self.adaptive_retrieval and len(candidates) > 1:
            # Use the adjusted adaptive_k_factor
            adaptive_k_factor = params.get("adaptive_k_factor", self.adaptive_k_factor)
            scores = np.array([c["relevance_score"] for c in candidates])
            diffs = np.diff(scores)

            # Find significant drops
            significance_threshold = adaptive_k_factor * scores[0]
            significant_drops = np.where((-diffs) > significance_threshold)[0]

            if len(significant_drops) > 0:
                # Use the first significant drop as the cut point
                cut_idx = significant_drops[0] + 1
                candidates = candidates[:cut_idx]

        # Take top_k from the re-ranked candidates
        return candidates[:top_k]

    def _single_stage_retrieval(
        self,
        query_embedding: np.ndarray,
        query_type: str,
        important_keywords: set[str],
        top_k: int,
        params: dict[str, Any],
    ) -> list[dict]:
        """
        Perform single-stage retrieval.

        Args:
            query_embedding: Query embedding
            query_type: Type of query (personal, factual, etc.)
            important_keywords: Important keywords from the query
            top_k: Number of results to return
            params: Adjusted parameters for this query type

        Returns:
            list of retrieved memories
        """
        confidence_threshold = params.get("confidence_threshold", self.confidence_threshold)
        adaptive_k_factor = params.get("adaptive_k_factor", self.adaptive_k_factor)

        # Expand keywords for factual queries if enabled
        expanded_keywords = important_keywords
        if self.enable_keyword_expansion and query_type == "factual":
            expanded_keywords = self._expand_keywords(important_keywords)

        # Use the appropriate retrieval strategy
        if self.retrieval_strategy == "similarity":
            memories = self._retrieve_by_similarity(
                query_embedding, top_k, expanded_keywords, confidence_threshold
            )
        elif self.retrieval_strategy == "temporal":
            memories = self._retrieve_by_recency(top_k)
        else:  # hybrid approach
            memories = self._retrieve_hybrid(
                query_embedding,
                top_k,
                expanded_keywords,
                confidence_threshold,
                adaptive_k_factor=adaptive_k_factor,
            )

        return memories

    def _expand_keywords(self, keywords: set[str]) -> set[str]:
        """
        Expand keywords with related terms for better recall.

        Args:
            keywords: Original keywords

        Returns:
            Expanded set of keywords
        """
        expanded = set(keywords)

        # Add singular/plural forms
        for keyword in keywords:
            # Simple pluralization (add 's')
            if not keyword.endswith("s"):
                expanded.add(f"{keyword}s")
            # Simple singularization (remove 's')
            elif keyword.endswith("s") and len(keyword) > 1:
                expanded.add(keyword[:-1])

        # Add common synonyms for certain terms
        synonym_map = {
            "job": ["occupation", "profession", "career", "work", "employment"],
            "home": ["house", "apartment", "residence", "place"],
            "car": ["vehicle", "automobile", "ride"],
            "color": ["colour", "hue", "shade"],
            "food": ["meal", "dish", "cuisine"],
            "hobby": ["pastime", "activity", "interest"],
        }

        for keyword in list(keywords):
            if keyword in synonym_map:
                expanded.update(synonym_map[keyword])

        return expanded

    def _update_conversation_state(
        self,
        current_input: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> None:
        """
        Update internal conversation state tracking.

        Args:
            current_input: The latest user input.
            conversation_history: A list of past conversation exchanges.
        """
        self.conversation_state["interaction_count"] += 1

        # Extract potential topics from current input (simplified example)
        potential_topics = current_input.split()[:5]
        self.conversation_state["recent_topics"] = potential_topics

        # Update user interests based on the last 3 exchanges
        if conversation_history:
            for exchange in conversation_history[-3:]:
                if "user" in exchange.get("speaker", "").lower():
                    words = exchange.get("message", "").split()
                    self.conversation_state["user_interests"].update(words[:3])

    def _update_personal_attributes(self, extracted_attributes: dict[str, Any]) -> None:
        """
        Update personal attributes with newly extracted attributes.

        Args:
            extracted_attributes: dictionary of extracted attributes
        """
        for category, items in extracted_attributes.items():
            if not items:
                continue

            if category not in self.personal_attributes:
                self.personal_attributes[category] = {}

            if isinstance(items, dict):
                for key, value in items.items():
                    self.personal_attributes[category][key] = value
            elif isinstance(items, list):
                if not isinstance(self.personal_attributes[category], list):
                    self.personal_attributes[category] = []
                for item in items:
                    if item not in self.personal_attributes[category]:
                        self.personal_attributes[category].append(item)
            else:
                self.personal_attributes[category] = items

    def _extract_personal_attributes(self, text: str) -> None:
        """
        Extract personal attributes from text and update the personal attributes dictionary.

        Args:
            text: Text to extract attributes from
        """
        if not text:
            return

        text_lower = text.lower()

        # Extract preferences
        self._extract_preferences(text_lower)

        # Extract demographic information
        self._extract_demographics(text_lower)

        # Extract traits and hobbies
        self._extract_traits(text_lower)

        # Extract relationships
        self._extract_relationships(text_lower)

    def _extract_preferences(self, text: str) -> None:
        """Extract user preferences from text."""
        # Process favorite patterns
        for pattern in self.favorite_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    category, value = match
                    self.personal_attributes["preferences"][category.strip()] = value.strip()
                elif isinstance(match, str):
                    # For the third pattern, try to categorize the preference
                    if "color" in text:
                        color_match = re.search(
                            r"(?:like|love|prefer|enjoy) (?:the color) ([a-z\s]+)", text
                        )
                        if color_match:
                            self.personal_attributes["preferences"]["color"] = color_match.group(
                                1
                            ).strip()
                    elif "food" in text or "eating" in text:
                        self.personal_attributes["preferences"]["food"] = match.strip()
                    elif "drink" in text or "drinking" in text:
                        self.personal_attributes["preferences"]["drink"] = match.strip()
                    elif "movie" in text or "watching" in text:
                        self.personal_attributes["preferences"]["movie"] = match.strip()
                    elif "book" in text or "reading" in text:
                        self.personal_attributes["preferences"]["book"] = match.strip()
                    elif "music" in text or "listening" in text:
                        self.personal_attributes["preferences"]["music"] = match.strip()

        # Direct statements about color preferences
        for pattern in self.color_patterns:
            matches = pattern.findall(text)
            for match in matches:
                self.personal_attributes["preferences"]["color"] = match.strip()

    def _extract_demographics(self, text: str) -> None:
        """Extract demographic information from text."""
        # Process location patterns
        for pattern in self.location_patterns:
            matches = pattern.findall(text)
            for match in matches:
                self.personal_attributes["demographics"]["location"] = match.strip()

        # Process occupation patterns
        for pattern in self.occupation_patterns:
            matches = pattern.findall(text)
            for match in matches:
                # Filter out common false positives
                if match.strip() not in ["bit", "lot", "fan", "user"]:
                    self.personal_attributes["demographics"]["occupation"] = match.strip()

    def _extract_traits(self, text: str) -> None:
        """Extract personality traits and hobbies from text."""
        # Process hobby patterns
        for pattern in self.hobby_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    activity, time = match
                    if time.strip() in ["weekends", "weekend", "evenings", "free time"]:
                        if "hobbies" not in self.personal_attributes["traits"]:
                            self.personal_attributes["traits"]["hobbies"] = []
                        # Add only if not already present
                        if activity.strip() not in self.personal_attributes["traits"]["hobbies"]:
                            self.personal_attributes["traits"]["hobbies"].append(activity.strip())
                elif isinstance(match, str):
                    if "hobbies" not in self.personal_attributes["traits"]:
                        self.personal_attributes["traits"]["hobbies"] = []
                    for hobby in match.split("and"):
                        hobby = hobby.strip().strip(",.")
                        if hobby and hobby not in self.personal_attributes["traits"]["hobbies"]:
                            self.personal_attributes["traits"]["hobbies"].append(hobby)

        # Check specifically for hiking in mountains on weekends
        if "hike" in text and "mountains" in text and "weekend" in text:
            if "hobbies" not in self.personal_attributes["traits"]:
                self.personal_attributes["traits"]["hobbies"] = []
            if "hiking in the mountains" not in self.personal_attributes["traits"]["hobbies"]:
                self.personal_attributes["traits"]["hobbies"].append("hiking in the mountains")

    def _extract_relationships(self, text: str) -> None:
        """Extract relationship information from text."""
        # Process family relationship patterns
        for pattern in self.family_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    relation, name = match
                    if "family" not in self.personal_attributes["relationships"]:
                        self.personal_attributes["relationships"]["family"] = {}
                    self.personal_attributes["relationships"]["family"][relation.strip()] = (
                        name.strip()
                    )

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

    def _extract_important_keywords(self, query: str) -> set[str]:
        """
        Extract important keywords for direct reference matching.

        Args:
            query: The user query

        Returns:
            set of important keywords
        """
        # Convert to lowercase for case-insensitive matching
        query_lower = query.lower()

        important_words = set()

        # Extract keywords from reference patterns
        for pattern in self.reference_patterns:
            matches = pattern.findall(query_lower)
            for match in matches:
                # Add each individual word from the match
                important_words.update(match.split())

        # Add specific preference and personal attribute keywords
        preference_terms = ["favorite", "like", "prefer", "love", "favorite color", "favorite food"]
        personal_terms = [
            "color",
            "food",
            "movie",
            "book",
            "hobby",
            "activity",
            "live",
            "work",
            "job",
            "occupation",
            "weekend",
        ]

        for term in preference_terms + personal_terms:
            if term in query_lower:
                important_words.add(term)

        # Filter out common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "was",
            "were",
            "be",
            "been",
            "being",
            "to",
            "of",
            "and",
            "or",
            "that",
            "this",
            "these",
            "those",
        }
        important_words = {word for word in important_words if word not in stop_words}

        return important_words

    def _calculate_keyword_boost(
        self, memory_metadata: dict, important_keywords: set[str]
    ) -> float:
        """
        Calculate a boost factor based on keyword matching between memory and important keywords.

        Args:
            memory_metadata: Metadata for a memory
            important_keywords: Important keywords from the query

        Returns:
            Boost factor (1.0 = no boost)
        """
        if not important_keywords:
            return 1.0

        # Combine relevant text fields from memory
        memory_text = ""

        # Check different text fields that might exist in the metadata
        for field in ["text", "content", "description", "name"]:
            if field in memory_metadata:
                memory_text += " " + str(memory_metadata[field]).lower()

        # For interaction type, check response field too
        if memory_metadata.get("type") == "interaction" and "response" in memory_metadata:
            memory_text += " " + str(memory_metadata["response"]).lower()

        # Count matching keywords
        matches = sum(1 for keyword in important_keywords if keyword in memory_text)

        # Calculate boost factor (more matches = higher boost)
        if matches > 0:
            # Exponential boost for multiple keyword matches
            boost = 1.0 + min(2.5, 0.7 * matches)  # More aggressive boosting
            return boost

        return 1.0

    def _enhance_with_personal_attributes(self, memories: list, query: str) -> list:
        """
        Enhance memory results with relevant personal attributes.

        Args:
            memories: Retrieved memories
            query: User query

        Returns:
            Enhanced memory list with personal attributes
        """
        # Clone the memories list to avoid modifying the original
        enhanced_memories = list(memories)

        # Check if the query is related to personal attributes
        query_lower = query.lower()

        # Check for relevant personal attributes based on query keywords
        relevant_attributes = {}

        # Extract keywords from query using NLP
        important_keywords = self.nlp_extractor.extract_important_keywords(query)

        # Check for preference-related queries
        preference_keywords = [
            "favorite",
            "like",
            "prefer",
            "love",
            "favorite color",
            "favorite food",
        ]
        if any(keyword in query_lower for keyword in preference_keywords):
            # Add all preferences
            for category, value in self.personal_attributes["preferences"].items():
                if category in query_lower or any(
                    keyword in query_lower for keyword in preference_keywords
                ):
                    relevant_attributes[f"preference_{category}"] = value

        # Check for location/demographic queries
        demographic_keywords = [
            "live",
            "location",
            "city",
            "town",
            "from",
            "work",
            "job",
            "occupation",
        ]
        if any(keyword in query_lower for keyword in demographic_keywords):
            # Add all demographics
            for category, value in self.personal_attributes["demographics"].items():
                if category in query_lower or any(
                    keyword in query_lower for keyword in demographic_keywords
                ):
                    relevant_attributes[f"demographic_{category}"] = value

        # Check for hobby/activity related queries
        activity_keywords = [
            "hobby",
            "hobbies",
            "activity",
            "enjoy",
            "weekend",
            "free time",
            "like to do",
        ]
        if any(keyword in query_lower for keyword in activity_keywords):
            # Add all hobbies/activities
            if "hobbies" in self.personal_attributes["traits"]:
                relevant_attributes["trait_hobbies"] = self.personal_attributes["traits"]["hobbies"]

        # Check for relationship queries
        relationship_keywords = ["family", "wife", "husband", "children", "spouse", "partner"]
        if any(keyword in query_lower for keyword in relationship_keywords):
            if "family" in self.personal_attributes["relationships"]:
                for relation, name in self.personal_attributes["relationships"]["family"].items():
                    if relation in query_lower or any(
                        keyword in query_lower for keyword in relationship_keywords
                    ):
                        relevant_attributes[f"relationship_{relation}"] = name

        # If we have relevant attributes, create a special "attribute memory" entry
        if relevant_attributes:
            # Create a special memory entry for personal attributes
            attribute_memory = {
                "memory_id": "personal_attributes",
                "type": "personal_attributes",
                "relevance_score": 10.0,  # Give it a high score to appear first
                "content": "User personal attributes",
                "attributes": relevant_attributes,
            }

            # Insert at the beginning for highest priority
            enhanced_memories.insert(0, attribute_memory)

        return enhanced_memories

    def _retrieve_by_similarity(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        important_keywords: set[str] = None,
        confidence_threshold: float = 0.0,
    ) -> list[dict]:
        """Retrieve memories based purely on contextual similarity."""
        # Pass the confidence_threshold and additional parameters for enhanced retrieval
        results = self.memory.retrieve_memories(
            query_embedding,
            top_k=top_k,
            activation_boost=True,
            confidence_threshold=confidence_threshold,
        )

        # Format results and apply keyword boost if needed
        formatted_results = []
        for idx, score, metadata in results:
            boost = 1.0
            if important_keywords:
                boost = self._calculate_keyword_boost(metadata, important_keywords)

            boosted_score = score * boost

            formatted_results.append(
                {
                    "memory_id": idx,
                    "relevance_score": boosted_score,
                    "original_score": score,
                    "keyword_boost": boost,
                    **metadata,
                }
            )

        # Re-sort by boosted score
        formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return formatted_results[:top_k]

    def _retrieve_by_recency(self, top_k: int) -> list[dict]:
        """Retrieve memories based on recency and activation."""
        # This is a simplified implementation
        # In a real system, this would be more sophisticated

        # Get memories sorted by temporal markers (most recent first)
        temporal_order = np.argsort(-self.memory.temporal_markers)[:top_k]

        results = []
        for idx in temporal_order:
            results.append(
                {
                    "memory_id": int(idx),
                    "relevance_score": float(self.memory.activation_levels[idx]),
                    **self.memory.memory_metadata[idx],
                }
            )

        return results

    def _retrieve_hybrid(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        important_keywords: set[str] = None,
        confidence_threshold: float = 0.0,
        adaptive_k_factor: float = None,
    ) -> list[dict]:
        """
        Hybrid retrieval combining similarity, recency, and keyword matching.
        """
        # Use the provided adaptive_k_factor or fall back to the instance variable
        if adaptive_k_factor is None:
            adaptive_k_factor = self.adaptive_k_factor

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

        # Apply confidence threshold filtering
        valid_indices = np.where(combined_scores >= confidence_threshold)[0]
        if len(valid_indices) == 0:
            return []

        # Get top-k indices from valid indices
        array_size = len(valid_indices)
        if top_k >= array_size:
            top_relative_indices = np.argsort(-combined_scores[valid_indices])
        else:
            # First get more candidates than needed for keyword boosting
            # Fix: Ensure candidate_k is less than array_size to avoid argpartition error
            candidate_k = min(array_size - 1, top_k * 2)  # Safely compute candidate_k
            if candidate_k <= 0:  # Additional safety check
                candidate_k = min(1, array_size - 1)

            candidate_indices = np.argpartition(-combined_scores[valid_indices], candidate_k)[
                :candidate_k
            ]

            # Convert back to original indices
            candidate_indices = valid_indices[candidate_indices]

            # Format preliminary results with potential for keyword boosting
            candidates = []
            for idx in candidate_indices:
                score = float(combined_scores[idx])
                if score <= 0:  # Skip non-positive scores
                    continue

                metadata = self.memory.memory_metadata[idx]
                boost = 1.0

                # Apply keyword boosting if needed
                if important_keywords:
                    boost = self._calculate_keyword_boost(metadata, important_keywords)
                    score = score * boost

                candidates.append(
                    {
                        "memory_id": int(idx),
                        "relevance_score": score,
                        "similarity": float(similarities[idx]),
                        "recency": float(temporal_factors[idx]),
                        "keyword_boost": boost,
                        **metadata,
                    }
                )

            # Sort by boosted score and take top-k
            candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Apply semantic coherence check if enabled
            if self.semantic_coherence_check and len(candidates) > 1:
                # Convert to proper format for memory's _apply_coherence_check
                candidate_tuples = [
                    (
                        c["memory_id"],
                        c["relevance_score"],
                        {k: v for k, v in c.items() if k not in ["memory_id", "relevance_score"]},
                    )
                    for c in candidates
                ]

                coherent_tuples = self.memory._apply_coherence_check(
                    candidate_tuples, query_embedding
                )

                # Convert back to our format
                coherent_candidates = []
                for memory_id, score, metadata in coherent_tuples:
                    coherent_candidates.append(
                        {
                            "memory_id": memory_id,
                            "relevance_score": score,
                            **metadata,
                        }
                    )

                candidates = coherent_candidates

            # Apply adaptive k selection if enabled
            if self.adaptive_retrieval and len(candidates) > 1:
                # Modified adaptive k selection algorithm - less conservative
                scores = np.array([c["relevance_score"] for c in candidates])
                diffs = np.diff(scores)

                # Find significant drops
                # Using adaptive_k_factor to control how conservative the algorithm is
                # Lower values = less conservative (returns more results)
                significance_threshold = adaptive_k_factor * scores[0]
                significant_drops = np.where((-diffs) > significance_threshold)[0]

                if len(significant_drops) > 0:
                    # Use the first significant drop as the cut point
                    cut_idx = significant_drops[0] + 1
                    candidates = candidates[:cut_idx]

            return candidates[:top_k]

        # Format results (if we didn't take the candidate path)
        results = []
        for idx in valid_indices[top_relative_indices]:
            score = float(combined_scores[idx])
            if score <= 0:  # Skip non-positive scores
                continue

            metadata = self.memory.memory_metadata[idx]
            boost = 1.0

            # Apply keyword boosting if needed
            if important_keywords:
                boost = self._calculate_keyword_boost(metadata, important_keywords)
                score = score * boost

            results.append(
                {
                    "memory_id": int(idx),
                    "relevance_score": score,
                    "similarity": float(similarities[idx]),
                    "recency": float(temporal_factors[idx]),
                    "keyword_boost": boost,
                    **metadata,
                }
            )

        # Re-sort by boosted score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return results[:top_k]

    def _compile_regex_patterns(self):
        """Pre-compile regex patterns for faster extraction."""
        # Preference patterns
        self.favorite_patterns = [
            re.compile(
                r"(?:my|I) (?:favorite|preferred) (color|food|drink|movie|book|music|song|artist|sport|game|place) (?:is|are) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"
            ),
            re.compile(
                r"(?:I|my) (?:like|love|prefer|enjoy) ([a-z0-9\s]+) (?:for|as) (?:my) (color|food|drink|movie|book|music|activity)"
            ),
            re.compile(
                r"(?:I|my) (?:like|love|prefer|enjoy) (?:the color|eating|drinking|watching|reading|listening to) ([a-z0-9\s]+)"
            ),
        ]

        self.color_patterns = [
            re.compile(r"(?:my|I) (?:favorite) color is ([a-z\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I|my) (?:like|love|prefer|enjoy) the color ([a-z\s]+)"),
        ]

        # Location patterns
        self.location_patterns = [
            re.compile(r"(?:I|my) (?:live|stay|reside) in ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I|my) (?:from|grew up in|was born in) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(
                r"(?:I|my) (?:city|town|state|country) (?:is) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"
            ),
        ]

        # Occupation patterns
        self.occupation_patterns = [
            re.compile(r"(?:I|my) (?:work as|am) (?:a|an) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(
                r"(?:I|my) (?:job|profession|occupation) (?:is) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"
            ),
        ]

        # Hobby patterns
        self.hobby_patterns = [
            re.compile(
                r"(?:I|my) (?:like to|love to|enjoy) ([a-z\s]+) (?:on|in|during) (?:my|the) ([a-z\s]+)(?:\.|\,|\!|\?|$)"
            ),
            re.compile(
                r"(?:I|my) (?:hobby|hobbies|pastime|activity) (?:is|are|include) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"
            ),
        ]

        # Family relationship patterns
        self.family_patterns = [
            re.compile(
                r"(?:my) (wife|husband|spouse|partner|girlfriend|boyfriend) (?:is|name is) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"
            ),
            re.compile(
                r"(?:my) (son|daughter|child|children|mother|father|brother|sister|sibling) (?:is|are|name is|names are) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"
            ),
        ]

        # Reference patterns for keyword extraction
        self.reference_patterns = [
            re.compile(r"what (?:was|is|were) (?:my|your|the) ([a-z\s]+)(?:\?|\.|$)"),
            re.compile(
                r"(?:did|do) (?:I|you) (?:mention|say|tell|share) (?:about|that) ([a-z\s]+)(?:\?|\.|$)"
            ),
            re.compile(r"(?:remind|tell) me (?:about|what) ([a-z\s]+)(?:\?|\.|$)"),
            re.compile(
                r"(?:what|which) ([a-z\s]+) (?:did|do) I (?:like|prefer|mention|say)(?:\?|\.|$)"
            ),
        ]


__all__ = ["ContextualRetriever"]
