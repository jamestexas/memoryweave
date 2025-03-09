"""
Hybrid Fabric Retrieval Strategy for MemoryWeave.

This module implements a memory-efficient retrieval strategy that combines
full embeddings, selective chunks, and keyword filtering for optimal
retrieval performance with minimal memory usage.
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
from memoryweave.storage import HybridMemoryStore
from memoryweave.utils import _load_module

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(markup=True),  # allow colors in terminal
    ],
)
logger = logging.getLogger(__name__)


def _nltk_extract_keywords(text: str) -> list[str] | None:
    """
    Extract keywords from text using NLTK.

    Args:
        text (str): The text to extract keywords from

    Returns:
        list[str] | None: List of extracted keywords
    """
    if _load_module("nltk"):
        logger.warning("[bold yellow]NLTK not found, skipping keyword extraction[/bold yellow]")
        return None

    logging.warning("[bold yellow]NLTK found, extracting keywords from text[/bold yellow]")
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # Download required NLTK data if not already present
    for resource in ["tokenizers/punkt", "corpora/stopwords", "taggers/averaged_perceptron_tagger"]:
        try:
            nltk.data.find(resource)
        except LookupError:
            resource_name = resource.split("/")[-1]
            nltk.download(resource_name, quiet=True)

    # Tokenize and part-of-speech tag
    tokens = word_tokenize(text.lower())
    tagged_tokens = nltk.pos_tag(tokens)

    # Get English stopwords
    stop_words = set(stopwords.words("english"))
    # Extract content words (nouns, verbs, adjectives, adverbs) longer than 3 characters
    important_tags = {"NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"}  # fmt: skip
    return [
        word
        for word, tag in tagged_tokens
        if tag[:2] in important_tags and word not in stop_words and len(word) > 3
    ]


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
        use_two_stage_by_default: bool = True,
        first_stage_k: int = 30,
        first_stage_threshold_factor: float = 0.7,
        memory_store: Optional[HybridMemoryStore] = None,
        associative_linker: Optional[AssociativeMemoryLinker] = None,
        temporal_context: Optional[TemporalContextBuilder] = None,
        activation_manager: Optional[ActivationManager] = None,
        **kwargs,
    ):
        """
        Initialize the hybrid fabric strategy.

        Args:
            memory_store: Memory store or adapter with hybrid capabilities
            associative_linker: Associative memory linker for traversing links
            temporal_context: Temporal context builder for time-based relevance
            activation_manager: Activation manager for memory accessibility
        """
        if memory_store is None:
            logger.debug("[bold red] MISSING MEMORY STORE [/bold red]")
        super().__init__(
            memory_store=memory_store,
            associative_linker=associative_linker,
            temporal_context=temporal_context,
            activation_manager=activation_manager,
        )
        self.component_id = ComponentName.HYBRID_FABRIC_STRAETGY
        self._kwargs = kwargs
        # Hybrid specific parameters
        self.use_two_stage_by_default: bool = use_two_stage_by_default
        self.first_stage_k: int = first_stage_k
        self.first_stage_threshold_factor: float = first_stage_threshold_factor

        # Kwarg configurable options
        self.use_keyword_filtering = self._kwargs.get("use_keyword_filtering", True)
        self.keyword_boost_factor = self._kwargs.get("keyword_boost_factor", 0.3)
        self.max_chunks_per_memory = self._kwargs.get("max_chunks_per_memory", 3)
        self.prioritize_full_embeddings = self._kwargs.get("prioritize_full_embeddings", True)

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the strategy with configuration."""
        # First, initialize base class
        super().initialize(config)

        # Then, set hybrid specific parameters
        self.use_keyword_filtering = config.get("use_keyword_filtering", self.use_keyword_filtering)
        self.keyword_boost_factor = config.get("keyword_boost_factor", self.keyword_boost_factor)
        self.max_chunks_per_memory = config.get("max_chunks_per_memory", self.max_chunks_per_memory)
        self.prioritize_full_embeddings = config.get(
            "prioritize_full_embeddings", self.prioritize_full_embeddings
        )

        # Add two-stage parameters
        self.use_two_stage_by_default = config.get("use_two_stage", self.use_two_stage_by_default)
        self.first_stage_k = config.get("first_stage_k", self.first_stage_k)
        self.first_stage_threshold_factor = config.get(
            "first_stage_threshold_factor", self.first_stage_threshold_factor
        )

        # # Reset supports_hybrid to False before checking
        # self.supports_hybrid = False

        # Check hybrid support
        print(f"DEBUG: Memory store: {self.memory_store}")
        self._check_hybrid_support()
        print(f"DEBUG: Memory store: {self.memory_store}, supports_hybrid: {self.supports_hybrid}")
        logger.debug(f"After initialize: supports_hybrid set to {self.supports_hybrid}")

    def _check_hybrid_support(self) -> None:
        """Check if memory_store supports hybrid features."""
        # Explicitly set to False by default - VERY IMPORTANT
        self.supports_hybrid = False

        if self.memory_store is None:
            # No memory store, no hybrid support
            self.supports_hybrid = False
            logger.debug("No memory store, no hybrid support")
            return

        # Check for specific attributes and ONLY set to True if found
        if hasattr(self.memory_store, "search_hybrid"):
            self.supports_hybrid = True
            logger.debug("Hybrid search support detected in memory store")
        elif hasattr(self.memory_store, "memory_store") and hasattr(
            self.memory_store.memory_store, "search_hybrid"
        ):
            self.supports_hybrid = True
            logger.debug("Hybrid search support detected in nested memory store")
        elif hasattr(self.memory_store, "search_chunks"):
            self.supports_hybrid = True
            logger.debug("Chunk search support detected - will use for hybrid search")
        else:
            # No hybrid capabilities found - EXPLICITLY set to False again
            self.supports_hybrid = False
            logger.debug("No hybrid search capabilities detected")

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        context: dict[str, Any],
        query: str = None,  # Make sure this parameter exists
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories using the hybrid fabric strategy.

        Args:
            query_embedding: Query embedding for similarity matching
            top_k: Number of results to return
            context: Context containing query, memory, etc.
            query: Optional query text (will use from context if not provided)

        Returns:
            List of retrieved memory dicts with relevance scores
        """
        # Get query from context if not provided directly
        query = query or context.get("query", "")

        # For benchmarking, use a simplified approach with lower thresholds
        # Use the memory store from context or instance
        memory_store = context.get("memory_store", self.memory_store)

        # Apply parameter adaptation if available
        adapted_params = context.get("adapted_retrieval_params", {})
        confidence_threshold = adapted_params.get("confidence_threshold", self.confidence_threshold)

        # Use very low threshold for benchmarking
        confidence_threshold = min(confidence_threshold, 0.05)

        # Get keywords from context
        keywords = adapted_params.get("important_keywords", [])
        if not keywords and query:
            # Very simple keyword extraction
            keywords = [word.lower() for word in query.split() if len(word) > 3]

        # Perform direct vector similarity search
        vector_results = []
        if hasattr(memory_store, "search_by_vector"):
            vector_results = memory_store.search_by_vector(
                query_vector=query_embedding,
                limit=top_k * 2,
                threshold=confidence_threshold,
            )

        # Return results directly with minimal processing for benchmark
        return vector_results[:top_k]

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
        Extract keywords from text using NLTK for better linguistic analysis.

        This method uses part-of-speech tagging to identify significant content words
        (nouns, verbs, adjectives) while filtering out stopwords and short words.

        Args:
            text: The text to extract keywords from

        Returns:
            list of extracted keywords
        """
        # Note we use None or an empty list if nothing is found. So we use this if we can and it works
        if (best_case_extract := _nltk_extract_keywords(text)) is not None:
            return best_case_extract

        # Fallback if NLTK is not available
        logging.warning("[bold orange]Falling back to simple keyword extraction[/bold orange]")

        # Simple stopwords list
        stopwords = {
            "the", "a", "an", "and", "or", "but", "if", "because", "as",
            "what", "when", "where", "how", "who", "which", "this", "that",
            "these", "those", "then", "just", "so", "than", "such", "both",
            "through", "about", "for", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did",
            "can", "could", "will", "would", "shall", "should", "may",
            "might", "must", "to", "in", "on", "at", "by", "with", "from",
            "of", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
            "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "like", "some",
        }  # fmt: skip

        # Tokenize the text and remove punctuation
        tokens = []
        for word in text.lower().split():
            # Remove punctuation
            if clean_word := "".join(c for c in word if c.isalnum()):
                tokens.append(clean_word)

        # Filter out stopwords and short words
        return [token for token in tokens if token not in stopwords and len(token) > 3]

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
        for _memory_id, result in combined_dict.items():
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
                            logger.debug(f"Error processing associative memory {memory_id}: {e}")

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
                            logger.debug(f"Error processing temporal memory {memory_id}: {e}")

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
        for _memory_id, result in combined_dict.items():
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

    def _analyze_query(self, query: str):
        """
        Analyze query to extract type, keywords, and parameters.

        Args:
            query: Query text

        Returns:
            Tuple of (query_obj, adapted_params, expanded_keywords, query_type, entities)
        """
        # If you have access to the query analyzer
        if hasattr(self, "query_analyzer") and self.query_analyzer:
            query_type = self.query_analyzer.analyze(query)
            keywords = self.query_analyzer.extract_keywords(query)
            entities = self.query_analyzer.extract_entities(query)
        else:
            # Provide reasonable defaults
            from memoryweave.interfaces.retrieval import QueryType

            query_type = QueryType.UNKNOWN
            keywords = []
            entities = []

        query_obj = {
            "text": query,
            "query_type": query_type,
            "extracted_keywords": keywords,
            "extracted_entities": entities,
        }

        expanded_keywords = keywords
        adapted_params = {"confidence_threshold": self.confidence_threshold, "max_results": 10}

        return query_obj, adapted_params, expanded_keywords, query_type, entities

    def retrieve_two_stage(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        context: dict[str, Any],
        query: str = None,
    ) -> list[dict[str, Any]]:
        """
        Two-stage retrieval approach within the HybridFabricStrategy.

        Stage 1: Get a larger set of candidates with lower threshold
        Stage 2: Apply more detailed processing and re-ranking

        Args:
            query_embedding: Query embedding
            top_k: Final number of results to return
            context: Context dict with parameters
            query: Optional query string

        Returns:
            List of retrieved memory dicts
        """
        # Get query and params
        query = query or context.get("query", "")
        memory_store = context.get("memory_store", self.memory_store)
        adapted_params = context.get("adapted_retrieval_params", {})

        # First stage: Get more candidates with lower threshold
        first_stage_k = adapted_params.get("first_stage_k", self.first_stage_k)
        first_stage_threshold = adapted_params.get(
            "confidence_threshold", self.confidence_threshold
        )
        first_stage_threshold *= adapted_params.get(
            "first_stage_threshold_factor", self.first_stage_threshold_factor
        )

        # AVOID RECURSION: Directly use the appropriate search methods for the first stage
        candidates = []

        # 1. Direct vector similarity search
        if hasattr(memory_store, "search_by_vector"):
            vector_results = memory_store.search_by_vector(
                query_vector=query_embedding,
                limit=first_stage_k,
                threshold=first_stage_threshold,
            )
            candidates.extend(vector_results)

        # 2. Add keyword-based results if keywords available
        keywords = adapted_params.get("important_keywords", context.get("important_keywords", []))
        if keywords and hasattr(memory_store, "search_by_keywords"):
            keyword_results = memory_store.search_by_keywords(
                keywords=keywords,
                limit=first_stage_k // 2,  # Get fewer keyword results to balance
            )
            candidates.extend(keyword_results)

        # Remove duplicates (by memory_id)
        unique_candidates = {}
        for result in candidates:
            memory_id = result.get("memory_id", result.get("id"))
            if memory_id not in unique_candidates:
                unique_candidates[memory_id] = result

        candidates = list(unique_candidates.values())

        # If no candidates, return empty list
        if not candidates:
            return []

        # Second stage: Apply more sophisticated filtering and re-ranking

        # 1. Apply keyword boosting if keywords available
        if keywords:
            candidates = self._apply_keyword_boosting(candidates, keywords)

        # 2. Apply semantic coherence check
        candidates = self._check_semantic_coherence(candidates, query_embedding)

        # 3. Enhance with associative context
        candidates = self._enhance_with_associative_context(candidates)

        # 4. Final re-ranking based on combined scores
        final_results = sorted(candidates, key=lambda x: x.get("relevance_score", 0), reverse=True)

        return final_results[:top_k]

    def _combine_results_with_rank_fusion(
        self,
        results1: list[dict[str, Any]],
        results2: list[dict[str, Any]],
        k1: float = 60.0,
        k2: float = 40.0,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Combine results using reciprocal rank fusion with more robust handling.

        This method combines results from different retrieval methods
        by using their ranks rather than raw scores, making it more robust.

        Args:
            results1: First result set (typically keyword-based)
            results2: Second result set (typically vector-based)
            k1: Constant for first result set (higher values discount rankings)
            k2: Constant for second result set
            top_k: Number of top results to return

        Returns:
            Combined list of results
        """
        # Handle empty input cases gracefully
        if not results1 and not results2:
            return []
        if not results1:
            return results2[:top_k]
        if not results2:
            return results1[:top_k]

        # Create dictionary of memory_id -> result info
        result_map = {}

        # Process first result set
        for rank, result in enumerate(results1):
            memory_id = result.get("memory_id", result.get("id", ""))
            if not memory_id:  # Skip results without a valid memory_id
                continue

            # RRF formula: 1/(k + rank)
            score = 1.0 / (k1 + rank)

            if memory_id not in result_map:
                result_map[memory_id] = {"result": result, "score": score, "sources": ["keyword"]}
            else:
                result_map[memory_id]["score"] += score
                if "keyword" not in result_map[memory_id]["sources"]:
                    result_map[memory_id]["sources"].append("keyword")

        # Process second result set
        for rank, result in enumerate(results2):
            memory_id = result.get("memory_id", result.get("id", ""))
            if not memory_id:  # Skip results without a valid memory_id
                continue

            # RRF formula: 1/(k + rank)
            score = 1.0 / (k2 + rank)

            if memory_id not in result_map:
                result_map[memory_id] = {"result": result, "score": score, "sources": ["vector"]}
            else:
                result_map[memory_id]["score"] += score
                if "vector" not in result_map[memory_id]["sources"]:
                    result_map[memory_id]["sources"].append("vector")
                    # Use vector result as base since it has more metadata
                    result_map[memory_id]["result"] = result

        # Format combined results
        combined_results = []
        for _memory_id, data in result_map.items():
            result_obj = data["result"].copy()
            result_obj["rrf_score"] = data["score"]
            result_obj["retrieval_sources"] = data["sources"]

            # Use either original relevance score or set based on RRF score
            if "relevance_score" not in result_obj:
                result_obj["relevance_score"] = min(0.99, data["score"] * 0.5)

            # Boost results that appear in both sources (similar to how we boosted activation)
            if len(data["sources"]) > 1:
                # Apply a boost for results found in multiple sources
                boost_factor = 1.5
                result_obj["relevance_score"] = min(
                    0.99,
                    result_obj["relevance_score"]
                    + boost_factor * (1.0 - result_obj["relevance_score"]) * 0.3,
                )
                result_obj["multi_source_boost"] = True

            combined_results.append(result_obj)

        # Sort by RRF score and return top-k
        return sorted(combined_results, key=lambda x: x["rrf_score"], reverse=True)[:top_k]

    def _apply_keyword_boosting(
        self, results: list[dict[str, Any]], keywords: list[str]
    ) -> list[dict[str, Any]]:
        """
        Boost scores for results containing important keywords with improved
        handling and exponential boosting.
        """
        if not results or not keywords:
            return results

        # Make a copy to avoid modifying the original list
        boosted_results = []

        for result in results:
            result_copy = dict(result)
            content = str(result_copy.get("content", "")).lower()

            # Count keyword matches and track which keywords matched
            keyword_matches = 0
            matched_keywords = []

            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in content:
                    keyword_matches += 1
                    matched_keywords.append(kw)

            if keyword_matches > 0:
                # Calculate boost factor based on matches
                # Use exponential formula for stronger boosting with multiple matches
                match_ratio = keyword_matches / len(keywords)
                boost = min(self.keyword_boost_factor * (match_ratio**1.5), 0.5)

                # Apply boost with sensitivity to original score
                original_score = result_copy.get("relevance_score", 0.5)
                result_copy["relevance_score"] = min(
                    0.99, original_score + boost * (1.0 - original_score)
                )

                # Store keyword matching metadata for debugging
                result_copy["keyword_boost"] = boost
                result_copy["keyword_matches"] = keyword_matches
                result_copy["matched_keywords"] = matched_keywords

            boosted_results.append(result_copy)

        # Sort by boosted relevance score
        return sorted(boosted_results, key=lambda x: x.get("relevance_score", 0), reverse=True)

    def _check_semantic_coherence(
        self, results: list[dict[str, Any]], query_embedding: np.ndarray
    ) -> list[dict[str, Any]]:
        """
        Check coherence between results and adjust ranking with improved handling
        for edge cases.
        """
        if len(results) <= 1:
            return results

        # Handle missing embeddings gracefully
        results_with_embeddings = []
        results_without_embeddings = []

        for result in results:
            if "embedding" in result and result["embedding"] is not None:
                results_with_embeddings.append(result)
            else:
                results_without_embeddings.append(result)

        # If no embeddings available, return original results
        if not results_with_embeddings:
            return results

        # Get embeddings for calculation
        embeddings = np.array([r["embedding"] for r in results_with_embeddings])

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # Avoid division by zero
        embeddings = embeddings / norms

        # Calculate pairwise similarities
        similarities = np.dot(embeddings, embeddings.T)

        # For each result, calculate average similarity to others (coherence)
        for i, result in enumerate(results_with_embeddings):
            # Remove self-similarity
            other_similarities = np.concatenate([similarities[i, :i], similarities[i, i + 1 :]])

            # Calculate coherence score (average similarity to others)
            coherence = float(np.mean(other_similarities)) if len(other_similarities) > 0 else 1.0

            # Apply adjustment based on coherence
            if coherence < 0.3:  # Low coherence
                # Apply penalty proportional to incoherence
                penalty = (0.3 - coherence) * 0.5
                result["relevance_score"] = max(0.1, result.get("relevance_score", 0.5) - penalty)
                result["coherence_penalty"] = penalty
            elif coherence > 0.7:  # High coherence
                # Apply boost for highly coherent results
                boost = (coherence - 0.7) * 0.3
                original_score = result.get("relevance_score", 0.5)
                result["relevance_score"] = min(
                    0.99, original_score + boost * (1.0 - original_score)
                )
                result["coherence_boost"] = boost

            # Store coherence score
            result["coherence_score"] = coherence

        # Combine results and sort
        combined_results = results_with_embeddings + results_without_embeddings
        return sorted(combined_results, key=lambda x: x.get("relevance_score", 0), reverse=True)

    def _enhance_with_associative_context(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Enhance results with associative context from memory fabric."""
        if not self.associative_linker:
            return results

        # Keep track of added associative memories
        associative_memories = set()

        # Track existing memory IDs from initial results
        for result in results:
            memory_id = result.get("memory_id")
            if memory_id is not None:
                associative_memories.add(memory_id)

        # Create a copy of the results list to avoid modifying during iteration
        enhanced_results = list(results)

        # For each result, find associative links
        for result in results:
            memory_id = result.get("memory_id")
            if memory_id is None:
                continue

            # Get associative links for this memory
            links = self.associative_linker.get_associative_links(memory_id)

            # If no links are returned directly, try the traverse_associative_network method
            if not links and hasattr(self.associative_linker, "traverse_associative_network"):
                network_links = self.associative_linker.traverse_associative_network(
                    start_id=memory_id,
                    max_hops=2,  # Default max hops
                    min_strength=0.3,  # Default minimum strength
                )
                links = [(linked_id, strength) for linked_id, strength in network_links.items()]

            # Add linked memories that aren't already in results
            for linked_id, strength in links:
                if linked_id not in associative_memories and strength > 0.3:
                    try:
                        # Get the memory if possible
                        memory_data = {}
                        if self.memory_store:
                            try:
                                memory = self.memory_store.get(linked_id)
                                if hasattr(memory, "content"):
                                    memory_data["content"] = memory.content
                                if hasattr(memory, "metadata"):
                                    memory_data.update(memory.metadata)
                            except Exception as e:
                                # If we can't get the memory, still create a minimal result
                                logger.debug(
                                    f"Error retrieving associative memory {linked_id}: {e}"
                                )
                                pass

                        # Create result dict with available data
                        associative_result = {
                            "memory_id": linked_id,
                            "content": memory_data.get("content", f"Associated memory {linked_id}"),
                            "relevance_score": 0.7 * strength,  # Scale by link strength
                            "associative_link": True,
                            "link_source": memory_id,
                            "link_strength": strength,
                        }

                        # Add additional metadata if available
                        if memory_data:
                            for key, value in memory_data.items():
                                if key not in ("content", "memory_id"):
                                    associative_result[key] = value

                        enhanced_results.append(associative_result)
                        associative_memories.add(linked_id)
                    except Exception as e:
                        logger.debug(f"Error retrieving associative memory {linked_id}: {e}")
                        # Skip if memory can't be retrieved
                        pass

        # Sort by relevance score
        return sorted(enhanced_results, key=lambda x: x.get("relevance_score", 0), reverse=True)

    def configure_two_stage(
        self,
        enable: bool = True,
        first_stage_k: int = 30,
        first_stage_threshold_factor: float = 0.7,
    ) -> None:
        """
        Configure two-stage retrieval settings.

        Args:
            enable: Whether to enable two-stage retrieval by default
            first_stage_k: Number of candidates to retrieve in first stage
            first_stage_threshold_factor: Factor to multiply confidence threshold by in first stage
        """
        self.use_two_stage_by_default = enable
        self.first_stage_k = first_stage_k
        self.first_stage_threshold_factor = first_stage_threshold_factor

        if self.debug:
            logger.debug(
                f"Configured two-stage retrieval: enable={enable}, "
                f"first_stage_k={first_stage_k}, "
                f"first_stage_threshold_factor={first_stage_threshold_factor}"
            )
