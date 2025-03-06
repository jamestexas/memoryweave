"""Test fixtures for consistent test setup and verification.

This module provides fixtures for creating test data, verifying results,
and setting up test environments in a consistent way.
"""

import hashlib
from typing import Any, Union, dict, list, tuple

import numpy as np

from memoryweave.components.retrieval_strategies import (
    HybridRetrievalStrategy,
    SimilarityRetrievalStrategy,
    TwoStageRetrievalStrategy,
)
from memoryweave.core.contextual_memory import ContextualMemory


class PredictableTestEmbeddings:
    """Class providing predictable test embeddings for specific queries.

    This ensures that tests using these embeddings will have consistent
    behavior and pass reliably.
    """

    @staticmethod
    def cat_query(dim: int = 768) -> np.ndarray:
        """Create a cat-related query embedding.

        This embedding is designed to have high similarity with
        cat-related content in test memories.
        """
        embedding = np.zeros(dim)

        # Pattern that will have high similarity with cat content
        embedding[0] = 0.8  # Primary signal for "cat"
        embedding[1 % dim] = 0.3  # Secondary signal

        # Add some noise to make it realistic
        for i in range(min(10, dim)):
            noise_idx = (i + 2) % dim
            embedding[noise_idx] = 0.1 * ((i % 5) / 10)

        # Normalize
        return embedding / np.linalg.norm(embedding)

    @staticmethod
    def color_query(dim: int = 768) -> np.ndarray:
        """Create a color-related query embedding."""
        embedding = np.zeros(dim)

        # Pattern for color content
        embedding[1 % dim] = 0.7
        embedding[2 % dim] = 0.4

        # Add some noise to make it realistic but work with small dimensions
        for i in range(min(3, dim)):
            noise_idx = (i + 1) % dim
            embedding[noise_idx] += 0.1 * ((i % 3) / 10)

        # Normalize
        return embedding / np.linalg.norm(embedding)

    @staticmethod
    def weather_query(dim: int = 768) -> np.ndarray:
        """Create a weather-related query embedding."""
        embedding = np.zeros(dim)

        # Pattern for weather content
        embedding[2 % dim] = 0.7
        embedding[3 % dim] = 0.4

        # Add some noise to make it realistic but work with small dimensions
        for i in range(min(3, dim)):
            noise_idx = (i + 2) % dim
            embedding[noise_idx] += 0.1 * ((i % 4) / 10)

        # Normalize
        return embedding / np.linalg.norm(embedding)


def create_test_embedding(text: str, dim: int = 768) -> np.ndarray:
    """Create a deterministic test embedding from text.

    Args:
        text: Text to encode into an embedding
        dim: Dimension of the embedding (works with any dimension)

    Returns:
        A numpy array with a normalized embedding
    """
    # Create deterministic embedding from text content
    embedding = np.zeros(dim)

    # For very small dimensions, create simple pattern
    if dim <= 4:
        # Use a simple but deterministic pattern for tiny embeddings
        for i, char in enumerate(text[:4]):
            char_val = ord(char) / 1000
            embedding[i % dim] += char_val

        # Ensure we have some signal
        if np.sum(embedding) < 0.1:
            # Use the first character of text to create a pattern
            if text:
                idx = ord(text[0]) % dim
                embedding[idx] = 0.8
                embedding[(idx + 1) % dim] = 0.3
            else:
                # Fallback pattern
                embedding[0] = 0.7
                embedding[1 % dim] = 0.3
    else:
        # For normal dimensions, use more complex pattern
        # Use text characteristics for basic embedding pattern
        for i, char in enumerate(text[: min(10, dim)]):
            embedding[i % dim] += ord(char) / 1000

        # Use hash of text for additional patterns
        text_hash = hashlib.md5(text.encode()).digest()  # noqa: S324
        for _i, byte in enumerate(text_hash):
            pos = byte % dim
            embedding[pos] += byte / 256

    # Normalize the embedding
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    else:
        # Fallback for zero vectors
        embedding[0] = 1.0

    return embedding


def create_test_memories(
    num_memories: int = 10, embedding_dim: int = 768
) -> tuple[list[np.ndarray], list[str], list[dict[str, Any]]]:
    """Create test memories with deterministic patterns.

    Args:
        num_memories: Number of memories to create
        embedding_dim: Dimension of the memory embeddings

    Returns:
        tuple of (embeddings, texts, metadata)
    """
    embeddings = []
    texts = []
    metadata = []

    # Create memories with predictable patterns
    for i in range(num_memories):
        # Create text with clear category patterns
        if i % 3 == 0:
            # Personal memories about preferences
            text = f"Memory {i}: I like the color {['blue', 'red', 'green'][i % 3]} and enjoy {['reading', 'hiking', 'cooking'][i % 3]}."
            category = "personal"
        elif i % 3 == 1:
            # Factual memories
            text = f"Memory {i}: The capital of {['France', 'Japan', 'Brazil'][i % 3]} is {['Paris', 'Tokyo', 'Brasília'][i % 3]}."
            category = "factual"
        else:
            # Memories about events
            text = f"Memory {i}: Yesterday I {['went to the park', 'had dinner with friends', 'watched a movie'][i % 3]}."
            category = "event"

        # Create embedding from text
        embedding = create_test_embedding(text, embedding_dim)

        # Create metadata
        meta = {
            "text": text,
            "category": category,
            "importance": 0.5 + (i * 0.05),
            "created_at": 1672531200 + i * 3600,  # Jan 1, 2023 + i hours
            "source": "test",
            "index": i,  # Useful for verification
        }

        embeddings.append(embedding)
        texts.append(text)
        metadata.append(meta)

    return embeddings, texts, metadata


def create_test_memory(embedding_dim: int = 768) -> ContextualMemory:
    """Create a test memory with predictable patterns.

    Creates a ContextualMemory instance with test memories that have
    predictable patterns for cat-related content, colors, weather, etc.

    Args:
        embedding_dim: Dimension for memory embeddings

    Returns:
        Populated ContextualMemory instance
    """
    # Create memory instance
    memory = ContextualMemory(embedding_dim=embedding_dim)

    # Add cat-related memories
    cat_embedding = np.zeros(embedding_dim)
    cat_embedding[0] = 0.8
    cat_embedding[1] = 0.2
    cat_embedding = cat_embedding / np.linalg.norm(cat_embedding)

    memory.add_memory(
        cat_embedding,
        "My cat Whiskers loves to sleep on the couch.",
        {"type": "personal", "category": "pets", "content": "cat", "index": 0},
    )

    memory.add_memory(
        cat_embedding * 0.95,
        "I feed my cat twice a day with premium food.",
        {"type": "personal", "category": "pets", "content": "cat", "index": 1},
    )

    # Add color-related memories
    color_embedding = np.zeros(embedding_dim)
    # Use modulo to ensure indices are in bounds for small embedding dimensions
    color_idx1 = 1 % embedding_dim
    color_idx2 = 2 % embedding_dim
    color_embedding[color_idx1] = 0.7
    color_embedding[color_idx2] = 0.3
    color_embedding = color_embedding / np.linalg.norm(color_embedding)

    memory.add_memory(
        color_embedding,
        "My favorite color is blue, I like blue shirts.",
        {"type": "personal", "category": "preferences", "content": "color", "index": 2},
    )

    memory.add_memory(
        color_embedding * 0.9,
        "The sky was a beautiful blue color yesterday.",
        {"type": "factual", "category": "observations", "content": "color", "index": 3},
    )

    # Add weather-related memories
    weather_embedding = np.zeros(embedding_dim)
    # Use modulo to ensure indices are in bounds for small embedding dimensions
    weather_idx1 = 2 % embedding_dim
    weather_idx2 = 3 % embedding_dim
    weather_embedding[weather_idx1] = 0.7
    weather_embedding[weather_idx2] = 0.3
    weather_embedding = weather_embedding / np.linalg.norm(weather_embedding)

    memory.add_memory(
        weather_embedding,
        "It was raining heavily in Seattle yesterday.",
        {"type": "factual", "category": "weather", "content": "weather", "index": 4},
    )

    memory.add_memory(
        weather_embedding * 0.9,
        "The forecast predicts sunny weather tomorrow.",
        {"type": "factual", "category": "weather", "content": "weather", "index": 5},
    )

    # Add general memories
    general_embedding = np.zeros(embedding_dim)
    # Use modulo to ensure indices are in bounds for small embedding dimensions
    general_idx1 = 0 % embedding_dim
    general_idx2 = 3 % embedding_dim
    general_embedding[general_idx1] = 0.6
    general_embedding[general_idx2] = 0.4
    general_embedding = general_embedding / np.linalg.norm(general_embedding)

    memory.add_memory(
        general_embedding,
        "The library has many interesting books.",
        {"type": "factual", "category": "general", "content": "general", "index": 6},
    )

    memory.add_memory(
        general_embedding * 0.9,
        "Coffee tastes best when freshly brewed.",
        {"type": "factual", "category": "food", "content": "general", "index": 7},
    )

    return memory


def create_retrieval_components(memory: ContextualMemory) -> dict[str, Any]:
    """Create a suite of retrieval components for testing.

    Args:
        memory: ContextualMemory instance to use for retrieval

    Returns:
        dictionary of retrieval components
    """
    # Create base retrieval strategies
    similarity_strategy = SimilarityRetrievalStrategy(memory)
    similarity_strategy.initialize(
        {
            "confidence_threshold": 0.3,
            "activation_boost": True,
            "min_results": 3,
        }
    )

    hybrid_strategy = HybridRetrievalStrategy(memory)
    hybrid_strategy.initialize(
        {
            "confidence_threshold": 0.3,
            "relevance_weight": 0.7,
            "recency_weight": 0.3,
        }
    )

    # Create mock post-processors
    keyword_processor = MockKeywordProcessor()
    keyword_processor.initialize({"keyword_boost_weight": 0.5})

    coherence_processor = MockCoherenceProcessor()
    coherence_processor.initialize({"coherence_threshold": 0.2})

    # Create two-stage strategies
    basic_two_stage = TwoStageRetrievalStrategy(
        memory, base_strategy=similarity_strategy, post_processors=[]
    )
    basic_two_stage.initialize(
        {
            "confidence_threshold": 0.3,
            "first_stage_k": 3,
            "first_stage_threshold_factor": 0.7,
        }
    )

    advanced_two_stage = TwoStageRetrievalStrategy(
        memory,
        base_strategy=hybrid_strategy,
        post_processors=[keyword_processor, coherence_processor],
    )
    advanced_two_stage.initialize(
        {
            "confidence_threshold": 0.3,
            "first_stage_k": 5,
            "first_stage_threshold_factor": 0.7,
        }
    )

    return {
        "similarity_strategy": similarity_strategy,
        "hybrid_strategy": hybrid_strategy,
        "keyword_processor": keyword_processor,
        "coherence_processor": coherence_processor,
        "basic_two_stage": basic_two_stage,
        "advanced_two_stage": advanced_two_stage,
    }


class MockKeywordProcessor:
    """Mock implementation of a keyword boost processor."""

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.keyword_boost_weight = config.get("keyword_boost_weight", 0.5)

    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process results by boosting keyword matches."""
        # Get keywords from context
        keywords = context.get("important_keywords", set())

        if not keywords:
            return results

        # Create new result list with modified scores
        processed_results = []

        for result in results:
            # Get content to check for keywords
            content = ""
            if "content" in result:
                content = result["content"].lower()
            elif "text" in result:
                content = result["text"].lower()

            # Check for keyword matches
            keyword_match = False
            for keyword in keywords:
                if keyword.lower() in content:
                    keyword_match = True
                    break

            # Create copy of result
            new_result = result.copy()

            # Apply boosting if keywords match
            if keyword_match:
                original_score = result.get("relevance_score", 0.0)
                new_score = original_score * (1.0 + self.keyword_boost_weight)
                new_result["relevance_score"] = min(1.0, new_score)
                new_result["keyword_boosted"] = True

            processed_results.append(new_result)

        return processed_results


class MockCoherenceProcessor:
    """Mock implementation of a semantic coherence processor."""

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.coherence_threshold = config.get("coherence_threshold", 0.2)

    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process results by filtering and adjusting for coherence."""
        # Check if semantic coherence is enabled
        if not context.get("enable_semantic_coherence", False):
            return results

        # Create new result list with coherence adjustments
        processed_results = []

        for result in results:
            # Create a copy of the result
            new_result = result.copy()

            # Add a fake coherence score
            content_type = result.get("type", "")
            query_type = context.get("primary_query_type", "")

            # Simulate coherence scoring
            coherence_score = 0.5
            if query_type and content_type:
                if query_type == content_type:
                    coherence_score = 0.8  # Higher score for matching types
                else:
                    coherence_score = 0.3  # Lower for mismatched types

            # Add coherence information
            new_result["coherence_score"] = coherence_score

            # Apply coherence penalty to relevance score if below threshold
            if coherence_score < self.coherence_threshold:
                original_score = result.get("relevance_score", 0.0)
                new_score = original_score * coherence_score
                new_result["relevance_score"] = new_score
                new_result["coherence_penalty_applied"] = True

            processed_results.append(new_result)

        return processed_results


def create_test_queries(num_queries: int = 3, embedding_dim: int = 768) -> list[dict[str, Any]]:
    """Create test queries with predictable patterns.

    Args:
        num_queries: Number of queries to create
        embedding_dim: Dimension of the query embeddings

    Returns:
        list of query dictionaries
    """
    queries = []

    query_texts = [
        "What's my favorite color?",
        "Tell me about the capital of France",
        "What did I do yesterday?",
    ]

    query_types = ["personal", "factual", "event"]

    # Expected relevant indices for each query
    # These correspond to the categories in create_test_memories
    relevant_indices = [
        [0, 3, 6],  # Personal memories
        [1, 4, 7],  # Factual memories
        [2, 5, 8],  # Event memories
    ]

    # Create query set
    for i in range(min(num_queries, len(query_texts))):
        query_text = query_texts[i]

        # Create embedding directly from text for consistency
        query_embedding = create_test_embedding(query_text, embedding_dim)

        # Add keywords for testing keyword expansion
        keywords = set(query_text.lower().replace("?", "").replace(".", "").split())

        # Create query object
        query = {
            "text": query_text,
            "embedding": query_embedding,
            "type": query_types[i],
            "relevant_indices": relevant_indices[i],
            "keywords": keywords,
        }

        queries.append(query)

    return queries


def verify_retrieval_results(
    results: list[dict[str, Any]],
    expected_content: Union[list[int], list[str]],
    require_all: bool = False,
    check_order: bool = False,
) -> Union[bool, tuple[bool, dict[str, Any]]]:
    """Verify retrieval results against expected content.

    This function can check for either:
    1. Specific memory indices in the results
    2. Content keywords in the retrieved memories

    Args:
        results: list of retrieval results
        expected_content: list of expected memory indices or content keywords
        require_all: Whether all expected content must be present
        check_order: Whether to check the order of results

    Returns:
        Either a boolean success indicator, or a tuple of (success, metrics)
    """
    # Check if we're matching indices or content keywords
    if expected_content and isinstance(expected_content[0], str):
        # We're matching content keywords (strings)
        success = _verify_content_keywords(results, expected_content, require_all)
        return success
    else:
        # We're matching memory indices (integers)
        return _verify_indices(results, expected_content, require_all, check_order)


def _verify_content_keywords(
    results: list[dict[str, Any]], expected_keywords: list[str], require_all: bool = False
) -> bool:
    """Check if result content contains the expected keywords.

    Args:
        results: list of retrieval results
        expected_keywords: Keywords to look for in content
        require_all: Whether all keywords must be found

    Returns:
        Boolean indicating success
    """
    if not results or not expected_keywords:
        return False

    # Extract content from results
    contents = []
    for r in results:
        if "content" in r:
            contents.append(r["content"].lower())
        elif "text" in r:
            contents.append(r["text"].lower())

    # Check if keywords are in any content
    found_keywords = set()
    for keyword in expected_keywords:
        for content in contents:
            if keyword.lower() in content:
                found_keywords.add(keyword)
                break

    # Success criteria
    if require_all:
        return len(found_keywords) == len(expected_keywords)
    else:
        return len(found_keywords) > 0


def _verify_indices(
    results: list[dict[str, Any]],
    expected_indices: list[int],
    require_all: bool = False,
    check_order: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Verify retrieval results against expected indices.

    Args:
        results: list of retrieval results
        expected_indices: list of expected memory indices
        require_all: Whether all expected indices must be present
        check_order: Whether to check the order of results

    Returns:
        tuple of (success, metrics)
    """
    # Extract indices from results
    retrieved_indices = []
    for r in results:
        # Get index from metadata
        if "index" in r:
            retrieved_indices.append(r["index"])
        elif "memory_id" in r and isinstance(r["memory_id"], int):
            # Direct memory_id as index
            retrieved_indices.append(r["memory_id"])

    # Convert to sets for comparison
    expected_set = set(expected_indices)
    retrieved_set = set(retrieved_indices)

    # Calculate metrics
    intersection = expected_set.intersection(retrieved_set)

    # Precision: what fraction of retrieved items are relevant
    precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0

    # Recall: what fraction of relevant items are retrieved
    recall = len(intersection) / len(expected_set) if expected_set else 1.0

    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Order check if requested
    order_correct = True
    if check_order and len(intersection) > 1:
        # Check if the order of expected items matches their order in retrieved
        expected_positions = {idx: i for i, idx in enumerate(expected_indices)}
        retrieved_order = [idx for idx in retrieved_indices if idx in expected_set]

        # Check if retrieved order preserves expected order
        for i in range(len(retrieved_order) - 1):
            idx1, idx2 = retrieved_order[i], retrieved_order[i + 1]
            if idx1 in expected_positions and idx2 in expected_positions:
                if expected_positions[idx1] > expected_positions[idx2]:
                    order_correct = False
                    break

    # Success criteria
    success = True
    if require_all:
        success = expected_set.issubset(retrieved_set)
    else:
        success = len(intersection) > 0

    if check_order:
        success = success and order_correct

    # Create metrics dictionary
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "retrieved_count": len(retrieved_set),
        "expected_count": len(expected_set),
        "intersection": len(intersection),
        "order_correct": order_correct,
    }

    return success, metrics


def assert_specific_difference(
    results1: list[dict[str, Any]], results2: list[dict[str, Any]], message_prefix: str
) -> tuple[bool, str]:
    """Assert that two result sets have specific, meaningful differences.

    Args:
        results1: First set of retrieval results
        results2: Second set of retrieval results
        message_prefix: Prefix for the error message

    Returns:
        tuple of (difference_found, message)
    """
    differences = []

    # Check for count differences
    if len(results1) != len(results2):
        differences.append(f"result count ({len(results1)} vs {len(results2)})")

    # Check for content differences by comparing memory IDs
    ids1 = set(r.get("memory_id") for r in results1 if "memory_id" in r)
    ids2 = set(r.get("memory_id") for r in results2 if "memory_id" in r)

    if ids1 != ids2:
        differences.append("memory IDs")

    # Check for score differences
    avg_score1 = (
        sum(r.get("relevance_score", 0) for r in results1) / len(results1) if results1 else 0
    )
    avg_score2 = (
        sum(r.get("relevance_score", 0) for r in results2) / len(results2) if results2 else 0
    )

    if abs(avg_score1 - avg_score2) > 0.01:
        differences.append(f"average relevance scores ({avg_score1:.3f} vs {avg_score2:.3f})")

    # Check for rank differences by comparing memory IDs
    if len(ids1.intersection(ids2)) > 1:
        # Get items in both result sets
        common_ids = ids1.intersection(ids2)

        # Get rankings
        rank1 = {
            r.get("memory_id"): i
            for i, r in enumerate(results1)
            if r.get("memory_id") in common_ids
        }
        rank2 = {
            r.get("memory_id"): i
            for i, r in enumerate(results2)
            if r.get("memory_id") in common_ids
        }

        # Check if ranking order is different
        for id1 in common_ids:
            for id2 in common_ids:
                if id1 != id2:
                    # Check if relative order differs
                    if (rank1[id1] < rank1[id2] and rank2[id1] > rank2[id2]) or (
                        rank1[id1] > rank1[id2] and rank2[id1] < rank2[id2]
                    ):
                        differences.append("ranking order")
                        break

    # Return results
    if differences:
        return True, f"{message_prefix}. Found differences in: {', '.join(differences)}"
    else:
        return False, f"{message_prefix}, but no meaningful differences found"
