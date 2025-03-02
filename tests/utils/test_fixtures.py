"""
Test fixtures and utilities for MemoryWeave tests.

This module provides reusable test fixtures and utilities to ensure
consistent, predictable test behavior.
"""

import numpy as np
from typing import Dict, List, Any, Tuple

from memoryweave.core import ContextualMemory
from memoryweave.components.retrieval_strategies import (
    HybridRetrievalStrategy,
    SimilarityRetrievalStrategy,
    TwoStageRetrievalStrategy,
)
from memoryweave.components.post_processors import (
    KeywordBoostProcessor,
    SemanticCoherenceProcessor,
)


class PredictableTestEmbeddings:
    """Creates predictable, deterministic embeddings for testing."""
    
    @staticmethod
    def cat_embedding() -> np.ndarray:
        """Return a cat-related embedding."""
        return np.array([0.9, 0.1, 0.1, 0.1])
    
    @staticmethod
    def dog_embedding() -> np.ndarray:
        """Return a dog-related embedding."""
        return np.array([0.1, 0.9, 0.1, 0.1])
    
    @staticmethod
    def weather_embedding() -> np.ndarray:
        """Return a weather-related embedding."""
        return np.array([0.1, 0.1, 0.9, 0.1])
    
    @staticmethod
    def travel_embedding() -> np.ndarray:
        """Return a travel-related embedding."""
        return np.array([0.1, 0.1, 0.1, 0.9])
    
    @staticmethod
    def cat_query() -> np.ndarray:
        """Return a query embedding that should match cat content."""
        return np.array([0.8, 0.2, 0.0, 0.0])
    
    @staticmethod
    def dog_query() -> np.ndarray:
        """Return a query embedding that should match dog content."""
        return np.array([0.2, 0.8, 0.0, 0.0])
    
    @staticmethod
    def similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def create_test_memory() -> ContextualMemory:
    """Create a test memory with predictable content and embeddings."""
    memory = ContextualMemory(embedding_dim=4)
    
    # Add test memories with consistent, predictable embeddings
    # Ensure content is in both the metadata and at the top level
    cat_content1 = "My cat is named Whiskers and likes to sleep on the couch."
    memory.add_memory(
        PredictableTestEmbeddings.cat_embedding(), 
        cat_content1,
        {"type": "personal", "category": "pets", "created_at": 1, "content": cat_content1}
    )
    
    cat_content2 = "Whiskers enjoys playing with toy mice."
    memory.add_memory(
        PredictableTestEmbeddings.cat_embedding() * 0.95, 
        cat_content2,
        {"type": "personal", "category": "pets", "created_at": 2, "content": cat_content2}
    )
    
    dog_content = "I have a dog named Rover who likes to fetch balls."
    memory.add_memory(
        PredictableTestEmbeddings.dog_embedding(), 
        dog_content,
        {"type": "personal", "category": "pets", "created_at": 3, "content": dog_content}
    )
    
    weather_content = "It was rainy in Seattle yesterday."
    memory.add_memory(
        PredictableTestEmbeddings.weather_embedding(), 
        weather_content,
        {"type": "factual", "category": "weather", "created_at": 4, "content": weather_content}
    )
    
    travel_content = "I visited Paris last summer and saw the Eiffel Tower."
    memory.add_memory(
        PredictableTestEmbeddings.travel_embedding(), 
        travel_content,
        {"type": "personal", "category": "travel", "created_at": 5, "content": travel_content}
    )
    
    return memory


def create_retrieval_components(memory: ContextualMemory) -> Dict[str, Any]:
    """Create retrieval components with predictable configurations."""
    
    # Create base strategies
    similarity_strategy = SimilarityRetrievalStrategy(memory)
    similarity_strategy.initialize({"confidence_threshold": 0.3})
    
    hybrid_strategy = HybridRetrievalStrategy(memory)
    hybrid_strategy.initialize({"confidence_threshold": 0.3})
    
    # Create post processors
    keyword_processor = KeywordBoostProcessor()
    keyword_processor.initialize({"keyword_boost_weight": 0.2})
    
    coherence_processor = SemanticCoherenceProcessor()
    coherence_processor.initialize({
        "coherence_threshold": 0.2,
        "enable_query_type_filtering": True,
        "max_penalty": 0.3
    })
    
    # Create different configurations of two-stage strategy
    basic_two_stage = TwoStageRetrievalStrategy(
        memory,
        base_strategy=similarity_strategy,
        post_processors=[]
    )
    basic_two_stage.initialize({
        "confidence_threshold": 0.3,
        "first_stage_k": 3
    })
    
    advanced_two_stage = TwoStageRetrievalStrategy(
        memory,
        base_strategy=hybrid_strategy,
        post_processors=[keyword_processor, coherence_processor]
    )
    advanced_two_stage.initialize({
        "confidence_threshold": 0.3,
        "first_stage_k": 5
    })
    
    return {
        "similarity_strategy": similarity_strategy,
        "hybrid_strategy": hybrid_strategy, 
        "keyword_processor": keyword_processor,
        "coherence_processor": coherence_processor,
        "basic_two_stage": basic_two_stage,
        "advanced_two_stage": advanced_two_stage
    }


def verify_retrieval_results(results: List[Dict[str, Any]], expected_content_substrings: List[str]) -> bool:
    """Verify that retrieval results contain the expected content substrings."""
    if len(results) < len(expected_content_substrings):
        # Not enough results to contain all expected substrings
        print(f"Not enough results: {len(results)} < {len(expected_content_substrings)}")
        return False
    
    # Check if the results have 'content' field directly
    if all("content" in r for r in results):
        # Check that each expected substring appears in at least one result
        for substring in expected_content_substrings:
            if not any(substring.lower() in r.get("content", "").lower() for r in results):
                print(f"Missing substring '{substring}' in results")
                print(f"Available content: {[r.get('content', '') for r in results]}")
                return False
        return True
    
    # Results might have content in metadata
    if all(isinstance(r.get("metadata", {}), dict) for r in results):
        # Check that each expected substring appears in at least one result's metadata
        for substring in expected_content_substrings:
            if not any(substring.lower() in r.get("metadata", {}).get("content", "").lower() for r in results):
                print(f"Missing substring '{substring}' in metadata content")
                print(f"Available metadata content: {[r.get('metadata', {}).get('content', '') for r in results]}")
                return False
        return True
    
    # Content might be nested deeper in the metadata
    print(f"Could not find content in results: {results}")
    return False


def assert_specific_difference(result_a: List[Dict[str, Any]], 
                            result_b: List[Dict[str, Any]], 
                            difference_description: str) -> Tuple[bool, str]:
    """
    Assert that there are specific, expected differences between two result sets.
    Returns (passed, message) tuple.
    """
    # Check for length differences
    if len(result_a) != len(result_b):
        return True, f"Results differ in length: {len(result_a)} vs {len(result_b)}"
    
    # Check for score differences
    a_scores = [r.get("relevance_score", 0) for r in result_a]
    b_scores = [r.get("relevance_score", 0) for r in result_b]
    
    if a_scores != b_scores:
        return True, f"Results differ in relevance scores"
    
    # Check for content differences
    a_content = [r.get("content", "") for r in result_a]
    b_content = [r.get("content", "") for r in result_b]
    
    if a_content != b_content:
        return True, f"Results differ in content"
    
    # No significant differences found
    return False, f"No specific differences found: {difference_description}"