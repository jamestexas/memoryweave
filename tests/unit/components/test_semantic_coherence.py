"""
Tests for the SemanticCoherenceProcessor component.

This module tests the SemanticCoherenceProcessor component to ensure:
1. It applies penalties for incoherent results when enabled
2. It properly skips processing when disabled
3. It identifies outliers and promotes coherent result clusters
4. Different configurations use different coherence parameters
"""

import pytest
import numpy as np
from memoryweave.components.post_processors import SemanticCoherenceProcessor


class TestSemanticCoherenceProcessor:
    """Test suite for SemanticCoherenceProcessor component."""

    def setup_method(self):
        """Setup for each test."""
        self.processor = SemanticCoherenceProcessor()
        self.processor.initialize({
            "coherence_threshold": 0.2,
            "enable_query_type_filtering": True,
            "enable_pairwise_coherence": True,
            "enable_clustering": False,
            "max_penalty": 0.3,
            "boost_coherent_results": True
        })
        
        # Sample results for testing
        self.sample_results = [
            {
                "memory_id": 1,
                "relevance_score": 0.8,
                "content": "The cat sat on the mat and purred happily.",
                "type": "personal"
            },
            {
                "memory_id": 2,
                "relevance_score": 0.7,
                "content": "My cat enjoys sleeping on the sofa.",
                "type": "personal" 
            },
            {
                "memory_id": 3,
                "relevance_score": 0.6,
                "content": "The capital of France is Paris.",
                "type": "factual"  # This is semantically different
            },
            {
                "memory_id": 4,
                "relevance_score": 0.5,
                "content": "I bought cat food yesterday.",
                "type": "personal"
            }
        ]

    def test_disabled_skips_processing(self):
        """Test that when semantic coherence is disabled, processing is skipped."""
        # Create a copy of results to compare
        original_results = self.sample_results.copy()
        
        # Process with semantic coherence disabled
        context = {
            "enable_semantic_coherence": False,
            "config_name": "Basic"
        }
        
        processed_results = self.processor.process_results(
            self.sample_results, "Tell me about my cat", context
        )
        
        # Verify original scores are preserved and semantic_coherence_skipped flag is set
        for i, result in enumerate(processed_results):
            assert "semantic_coherence_skipped" in result
            assert result["relevance_score"] == original_results[i]["relevance_score"]

    def test_enabled_applies_penalties(self):
        """Test that when enabled, penalties are applied to incoherent results."""
        # Process with semantic coherence enabled
        context = {
            "enable_semantic_coherence": True,
            "config_name": "Semantic-Coherence",
            "primary_query_type": "personal"
        }
        
        processed_results = self.processor.process_results(
            self.sample_results, "Tell me about my cat", context
        )
        
        # Find the factual result (should be penalized)
        factual_result = next(r for r in processed_results if r["type"] == "factual")
        
        # Find a personal result about cats (should not be penalized)
        cat_result = next(r for r in processed_results if "cat" in r["content"] and r["type"] == "personal")
        
        # The factual result should have a lower score than original
        original_factual = next(r for r in self.sample_results if r["type"] == "factual")
        assert factual_result["relevance_score"] < original_factual["relevance_score"]
        
        # Track which results had penalties applied
        has_type_coherence_applied = any("type_coherence_applied" in r for r in processed_results)
        assert has_type_coherence_applied, "No type coherence penalties were applied"

    def test_different_configurations_use_different_parameters(self):
        """Test that different configurations use different coherence parameters."""
        # Create two processors with different parameters
        basic_processor = SemanticCoherenceProcessor()
        basic_processor.initialize({
            "coherence_threshold": 0.2,
            "max_penalty": 0.1  # Small penalty
        })
        
        advanced_processor = SemanticCoherenceProcessor()
        advanced_processor.initialize({
            "coherence_threshold": 0.2,
            "max_penalty": 0.5  # Large penalty
        })
        
        # Process with both processors
        context = {
            "enable_semantic_coherence": True,
            "config_name": "Test",
            "primary_query_type": "personal"
        }
        
        basic_results = basic_processor.process_results(
            self.sample_results.copy(), "Tell me about my cat", context
        )
        
        advanced_results = advanced_processor.process_results(
            self.sample_results.copy(), "Tell me about my cat", context
        )
        
        # Find the factual result in both result sets
        basic_factual = next(r for r in basic_results if r["type"] == "factual")
        advanced_factual = next(r for r in advanced_results if r["type"] == "factual")
        
        # Advanced processor should apply larger penalty
        assert advanced_factual["relevance_score"] < basic_factual["relevance_score"]

    def test_reranking_based_on_coherence(self):
        """Test that results are reranked based on coherence."""
        # Create results with artificial coherence relationship
        # The first result is unrelated to the query, others are related
        coherence_test_results = [
            {
                "memory_id": 1,
                "relevance_score": 0.9,  # High score but unrelated
                "content": "The weather in London is rainy today.",
                "type": "factual"
            },
            {
                "memory_id": 2,
                "relevance_score": 0.7,
                "content": "My cat is named Whiskers and he likes fish.",
                "type": "personal"
            },
            {
                "memory_id": 3,
                "relevance_score": 0.6,
                "content": "I bought special cat food for Whiskers yesterday.",
                "type": "personal"
            },
            {
                "memory_id": 4,
                "relevance_score": 0.5,
                "content": "Whiskers sleeps on my bed every night.",
                "type": "personal"
            }
        ]
        
        # Enable coherence processing with pairwise coherence
        context = {
            "enable_semantic_coherence": True,
            "config_name": "Semantic-Coherence",
            "primary_query_type": "personal"
        }
        
        # Create a processor with pairwise coherence enabled
        pairwise_processor = SemanticCoherenceProcessor()
        pairwise_processor.initialize({
            "coherence_threshold": 0.2,
            "enable_query_type_filtering": True,
            "enable_pairwise_coherence": True,
            "max_penalty": 0.5,
            "boost_coherent_results": True
        })
        
        # Add embeddings to simulate coherence calculation
        # Cat-related embeddings are similar to each other but different from weather
        weather_embedding = np.array([0.1, 0.1, 0.1, 0.9])
        cat_embedding = np.array([0.9, 0.9, 0.9, 0.1])
        
        # Add embeddings to the results
        coherence_test_results[0]["embedding"] = weather_embedding
        coherence_test_results[1]["embedding"] = cat_embedding 
        coherence_test_results[2]["embedding"] = cat_embedding * 0.95  # Similar
        coherence_test_results[3]["embedding"] = cat_embedding * 0.9   # Similar
        
        # Process results
        processed_results = pairwise_processor.process_results(
            coherence_test_results, "Tell me about my cat Whiskers", context
        )
        
        # Sort results by relevance score
        processed_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Check if weather result was penalized and no longer top result
        top_result = processed_results[0]
        assert "weather" not in top_result["content"].lower(), "Weather result should not be top anymore"
        
        # Weather result should have a coherence penalty applied
        weather_result = next(r for r in processed_results if "weather" in r["content"].lower())
        assert weather_result["relevance_score"] < 0.9, "Weather result should be penalized"


if __name__ == "__main__":
    pytest.main()