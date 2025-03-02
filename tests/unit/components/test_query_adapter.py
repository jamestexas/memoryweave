"""
Tests for the QueryTypeAdapter component.

This module tests the QueryTypeAdapter component to ensure:
1. Parameters are correctly adapted based on query type
2. Default parameters are returned when adaptation is disabled
3. Different configurations lead to distinctly different parameters
"""

import pytest
import numpy as np
from memoryweave.components.query_adapter import QueryTypeAdapter


class TestQueryTypeAdapter:
    """Test suite for QueryTypeAdapter component."""

    def setup_method(self):
        """Setup for each test."""
        self.adapter = QueryTypeAdapter()
        self.adapter.initialize({
            "adaptation_strength": 1.0,
            "confidence_threshold": 0.3,
            "adaptive_k_factor": 0.3,
            "first_stage_k": 20,
            "first_stage_threshold_factor": 0.7,
            "keyword_boost_weight": 0.5
        })

    def test_disabled_adaptation_returns_defaults(self):
        """Test that when adaptation is disabled, default parameters are returned."""
        # Set up context with adaptation disabled
        context = {
            "enable_query_type_adaptation": False,
            "config_name": "test-config"
        }
        
        # Process a query
        result = self.adapter.process_query("What is my favorite color?", context)
        
        # Verify default parameters are returned
        assert "adapted_retrieval_params" in result
        assert result["adapted_retrieval_params"]["confidence_threshold"] == 0.3
        assert result["adapted_retrieval_params"]["adapted_by_query_type"] is False

    def test_enabled_adaptation_modifies_parameters(self):
        """Test that when adaptation is enabled, parameters are modified."""
        # Set up context with adaptation enabled and personal query type
        context = {
            "enable_query_type_adaptation": True,
            "primary_query_type": "personal",
            "config_name": "test-config" 
        }
        
        # Process a personal query
        result = self.adapter.process_query("What is my favorite color?", context)
        
        # Verify parameters are adapted
        assert "adapted_retrieval_params" in result
        assert result["adapted_retrieval_params"]["adapted_by_query_type"] is True
        
        # Personal queries should have higher confidence thresholds
        assert result["adapted_retrieval_params"]["confidence_threshold"] > 0.3

    def test_query_adaptation_config_has_stronger_effect(self):
        """Test that Query-Adaptation config has stronger parameter modifications."""
        # Create another adapter with config name that should trigger stronger effects
        special_adapter = QueryTypeAdapter()
        special_adapter.initialize({
            "adaptation_strength": 1.0,
            "confidence_threshold": 0.3,
            "adaptive_k_factor": 0.3,
            "first_stage_k": 20,
            "first_stage_threshold_factor": 0.7,
            "keyword_boost_weight": 0.5
        })
        special_adapter.config_name = "Query-Adaptation"
        
        # Regular context
        basic_context = {
            "enable_query_type_adaptation": True,
            "primary_query_type": "factual", 
            "config_name": "Basic"
        }
        
        # Special context
        special_context = {
            "enable_query_type_adaptation": True,
            "primary_query_type": "factual",
            "config_name": "Query-Adaptation"
        }
        
        # Process with both adapters
        basic_result = self.adapter.process_query("What is the capital of France?", basic_context)
        special_result = special_adapter.process_query("What is the capital of France?", special_context)
        
        basic_params = basic_result["adapted_retrieval_params"]
        special_params = special_result["adapted_retrieval_params"]
        
        # Special configuration should have more aggressive parameter adjustments
        # For factual queries, first_stage_k should be larger in special config
        assert special_params["first_stage_k"] > basic_params["first_stage_k"]
        
        # Special configuration should expand keywords for factual queries
        assert special_params["expand_keywords"] is True

    def test_different_query_types_produce_different_params(self):
        """Test that different query types result in different adaptation parameters."""
        # Context for personal query
        personal_context = {
            "enable_query_type_adaptation": True,
            "primary_query_type": "personal",
            "config_name": "test-config"
        }
        
        # Context for factual query  
        factual_context = {
            "enable_query_type_adaptation": True,
            "primary_query_type": "factual",
            "config_name": "test-config"
        }
        
        # Process both queries
        personal_result = self.adapter.process_query("What is my favorite color?", personal_context)
        factual_result = self.adapter.process_query("What is the capital of France?", factual_context)
        
        personal_params = personal_result["adapted_retrieval_params"]
        factual_params = factual_result["adapted_retrieval_params"]
        
        # Personal queries should have higher threshold (precision-focused)
        assert personal_params["confidence_threshold"] > factual_params["confidence_threshold"]
        
        # Factual queries should have higher first_stage_k (recall-focused)
        assert factual_params["first_stage_k"] > personal_params["first_stage_k"]


if __name__ == "__main__":
    pytest.main()