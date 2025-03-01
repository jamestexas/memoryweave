"""
Unit tests for the MinimumResultGuaranteeProcessor.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from memoryweave.components.post_processors import MinimumResultGuaranteeProcessor


class MinimumResultGuaranteeProcessorTest(unittest.TestCase):
    """
    Unit tests for the MinimumResultGuaranteeProcessor component.
    """

    def setUp(self):
        """Set up test environment before each test."""
        self.processor = MinimumResultGuaranteeProcessor()
        
        # Create mock memory with search capability
        self.mock_memory = MagicMock()
        
        # Configure processor
        self.processor.initialize({
            "min_results": 3,
            "fallback_threshold_factor": 0.5,
            "min_fallback_threshold": 0.05,
            "memory": self.mock_memory
        })
        
    def test_no_action_when_enough_results(self):
        """Test that no fallback retrieval is performed when enough results exist."""
        # Create test results with the required minimum
        results = [
            {"id": "mem1", "content": "Test memory 1", "relevance_score": 0.8},
            {"id": "mem2", "content": "Test memory 2", "relevance_score": 0.7},
            {"id": "mem3", "content": "Test memory 3", "relevance_score": 0.6},
        ]
        
        # Mock search_by_embedding function
        self.mock_memory.search_by_embedding = MagicMock()
        
        # Process results
        processed_results = self.processor.process_results(
            results,
            "test query",
            {"confidence_threshold": 0.5, "query_embedding": np.array([0.1, 0.2, 0.3])}
        )
        
        # Verify that search_by_embedding wasn't called
        self.mock_memory.search_by_embedding.assert_not_called()
        
        # Verify that results are unchanged
        self.assertEqual(len(processed_results), 3)
        
    def test_fallback_retrieval(self):
        """Test fallback retrieval when not enough results meet the threshold."""
        # Create test results with fewer than minimum
        results = [
            {"id": "mem1", "content": "Test memory 1", "relevance_score": 0.8},
        ]
        
        # Mock fallback results
        fallback_results = [
            {"id": "mem2", "content": "Test memory 2", "relevance_score": 0.4},
            {"id": "mem3", "content": "Test memory 3", "relevance_score": 0.3},
            {"id": "mem4", "content": "Test memory 4", "relevance_score": 0.2},
        ]
        
        # Configure mock search_by_embedding to return fallback results
        self.mock_memory.search_by_embedding = MagicMock(return_value=fallback_results)
        
        # Process results
        processed_results = self.processor.process_results(
            results,
            "test query",
            {"confidence_threshold": 0.5, "query_embedding": np.array([0.1, 0.2, 0.3])}
        )
        
        # Verify that search_by_embedding was called with expected parameters
        self.mock_memory.search_by_embedding.assert_called_once()
        call_args = self.mock_memory.search_by_embedding.call_args[0]
        call_kwargs = self.mock_memory.search_by_embedding.call_args[1]
        
        # Check that query_embedding was passed correctly
        self.assertTrue(np.array_equal(call_args[0], np.array([0.1, 0.2, 0.3])))
        
        # Check that threshold was halved (0.5 * 0.5 = 0.25)
        self.assertEqual(call_kwargs["threshold"], 0.25)
        
        # Verify that we got the required 3 results
        self.assertEqual(len(processed_results), 3)
        
        # Verify that fallback results were added
        self.assertEqual(processed_results[0]["id"], "mem1")  # Original result
        self.assertEqual(processed_results[1]["id"], "mem2")  # Fallback result
        self.assertEqual(processed_results[2]["id"], "mem3")  # Fallback result
        
        # Verify fallback flag was added
        self.assertTrue(processed_results[1]["from_fallback"])
        self.assertTrue(processed_results[2]["from_fallback"])
        
    def test_minimum_fallback_threshold(self):
        """Test that fallback threshold has a minimum value."""
        # Create test results with fewer than minimum
        results = [
            {"id": "mem1", "content": "Test memory 1", "relevance_score": 0.8},
        ]
        
        # Configure mock search_by_embedding to return fallback results
        self.mock_memory.search_by_embedding = MagicMock(return_value=[])
        
        # Process results with very low confidence threshold
        processed_results = self.processor.process_results(
            results,
            "test query",
            {"confidence_threshold": 0.01, "query_embedding": np.array([0.1, 0.2, 0.3])}
        )
        
        # Verify that search_by_embedding was called with the minimum threshold
        call_kwargs = self.mock_memory.search_by_embedding.call_args[1]
        self.assertEqual(call_kwargs["threshold"], 0.05)  # Should use min_fallback_threshold
        
    def test_handles_missing_dependencies(self):
        """Test graceful handling when dependencies are missing."""
        # Create test results with fewer than minimum
        results = [
            {"id": "mem1", "content": "Test memory 1", "relevance_score": 0.8},
        ]
        
        # Set memory to None to simulate missing dependency
        self.processor.memory = None
        
        # Process results without necessary context
        processed_results = self.processor.process_results(
            results,
            "test query",
            {"confidence_threshold": 0.5}  # Missing query_embedding
        )
        
        # Verify that results are unchanged
        self.assertEqual(len(processed_results), 1)
        self.assertEqual(processed_results[0]["id"], "mem1")


if __name__ == "__main__":
    unittest.main()