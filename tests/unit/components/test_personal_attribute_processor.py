"""
Unit tests for the PersonalAttributeProcessor.
"""

import unittest

from memoryweave.components.personal_attributes import PersonalAttributeManager
from memoryweave.components.post_processors import PersonalAttributeProcessor


class PersonalAttributeProcessorTest(unittest.TestCase):
    """
    Unit tests for the PersonalAttributeProcessor component.
    """

    def setUp(self):
        """Set up test environment before each test."""
        self.attribute_manager = PersonalAttributeManager()
        self.attribute_manager.initialize({})
        
        # Initialize the processor
        self.processor = PersonalAttributeProcessor()
        self.processor.initialize({})
        
        # Set up some test attributes
        self.attribute_manager.process({"text": "My favorite color is blue"}, {})
        self.attribute_manager.process({"text": "I live in Seattle"}, {})
        self.attribute_manager.process({"text": "I work as a software engineer"}, {})
        
    def test_attribute_boost(self):
        """Test that results containing attributes get boosted."""
        # Process a query to get attributes
        query = "What's my favorite color?"
        attribute_context = self.attribute_manager.process_query(query, {})
        
        # Sample results
        test_results = [
            {
                "content": "Blue is the color of the sky.",
                "relevance_score": 0.5,
                "id": "mem1"
            },
            {
                "content": "Red is a vibrant color.",
                "relevance_score": 0.6,
                "id": "mem2"
            }
        ]
        
        # Process results with the processor
        processed_results = self.processor.process_results(
            test_results, 
            query, 
            attribute_context
        )
        
        # Check that the blue result got boosted
        blue_result = next(r for r in processed_results if "blue" in r["content"].lower())
        self.assertIn("attribute_boost_applied", blue_result)
        self.assertTrue(blue_result["attribute_boost_applied"])
        self.assertGreater(blue_result["relevance_score"], 0.5)  # Should be boosted
        
        # Red result should not be boosted
        red_result = next(r for r in processed_results if "red" in r["content"].lower())
        self.assertNotIn("attribute_boost_applied", red_result)
        self.assertEqual(red_result["relevance_score"], 0.6)  # Should remain the same
        
    def test_direct_response_creation(self):
        """Test creation of direct responses for attribute questions."""
        # Process a query to get attributes
        query = "What's my favorite color?"
        attribute_context = self.attribute_manager.process_query(query, {})
        
        # Empty results to trigger synthetic generation
        test_results = []
        
        # Process results with the processor
        processed_results = self.processor.process_results(
            test_results, 
            query, 
            attribute_context
        )
        
        # Should have created a synthetic result
        self.assertEqual(len(processed_results), 1)
        self.assertEqual(processed_results[0]["relevance_score"], 1.0)
        self.assertEqual(processed_results[0]["type"], "attribute")
        self.assertTrue(processed_results[0]["is_synthetic"])
        self.assertIn("blue", processed_results[0]["content"].lower())
        
    def test_location_query(self):
        """Test location-based attribute queries."""
        # Process a query to get attributes
        query = "Where do I live?"
        attribute_context = self.attribute_manager.process_query(query, {})
        
        # Sample results with location mention
        test_results = [
            {
                "content": "Seattle is a city in Washington state.",
                "relevance_score": 0.4,
                "id": "mem1"
            }
        ]
        
        # Process results with the processor
        processed_results = self.processor.process_results(
            test_results, 
            query, 
            attribute_context
        )
        
        # Check that Seattle result got boosted
        self.assertIn("attribute_boost_applied", processed_results[0])
        self.assertTrue(processed_results[0]["attribute_boost_applied"])
        self.assertGreater(processed_results[0]["relevance_score"], 0.4)
        
    def test_no_relevant_attributes(self):
        """Test behavior when no relevant attributes exist."""
        # Query unrelated to stored attributes
        query = "What is machine learning?"
        attribute_context = self.attribute_manager.process_query(query, {})
        
        # Sample results
        test_results = [
            {
                "content": "Machine learning is a branch of AI.",
                "relevance_score": 0.7,
                "id": "mem1"
            }
        ]
        
        # Process results with the processor
        processed_results = self.processor.process_results(
            test_results, 
            query, 
            attribute_context
        )
        
        # Results should be unchanged
        self.assertEqual(len(processed_results), 1)
        self.assertEqual(processed_results[0]["relevance_score"], 0.7)
        self.assertNotIn("attribute_boost_applied", processed_results[0])


if __name__ == "__main__":
    unittest.main()