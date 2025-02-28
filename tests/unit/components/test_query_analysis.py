"""
Unit tests for the QueryAnalyzer component.
"""

import unittest

from memoryweave.components.query_analysis import QueryAnalyzer


class QueryAnalyzerTest(unittest.TestCase):
    """
    Unit tests for the QueryAnalyzer component.
    """
    
    def setUp(self):
        """Set up test environment before each test."""
        self.analyzer = QueryAnalyzer()
        self.analyzer.initialize({})
    
    def test_factual_query_identification(self):
        """Test identification of factual queries."""
        factual_queries = [
            "What is Python?",
            "Who invented the internet?",
            "When was the first computer created?",
            "Tell me about machine learning",
            "Explain how memory management works"
        ]
        
        for query in factual_queries:
            result = self.analyzer.process_query(query, {})
            self.assertIn("primary_query_type", result)
            self.assertEqual(result["primary_query_type"], "factual",
                            f"Failed to identify factual query: {query}")
    
    def test_personal_query_identification(self):
        """Test identification of personal queries."""
        personal_queries = [
            "What's my name?",
            "Where do I live?",
            "What is my favorite color?",
            "Tell me about my job",
            "Who is my wife?"
        ]
        
        for query in personal_queries:
            result = self.analyzer.process_query(query, {})
            self.assertIn("primary_query_type", result)
            self.assertEqual(result["primary_query_type"], "personal",
                            f"Failed to identify personal query: {query}")
    
    def test_opinion_query_identification(self):
        """Test identification of opinion queries."""
        opinion_queries = [
            "What do you think about Python?",
            "Do you believe AI will replace humans?",
            "What's your opinion on climate change?",
            "How do you feel about self-driving cars?"
        ]
        
        for query in opinion_queries:
            result = self.analyzer.process_query(query, {})
            self.assertIn("primary_query_type", result)
            self.assertEqual(result["primary_query_type"], "opinion",
                            f"Failed to identify opinion query: {query}")
    
    def test_instruction_query_identification(self):
        """Test identification of instruction queries."""
        instruction_queries = [
            "Write a Python function to sort a list",
            "Create a new file called test.py",
            "Please summarize this article",
            "Find all instances of the word 'memory'"
        ]
        
        for query in instruction_queries:
            result = self.analyzer.process_query(query, {})
            self.assertIn("primary_query_type", result)
            self.assertEqual(result["primary_query_type"], "instruction",
                            f"Failed to identify instruction query: {query}")
    
    def test_keyword_extraction(self):
        """Test extraction of important keywords."""
        query = "Tell me about Python programming and memory management"
        result = self.analyzer.process_query(query, {})
        
        self.assertIn("important_keywords", result)
        keywords = result["important_keywords"]
        
        # Check that important keywords are extracted
        self.assertIn("python", keywords)
        self.assertIn("programming", keywords)
        self.assertIn("memory", keywords)
        self.assertIn("management", keywords)
        
        # Check that stop words are not included
        self.assertNotIn("about", keywords)
        self.assertNotIn("and", keywords)
        self.assertNotIn("me", keywords)
        self.assertNotIn("tell", keywords)


if __name__ == "__main__":
    unittest.main()
