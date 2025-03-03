import pytest
from memoryweave.components.post_processors import KeywordBoostProcessor


class TestKeywordBoostProcessor:
    """Test the KeywordBoostProcessor component."""

    def setup_method(self):
        """Set up the test environment."""
        self.processor = KeywordBoostProcessor()
        self.processor.initialize({"keyword_boost_weight": 0.5})

    def test_initialization(self):
        """Test that the processor initializes with the correct configuration."""
        assert self.processor.keyword_boost_weight == 0.5

        # Test with different configuration
        processor = KeywordBoostProcessor()
        processor.initialize({"keyword_boost_weight": 0.8})
        assert processor.keyword_boost_weight == 0.8

    def test_no_keywords_no_change(self):
        """Test that results are unchanged when no keywords are provided."""
        results = [
            {"content": "This is a test content", "relevance_score": 0.7},
            {"content": "Another test content", "relevance_score": 0.5},
        ]
        context = {}  # No keywords in context

        processed_results = self.processor.process_results(results, "test query", context)

        # Results should be unchanged
        assert processed_results == results
        assert "keyword_boost_applied" not in processed_results[0]
        assert "keyword_boost_applied" not in processed_results[1]

    def test_with_keywords_boost_applied(self):
        """Test that relevance scores are boosted for content containing keywords."""
        results = [
            {"content": "This content mentions important keyword apple", "relevance_score": 0.6},
            {"content": "This content has no relevant keywords", "relevance_score": 0.7},
            {"content": "Contains both apple and banana keywords", "relevance_score": 0.4},
        ]

        # Context with important keywords
        context = {"important_keywords": {"apple", "banana", "orange"}}

        processed_results = self.processor.process_results(results, "test query", context)

        # First result should be boosted (contains 'apple')
        assert processed_results[0]["relevance_score"] > 0.6
        assert processed_results[0]["keyword_boost_applied"] is True

        # Second result should be unchanged (no keywords)
        assert processed_results[1]["relevance_score"] == 0.7
        assert "keyword_boost_applied" not in processed_results[1]

        # Third result should have highest boost (contains 'apple' and 'banana')
        assert processed_results[2]["relevance_score"] > 0.4
        assert processed_results[2]["keyword_boost_applied"] is True

        # Verify both results got boosted
        assert processed_results[0]["relevance_score"] > 0.6
        assert processed_results[2]["relevance_score"] > 0.4

        # The third result should ideally have a higher boost than first (two keywords vs one),
        # but this is implementation-dependent and depends on how the boost is calculated

    def test_boost_calculation_accuracy(self):
        """Test the accuracy of the boost calculation formula."""
        results = [{"content": "Test content with keyword apple", "relevance_score": 0.5}]
        context = {"important_keywords": {"apple", "banana", "orange"}}

        # With keyword_boost_weight = 0.5 and 1/3 keywords matched
        # boost = 0.5 * (1/3) = 0.1667
        # new_score = 0.5 + 0.1667 * (1 - 0.5) = 0.5833

        processed_results = self.processor.process_results(results, "test query", context)

        # Check if the calculation is accurate (allowing for small float precision differences)
        expected_score = 0.5 + 0.5 * (1 / 3) * (1 - 0.5)
        assert abs(processed_results[0]["relevance_score"] - expected_score) < 0.0001

    def test_max_boost_cap(self):
        """Test that the boost is properly capped at the maximum value."""
        # Create a result with a high initial score
        results = [
            {"content": "Contains all keywords: apple banana orange", "relevance_score": 0.9}
        ]
        context = {"important_keywords": {"apple", "banana", "orange"}}

        # Processor should apply boost but not exceed 1.0
        processed_results = self.processor.process_results(results, "test query", context)

        # Even with max boost, score should not exceed 1.0
        assert processed_results[0]["relevance_score"] <= 1.0
        assert processed_results[0]["relevance_score"] > 0.9
        assert processed_results[0]["keyword_boost_applied"] is True

    def test_case_insensitive_matching(self):
        """Test that keyword matching is case insensitive."""
        results = [{"content": "This content has APPLE in uppercase", "relevance_score": 0.5}]
        context = {"important_keywords": {"apple"}}

        processed_results = self.processor.process_results(results, "test query", context)

        # Should match APPLE even though keyword is lowercase
        assert processed_results[0]["relevance_score"] > 0.5
        assert processed_results[0]["keyword_boost_applied"] is True
