"""
Unit tests for the KeywordExpander component.
"""

import unittest

from memoryweave.components.keyword_expander import KeywordExpander
from memoryweave.interfaces.retrieval import Query


class KeywordExpanderTest(unittest.TestCase):
    """
    Unit tests for the KeywordExpander component.
    """

    def setUp(self):
        """Set up test environment before each test."""
        self.expander = KeywordExpander()

    def test_initialization(self):
        """Test initialization with configuration."""
        config = {
            "enable_expansion": True,
            "max_expansions_per_keyword": 3,
            "custom_synonyms": {"test": ["experiment", "trial", "examination"]},
        }

        self.expander.initialize(config)

        self.assertTrue(self.expander.enable_expansion)
        self.assertEqual(self.expander.max_expansions_per_keyword, 3)
        self.assertEqual(self.expander.synonyms["test"], ["experiment", "trial", "examination"])

    def test_singular_plural_forms(self):
        """Test generation of singular/plural forms."""
        # Regular forms
        singular, plural = self.expander._get_singular_plural("book")
        self.assertEqual(singular, "book")
        self.assertEqual(plural, "books")

        singular, plural = self.expander._get_singular_plural("books")
        self.assertEqual(singular, "book")
        self.assertEqual(plural, "books")

        # Irregular forms
        singular, plural = self.expander._get_singular_plural("child")
        self.assertEqual(singular, "child")
        self.assertEqual(plural, "children")

        singular, plural = self.expander._get_singular_plural("children")
        self.assertEqual(singular, "child")
        self.assertEqual(plural, "children")

    def test_keyword_expansion(self):
        """Test expanding a set of keywords."""
        keywords = {"book", "person", "programming"}

        expanded = self.expander.expand_keywords(keywords)

        # Original keywords should be included
        self.assertIn("book", expanded)
        self.assertIn("person", expanded)
        self.assertIn("programming", expanded)

        # Regular singular/plural forms
        self.assertIn("books", expanded)

        # Irregular singular/plural forms
        self.assertIn("people", expanded)

        # Added 's' for regular singular
        self.assertIn("programmings", expanded)

    def test_synonym_expansion(self):
        """Test expanding keywords with synonyms."""
        # Choose keywords that are in the predefined synonym map
        keywords = {"happy", "car", "computer"}

        expanded = self.expander.expand_keywords(keywords)

        # Check for synonyms
        for keyword in keywords:
            for synonym in self.expander.synonyms.get(keyword, [])[
                : self.expander.max_expansions_per_keyword
            ]:
                self.assertIn(synonym, expanded)

        # Specifically check some expected synonyms
        self.assertIn("joyful", expanded)  # Synonym for "happy"
        self.assertIn("vehicle", expanded)  # Synonym for "car"
        self.assertIn("laptop", expanded)  # Synonym for "computer"

    def test_processing(self):
        """Test processing with context."""
        # Setup
        data = {}
        context = {"important_keywords": ["book", "computer", "person"]}

        # Process
        result = self.expander.process(data, context)

        # Check that original and expanded keywords are in result
        self.assertIn("original_keywords", result)
        self.assertIn("expanded_keywords", result)

        # Check that expansions include expected forms
        expanded = result["expanded_keywords"]
        expected_expansions = ["books", "laptop", "people"]
        for expected in expected_expansions:
            self.assertIn(expected, expanded)

        # Check context was updated too
        self.assertIn("expanded_keywords", context)

    def test_disabled_expansion(self):
        """Test with expansion disabled."""
        # Setup
        self.expander.enable_expansion = False
        data = {}
        context = {"important_keywords": ["book", "computer", "person"]}

        # Process
        result = self.expander.process(data, context)

        # Check that nothing was expanded
        self.assertNotIn("expanded_keywords", result)

    def test_expand_query(self):
        """Test expanding a Query object."""
        # Create a Query object with some keywords
        query = Query(
            text="books and computers",
            extracted_keywords=["book", "computer", "person"],
            embedding=None,  # Not needed for this test
            query_type=None,  # Not needed for this test
            extracted_entities=[],  # Required parameter
        )

        # Expand the query
        expanded_query = self.expander.expand(query)

        # Check that original keywords are preserved
        for keyword in query.extracted_keywords:
            self.assertIn(keyword, expanded_query.extracted_keywords)

        # Check that expansions include expected forms
        expected_expansions = ["books", "laptop", "people"]
        for expected in expected_expansions:
            self.assertIn(expected, expanded_query.extracted_keywords)

    def test_word_embedding_expansion(self):
        """Test expanding keywords using word embeddings."""
        # Create a simple word embedding dictionary
        word_embeddings = {
            "book": [1.0, 0.0, 0.0],
            "text": [0.9, 0.1, 0.0],  # Similar to "book"
            "volume": [0.85, 0.1, 0.05],  # Similar to "book"
            "computer": [0.0, 1.0, 0.0],
            "laptop": [0.1, 0.9, 0.0],  # Similar to "computer"
            "desktop": [0.05, 0.85, 0.1],  # Similar to "computer"
        }

        # Create expander with embeddings
        expander_with_embeddings = KeywordExpander(word_embeddings)
        expander_with_embeddings.initialize({"use_embeddings": True, "min_similarity": 0.7})

        # Test expansion
        expanded = expander_with_embeddings.expand_keywords({"book", "computer"})

        # Check embedding-based expansions
        self.assertIn("text", expanded)
        self.assertIn("volume", expanded)
        self.assertIn("laptop", expanded)
        self.assertIn("desktop", expanded)

    def test_configure_compat(self):
        """Test the configure method (IQueryExpander compatibility)."""
        # This tests the compatibility with IQueryExpander interface
        config = {"max_expansions_per_keyword": 4, "min_similarity": 0.6}

        # Use configure method (which should delegate to initialize)
        self.expander.configure(config)

        # Check that values were properly set
        self.assertEqual(self.expander.max_expansions_per_keyword, 4)
        self.assertEqual(self.expander.min_similarity, 0.6)


if __name__ == "__main__":
    unittest.main()
