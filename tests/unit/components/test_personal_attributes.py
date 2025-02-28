"""
Unit tests for the PersonalAttributeManager component.
"""

import unittest

from memoryweave.components.personal_attributes import PersonalAttributeManager


class PersonalAttributeManagerTest(unittest.TestCase):
    """
    Unit tests for the PersonalAttributeManager component.
    """

    def setUp(self):
        """Set up test environment before each test."""
        self.attribute_manager = PersonalAttributeManager()
        self.attribute_manager.initialize({})

    def test_preference_extraction(self):
        """Test extraction of preferences."""
        # Test with direct statement
        text = "My favorite color is blue"
        result = self.attribute_manager.process({"text": text}, {})

        self.assertIn("personal_attributes", result)
        self.assertIn("preferences", result["personal_attributes"])
        self.assertIn("color", result["personal_attributes"]["preferences"])
        self.assertEqual(result["personal_attributes"]["preferences"]["color"], "blue")

        # Test with indirect statement
        text = "I really love eating pizza"
        result = self.attribute_manager.process({"text": text}, {})

        self.assertIn("personal_attributes", result)
        self.assertIn("preferences", result["personal_attributes"])
        self.assertIn("food", result["personal_attributes"]["preferences"])
        self.assertEqual(result["personal_attributes"]["preferences"]["food"], "pizza")

    def test_demographic_extraction(self):
        """Test extraction of demographic information."""
        # Test location extraction
        text = "I live in Seattle"
        result = self.attribute_manager.process({"text": text}, {})

        self.assertIn("personal_attributes", result)
        self.assertIn("demographics", result["personal_attributes"])
        self.assertIn("location", result["personal_attributes"]["demographics"])
        self.assertEqual(result["personal_attributes"]["demographics"]["location"], "seattle")

        # Test occupation extraction
        text = "I work as a software engineer"
        result = self.attribute_manager.process({"text": text}, {})

        self.assertIn("personal_attributes", result)
        self.assertIn("demographics", result["personal_attributes"])
        self.assertIn("occupation", result["personal_attributes"]["demographics"])
        self.assertEqual(
            result["personal_attributes"]["demographics"]["occupation"], "software engineer"
        )

    def test_trait_extraction(self):
        """Test extraction of traits and hobbies."""
        text = "I enjoy hiking in the mountains on weekends"
        result = self.attribute_manager.process({"text": text}, {})

        self.assertIn("personal_attributes", result)
        self.assertIn("traits", result["personal_attributes"])
        self.assertIn("hobbies", result["personal_attributes"]["traits"])

        hobbies = result["personal_attributes"]["traits"]["hobbies"]
        self.assertIsInstance(hobbies, list)
        self.assertIn("hiking", " ".join(hobbies).lower())

    def test_relationship_extraction(self):
        """Test extraction of relationships."""
        text = "My wife's name is Sarah"
        result = self.attribute_manager.process({"text": text}, {})

        self.assertIn("personal_attributes", result)
        self.assertIn("relationships", result["personal_attributes"])
        self.assertIn("family", result["personal_attributes"]["relationships"])
        self.assertIn("wife", result["personal_attributes"]["relationships"]["family"])
        self.assertEqual(result["personal_attributes"]["relationships"]["family"]["wife"], "sarah")

    def test_query_processing(self):
        """Test processing of queries for relevant attributes."""
        # First add some attributes
        self.attribute_manager.process({"text": "My favorite color is blue"}, {})
        self.attribute_manager.process({"text": "I live in Seattle"}, {})

        # Test query for color
        query = "What's my favorite color?"
        result = self.attribute_manager.process_query(query, {})

        self.assertIn("relevant_attributes", result)
        self.assertIn("preference_color", result["relevant_attributes"])
        self.assertEqual(result["relevant_attributes"]["preference_color"], "blue")

        # Test query for location
        query = "Where do I live?"
        result = self.attribute_manager.process_query(query, {})

        self.assertIn("relevant_attributes", result)
        self.assertTrue(
            any(key.startswith("demographic_") for key in result["relevant_attributes"])
        )

    def test_attribute_update(self):
        """Test updating of attributes."""
        # Add initial attribute
        self.attribute_manager.process({"text": "My favorite color is blue"}, {})

        # Update the attribute
        self.attribute_manager.process({"text": "Actually, my favorite color is green"}, {})

        # Check that the attribute was updated
        self.assertEqual(
            self.attribute_manager.personal_attributes["preferences"]["color"], "green"
        )


if __name__ == "__main__":
    unittest.main()
