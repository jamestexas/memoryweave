"""
Unit tests for the PersonalAttributeManager component with debug logging.
"""

import pprint
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
        self.pp = pprint.PrettyPrinter(indent=2)

    def test_preference_extraction(self):
        """Test extraction of preferences."""
        text = "My favorite color is blue"
        result = self.attribute_manager.process({"text": text}, {})
        print("\n[DEBUG] After extracting color preference:")
        self.pp.pprint(self.attribute_manager.personal_attributes)

        self.assertIn("personal_attributes", result)
        self.assertIn("preferences", result["personal_attributes"])
        self.assertIn("color", result["personal_attributes"]["preferences"])
        self.assertEqual(result["personal_attributes"]["preferences"]["color"], "blue")

        text = "I really love eating pizza"
        result = self.attribute_manager.process({"text": text}, {})
        print("\n[DEBUG] After extracting food preference:")
        self.pp.pprint(self.attribute_manager.personal_attributes)

        self.assertIn("personal_attributes", result)
        self.assertIn("preferences", result["personal_attributes"])
        self.assertIn("food", result["personal_attributes"]["preferences"])
        self.assertEqual(result["personal_attributes"]["preferences"]["food"], "pizza")

    def test_demographic_extraction(self):
        """Test extraction of demographic information."""
        text = "I live in Seattle"
        result = self.attribute_manager.process({"text": text}, {})
        print("\n[DEBUG] After extracting location:")
        self.pp.pprint(self.attribute_manager.personal_attributes)

        self.assertIn("personal_attributes", result)
        self.assertIn("demographics", result["personal_attributes"])
        self.assertIn("location", result["personal_attributes"]["demographics"])
        self.assertEqual(
            result["personal_attributes"]["demographics"]["location"].lower(), "seattle"
        )

        text = "I work as a software engineer"
        result = self.attribute_manager.process({"text": text}, {})
        print("\n[DEBUG] After extracting occupation:")
        self.pp.pprint(self.attribute_manager.personal_attributes)

        self.assertIn("personal_attributes", result)
        self.assertIn("demographics", result["personal_attributes"])
        self.assertIn("occupation", result["personal_attributes"]["demographics"])
        self.assertEqual(
            result["personal_attributes"]["demographics"]["occupation"].lower(), "software engineer"
        )

    def test_trait_extraction(self):
        """Test extraction of traits and hobbies."""
        text = "I enjoy hiking in the mountains on weekends"
        result = self.attribute_manager.process({"text": text}, {})
        print("\n[DEBUG] After extracting hobbies:")
        self.pp.pprint(self.attribute_manager.personal_attributes)

        self.assertIn("personal_attributes", result)
        self.assertIn("traits", result["personal_attributes"])
        self.assertIn("hobbies", result["personal_attributes"]["traits"])

        hobbies = result["personal_attributes"]["traits"]["hobbies"]
        self.assertIsInstance(hobbies, list)
        self.assertTrue(any("hiking" in hobby.lower() for hobby in hobbies))

    def test_relationship_extraction(self):
        """Test extraction of relationships."""
        text = "My wife's name is Sarah"
        result = self.attribute_manager.process({"text": text}, {})
        print("\n[DEBUG] After extracting relationships:")
        self.pp.pprint(self.attribute_manager.personal_attributes)

        self.assertIn("personal_attributes", result)
        self.assertIn("relationships", result["personal_attributes"])
        self.assertIn("family", result["personal_attributes"]["relationships"])
        self.assertIn("wife", result["personal_attributes"]["relationships"]["family"])
        self.assertEqual(
            result["personal_attributes"]["relationships"]["family"]["wife"].lower(), "sarah"
        )

    def test_query_processing(self):
        """Test processing of queries for relevant attributes."""
        self.attribute_manager.process(dict(text="My favorite color is blue"), {})
        self.attribute_manager.process(dict(text="I live in Seattle"), {})

        print("\n[DEBUG] Before querying:")
        self.pp.pprint(self.attribute_manager.personal_attributes)

        query = "What's my favorite color?"
        result = self.attribute_manager.process_query(query, {})
        print("\n[DEBUG] Query result for color:")
        self.pp.pprint(result)

        self.assertIn("relevant_attributes", result)
        self.assertIn("preferences_color", result["relevant_attributes"])
        self.assertEqual(result["relevant_attributes"]["preferences_color"], "blue")

        query = "Where do I live?"
        result = self.attribute_manager.process_query(query, {})
        print("\n[DEBUG] Query result for location:")
        self.pp.pprint(result)

        self.assertIn("relevant_attributes", result)
        self.assertTrue(
            any(key.startswith("demographic_") for key in result["relevant_attributes"])
        )

    def test_attribute_update(self):
        """Test updating of attributes."""
        self.attribute_manager.process({"text": "My favorite color is blue"}, {})
        print("\n[DEBUG] Before updating color preference:")
        self.pp.pprint(self.attribute_manager.personal_attributes)

        self.attribute_manager.process({"text": "Actually, my favorite color is green"}, {})
        print("\n[DEBUG] After updating color preference:")
        self.pp.pprint(self.attribute_manager.personal_attributes)

        self.assertEqual(
            self.attribute_manager.personal_attributes["preferences"]["color"], "green"
        )


if __name__ == "__main__":
    unittest.main()
