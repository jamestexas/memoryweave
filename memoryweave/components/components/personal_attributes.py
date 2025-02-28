"""
Personal attribute management component for MemoryWeave.
"""

import re

import spacy

try:
    nlp = spacy.load("en_core_web_sm")
    print("Successfully loaded spaCy model: en_core_web_sm")
except Exception as e:
    print(f"Failed to load spaCy model: {e}")
    nlp = None

from .base import Component


class PersonalAttributeManager(Component):
    """
    Component for extracting and managing personal attributes from text.
    """

    def __init__(self):
        """Initialize the personal attribute manager."""
        self.personal_attributes = {
            "preferences": {},
            "demographics": {},
            "traits": {"hobbies": []},
            "relationships": {"family": {}, "friends": {}},
        }

    def initialize(self, config):
        """
        Initialize with configuration.

        Args:
            config: Configuration dictionary
        """
        pass

    def process(self, data, context):
        """
        Process text to extract personal attributes.

        Args:
            data: Dictionary containing text to process
            context: Additional context

        Returns:
            Updated context with extracted attributes
        """
        if "text" not in data:
            return context

        text = data["text"]

        # Extract preferences
        self._extract_preferences(text)

        # Extract demographic information
        self._extract_demographics(text)

        # Extract traits and hobbies
        self._extract_traits(text)

        # Extract relationships
        self._extract_relationships(text)

        # Add attributes to context
        context["personal_attributes"] = self.personal_attributes
        return context

    def process_query(self, query, context):
        """
        Process a query to find relevant personal attributes.

        Args:
            query: Query string
            context: Additional context

        Returns:
            Updated context with relevant attributes
        """
        # Initialize relevant attributes
        relevant_attributes = {}

        # Check for color preference queries
        if any(word in query.lower() for word in ["color", "favourite", "favorite"]):
            color = self.personal_attributes["preferences"].get("color")
            if color:
                relevant_attributes["preference_color"] = color

        # Check for location queries
        if any(word in query.lower() for word in ["live", "location", "where", "city"]):
            location = self.personal_attributes["demographics"].get("location")
            if location:
                relevant_attributes["demographic_location"] = location

        # Check for occupation queries
        if any(word in query.lower() for word in ["job", "work", "profession", "occupation"]):
            occupation = self.personal_attributes["demographics"].get("occupation")
            if occupation:
                relevant_attributes["demographic_occupation"] = occupation

        # Check for relationship queries
        if any(word in query.lower() for word in ["wife", "husband", "partner", "spouse"]):
            partner = self.personal_attributes["relationships"]["family"].get(
                "wife"
            ) or self.personal_attributes["relationships"]["family"].get("husband")
            if partner:
                relevant_attributes["relationship_partner"] = partner

        # Add attributes to context
        context["relevant_attributes"] = relevant_attributes
        return context

    def _extract_preferences(self, text):
        """
        Extract preferences from text.

        Args:
            text: Text to process
        """
        # Extract favorite color
        color_match = re.search(r"(?:favorite|favourite) color is (\w+)", text, re.IGNORECASE)
        if color_match:
            self.personal_attributes["preferences"]["color"] = color_match.group(1)

        # Extract food preferences
        food_patterns = [
            r"(?:favorite|favourite) food is (\w+)",
            r"I (?:really |)(?:like|love) (?:eating |)(\w+)",
        ]

        for pattern in food_patterns:
            food_match = re.search(pattern, text, re.IGNORECASE)
            if food_match:
                food = food_match.group(1)
                # Check if it's actually a food (simple check)
                if food.lower() not in ["to", "and", "or", "the", "a", "an"]:
                    self.personal_attributes["preferences"]["food"] = food
                    break

    def _extract_demographics(self, text):
        """
        Extract demographic information from text.

        Args:
            text: Text to process
        """
        # Extract location
        location_match = re.search(r"(?:I|we) live in (\w+)", text, re.IGNORECASE)
        if location_match:
            self.personal_attributes["demographics"]["location"] = location_match.group(1).lower()

        # Extract occupation
        occupation_match = re.search(
            r"(?:work|job) as (?:a |an |)(.+?)(?:\.|\,|\s$|$)", text, re.IGNORECASE
        )
        if occupation_match:
            self.personal_attributes["demographics"]["occupation"] = occupation_match.group(
                1
            ).strip()

    def _extract_traits(self, text):
        """
        Extract traits and hobbies from text.

        Args:
            text: Text to process
        """
        # Extract hobbies
        hobby_patterns = [
            r"I enjoy (\w+ing)",
            r"I like to (\w+)",
            r"My hobby is (\w+ing)",
            r"I love (\w+ing)",
        ]

        for pattern in hobby_patterns:
            hobby_match = re.search(pattern, text, re.IGNORECASE)
            if hobby_match:
                hobby = hobby_match.group(1)
                if hobby.lower() not in ["something", "anything", "nothing", "doing"]:
                    if "hobbies" not in self.personal_attributes["traits"]:
                        self.personal_attributes["traits"]["hobbies"] = []
                    self.personal_attributes["traits"]["hobbies"].append(hobby)

    def _extract_relationships(self, text):
        """
        Extract relationships from text.

        Args:
            text: Text to process
        """
        # Initialize family dictionary if not present
        if "family" not in self.personal_attributes["relationships"]:
            self.personal_attributes["relationships"]["family"] = {}

        # Extract spouse
        spouse_patterns = [
            (r"(?:my|My) wife(?:'s|s'|) (?:name is |)(\w+)", "wife"),
            (r"(?:my|My) husband(?:'s|s'|) (?:name is |)(\w+)", "husband"),
            (r"(?:my|My) partner(?:'s|s'|) (?:name is |)(\w+)", "partner"),
        ]

        for pattern, relationship in spouse_patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1)
                self.personal_attributes["relationships"]["family"][relationship] = name.lower()
