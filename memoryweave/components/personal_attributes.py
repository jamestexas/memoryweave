# memoryweave/components/personal_attributes.py
from typing import Any

from memoryweave.components.base import MemoryComponent, RetrievalComponent
from memoryweave.utils.nlp_extraction import NLPExtractor


class PersonalAttributeManager(RetrievalComponent):
    """
    Manages extraction and storage of personal attributes.
    """

    def __init__(self, nlp_model_name: str = "en_core_web_sm"):
        self.nlp_extractor = NLPExtractor(model_name=nlp_model_name)
        self.personal_attributes = {
            "preferences": {},
            "demographics": {},
            "traits": {},
            "relationships": {},
        }

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        if "initial_attributes" in config:
            self._update_attributes(config["initial_attributes"])

    def process(self, data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Process text data to extract attributes."""
        text = data.get("text", "")
        if not text:
            return {"personal_attributes": self.personal_attributes}

        extracted_attributes = self.nlp_extractor.extract_personal_attributes(text)
        self._update_attributes(extracted_attributes)

        return {"personal_attributes": self.personal_attributes}

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Process a query to identify and extract personal attributes."""
        # Extract attributes from query
        extracted_attributes = self.nlp_extractor.extract_personal_attributes(query)
        self._update_attributes(extracted_attributes)

        # Extract relevant attributes for the query
        relevant_attributes = self._get_relevant_attributes(query)

        return {
            "personal_attributes": self.personal_attributes,
            "relevant_attributes": relevant_attributes,
        }

    def _update_attributes(self, attributes: dict[str, Any]) -> None:
        """Update stored attributes with new information."""
        for category, items in attributes.items():
            if not items:
                continue

            if category not in self.personal_attributes:
                self.personal_attributes[category] = {}

            if isinstance(items, dict):
                for key, value in items.items():
                    self.personal_attributes[category][key] = value
            elif isinstance(items, list):
                if not isinstance(self.personal_attributes[category], list):
                    self.personal_attributes[category] = []
                for item in items:
                    if item not in self.personal_attributes[category]:
                        self.personal_attributes[category].append(item)
            else:
                self.personal_attributes[category] = items

    def _get_relevant_attributes(self, query: str) -> dict[str, Any]:
        """Get attributes relevant to the query."""
        query_lower = query.lower()
        relevant_attributes = {}

        # Check for preference-related queries
        preference_keywords = ["favorite", "like", "prefer", "love"]
        if any(keyword in query_lower for keyword in preference_keywords):
            for category, value in self.personal_attributes["preferences"].items():
                if category in query_lower or any(
                    keyword in query_lower for keyword in preference_keywords
                ):
                    relevant_attributes[f"preference_{category}"] = value

        # Similar logic for other attribute types...

        return relevant_attributes
