from typing import Any

from memoryweave.components.base import RetrievalComponent
from memoryweave.utils.nlp_extraction import NLPExtractor


class PersonalAttributeManager(RetrievalComponent):
    """
    Manages extraction and storage of personal attributes.
    """

    def __init__(self):
        self.nlp_extractor = NLPExtractor()  # Shared NLP instance for efficiency
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
        text = data.get("text", "").strip()
        if not text:
            return {"personal_attributes": self.personal_attributes}

        extracted_attributes = self.nlp_extractor.extract_personal_attributes(text)
        self._update_attributes(extracted_attributes)

        return {"personal_attributes": self.personal_attributes}

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Process a query to identify and extract personal attributes."""
        extracted_attributes = self.nlp_extractor.extract_personal_attributes(query)
        self._update_attributes(extracted_attributes)

        relevant_attributes = self._get_relevant_attributes(query)

        return {
            "personal_attributes": self.personal_attributes,
            "relevant_attributes": relevant_attributes,
        }

    def _update_attributes(self, attributes: dict[str, Any]) -> None:
        """Ensure extracted attributes are properly stored and merged."""
        for category, items in attributes.items():
            if not items:
                continue  # Skip empty attributes

            # Ensure category exists
            if category not in self.personal_attributes:
                self.personal_attributes[category] = {}

            if isinstance(items, dict):
                for sub_key, sub_value in items.items():
                    if sub_key not in self.personal_attributes[category]:
                        self.personal_attributes[category][sub_key] = sub_value
                    else:
                        existing_value = self.personal_attributes[category][sub_key]
                        if isinstance(existing_value, list):
                            if sub_value not in existing_value:
                                existing_value.append(sub_value)
                        elif isinstance(existing_value, str) and existing_value != sub_value:
                            self.personal_attributes[category][sub_key] = sub_value

            elif isinstance(items, list):
                if not isinstance(self.personal_attributes[category], list):
                    self.personal_attributes[category] = []
                self.personal_attributes[category] = list(
                    set(self.personal_attributes[category] + items)
                )

            else:
                self.personal_attributes[category] = items

    def _get_relevant_attributes(self, query: str) -> dict[str, Any]:
        """Retrieve attributes relevant to the query dynamically."""
        query_lower = query.lower()
        relevant_attributes = {}

        for category, attributes in self.personal_attributes.items():
            for attr_key, attr_value in attributes.items():
                if attr_key in query_lower:
                    relevant_attributes[f"{category}_{attr_key}"] = attr_value
                elif isinstance(attr_value, list) and any(
                    term in query_lower for term in attr_value
                ):
                    relevant_attributes[f"{category}_{attr_key}"] = attr_value

        return relevant_attributes
