from typing import Any

from memoryweave.components.base import RetrievalComponent
from memoryweave.nlp.extraction import NLPExtractor


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

        extracted_attribute_list = self.nlp_extractor.extract_personal_attributes(text)

        # Convert the list of ExtractedAttribute objects to a dictionary
        attributes_dict = {}
        for attr in extracted_attribute_list:
            category = attr.attribute.split("_")[0] if "_" in attr.attribute else "preferences"
            attr_type = attr.attribute.split("_")[1] if "_" in attr.attribute else attr.attribute

            if category not in attributes_dict:
                attributes_dict[category] = {}

            attributes_dict[category][attr_type] = attr.value

        self._update_attributes(attributes_dict)

        return {"personal_attributes": self.personal_attributes}

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Process a query to identify and extract personal attributes."""
        extracted_attribute_list = self.nlp_extractor.extract_personal_attributes(query)

        # Convert the list of ExtractedAttribute objects to a dictionary
        attributes_dict = {}
        for attr in extracted_attribute_list:
            category = attr.attribute.split("_")[0] if "_" in attr.attribute else "preferences"
            attr_type = attr.attribute.split("_")[1] if "_" in attr.attribute else attr.attribute

            if category not in attributes_dict:
                attributes_dict[category] = {}

            attributes_dict[category][attr_type] = attr.value

        self._update_attributes(attributes_dict)
        relevant_attributes = self._get_relevant_attributes(query)

        return {
            "personal_attributes": self.personal_attributes,
            "relevant_attributes": relevant_attributes,
        }

    def _update_attributes(self, attributes: dict[str, Any]) -> None:
        """Ensure extracted attributes are properly stored and merged."""
        # Handle attributes with category_subcategory format (e.g., preferences_color)
        if "preferences_color" in attributes:
            self._ensure_category_exists("preferences")
            self.personal_attributes["preferences"]["color"] = attributes["preferences_color"]

        # Handle structured attributes with nested dictionaries
        if (
            "favorite" in attributes
            and isinstance(attributes["favorite"], dict)
            and "color" in attributes["favorite"]
        ):
            self._ensure_category_exists("preferences")
            self.personal_attributes["preferences"]["color"] = attributes["favorite"]["color"]

        # Standard processing for attributes
        for category, items in attributes.items():
            if not items:
                continue  # Skip empty attributes

            # Special handling for attributes with underscore format (from extracted attributes)
            if "_" in category and category not in self.personal_attributes:
                main_category, sub_key = category.split("_", 1)

                # Create category if it doesn't exist
                self._ensure_category_exists(main_category)

                # Handle family relationships
                if main_category == "relationships" and sub_key == "family":
                    if "family" not in self.personal_attributes["relationships"]:
                        self.personal_attributes["relationships"]["family"] = {}

                    # Update family relationships based on input format
                    if isinstance(items, dict):
                        for k, v in items.items():
                            self.personal_attributes["relationships"]["family"][k] = v
                    elif isinstance(items, str):
                        # Skip if it's just a string with no relation specified
                        pass

                # Handle hobbies (always ensure it's a list)
                elif main_category == "traits" and sub_key == "hobbies":
                    if sub_key not in self.personal_attributes[main_category]:
                        if isinstance(items, list):
                            self.personal_attributes[main_category][sub_key] = items
                        else:
                            self.personal_attributes[main_category][sub_key] = [items]
                else:
                    # Regular attribute
                    self.personal_attributes[main_category][sub_key] = items
                continue

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

        # Ensure all standard categories exist
        self._ensure_category_exists("preferences")
        self._ensure_category_exists("demographics")
        self._ensure_category_exists("traits")
        self._ensure_category_exists("relationships")

    def _ensure_category_exists(self, category: str) -> None:
        """Ensure a category exists in the personal attributes."""
        if category not in self.personal_attributes:
            self.personal_attributes[category] = {}

    def _get_relevant_attributes(self, query: str) -> dict[str, Any]:
        """Retrieve attributes relevant to the query dynamically."""
        query_lower = query.lower().strip()
        relevant_attributes = {}

        # Extract important keywords from query
        keywords = self.nlp_extractor.extract_important_keywords(query)

        # Check for favorite color questions
        if "what's my favorite color" in query_lower or "what is my favorite color" in query_lower:
            if self.personal_attributes.get("preferences", {}).get("color"):
                relevant_attributes["preferences_color"] = self.personal_attributes["preferences"][
                    "color"
                ]
            return relevant_attributes

        # Check for location questions
        if "where do i live" in query_lower or ("where" in query_lower and "live" in query_lower):
            if self.personal_attributes.get("demographics", {}).get("location"):
                relevant_attributes["demographic_location"] = self.personal_attributes[
                    "demographics"
                ]["location"]
            return relevant_attributes

        # Check different attribute categories based on query keywords

        # Check preferences
        preference_keywords = ["favorite", "like", "prefer", "love", "color", "food"]
        if any(kw in query_lower for kw in preference_keywords):
            for key, value in self.personal_attributes.get("preferences", {}).items():
                if key in query_lower or key in keywords:
                    relevant_attributes[f"preferences_{key}"] = value

        # Check demographics
        demographic_keywords = ["live", "location", "city", "work", "job", "occupation"]
        if any(kw in query_lower for kw in demographic_keywords):
            for key, value in self.personal_attributes.get("demographics", {}).items():
                if key in query_lower or key in keywords:
                    relevant_attributes[f"demographic_{key}"] = value
                # Special case for location when asking about living
                elif key == "location" and ("live" in query_lower or "where" in query_lower):
                    relevant_attributes[f"demographic_{key}"] = value

        # Check relationships
        relationship_keywords = ["wife", "husband", "family", "spouse", "partner"]
        if any(kw in query_lower for kw in relationship_keywords):
            # Check if family exists in relationships
            family_dict = self.personal_attributes.get("relationships", {}).get("family", {})
            if family_dict:
                for key, value in family_dict.items():
                    relevant_attributes[f"relationship_{key}"] = value

        # Check traits/hobbies
        trait_keywords = ["hobby", "hobbies", "enjoy", "activity", "like to do"]
        if any(kw in query_lower for kw in trait_keywords):
            # Check if hobbies exists in traits
            hobbies = self.personal_attributes.get("traits", {}).get("hobbies")
            if hobbies:
                relevant_attributes["trait_hobbies"] = hobbies

        return relevant_attributes
