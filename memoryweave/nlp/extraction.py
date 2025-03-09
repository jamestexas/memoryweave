# memoryweave/nlp/extraction.py
from typing import Any

from pydantic import BaseModel

from memoryweave.nlp.factories import NLPFactory
from memoryweave.nlp.keywords import Keyword


class ExtractedEntity(BaseModel):
    """A named entity extracted from text."""

    text: str
    label: str
    start: int
    end: int


class ExtractedAttribute(BaseModel):
    """A personal attribute extracted from text."""

    attribute: str
    value: str | dict[str, Any] | list[str]
    confidence: float = 0.9


class NLPExtractor:
    """Core NLP extraction functionality for MemoryWeave."""

    def __init__(self):
        """Initialize the NLP extractor with the best available implementations."""
        # Create components using factory
        self._entity_extractor = NLPFactory.create_entity_extractor()
        self._attribute_extractor = NLPFactory.create_attribute_extractor()
        self._keyword_extractor = NLPFactory.create_keyword_extractor()
        self._query_classifier = NLPFactory.create_query_classifier()

        # Configuration
        self._config = {"confidence_threshold": 0.7, "max_entities": 10}

    def extract_personal_attributes(self, text: str) -> list[ExtractedAttribute]:
        """
        Extract personal attributes from text.

        Args:
            text: The text to extract attributes from

        Returns:
            List of extracted attributes
        """
        if not text:
            return []

        # Use attribute extractor to get attributes
        attributes_dict = self._attribute_extractor.extract_attributes(text)

        # Convert to standard format
        results = []
        for attr_name, attr_value in attributes_dict.items():
            # Handle different value formats from different extractors
            if isinstance(attr_value, tuple):
                value, confidence = attr_value
            else:
                value = attr_value
                confidence = 0.9

            # Map attribute names to the expected format
            if attr_name.startswith("favorite_"):
                attribute = f"preferences_{attr_name.split('_')[1]}"
            elif attr_name in ["location"]:
                attribute = f"demographics_{attr_name}"
            elif attr_name in ["occupation", "job", "profession"]:
                attribute = "demographics_occupation"
            elif attr_name == "likes":
                attribute = "preferences_likes"
            else:
                attribute = attr_name

            # Create ExtractedAttribute object
            attr = ExtractedAttribute(attribute=attribute, value=value, confidence=confidence)

            results.append(attr)

        return results

    def extract_entities(self, text: str) -> list[ExtractedEntity]:
        """
        Extract named entities from text.

        Args:
            text: The text to extract entities from

        Returns:
            List of extracted entities
        """
        if not text:
            return []

        # Use entity extractor to get entities
        entities = self._entity_extractor.extract_entities(text)

        # Convert to standard format
        return [
            ExtractedEntity(
                text=entity.text, label=entity.label, start=entity.start, end=entity.end
            )
            for entity in entities
        ]

    def extract_keywords(self, text: str, stopwords: set[str] | None = None) -> list[str]:
        """
        Extract keywords from text.

        Args:
            text: The text to extract keywords from
            stopwords: Optional set of stopwords to filter out

        Returns:
            List of extracted keywords
        """
        if not text:
            return []

        # Use keyword extractor to get keywords
        keywords = self._keyword_extractor.extract_keywords(text)

        if isinstance(keywords, list) and keywords and isinstance(keywords[0], tuple):
            # Extract just the keywords from (keyword, score) tuples
            return [kw for kw, _ in keywords]

        return keywords

    def extract_important_keywords(self, query: str) -> set[str]:
        """
        Extract important keywords from a query.

        Args:
            query: The query text

        Returns:
            Set of important keywords
        """
        if not query:
            return set()

        # Use keyword extractor to get keywords
        extracted = self._keyword_extractor.extract(query)

        # Convert to set of strings
        if extracted and isinstance(extracted[0], Keyword):
            return {keyword.text for keyword in extracted}
        elif extracted and isinstance(extracted[0], str):
            return set(extracted)
        elif extracted and isinstance(extracted[0], tuple):
            return {kw for kw, _ in extracted}

        return set()

    def identify_query_type(self, query: str) -> dict[str, float]:
        """
        Identify the type of query.

        Args:
            query: The query text

        Returns:
            Dictionary with query type probabilities
        """
        if not query:
            return {"factual": 0.0, "personal": 0.0, "opinion": 0.0, "instruction": 0.0}

        # Use query classifier to classify query
        return self._query_classifier.classify(query)

    def extract_relationships(self, text: str) -> list[dict[str, Any]]:
        """
        Extract relationships between entities.

        Args:
            text: The text to extract relationships from

        Returns:
            List of relationship dictionaries
        """
        if not text:
            return []

        # Extract entities first
        entities = self.extract_entities(text)

        # Simple relationship extraction based on proximity
        relationships = []

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                entity1 = entities[i]
                entity2 = entities[j]

                # Check if they are close enough (within 50 characters)
                if entity2.start - entity1.end <= 50:
                    # Extract the text between them
                    relation_text = text[entity1.end : entity2.start].strip()

                    # If there's meaningful text between them, consider it a relationship
                    if len(relation_text) >= 3:
                        relationship = {
                            "entity1": entity1.text,
                            "entity1_type": entity1.label,
                            "entity2": entity2.text,
                            "entity2_type": entity2.label,
                            "relation": relation_text,
                        }
                        relationships.append(relationship)

        return relationships

    def configure(self, config: dict[str, Any]) -> None:
        """
        Configure the NLP extractor.

        Args:
            config: Configuration dictionary
        """
        if "confidence_threshold" in config:
            self._config["confidence_threshold"] = config["confidence_threshold"]

        if "max_entities" in config:
            self._config["max_entities"] = config["max_entities"]
