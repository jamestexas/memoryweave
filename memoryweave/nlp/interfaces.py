# memoryweave/nlp/interfaces.py
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

# Generic type for extracted items
T = TypeVar("T")


class Extractor(ABC, Generic[T]):
    """Base interface for all extractors."""

    @abstractmethod
    def extract(self, text: str) -> list[T]:
        """Extract information from text."""
        pass

    @property
    @abstractmethod
    def available(self) -> bool:
        """Check if this extractor is available for use."""
        pass


class EntityExtractor(Extractor):
    """Interface for extracting named entities from text."""

    @abstractmethod
    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from text."""
        pass


class AttributeExtractor(Extractor):
    """Interface for extracting personal attributes from text."""

    @abstractmethod
    def extract_attributes(self, text: str) -> dict[str, Any]:
        """Extract personal attributes from text."""
        pass


class KeywordExtractor(Extractor):
    """Interface for extracting keywords from text."""

    @abstractmethod
    def extract_keywords(self, text: str, **kwargs) -> list[str]:
        """Extract keywords from text."""
        pass


class QueryTypeClassifier(ABC):
    """Interface for classifying query types."""

    @abstractmethod
    def classify(self, query: str) -> dict[str, float]:
        """
        Classify a query into different types.

        Returns:
            Dictionary mapping query types to confidence scores
        """
        pass
