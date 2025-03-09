# memoryweave/nlp/factories.py
from memoryweave.nlp.attributes import PatternAttributeExtractor, SpacyAttributeExtractor
from memoryweave.nlp.entity import RegexEntityExtractor, SpacyEntityExtractor
from memoryweave.nlp.interfaces import (
    AttributeExtractor,
    EntityExtractor,
    KeywordExtractor,
    QueryTypeClassifier,
)
from memoryweave.nlp.keywords import StatisticalKeywordExtractor, YakeKeywordExtractor
from memoryweave.nlp.query import PatternQueryClassifier, SpacyQueryClassifier


class NLPFactory:
    """Factory for creating NLP components."""

    @staticmethod
    def create_entity_extractor() -> EntityExtractor:
        """Create the best available entity extractor."""
        # Try spaCy first
        extractor = SpacyEntityExtractor()
        if extractor.available:
            return extractor

        # Fallback to regex-based extractor
        return RegexEntityExtractor()

    @staticmethod
    def create_attribute_extractor() -> AttributeExtractor:
        """Create the best available attribute extractor."""
        # Try spaCy first
        extractor = SpacyAttributeExtractor()
        if extractor.available:
            return extractor

        # Fallback to pattern-based extractor
        return PatternAttributeExtractor()

    @staticmethod
    def create_keyword_extractor() -> KeywordExtractor:
        """Create the best available keyword extractor."""
        # Try YAKE first
        extractor = YakeKeywordExtractor()
        if extractor.available:
            return extractor

        # Fallback to statistical extractor
        return StatisticalKeywordExtractor()

    @staticmethod
    def create_query_classifier() -> QueryTypeClassifier:
        """Create the best available query classifier."""
        # Try spaCy first
        classifier = SpacyQueryClassifier()
        if hasattr(classifier, "_initialized") and classifier._initialized:
            return classifier

        # Fallback to pattern-based classifier
        return PatternQueryClassifier()
