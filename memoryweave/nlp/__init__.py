# memoryweave/nlp/__init__.py
"""NLP utility components for MemoryWeave.

This package contains implementations of NLP utilities,
including text extraction, pattern matching, and keyword management.
"""

from memoryweave.nlp.attributes import Attribute
from memoryweave.nlp.entity import Entity
from memoryweave.nlp.extraction import ExtractedAttribute, ExtractedEntity, NLPExtractor
from memoryweave.nlp.factories import NLPFactory
from memoryweave.nlp.keywords import Keyword

__all__ = [
    "NLPExtractor",
    "ExtractedEntity",
    "ExtractedAttribute",
    "Entity",
    "Attribute",
    "Keyword",
    "NLPFactory",
]
