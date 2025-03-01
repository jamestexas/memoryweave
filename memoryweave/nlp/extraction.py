"""NLP extraction utilities for MemoryWeave.

This module provides core extraction functionality for extracting
information from text, such as entities, attributes, and relationships.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from memoryweave.nlp.patterns import PERSONAL_ATTRIBUTE_PATTERNS


@dataclass
class ExtractedEntity:
    """A named entity extracted from text."""
    text: str
    label: str
    start: int
    end: int


@dataclass
class ExtractedAttribute:
    """A personal attribute extracted from text."""
    attribute: str
    value: str
    confidence: float


class NLPExtractor:
    """Core NLP extraction functionality for MemoryWeave."""

    def __init__(self):
        """Initialize the NLP extractor."""
        # Compile patterns for personal attributes
        self._compiled_patterns = {}
        for attr_type, patterns in PERSONAL_ATTRIBUTE_PATTERNS.items():
            self._compiled_patterns[attr_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        # Entity extraction patterns
        self._entity_patterns = {
            'PERSON': re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'),
            'LOCATION': re.compile(r'\b([A-Z][a-z]+(?:,\s+[A-Z][a-z]+)*)\b'),
            'ORGANIZATION': re.compile(r'\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)+)\b'),
            'DATE': re.compile(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b|\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:[a-z]*)?(?:,\s+\d{4})?)\b')
        }

        # Default configuration
        self._config = {
            'confidence_threshold': 0.7,
            'max_entities': 10
        }

    def extract_personal_attributes(self, text: str) -> List[ExtractedAttribute]:
        """Extract personal attributes from text."""
        results = []

        # Search for matches in each attribute type
        for attr_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    # Get match groups
                    groups = match.groups()

                    # Basic sanity check on the match
                    if not groups or all(not g for g in groups if g is not None):
                        continue

                    # The last non-None group is typically the attribute value
                    value = next((g for g in reversed(groups) if g is not None), "")

                    # Calculate a confidence score based on:
                    # - Length of the match (longer matches are more likely to be correct)
                    # - Pattern specificity (more specific patterns get higher confidence)
                    match_len_factor = min(len(value) / 10.0, 1.0)
                    pattern_specificity = 0.8  # Default specificity
                    confidence = 0.5 + (match_len_factor * 0.3) + (pattern_specificity * 0.2)

                    # Create attribute
                    attribute = ExtractedAttribute(
                        attribute=attr_type,
                        value=value.strip(),
                        confidence=confidence
                    )

                    # Add to results if confidence exceeds threshold
                    if confidence >= self._config['confidence_threshold']:
                        results.append(attribute)

        return results

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract named entities from text."""
        results = []

        # Search for entity matches
        for entity_type, pattern in self._entity_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                # Get match span and text
                start, end = match.span()
                entity_text = match.group(0)

                # Create entity
                entity = ExtractedEntity(
                    text=entity_text,
                    label=entity_type,
                    start=start,
                    end=end
                )

                results.append(entity)

        # Sort by position in text
        results.sort(key=lambda e: e.start)

        # Remove duplicates and overlaps
        filtered_results = self._remove_overlapping_entities(results)

        # Limit to max entities
        return filtered_results[:self._config['max_entities']]

    def extract_keywords(self, text: str, stopwords: Optional[Set[str]] = None) -> List[str]:
        """Extract keywords from text."""
        from memoryweave.nlp.keywords import extract_keywords

        # Use the keyword extraction module
        return extract_keywords(text, stopwords)

    def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
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
                    relation_text = text[entity1.end:entity2.start].strip()

                    # If there's meaningful text between them, consider it a relationship
                    if len(relation_text) >= 3:
                        relationship = {
                            'entity1': entity1.text,
                            'entity1_type': entity1.label,
                            'entity2': entity2.text,
                            'entity2_type': entity2.label,
                            'relation': relation_text
                        }
                        relationships.append(relationship)

        return relationships

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the NLP extractor."""
        if 'confidence_threshold' in config:
            self._config['confidence_threshold'] = config['confidence_threshold']

        if 'max_entities' in config:
            self._config['max_entities'] = config['max_entities']

    def _remove_overlapping_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove overlapping entities, keeping the longest ones."""
        if not entities:
            return []

        # Sort by length (descending) to prefer longer entities
        sorted_entities = sorted(entities, key=lambda e: e.end - e.start, reverse=True)

        # Keep track of used character positions
        used_positions = set()
        filtered_entities = []

        for entity in sorted_entities:
            # Check if this entity overlaps with already selected entities
            entity_positions = set(range(entity.start, entity.end))
            if not entity_positions.intersection(used_positions):
                # No overlap, add to filtered list
                filtered_entities.append(entity)
                used_positions.update(entity_positions)

        # Sort by position in text
        filtered_entities.sort(key=lambda e: e.start)

        return filtered_entities
