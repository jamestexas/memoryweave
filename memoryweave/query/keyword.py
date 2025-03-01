"""Keyword extraction and expansion for MemoryWeave.

This module provides implementations for keyword extraction and expansion
to enhance query understanding and retrieval relevance.
"""

import math
from typing import Any, Dict, List, Optional

from memoryweave.interfaces.query import IQueryExpander
from memoryweave.interfaces.retrieval import Query


class KeywordExpander(IQueryExpander):
    """Expands queries with additional related keywords."""

    def __init__(self, word_embeddings: Optional[Dict[str, List[float]]] = None):
        """Initialize the keyword expander.

        Args:
            word_embeddings: Optional dictionary mapping words to embedding vectors
        """
        self._word_embeddings = word_embeddings or {}
        self._word_to_id: Dict[str, int] = {
            word: i for i, word in enumerate(self._word_embeddings.keys())
        }

        # Word relationships (if no embeddings provided)
        self._word_relationships: Dict[str, List[str]] = {}

        # Default configuration
        self._config = {
            "expansion_count": 3,  # Number of keywords to add per original keyword
            "min_similarity": 0.7,  # Minimum similarity for expansion
            "use_embeddings": bool(self._word_embeddings),  # Use embeddings if available
        }

    def expand(self, query: Query) -> Query:
        """Expand a query with additional keywords or concepts."""
        if not query.extracted_keywords:
            return query

        # Create copy of the query to modify
        expanded_query = Query(
            text=query.text,
            embedding=query.embedding,
            query_type=query.query_type,
            extracted_keywords=query.extracted_keywords.copy(),
            extracted_entities=query.extracted_entities,
            context=query.context,
        )

        # Expand each keyword
        expansion_count = self._config["expansion_count"]
        expanded_keywords = set(expanded_query.extracted_keywords)

        for keyword in expanded_query.extracted_keywords:
            related_keywords = self._find_related_keywords(
                keyword, expansion_count, self._config["min_similarity"]
            )
            expanded_keywords.update(related_keywords)

        # Update the query with expanded keywords
        expanded_query.extracted_keywords = list(expanded_keywords)

        return expanded_query

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the keyword expander."""
        if "expansion_count" in config:
            self._config["expansion_count"] = config["expansion_count"]

        if "min_similarity" in config:
            self._config["min_similarity"] = config["min_similarity"]

        if "use_embeddings" in config:
            self._config["use_embeddings"] = config["use_embeddings"]

        # Add word relationships (for use when embeddings not available)
        if "word_relationships" in config:
            for word, related in config["word_relationships"].items():
                self._word_relationships[word] = related

    def _find_related_keywords(self, keyword: str, count: int, min_similarity: float) -> List[str]:
        """Find related keywords for a given keyword."""
        if self._config["use_embeddings"] and keyword in self._word_embeddings:
            return self._find_related_by_embedding(keyword, count, min_similarity)
        else:
            return self._find_related_by_relationships(keyword, count)

    def _find_related_by_embedding(
        self, keyword: str, count: int, min_similarity: float
    ) -> List[str]:
        """Find related keywords using word embeddings."""
        if not self._word_embeddings or keyword not in self._word_embeddings:
            return []

        keyword_vec = self._word_embeddings[keyword]
        similarities = {}

        # Compute similarities with all other words
        for word, vec in self._word_embeddings.items():
            if word != keyword:
                similarity = self._cosine_similarity(keyword_vec, vec)
                if similarity >= min_similarity:
                    similarities[word] = similarity

        # Sort by similarity (descending) and take top k
        related = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in related[:count]]

    def _find_related_by_relationships(self, keyword: str, count: int) -> List[str]:
        """Find related keywords using predefined relationships."""
        related = self._word_relationships.get(keyword.lower(), [])
        return related[:count]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
