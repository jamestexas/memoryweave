"""
Query analysis component for MemoryWeave.
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


class QueryAnalyzer(Component):
    """
    Component for analyzing user queries.

    This component:
    1. Determines the primary query type (personal, factual, opinion, etc.)
    2. Extracts important keywords from the query
    3. Identifies entities and concepts
    """

    def __init__(self):
        """Initialize the query analyzer."""
        pass

    def initialize(self, config):
        """
        Initialize with configuration.

        Args:
            config: Configuration dictionary
        """
        pass

    def process_query(self, query, context):
        """
        Analyze a query to determine its type and extract features.

        Args:
            query: Query string
            context: Additional context

        Returns:
            Updated context with query analysis results
        """
        # Determine query type
        query_type = self._determine_query_type(query)

        # Extract important keywords
        keywords = self._extract_keywords(query)

        # Add results to context
        context["primary_query_type"] = query_type
        context["important_keywords"] = keywords

        return context

    def _determine_query_type(self, query):
        """
        Determine the primary type of the query.

        Args:
            query: Query string

        Returns:
            Query type string: "personal", "factual", "opinion", or "instruction"
        """
        query = query.lower()

        # Check for personal queries
        personal_patterns = [
            r"\bmy\b",
            r"\bmine\b",
            r"\bour\b",
            r"\bme\b",
            r"\bwe\b",
            r"what.*\bi\b.*\?",
            r"where.*\bi\b.*\?",
            r"who.*\bmy\b.*\?",
            r"when.*\bi\b.*\?",
        ]

        for pattern in personal_patterns:
            if re.search(pattern, query):
                return "personal"

        # Check for opinion queries
        opinion_patterns = [
            r"what.*\byou think\b",
            r"what.*\byour opinion\b",
            r"do you believe",
            r"how.*\byou feel\b",
            r"\byour thoughts\b",
        ]

        for pattern in opinion_patterns:
            if re.search(pattern, query):
                return "opinion"

        # Check for instruction queries
        instruction_patterns = [
            r"^(?:please |)(?:write|create|make|generate|code|implement)",
            r"^(?:please |)(?:find|search|locate|identify)",
            r"^(?:please |)(?:explain|describe|summarize)",
            r"^(?:please |)(?:calculate|compute)",
        ]

        for pattern in instruction_patterns:
            if re.search(pattern, query):
                return "instruction"

        # Default to factual
        return "factual"

    def _extract_keywords(self, query):
        """
        Extract important keywords from the query.

        Args:
            query: Query string

        Returns:
            Set of important keywords
        """
        if not nlp:
            # Fallback if spaCy not available
            words = query.lower().split()
            return set(words)

        # Process the query with spaCy
        doc = nlp(query)

        # Extract nouns, verbs, and named entities
        important_tokens = set()

        # Add nouns and verbs
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN", "VERB"]:
                if not token.is_stop and len(token.text) > 1:
                    important_tokens.add(token.text.lower())

        # Add named entities
        for ent in doc.ents:
            important_tokens.add(ent.text.lower())

        # If no important tokens found, add all non-stop words
        if not important_tokens:
            important_tokens = {
                token.text.lower() for token in doc if not token.is_stop and len(token.text) > 1
            }

        return important_tokens
