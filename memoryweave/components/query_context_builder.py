"""Query context building for memory retrieval.

This module provides components for enriching query context with relevant information
to improve memory retrieval performance.
"""

from typing import Any, Dict, Optional

import numpy as np

from memoryweave.components.base import Component
from memoryweave.nlp.extraction import NLPExtractor


class QueryContextBuilder(Component):
    """
    Enriches query context with relevant information for improved memory retrieval.

    This component analyzes the query and builds additional context information such as:
    - Extracting temporal markers to prioritize relevant time periods
    - Identifying entities and concepts for better matching
    - Building context from recent interactions
    - Including conversation history if available
    """

    def __init__(self):
        """Initialize the query context builder."""
        self.nlp_extractor = NLPExtractor()

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.max_history_turns = config.get("max_history_turns", 3)
        self.max_history_tokens = config.get("max_history_tokens", 1000)
        self.include_entities = config.get("include_entities", True)
        self.include_temporal_markers = config.get("include_temporal_markers", True)
        self.include_conversation_history = config.get("include_conversation_history", True)
        self.extract_implied_timeframe = config.get("extract_implied_timeframe", True)

    def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a query to build extended context."""
        # Start with existing context
        updated_context = context.copy()

        # Extract entities if enabled
        if self.include_entities:
            entities = self.nlp_extractor.extract_entities(query)
            updated_context["entities"] = entities

        # Extract temporal markers if enabled
        if self.include_temporal_markers:
            temporal_markers = self._extract_temporal_information(query)
            if temporal_markers:
                updated_context["temporal_markers"] = temporal_markers

        # Include conversation history if available and enabled
        if self.include_conversation_history:
            conversation_context = self._build_conversation_context(query, context)
            if conversation_context:
                updated_context["conversation_context"] = conversation_context

        # Enrich query embedding with context if needed
        if "query_embedding" in updated_context and any(
            k in updated_context for k in ["entities", "temporal_markers", "conversation_context"]
        ):
            updated_context["original_query_embedding"] = updated_context["query_embedding"].copy()
            updated_context["query_embedding"] = self._enrich_embedding(
                updated_context["query_embedding"], updated_context
            )

        return updated_context

    def _extract_temporal_information(self, query: str) -> Dict[str, Any]:
        """Extract temporal information from the query."""
        # Extract explicit time references
        time_references = self.nlp_extractor.extract_time_references(query)

        result = {}
        if time_references:
            result["explicit_time_references"] = time_references

        # Extract implied timeframe if enabled
        if self.extract_implied_timeframe:
            timeframe = self._extract_implied_timeframe(query)
            if timeframe:
                result["implied_timeframe"] = timeframe

        return result

    def _extract_implied_timeframe(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract implied timeframe from query patterns."""
        query_lower = query.lower()

        # Look for patterns indicating temporal focus
        if any(marker in query_lower for marker in ["recent", "lately", "today", "now"]):
            return {"focus": "recent", "weight": 0.8}

        if any(marker in query_lower for marker in ["earlier", "before", "previously", "past"]):
            return {"focus": "past", "weight": 0.6}

        if any(marker in query_lower for marker in ["first time", "initially", "originally"]):
            return {"focus": "initial", "weight": 0.7}

        # Look for verbs indicating timeframe
        past_tense_heavy = any(
            verb in query_lower
            for verb in ["happened", "occurred", "said", "told", "mentioned", "was", "were", "did"]
        )
        present_tense_heavy = any(
            verb in query_lower
            for verb in ["is", "are", "do", "does", "am", "know", "think", "feel"]
        )

        if past_tense_heavy and not present_tense_heavy:
            return {"focus": "past", "weight": 0.5}

        if present_tense_heavy and not past_tense_heavy:
            return {"focus": "recent", "weight": 0.5}

        return None

    def _build_conversation_context(
        self, query: str, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Build context from conversation history if available."""
        # Check if conversation history is available
        conversation_history = context.get("conversation_history", [])
        if not conversation_history:
            return None

        # Limit history by number of turns
        recent_history = conversation_history[-self.max_history_turns :]

        # Extract important entities and topics from history
        entities = set()
        topics = set()

        for turn in recent_history:
            # Process both user queries and system responses
            for text in [turn.get("user", ""), turn.get("system", "")]:
                if not text:
                    continue

                # Extract entities
                turn_entities = self.nlp_extractor.extract_entities(text)
                entities.update(turn_entities)

                # Extract topics/keywords
                turn_topics = self.nlp_extractor.extract_keywords(text)
                topics.update(turn_topics)

        # Build context
        conversation_context = {
            "recent_entities": list(entities),
            "recent_topics": list(topics),
            "turns": len(recent_history),
        }

        return conversation_context

    def _enrich_embedding(self, query_embedding: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """Enrich query embedding with contextual information."""
        # This is a simplified implementation. In a real system,
        # you would probably use a more sophisticated approach.

        # Get embedding model from context
        embedding_model = context.get("embedding_model")
        if not embedding_model:
            # Can't enrich without an embedding model
            return query_embedding

        # Entities to include in enriched context
        enrichment_texts = []

        # Add entities if available
        if "entities" in context and context["entities"]:
            enrichment_texts.append("Entities: " + ", ".join(context["entities"]))

        # Add conversation context if available
        if "conversation_context" in context:
            conv_context = context["conversation_context"]
            if "recent_topics" in conv_context and conv_context["recent_topics"]:
                enrichment_texts.append("Topics: " + ", ".join(conv_context["recent_topics"][:5]))

            if "recent_entities" in conv_context and conv_context["recent_entities"]:
                enrichment_texts.append(
                    "Context entities: " + ", ".join(conv_context["recent_entities"][:5])
                )

        # If no enrichment, return original
        if not enrichment_texts:
            return query_embedding

        # Create enriched text
        enriched_text = " ".join(enrichment_texts)

        # Get embedding for enriched text
        try:
            enrichment_embedding = embedding_model.encode(enriched_text)

            # Combine original and enrichment (weighted average)
            enriched_embedding = 0.7 * query_embedding + 0.3 * enrichment_embedding

            # Normalize
            enriched_embedding = enriched_embedding / np.linalg.norm(enriched_embedding)

            return enriched_embedding
        except Exception:
            # On any error, return original embedding
            return query_embedding
