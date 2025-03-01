"""Query analysis components for MemoryWeave.

This module provides implementations for query analysis,
including query type classification and keyword extraction.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import re
from collections import Counter

from memoryweave.interfaces.query import IQueryAnalyzer
from memoryweave.interfaces.retrieval import QueryType


class SimpleQueryAnalyzer(IQueryAnalyzer):
    """Simple rule-based query analyzer implementation."""
    
    def __init__(self):
        """Initialize the query analyzer."""
        # Patterns for different query types
        self._personal_patterns = [
            r'\b(?:my|your|I|me|mine|you|yours)\b',
            r'\b(?:remember|told|said|mentioned|talked about)\b',
            r'\b(?:like|enjoy|love|hate|prefer)\b',
            r'\b(?:favorite|opinion|think|feel|believe)\b',
            r'\b(?:family|friend|relative|parent|child|spouse)\b'
        ]
        
        self._factual_patterns = [
            r'\b(?:what is|who is|where is|when is|why is|how is)\b',
            r'\b(?:define|explain|describe|tell me about)\b',
            r'\b(?:fact|information|knowledge|data)\b'
        ]
        
        self._temporal_patterns = [
            r'\b(?:when|time|date|period|era|century|year|month|week|day)\b',
            r'\b(?:before|after|during|while|since|until|ago|past|future)\b',
            r'\b(?:recent|latest|newest|oldest|previous|next|last|first)\b'
        ]
        
        # Common stopwords to exclude from keywords
        self._stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
            'at', 'from', 'by', 'on', 'off', 'for', 'in', 'out', 'over', 'under',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
            'can', 'could', 'may', 'might', 'must', 'to', 'of', 'with'
        }
        
        # Compiled patterns
        self._compiled_personal = [re.compile(pattern, re.IGNORECASE) for pattern in self._personal_patterns]
        self._compiled_factual = [re.compile(pattern, re.IGNORECASE) for pattern in self._factual_patterns]
        self._compiled_temporal = [re.compile(pattern, re.IGNORECASE) for pattern in self._temporal_patterns]
        
        # Default configuration
        self._config = {
            'min_keyword_length': 3,
            'max_keywords': 10
        }
    
    def analyze(self, query_text: str) -> QueryType:
        """Analyze a query to determine its type."""
        # Count matches for each type
        personal_matches = sum(1 for pattern in self._compiled_personal if pattern.search(query_text))
        factual_matches = sum(1 for pattern in self._compiled_factual if pattern.search(query_text))
        temporal_matches = sum(1 for pattern in self._compiled_temporal if pattern.search(query_text))
        
        # Determine type based on match counts
        if personal_matches > factual_matches and personal_matches > temporal_matches:
            return QueryType.PERSONAL
        elif factual_matches > personal_matches and factual_matches > temporal_matches:
            return QueryType.FACTUAL
        elif temporal_matches > personal_matches and temporal_matches > factual_matches:
            return QueryType.TEMPORAL
        elif personal_matches > 0 or factual_matches > 0 or temporal_matches > 0:
            # If there are matches but no clear winner, return the type with the most matches
            max_matches = max(personal_matches, factual_matches, temporal_matches)
            if max_matches == personal_matches:
                return QueryType.PERSONAL
            elif max_matches == factual_matches:
                return QueryType.FACTUAL
            else:
                return QueryType.TEMPORAL
        else:
            # Default to unknown if no matches
            return QueryType.UNKNOWN
    
    def extract_keywords(self, query_text: str) -> List[str]:
        """Extract keywords from a query."""
        # Tokenize and clean the query
        words = re.findall(r'\b\w+\b', query_text.lower())
        
        # Filter out stopwords and short words
        min_length = self._config['min_keyword_length']
        filtered_words = [word for word in words if word not in self._stopwords and len(word) >= min_length]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Sort by frequency (descending) and then by word (ascending)
        sorted_keywords = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
        
        # Take top k keywords
        max_keywords = self._config['max_keywords']
        top_keywords = [word for word, _ in sorted_keywords[:max_keywords]]
        
        return top_keywords
    
    def extract_entities(self, query_text: str) -> List[str]:
        """Extract entities from a query.
        
        Note:
            This is a simple implementation that looks for capitalized words
            and multi-word phrases. For production use, consider using a
            dedicated NER system.
        """
        # Simple pattern for potential named entities (capitalized words)
        entity_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
        entities = entity_pattern.findall(query_text)
        
        # Remove duplicates while preserving order
        unique_entities = []
        seen = set()
        for entity in entities:
            if entity.lower() not in seen:
                unique_entities.append(entity)
                seen.add(entity.lower())
        
        return unique_entities
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the query analyzer."""
        if 'min_keyword_length' in config:
            self._config['min_keyword_length'] = config['min_keyword_length']
        
        if 'max_keywords' in config:
            self._config['max_keywords'] = config['max_keywords']
        
        # Additional patterns can be added through configuration
        if 'personal_patterns' in config:
            for pattern in config['personal_patterns']:
                self._compiled_personal.append(re.compile(pattern, re.IGNORECASE))
        
        if 'factual_patterns' in config:
            for pattern in config['factual_patterns']:
                self._compiled_factual.append(re.compile(pattern, re.IGNORECASE))
        
        if 'temporal_patterns' in config:
            for pattern in config['temporal_patterns']:
                self._compiled_temporal.append(re.compile(pattern, re.IGNORECASE))
        
        if 'stopwords' in config:
            self._stopwords.update(config['stopwords'])