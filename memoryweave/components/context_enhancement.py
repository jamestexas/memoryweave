# memoryweave/components/context_enhancement.py
"""
Contextual embedding enhancement components for MemoryWeave.

These components enhance memory embeddings with contextual information from various sources
to improve retrieval relevance and capture relationships between memories.
"""

import time
from typing import Any, Dict, List, Set

import numpy as np

from memoryweave.components.base import Component, MemoryComponent
from memoryweave.components.component_names import ComponentName


class ContextualEmbeddingEnhancer(MemoryComponent):
    """
    Component that enhances memory embeddings with contextual information.
    
    This component modifies memory embeddings by incorporating:
    1. Conversation history context
    2. Temporal context
    3. Topical context
    
    The enhanced embeddings improve retrieval by capturing the multi-dimensional
    relationships between memories beyond pure semantic similarity.
    """

    def __init__(self):
        """Initialize the contextual embedding enhancer."""
        self.conversation_weight = 0.2
        self.temporal_weight = 0.15
        self.topical_weight = 0.25
        self.context_window_size = 5
        self.decay_factor = 0.8
        self.context_history: List[Dict[str, Any]] = []
        self.max_history_items = 20
        self.component_id = ComponentName.CONTEXTUAL_EMBEDDING_ENHANCER

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the component with configuration.
        
        Args:
            config: Configuration dictionary with parameters:
                - conversation_weight: Weight for conversation context (default: 0.2)
                - temporal_weight: Weight for temporal context (default: 0.15)
                - topical_weight: Weight for topical context (default: 0.25)
                - context_window_size: Number of conversation turns to consider (default: 5)
                - decay_factor: How quickly context importance decays (default: 0.8)
                - max_history_items: Maximum number of history items to store (default: 20)
        """
        self.conversation_weight = config.get("conversation_weight", 0.2)
        self.temporal_weight = config.get("temporal_weight", 0.15)
        self.topical_weight = config.get("topical_weight", 0.25)
        self.context_window_size = config.get("context_window_size", 5)
        self.decay_factor = config.get("decay_factor", 0.8)
        self.max_history_items = config.get("max_history_items", 20)

    def process(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process memory data to enhance embeddings with contextual information.
        
        Args:
            data: Memory data to process
            context: Context information including:
                - conversation_history: List of previous conversation turns
                - current_time: Current timestamp
                - topics: Current or detected topics
                
        Returns:
            Updated memory data with enhanced embeddings
        """
        # Extract the base embedding
        embedding = data.get("embedding")
        if embedding is None:
            return data

        # Create a deep copy of the data to avoid modifying the original
        result = dict(data)

        # Extract context information
        conversation_history = context.get("conversation_history", [])
        current_time = context.get("current_time", time.time())
        topics = context.get("topics", set())

        # Store context for future use
        self._update_context_history(data, context)

        # Apply conversation context enhancement
        conversation_embedding = self._extract_conversation_context(conversation_history)

        # Apply temporal context enhancement
        temporal_embedding = self._extract_temporal_context(current_time, data)

        # Apply topical context enhancement
        topical_embedding = self._extract_topical_context(topics, data)

        # Combine all contextual enhancements
        enhanced_embedding = self._combine_embeddings(
            embedding,
            conversation_embedding,
            temporal_embedding,
            topical_embedding
        )

        # Store both original and enhanced embeddings
        result["original_embedding"] = embedding
        result["embedding"] = enhanced_embedding
        result["contextual_enhancements"] = {
            "conversation_weight": self.conversation_weight,
            "temporal_weight": self.temporal_weight,
            "topical_weight": self.topical_weight
        }

        return result

    def _update_context_history(self, data: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Update the internal context history with new information.
        
        Args:
            data: Current memory data
            context: Current context information
        """
        # Create history item
        history_item = {
            "timestamp": context.get("current_time", time.time()),
            "topics": list(context.get("topics", set())),
            "memory_id": data.get("id"),
            "query": context.get("query", "")
        }

        # Add to history
        self.context_history.append(history_item)

        # Limit history size
        if len(self.context_history) > self.max_history_items:
            self.context_history = self.context_history[-self.max_history_items:]

    def _extract_conversation_context(self, conversation_history: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract context embedding from conversation history.
        
        Args:
            conversation_history: List of previous conversation turns with embeddings
            
        Returns:
            Context embedding extracted from conversation history
        """
        # If no conversation history, return zero vector
        if not conversation_history:
            # Return a zero vector of standard embedding size
            return np.zeros(768)

        # Get recent history within window size
        recent_history = conversation_history[-self.context_window_size:]

        # Extract embeddings with recency weighting
        weighted_embeddings = []

        for i, turn in enumerate(recent_history):
            if "embedding" in turn:
                # Apply recency weight: more recent turns have higher weight
                recency_weight = self.decay_factor ** (len(recent_history) - i - 1)
                weighted_embedding = turn["embedding"] * recency_weight
                weighted_embeddings.append(weighted_embedding)

        # If no valid embeddings found, return zero vector
        if not weighted_embeddings:
            # Return a zero vector of standard embedding size
            return np.zeros(768)

        # Combine weighted embeddings
        combined_embedding = np.sum(weighted_embeddings, axis=0)

        # Normalize the combined embedding
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm

        return combined_embedding

    def _extract_temporal_context(self, current_time: float, data: Dict[str, Any]) -> np.ndarray:
        """
        Extract temporal context embedding.
        
        Args:
            current_time: Current timestamp
            data: Memory data containing creation time
            
        Returns:
            Temporal context embedding
        """
        # Extract memory creation time
        memory_time = data.get("created_at", 0)

        # If no valid time, return zero vector
        if not memory_time:
            return np.zeros(768)

        # Calculate time-based features
        time_diff = current_time - memory_time

        # Create time features vector (similar to positional encoding)
        # This creates a distinctive pattern based on recency
        time_vector = np.zeros(768)

        # Extract time units (days, hours, etc.)
        seconds_in_day = 86400
        days_diff = time_diff / seconds_in_day

        # Fill first part of vector with time-based patterns
        for i in range(min(50, len(time_vector))):
            if i % 2 == 0:
                time_vector[i] = np.sin(days_diff / (10000 ** (i / 50)))
            else:
                time_vector[i] = np.cos(days_diff / (10000 ** ((i-1) / 50)))

        # Add recency marker (more recent = higher value)
        recency_value = np.exp(-time_diff / (30 * seconds_in_day))  # 30-day scale
        time_vector[50:60] = recency_value

        # Normalize
        norm = np.linalg.norm(time_vector)
        if norm > 0:
            time_vector = time_vector / norm

        return time_vector

    def _extract_topical_context(self, topics, data: Dict[str, Any]) -> np.ndarray:
        """
        Extract topical context embedding.
        
        Args:
            topics: Set or list of current topics
            data: Memory data with potential topic information
            
        Returns:
            Topical context embedding
        """
        # If no topics, return zero vector
        if not topics:
            return np.zeros(768)

        # Get memory topics and ensure it's a set
        memory_topics = set(data.get("topics", []))
        
        # Convert topics to set if it's not already
        if not isinstance(topics, set):
            topics_set = set(topics)
        else:
            topics_set = topics

        # Calculate topic overlap
        common_topics = topics_set.intersection(memory_topics)
        topic_overlap_ratio = len(common_topics) / max(1, len(topics_set))

        # Create a topic context vector
        topic_vector = np.zeros(768)

        # Encode topic overlap in the vector
        topic_vector[0:10] = topic_overlap_ratio

        # Encode each topic using a hash-based approach
        for topic in topics_set:
            # Use hash of topic to determine which dimensions to affect
            hash_val = hash(topic) % 100  # Use modulo to limit to first 100 dimensions
            start_idx = 100 + hash_val
            end_idx = min(start_idx + 10, 768)

            # Set values in those dimensions
            for i in range(start_idx, end_idx):
                if i < 768:  # Safety check
                    topic_vector[i] = 0.5

        # Normalize
        norm = np.linalg.norm(topic_vector)
        if norm > 0:
            topic_vector = topic_vector / norm

        return topic_vector

    def _combine_embeddings(
        self,
        base_embedding: np.ndarray,
        conversation_embedding: np.ndarray,
        temporal_embedding: np.ndarray,
        topical_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Combine all embeddings into a single enhanced embedding.
        
        Args:
            base_embedding: Original memory embedding
            conversation_embedding: Conversation context embedding
            temporal_embedding: Temporal context embedding
            topical_embedding: Topical context embedding
            
        Returns:
            Combined enhanced embedding
        """
        # Ensure all embeddings have the same dimension
        dim = len(base_embedding)

        # Resize contextual embeddings if needed
        if len(conversation_embedding) != dim:
            conversation_embedding = np.resize(conversation_embedding, dim)
        if len(temporal_embedding) != dim:
            temporal_embedding = np.resize(temporal_embedding, dim)
        if len(topical_embedding) != dim:
            topical_embedding = np.resize(topical_embedding, dim)

        # Apply weighted combination
        base_weight = 1.0 - (self.conversation_weight + self.temporal_weight + self.topical_weight)
        enhanced_embedding = (
            base_weight * base_embedding +
            self.conversation_weight * conversation_embedding +
            self.temporal_weight * temporal_embedding +
            self.topical_weight * topical_embedding
        )

        # Normalize the enhanced embedding
        norm = np.linalg.norm(enhanced_embedding)
        if norm > 0:
            enhanced_embedding = enhanced_embedding / norm

        return enhanced_embedding


class ContextSignalExtractor(Component):
    """
    Utility component for extracting contextual signals from different sources.
    
    This component provides methods to extract contextual information from:
    - Conversation history
    - User behavior
    - Documents and content
    - Temporal patterns
    
    It can be used by other contextual components to extract meaningful signals.
    """

    def __init__(self):
        """Initialize the context signal extractor."""
        self.component_id = "context_signal_extractor"

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the component with configuration.
        
        Args:
            config: Configuration dictionary
        """
        pass

    def extract_conversation_signals(
        self,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract signals from conversation history.
        
        Args:
            conversation_history: List of conversation turns
            
        Returns:
            Dictionary of extracted signals
        """
        signals = {
            "topics": set(),
            "entities": set(),
            "sentiment": 0.0,
            "question_count": 0,
            "average_turn_length": 0
        }

        if not conversation_history:
            return signals

        # Extract basic statistics
        total_length = 0
        question_count = 0

        for turn in conversation_history:
            text = turn.get("text", "")
            total_length += len(text)

            # Simple question detection
            if "?" in text:
                question_count += 1

            # Add topics and entities if available
            signals["topics"].update(turn.get("topics", []))
            signals["entities"].update(turn.get("entities", []))

        # Calculate averages
        signals["question_count"] = question_count
        signals["average_turn_length"] = total_length / len(conversation_history) if conversation_history else 0

        return signals

    def extract_temporal_signals(self, timestamps: List[float]) -> Dict[str, Any]:
        """
        Extract temporal signals from a list of timestamps.
        
        Args:
            timestamps: List of timestamps
            
        Returns:
            Dictionary of temporal signals
        """
        if not timestamps:
            return {"pattern": "none", "frequency": 0, "recency": 0}

        current_time = time.time()
        timestamps.sort()

        # Calculate recency (how recent is the latest timestamp)
        recency = current_time - timestamps[-1] if timestamps else float('inf')

        # Calculate intervals between timestamps
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]

        # Calculate frequency (average interval)
        frequency = sum(intervals) / len(intervals) if intervals else 0

        # Detect temporal patterns
        pattern = "random"
        if len(intervals) >= 3:
            # Check for regular intervals (consistent frequency)
            variance = np.var(intervals) if len(intervals) > 1 else float('inf')
            mean = np.mean(intervals)

            if variance / mean < 0.2:  # Low variance indicates regularity
                pattern = "regular"
            elif all(intervals[i] < intervals[i-1] for i in range(1, len(intervals))):
                pattern = "accelerating"
            elif all(intervals[i] > intervals[i-1] for i in range(1, len(intervals))):
                pattern = "decelerating"

        return {
            "pattern": pattern,
            "frequency": frequency,
            "recency": recency
        }

    def extract_content_signals(self, content: str) -> Dict[str, Any]:
        """
        Extract signals from content text.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dictionary of content signals
        """
        # Basic content analysis
        word_count = len(content.split())

        # Simple topic extraction (just keywords for now)
        import re
        words = re.findall(r'\b\w{4,}\b', content.lower())
        keywords = set([word for word in words if len(word) > 4])

        # Simple entity extraction (capitalized words)
        entity_matches = re.findall(r'\b[A-Z][a-z]{2,}\b', content)
        entities = set(entity_matches)

        # Calculate reading level (very basic approximation)
        avg_word_length = sum(len(word) for word in content.split()) / max(1, word_count)

        return {
            "word_count": word_count,
            "keywords": keywords,
            "entities": entities,
            "avg_word_length": avg_word_length
        }
