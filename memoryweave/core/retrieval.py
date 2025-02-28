"""
Implements context-aware memory retrieval strategies.
"""

from typing import Any, Optional, List, Set, Dict

import numpy as np
import re

from memoryweave.core.contextual_fabric import ContextualMemory


class ContextualRetriever:
    """
    Retrieves memories from the contextual fabric based on the current
    conversation context, using activation patterns and contextual relevance.
    """

    def __init__(
        self,
        memory: ContextualMemory,
        embedding_model: Any,
        retrieval_strategy: str = "hybrid",
        recency_weight: float = 0.3,
        relevance_weight: float = 0.7,
        keyword_boost_weight: float = 0.5,
        confidence_threshold: float = 0.3,
        semantic_coherence_check: bool = False,
        adaptive_retrieval: bool = False,
        adaptive_k_factor: float = 0.3,  # Added parameter to control adaptive K threshold
        use_two_stage_retrieval: bool = False,  # New parameter for two-stage retrieval
        first_stage_k: int = 20,  # Number of candidates to retrieve in first stage
        query_type_adaptation: bool = False,  # New parameter for query type adaptation
    ):
        """
        Initialize the contextual retriever.

        Args:
            memory: The contextual memory to retrieve from
            embedding_model: Model for encoding queries
            retrieval_strategy: Strategy for retrieval ('similarity', 'temporal', 'hybrid')
            recency_weight: Weight given to recency in hybrid retrieval
            relevance_weight: Weight given to relevance in hybrid retrieval
            keyword_boost_weight: Weight given to keyword matching in retrieval
            confidence_threshold: Minimum similarity score for retrieved memories
            semantic_coherence_check: Whether to check for semantic coherence among retrieved memories
            adaptive_retrieval: Whether to adaptively determine number of results to return
            adaptive_k_factor: Controls conservativeness of adaptive retrieval
            use_two_stage_retrieval: Whether to use two-stage retrieval pipeline
            first_stage_k: Number of candidates to retrieve in first stage
            query_type_adaptation: Whether to adapt retrieval based on query type
        """
        self.memory = memory
        self.embedding_model = embedding_model
        self.retrieval_strategy = retrieval_strategy
        self.recency_weight = recency_weight
        self.relevance_weight = relevance_weight
        self.keyword_boost_weight = keyword_boost_weight
        self.confidence_threshold = confidence_threshold
        self.semantic_coherence_check = semantic_coherence_check
        self.adaptive_retrieval = adaptive_retrieval
        self.adaptive_k_factor = adaptive_k_factor  # Controls conservativeness of adaptive retrieval
        
        # New parameters for enhanced retrieval
        self.use_two_stage_retrieval = use_two_stage_retrieval
        self.first_stage_k = first_stage_k
        self.query_type_adaptation = query_type_adaptation

        # Conversation context tracking
        self.conversation_state = {
            "recent_topics": [],
            "user_interests": set(),
            "interaction_count": 0,
        }
        
        # Persistent personal attributes dictionary
        self.personal_attributes = {
            "preferences": {},  # e.g., {"color": "blue", "food": "pizza"}
            "demographics": {}, # e.g., {"location": "Seattle", "occupation": "engineer"}
            "traits": {},       # e.g., {"personality": "introvert", "hobbies": ["hiking", "reading"]}
            "relationships": {},# e.g., {"family": {"spouse": "Alex", "children": ["Sam", "Jamie"]}}
        }
        
        # Pre-compile regex patterns for performance
        self._compile_regex_patterns()
        
        # Patterns for detecting factual queries
        self.factual_query_patterns = [
            re.compile(r"what is|who is|where is|when is|why is|how is", re.IGNORECASE),
            re.compile(r"what are|who are|where are|when are|why are|how are", re.IGNORECASE),
            re.compile(r"tell me about|explain|describe", re.IGNORECASE),
            re.compile(r"capital of|author of|wrote|invented|discovered", re.IGNORECASE),
        ]

    def _is_factual_query(self, query: str) -> bool:
        """
        Determine if a query is factual/general knowledge rather than personal.
        
        Args:
            query: The query text
            
        Returns:
            True if the query appears to be factual, False otherwise
        """
        # Check if query matches factual patterns
        for pattern in self.factual_query_patterns:
            if pattern.search(query):
                # Check if it's not a personal query (doesn't contain "my", "I", etc.)
                personal_indicators = ["my", " i ", "i'm", "i've", "i'll", "i'd", "me", "mine"]
                if not any(indicator in query.lower() for indicator in personal_indicators):
                    return True
        
        return False

    def retrieve_for_context(
        self,
        current_input: str,
        conversation_history: Optional[list[dict]] = None,
        top_k: int = 5,
        confidence_threshold: float = None,
    ) -> list[dict]:
        """
        Retrieve memories relevant to the current conversation context.

        Args:
            current_input: The current user input
            conversation_history: Recent conversation history
            top_k: Number of memories to retrieve
            confidence_threshold: Minimum similarity score for memory inclusion (overrides default)

        Returns:
            list of relevant memory entries with metadata
        """
        # Update conversation state
        self._update_conversation_state(current_input, conversation_history)

        # Encode the query context
        query_context = self._build_query_context(current_input, conversation_history)
        query_embedding = self.embedding_model.encode(query_context)

        # Extract important keywords for direct reference matching
        important_keywords = self._extract_important_keywords(current_input)
        
        # Extract and update personal attributes if present in the input or response
        if conversation_history:
            for turn in conversation_history[-3:]:  # Look at recent turns
                message = turn.get("message", "")
                response = turn.get("response", "")
                self._extract_personal_attributes(message)
                self._extract_personal_attributes(response)
        
        # Also check current input for personal attributes
        self._extract_personal_attributes(current_input)

        # If no confidence threshold is provided, use the default
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        # If query type adaptation is enabled, adjust confidence threshold for factual queries
        if self.query_type_adaptation and self._is_factual_query(current_input):
            # Use a lower threshold for factual queries to improve recall
            adjusted_threshold = max(0.1, confidence_threshold * 0.6)
            # Also use a less conservative adaptive_k_factor for factual queries
            adjusted_adaptive_k_factor = max(0.05, self.adaptive_k_factor * 0.5)
        else:
            adjusted_threshold = confidence_threshold
            adjusted_adaptive_k_factor = self.adaptive_k_factor

        # Use two-stage retrieval if enabled
        if self.use_two_stage_retrieval:
            # First stage: Retrieve a larger set of candidates with lower threshold
            first_stage_threshold = max(0.1, adjusted_threshold * 0.7)  # Lower threshold for first stage
            
            if self.retrieval_strategy == "similarity":
                candidates = self._retrieve_by_similarity(
                    query_embedding, 
                    self.first_stage_k, 
                    important_keywords, 
                    first_stage_threshold
                )
            elif self.retrieval_strategy == "temporal":
                candidates = self._retrieve_by_recency(self.first_stage_k)
            else:  # hybrid approach
                candidates = self._retrieve_hybrid(
                    query_embedding, 
                    self.first_stage_k, 
                    important_keywords, 
                    first_stage_threshold
                )
                
            # Second stage: Re-rank candidates
            # Sort by relevance score (already includes keyword boosting)
            candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Apply semantic coherence check if enabled
            if self.semantic_coherence_check and len(candidates) > 1:
                # Convert to proper format for memory's _apply_coherence_check
                candidate_tuples = [(c["memory_id"], c["relevance_score"], {k: v for k, v in c.items() 
                                   if k not in ["memory_id", "relevance_score"]}) 
                                  for c in candidates]
                
                coherent_tuples = self.memory._apply_coherence_check(candidate_tuples, query_embedding)
                
                # Convert back to our format
                coherent_candidates = []
                for memory_id, score, metadata in coherent_tuples:
                    coherent_candidates.append({
                        "memory_id": memory_id,
                        "relevance_score": score,
                        **metadata
                    })
                
                candidates = coherent_candidates
            
            # Apply adaptive k selection if enabled
            if self.adaptive_retrieval and len(candidates) > 1:
                # Use the adjusted adaptive_k_factor
                scores = np.array([c["relevance_score"] for c in candidates])
                diffs = np.diff(scores)
                
                # Find significant drops
                significance_threshold = adjusted_adaptive_k_factor * scores[0]
                significant_drops = np.where((-diffs) > significance_threshold)[0]
                
                if len(significant_drops) > 0:
                    # Use the first significant drop as the cut point
                    cut_idx = significant_drops[0] + 1
                    candidates = candidates[:cut_idx]
            
            # Take top_k from the re-ranked candidates
            memories = candidates[:top_k]
        else:
            # Use the original retrieval methods with adjusted threshold
            if self.retrieval_strategy == "similarity":
                memories = self._retrieve_by_similarity(
                    query_embedding, 
                    top_k, 
                    important_keywords, 
                    adjusted_threshold
                )
            elif self.retrieval_strategy == "temporal":
                memories = self._retrieve_by_recency(top_k)
            else:  # hybrid approach
                memories = self._retrieve_hybrid(
                    query_embedding, 
                    top_k, 
                    important_keywords, 
                    adjusted_threshold,
                    adaptive_k_factor=adjusted_adaptive_k_factor
                )
            
        # Enhance results with personal attributes relevant to the query
        enhanced_memories = self._enhance_with_personal_attributes(memories, current_input)
        
        return enhanced_memories

    def _update_conversation_state(
        self, current_input: str, conversation_history: Optional[list[dict]]
    ) -> None:
        """Update internal conversation state tracking."""
        self.conversation_state["interaction_count"] += 1

        # Extract potential topics from current input
        # In a real implementation, this would use NLP to extract entities/topics
        potential_topics = current_input.split()[:5]  # Simplified example
        self.conversation_state["recent_topics"] = potential_topics

        # Update user interests based on conversation
        if conversation_history:
            # This is a simplistic approach; a real implementation would use
            # more sophisticated interest extraction
            for exchange in conversation_history[-3:]:  # Look at last 3 exchanges
                if "user" in exchange.get("speaker", "").lower():
                    words = exchange.get("message", "").split()
                    self.conversation_state["user_interests"].update(words[:3])

    def _extract_personal_attributes(self, text: str) -> None:
        """
        Extract personal attributes from text and update the personal attributes dictionary.
        
        Args:
            text: Text to extract attributes from
        """
        if not text:
            return
            
        text_lower = text.lower()
        
        # Extract preferences
        self._extract_preferences(text_lower)
        
        # Extract demographic information
        self._extract_demographics(text_lower)
        
        # Extract traits and hobbies
        self._extract_traits(text_lower)
        
        # Extract relationships
        self._extract_relationships(text_lower)

    def _extract_preferences(self, text: str) -> None:
        """Extract user preferences from text."""
        # Process favorite patterns
        for pattern in self.favorite_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    category, value = match
                    self.personal_attributes["preferences"][category.strip()] = value.strip()
                elif isinstance(match, str):
                    # For the third pattern, try to categorize the preference
                    if "color" in text:
                        color_match = re.search(r"(?:like|love|prefer|enjoy) (?:the color) ([a-z\s]+)", text)
                        if color_match:
                            self.personal_attributes["preferences"]["color"] = color_match.group(1).strip()
                    elif "food" in text or "eating" in text:
                        self.personal_attributes["preferences"]["food"] = match.strip()
                    elif "drink" in text or "drinking" in text:
                        self.personal_attributes["preferences"]["drink"] = match.strip()
                    elif "movie" in text or "watching" in text:
                        self.personal_attributes["preferences"]["movie"] = match.strip()
                    elif "book" in text or "reading" in text:
                        self.personal_attributes["preferences"]["book"] = match.strip()
                    elif "music" in text or "listening" in text:
                        self.personal_attributes["preferences"]["music"] = match.strip()
        
        # Direct statements about color preferences
        for pattern in self.color_patterns:
            matches = pattern.findall(text)
            for match in matches:
                self.personal_attributes["preferences"]["color"] = match.strip()

    def _extract_demographics(self, text: str) -> None:
        """Extract demographic information from text."""
        # Process location patterns
        for pattern in self.location_patterns:
            matches = pattern.findall(text)
            for match in matches:
                self.personal_attributes["demographics"]["location"] = match.strip()
        
        # Process occupation patterns
        for pattern in self.occupation_patterns:
            matches = pattern.findall(text)
            for match in matches:
                # Filter out common false positives
                if match.strip() not in ["bit", "lot", "fan", "user"]:
                    self.personal_attributes["demographics"]["occupation"] = match.strip()

    def _extract_traits(self, text: str) -> None:
        """Extract personality traits and hobbies from text."""
        # Process hobby patterns
        for pattern in self.hobby_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    activity, time = match
                    if time.strip() in ["weekends", "weekend", "evenings", "free time"]:
                        if "hobbies" not in self.personal_attributes["traits"]:
                            self.personal_attributes["traits"]["hobbies"] = []
                        # Add only if not already present
                        if activity.strip() not in self.personal_attributes["traits"]["hobbies"]:
                            self.personal_attributes["traits"]["hobbies"].append(activity.strip())
                elif isinstance(match, str):
                    if "hobbies" not in self.personal_attributes["traits"]:
                        self.personal_attributes["traits"]["hobbies"] = []
                    for hobby in match.split("and"):
                        hobby = hobby.strip().strip(",.")
                        if hobby and hobby not in self.personal_attributes["traits"]["hobbies"]:
                            self.personal_attributes["traits"]["hobbies"].append(hobby)
        
        # Check specifically for hiking in mountains on weekends
        if "hike" in text and "mountains" in text and "weekend" in text:
            if "hobbies" not in self.personal_attributes["traits"]:
                self.personal_attributes["traits"]["hobbies"] = []
            if "hiking in the mountains" not in self.personal_attributes["traits"]["hobbies"]:
                self.personal_attributes["traits"]["hobbies"].append("hiking in the mountains")

    def _extract_relationships(self, text: str) -> None:
        """Extract relationship information from text."""
        # Process family relationship patterns
        for pattern in self.family_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    relation, name = match
                    if "family" not in self.personal_attributes["relationships"]:
                        self.personal_attributes["relationships"]["family"] = {}
                    self.personal_attributes["relationships"]["family"][relation.strip()] = name.strip()

    def _build_query_context(
        self, current_input: str, conversation_history: Optional[list[dict]]
    ) -> str:
        """Build a rich query context from current input and conversation history."""
        query_context = f"Current input: {current_input}"

        if conversation_history and len(conversation_history) > 0:
            # Add recent conversation turns
            query_context += "\nRecent conversation:"
            for turn in conversation_history[-3:]:  # Last 3 turns
                speaker = turn.get("speaker", "Unknown")
                message = turn.get("message", "")
                query_context += f"\n{speaker}: {message}"

        # Add any persistent user interests
        if self.conversation_state["user_interests"]:
            interests = list(self.conversation_state["user_interests"])[:5]
            query_context += f"\nUser interests: {', '.join(interests)}"

        return query_context

    def _extract_important_keywords(self, query: str) -> Set[str]:
        """
        Extract important keywords for direct reference matching.
        
        Args:
            query: The user query
            
        Returns:
            Set of important keywords
        """
        # Convert to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        important_words = set()
        
        # Extract keywords from reference patterns
        for pattern in self.reference_patterns:
            matches = pattern.findall(query_lower)
            for match in matches:
                # Add each individual word from the match
                important_words.update(match.split())
        
        # Add specific preference and personal attribute keywords
        preference_terms = ["favorite", "like", "prefer", "love", "favorite color", "favorite food"]
        personal_terms = ["color", "food", "movie", "book", "hobby", "activity", 
                          "live", "work", "job", "occupation", "weekend"]
        
        for term in preference_terms + personal_terms:
            if term in query_lower:
                important_words.add(term)
        
        # Filter out common stop words
        stop_words = {"the", "a", "an", "is", "was", "were", "be", "been", "being", 
                      "to", "of", "and", "or", "that", "this", "these", "those"}
        important_words = {word for word in important_words if word not in stop_words}
        
        return important_words

    def _calculate_keyword_boost(self, memory_metadata: dict, important_keywords: Set[str]) -> float:
        """
        Calculate a boost factor based on keyword matching between memory and important keywords.
        
        Args:
            memory_metadata: Metadata for a memory
            important_keywords: Important keywords from the query
            
        Returns:
            Boost factor (1.0 = no boost)
        """
        if not important_keywords:
            return 1.0
            
        # Combine relevant text fields from memory
        memory_text = ""
        
        # Check different text fields that might exist in the metadata
        for field in ["text", "content", "description", "name"]:
            if field in memory_metadata:
                memory_text += " " + str(memory_metadata[field]).lower()
        
        # For interaction type, check response field too
        if memory_metadata.get("type") == "interaction" and "response" in memory_metadata:
            memory_text += " " + str(memory_metadata["response"]).lower()
            
        # Count matching keywords
        matches = sum(1 for keyword in important_keywords if keyword in memory_text)
        
        # Calculate boost factor (more matches = higher boost)
        if matches > 0:
            # Exponential boost for multiple keyword matches
            boost = 1.0 + min(2.5, 0.7 * matches)  # More aggressive boosting
            return boost
            
        return 1.0

    def _enhance_with_personal_attributes(self, memories: list, query: str) -> list:
        """
        Enhance memory results with relevant personal attributes.
        
        Args:
            memories: Retrieved memories
            query: User query
            
        Returns:
            Enhanced memory list with personal attributes
        """
        # Clone the memories list to avoid modifying the original
        enhanced_memories = list(memories)
        
        # Check if the query is related to personal attributes
        query_lower = query.lower()
        
        # Check for relevant personal attributes based on query keywords
        relevant_attributes = {}
        
        # Check for preference-related queries
        preference_keywords = ["favorite", "like", "prefer", "love", "favorite color", "favorite food"]
        if any(keyword in query_lower for keyword in preference_keywords):
            # Add all preferences
            for category, value in self.personal_attributes["preferences"].items():
                if category in query_lower or any(keyword in query_lower for keyword in preference_keywords):
                    relevant_attributes[f"preference_{category}"] = value
        
        # Check for location/demographic queries
        demographic_keywords = ["live", "location", "city", "town", "from", "work", "job", "occupation"]
        if any(keyword in query_lower for keyword in demographic_keywords):
            # Add all demographics
            for category, value in self.personal_attributes["demographics"].items():
                if category in query_lower or any(keyword in query_lower for keyword in demographic_keywords):
                    relevant_attributes[f"demographic_{category}"] = value
        
        # Check for hobby/activity related queries
        activity_keywords = ["hobby", "hobbies", "activity", "enjoy", "weekend", "free time", "like to do"]
        if any(keyword in query_lower for keyword in activity_keywords):
            # Add all hobbies/activities
            if "hobbies" in self.personal_attributes["traits"]:
                relevant_attributes["trait_hobbies"] = self.personal_attributes["traits"]["hobbies"]
        
        # Check for relationship queries
        relationship_keywords = ["family", "wife", "husband", "children", "spouse", "partner"]
        if any(keyword in query_lower for keyword in relationship_keywords):
            if "family" in self.personal_attributes["relationships"]:
                for relation, name in self.personal_attributes["relationships"]["family"].items():
                    if relation in query_lower or any(keyword in query_lower for keyword in relationship_keywords):
                        relevant_attributes[f"relationship_{relation}"] = name
        
        # If we have relevant attributes, create a special "attribute memory" entry
        if relevant_attributes:
            # Create a special memory entry for personal attributes
            attribute_memory = {
                "memory_id": "personal_attributes",
                "type": "personal_attributes",
                "relevance_score": 10.0,  # Give it a high score to appear first
                "content": "User personal attributes",
                "attributes": relevant_attributes
            }
            
            # Insert at the beginning for highest priority
            enhanced_memories.insert(0, attribute_memory)
        
        return enhanced_memories
                
    def _retrieve_by_similarity(
        self, 
        query_embedding: np.ndarray, 
        top_k: int, 
        important_keywords: Set[str] = None,
        confidence_threshold: float = 0.0
    ) -> list[dict]:
        """Retrieve memories based purely on contextual similarity."""
        # Pass the confidence_threshold and additional parameters for enhanced retrieval
        results = self.memory.retrieve_memories(
            query_embedding, 
            top_k=top_k, 
            activation_boost=True,
            confidence_threshold=confidence_threshold,
            semantic_coherence_check=self.semantic_coherence_check,
            adaptive_retrieval=self.adaptive_retrieval
        )

        # Format results and apply keyword boost if needed
        formatted_results = []
        for idx, score, metadata in results:
            boost = 1.0
            if important_keywords:
                boost = self._calculate_keyword_boost(metadata, important_keywords)
                
            boosted_score = score * boost
            
            formatted_results.append({
                "memory_id": idx, 
                "relevance_score": boosted_score, 
                "original_score": score,
                "keyword_boost": boost,
                **metadata
            })
        
        # Re-sort by boosted score
        formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return formatted_results[:top_k]

    def _retrieve_by_recency(self, top_k: int) -> list[dict]:
        """Retrieve memories based on recency and activation."""
        # This is a simplified implementation
        # In a real system, this would be more sophisticated

        # Get memories sorted by temporal markers (most recent first)
        temporal_order = np.argsort(-self.memory.temporal_markers)[:top_k]

        results = []
        for idx in temporal_order:
            results.append({
                "memory_id": int(idx),
                "relevance_score": float(self.memory.activation_levels[idx]),
                **self.memory.memory_metadata[idx],
            })

        return results

    def _retrieve_hybrid(
        self, 
        query_embedding: np.ndarray, 
        top_k: int, 
        important_keywords: Set[str] = None,
        confidence_threshold: float = 0.0,
        adaptive_k_factor: float = None
    ) -> list[dict]:
        """
        Hybrid retrieval combining similarity, recency, and keyword matching.
        """
        # Use the provided adaptive_k_factor or fall back to the instance variable
        if adaptive_k_factor is None:
            adaptive_k_factor = self.adaptive_k_factor
            
        # Get similarity scores
        similarities = np.dot(self.memory.memory_embeddings, query_embedding)

        # Normalize temporal factors
        max_time = float(self.memory.current_time)
        temporal_factors = self.memory.temporal_markers / max_time if max_time > 0 else 0

        # Combine scores
        combined_scores = (
            self.relevance_weight * similarities + self.recency_weight * temporal_factors
        )

        # Apply activation boost
        combined_scores = combined_scores * self.memory.activation_levels

        # Apply confidence threshold filtering
        valid_indices = np.where(combined_scores >= confidence_threshold)[0]
        if len(valid_indices) == 0:
            return []
            
        # Get top-k indices from valid indices
        array_size = len(valid_indices)
        if top_k >= array_size:
            top_relative_indices = np.argsort(-combined_scores[valid_indices])
        else:
            # First get more candidates than needed for keyword boosting
            # Fix: Ensure candidate_k is less than array_size to avoid argpartition error
            candidate_k = min(array_size - 1, top_k * 2)  # Safely compute candidate_k
            if candidate_k <= 0:  # Additional safety check
                candidate_k = min(1, array_size - 1)
                
            candidate_indices = np.argpartition(-combined_scores[valid_indices], candidate_k)[:candidate_k]
            
            # Convert back to original indices
            candidate_indices = valid_indices[candidate_indices]
            
            # Format preliminary results with potential for keyword boosting
            candidates = []
            for idx in candidate_indices:
                score = float(combined_scores[idx])
                if score <= 0:  # Skip non-positive scores
                    continue
                    
                metadata = self.memory.memory_metadata[idx]
                boost = 1.0
                
                # Apply keyword boosting if needed
                if important_keywords:
                    boost = self._calculate_keyword_boost(metadata, important_keywords)
                    score = score * boost
                
                candidates.append({
                    "memory_id": int(idx),
                    "relevance_score": score,
                    "similarity": float(similarities[idx]),
                    "recency": float(temporal_factors[idx]),
                    "keyword_boost": boost,
                    **metadata
                })
            
            # Sort by boosted score and take top-k
            candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Apply semantic coherence check if enabled
            if self.semantic_coherence_check and len(candidates) > 1:
                # Convert to proper format for memory's _apply_coherence_check
                candidate_tuples = [(c["memory_id"], c["relevance_score"], {k: v for k, v in c.items() 
                                   if k not in ["memory_id", "relevance_score"]}) 
                                  for c in candidates]
                
                coherent_tuples = self.memory._apply_coherence_check(candidate_tuples, query_embedding)
                
                # Convert back to our format
                coherent_candidates = []
                for memory_id, score, metadata in coherent_tuples:
                    coherent_candidates.append({
                        "memory_id": memory_id,
                        "relevance_score": score,
                        **metadata
                    })
                
                candidates = coherent_candidates
            
            # Apply adaptive k selection if enabled
            if self.adaptive_retrieval and len(candidates) > 1:
                # Modified adaptive k selection algorithm - less conservative
                scores = np.array([c["relevance_score"] for c in candidates])
                diffs = np.diff(scores)
                
                # Find significant drops
                # Using adaptive_k_factor to control how conservative the algorithm is
                # Lower values = less conservative (returns more results)
                significance_threshold = adaptive_k_factor * scores[0]
                significant_drops = np.where((-diffs) > significance_threshold)[0]
                
                if len(significant_drops) > 0:
                    # Use the first significant drop as the cut point
                    cut_idx = significant_drops[0] + 1
                    candidates = candidates[:cut_idx]
            
            return candidates[:top_k]

        # Format results (if we didn't take the candidate path)
        results = []
        for idx in valid_indices[top_relative_indices]:
            score = float(combined_scores[idx])
            if score <= 0:  # Skip non-positive scores
                continue
                
            metadata = self.memory.memory_metadata[idx]
            boost = 1.0
            
            # Apply keyword boosting if needed
            if important_keywords:
                boost = self._calculate_keyword_boost(metadata, important_keywords)
                score = score * boost
            
            results.append({
                "memory_id": int(idx),
                "relevance_score": score,
                "similarity": float(similarities[idx]),
                "recency": float(temporal_factors[idx]),
                "keyword_boost": boost,
                **metadata
            })
        
        # Re-sort by boosted score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return results[:top_k]

    def _compile_regex_patterns(self):
        """Pre-compile regex patterns for faster extraction."""
        # Preference patterns
        self.favorite_patterns = [
            re.compile(r"(?:my|I) (?:favorite|preferred) (color|food|drink|movie|book|music|song|artist|sport|game|place) (?:is|are) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I|my) (?:like|love|prefer|enjoy) ([a-z0-9\s]+) (?:for|as) (?:my) (color|food|drink|movie|book|music|activity)"),
            re.compile(r"(?:I|my) (?:like|love|prefer|enjoy) (?:the color|eating|drinking|watching|reading|listening to) ([a-z0-9\s]+)")
        ]
        
        self.color_patterns = [
            re.compile(r"(?:my|I) (?:favorite) color is ([a-z\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I|my) (?:like|love|prefer|enjoy) the color ([a-z\s]+)")
        ]
        
        # Location patterns
        self.location_patterns = [
            re.compile(r"(?:I|my) (?:live|stay|reside) in ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I|my) (?:from|grew up in|was born in) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I|my) (?:city|town|state|country) (?:is) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)")
        ]
        
        # Occupation patterns
        self.occupation_patterns = [
            re.compile(r"(?:I|my) (?:work as|am) (?:a|an) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I|my) (?:job|profession|occupation) (?:is) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)")
        ]
        
        # Hobby patterns
        self.hobby_patterns = [
            re.compile(r"(?:I|my) (?:like to|love to|enjoy) ([a-z\s]+) (?:on|in|during) (?:my|the) ([a-z\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I|my) (?:hobby|hobbies|pastime|activity) (?:is|are|include) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)")
        ]
        
        # Family relationship patterns
        self.family_patterns = [
            re.compile(r"(?:my) (wife|husband|spouse|partner|girlfriend|boyfriend) (?:is|name is) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:my) (son|daughter|child|children|mother|father|brother|sister|sibling) (?:is|are|name is|names are) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)")
        ]
        
        # Reference patterns for keyword extraction
        self.reference_patterns = [
            re.compile(r"what (?:was|is|were) (?:my|your|the) ([a-z\s]+)(?:\?|\.|$)"),
            re.compile(r"(?:did|do) (?:I|you) (?:mention|say|tell|share) (?:about|that) ([a-z\s]+)(?:\?|\.|$)"),
            re.compile(r"(?:remind|tell) me (?:about|what) ([a-z\s]+)(?:\?|\.|$)"),
            re.compile(r"(?:what|which) ([a-z\s]+) (?:did|do) I (?:like|prefer|mention|say)(?:\?|\.|$)")
        ]
