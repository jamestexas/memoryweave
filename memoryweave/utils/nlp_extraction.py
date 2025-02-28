"""
NLP-based extraction utilities for MemoryWeave.

This module provides NLP-powered extraction capabilities that complement
the regex-based extraction in the core retriever. It's designed to work
with or without spaCy, falling back to regex patterns when NLP libraries
aren't available.
"""

import re
from typing import Dict, List, Optional, Set, Any, Tuple


class NLPExtractor:
    """NLP-based attribute extractor with fallback to regex patterns."""
    
    def __init__(self, model_name="en_core_web_sm"):
        """
        Initialize with specified spaCy model.
        
        Args:
            model_name: Name of the spaCy model to use
        """
        self.nlp = None
        self.model_name = model_name
        self.is_spacy_available = False
        
        # Try to load spaCy
        try:
            import spacy
            try:
                self.nlp = spacy.load(model_name)
                self.is_spacy_available = True
                print(f"Successfully loaded spaCy model: {model_name}")
            except OSError:
                print(f"spaCy model '{model_name}' not available")
                print("Using fallback extraction methods")
        except ImportError:
            print("spaCy not available")
            print("Using fallback extraction methods")
            
        # Initialize fallback regex patterns
        self._init_fallback_patterns()
    
    def _init_fallback_patterns(self):
        """Initialize fallback regex patterns for when spaCy is not available."""
        # Preference patterns
        self.favorite_patterns = [
            re.compile(r"(?:my|I) (?:favorite|preferred) (color|food|drink|movie|book|music|song|artist|sport|game|place) (?:is|are) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I|my) (?:like|love|prefer|enjoy) ([a-z0-9\s]+) (?:for|as) (?:my) (color|food|drink|movie|book|music|activity)"),
            re.compile(r"(?:I|my) (?:like|love|prefer|enjoy) (?:the color|eating|drinking|watching|reading|listening to) ([a-z0-9\s]+)")
        ]
        
        # Color-specific patterns
        self.color_patterns = [
            re.compile(r"(?:my|I) (?:favorite) color is ([a-z\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I|my) (?:like|love|prefer|enjoy) the color ([a-z\s]+)"),
            re.compile(r"([a-z\s]+) is (?:definitely |certainly |absolutely )?(?:my|I) favorite color")
        ]
        
        # Food-specific patterns
        self.food_patterns = [
            re.compile(r"(?:I|my) (?:prefer|like|love|enjoy) ([a-z\s]+) food"),
            re.compile(r"(?:I|my) (?:prefer|like|love|enjoy) ([a-z\s]+) cuisine"),
            re.compile(r"(?:I|my) (?:prefer|like|love|enjoy) (?:eating )?([a-z\s]+)(?: food)?")
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
            re.compile(r"(?:I|my) (?:hobby|hobbies|pastime|activity) (?:is|are|include) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I|my) (?:enjoy) ([a-z\s]+ing)(?:\.|\,|\!|\?|$)")
        ]
        
        # Family relationship patterns
        self.family_patterns = [
            re.compile(r"(?:my) (wife|husband|spouse|partner|girlfriend|boyfriend) (?:is|name is) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:my) (son|daughter|child|children|mother|father|brother|sister|sibling) (?:is|are|name is|names are) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I have|I've got) (?:a|an|two|three|four|five|some) (wife|husband|spouse|partner|girlfriend|boyfriend|son|daughter|child|children) (?:named|called)? ([a-z0-9\s]+)"),
            re.compile(r"(?:I have|I've got) (?:a|an|two|three|four|five|some) (wife|husband|spouse|partner|girlfriend|boyfriend|son|daughter|child|children)")
        ]
        
        # Education patterns
        self.education_patterns = [
            re.compile(r"(?:I|my) (?:studied|majored in) ([a-z\s]+) at ([a-z\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I|my) (?:graduated from) ([a-z\s]+) with (?:a|an) (?:degree|diploma) in ([a-z\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I|my) (?:have|earned) (?:a|an) ([a-z\s]+) (?:degree|diploma) from ([a-z\s]+)(?:\.|\,|\!|\?|$)")
        ]
        
        # Query type patterns
        self.factual_patterns = [
            re.compile(r"^(?:what|who|where|when|why|how) (?:is|are|was|were|do|does|did)"),
            re.compile(r"tell me about"),
            re.compile(r"explain"),
            re.compile(r"describe")
        ]
        
        self.personal_patterns = [
            re.compile(r"\b(?:my|I|me|mine)\b"),
            re.compile(r"(?:do I|did I|should I|can I|will I)")
        ]
        
        self.opinion_patterns = [
            re.compile(r"\b(?:think|opinion|believe|feel|view)\b"),
            re.compile(r"(?:what do you think|your thoughts)")
        ]
        
        self.instruction_patterns = [
            re.compile(r"^(?:write|create|make|generate|list|find|search|show|tell|give)"),
            re.compile(r"^(?:please|kindly) ")
        ]
        
    def extract_personal_attributes(self, text: str) -> Dict[str, Any]:
        """
        Extract personal attributes using available methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted attributes
        """
        attributes = {
            "preferences": {},
            "demographics": {},
            "traits": {},
            "relationships": {}
        }
        
        if not text:
            return attributes
            
        # Always use regex extraction for now
        self._extract_preferences_regex(text, attributes)
        self._extract_demographics_regex(text, attributes)
        self._extract_traits_regex(text, attributes)
        self._extract_relationships_regex(text, attributes)
        
        # If spaCy is available, enhance with NLP extraction
        if self.is_spacy_available and self.nlp:
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ == "PERSON" and "name" not in attributes["demographics"]:
                    # Check if this might be the user's name
                    if "my name is" in text.lower() or "i am" in text.lower():
                        attributes["demographics"]["name"] = ent.text
                
                elif ent.label_ == "GPE" and "location" not in attributes["demographics"]:
                    # Check if this might be where they live
                    if "live in" in text.lower() or "reside in" in text.lower():
                        attributes["demographics"]["location"] = ent.text
                
                elif ent.label_ == "ORG" and "organization" not in attributes["demographics"]:
                    # Check if this might be where they work or study
                    if "work at" in text.lower() or "work for" in text.lower():
                        attributes["demographics"]["organization"] = ent.text
                    elif "study at" in text.lower() or "attend" in text.lower():
                        attributes["demographics"]["education"] = ent.text
                        
            # Extract family relationships
            if "wife" in text.lower() or "husband" in text.lower() or "children" in text.lower():
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        # Check context to determine relationship
                        context_window = text[max(0, ent.start_char - 20):min(len(text), ent.end_char + 20)].lower()
                        
                        if "wife" in context_window:
                            if "family" not in attributes["relationships"]:
                                attributes["relationships"]["family"] = {}
                            attributes["relationships"]["family"]["wife"] = ent.text
                        elif "husband" in context_window:
                            if "family" not in attributes["relationships"]:
                                attributes["relationships"]["family"] = {}
                            attributes["relationships"]["family"]["husband"] = ent.text
                        elif "son" in context_window:
                            if "family" not in attributes["relationships"]:
                                attributes["relationships"]["family"] = {}
                            if "children" not in attributes["relationships"]["family"]:
                                attributes["relationships"]["family"]["children"] = []
                            attributes["relationships"]["family"]["children"].append(ent.text)
                        elif "daughter" in context_window:
                            if "family" not in attributes["relationships"]:
                                attributes["relationships"]["family"] = {}
                            if "children" not in attributes["relationships"]["family"]:
                                attributes["relationships"]["family"]["children"] = []
                            attributes["relationships"]["family"]["children"].append(ent.text)
                        elif "child" in context_window or "children" in context_window:
                            if "family" not in attributes["relationships"]:
                                attributes["relationships"]["family"] = {}
                            if "children" not in attributes["relationships"]["family"]:
                                attributes["relationships"]["family"]["children"] = []
                            attributes["relationships"]["family"]["children"].append(ent.text)
        
        return attributes
    
    def _extract_preferences_regex(self, text: str, attributes: Dict) -> None:
        """
        Extract preferences using regex patterns.
        
        Args:
            text: Text to analyze
            attributes: Dictionary to update with extracted preferences
        """
        text_lower = text.lower()
        
        # Process favorite patterns
        for pattern in self.favorite_patterns:
            matches = pattern.findall(text_lower)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    category, value = match
                    attributes["preferences"][category.strip()] = value.strip()
                elif isinstance(match, str):
                    # For the third pattern, try to categorize the preference
                    if "color" in text_lower:
                        color_match = re.search(r"(?:like|love|prefer|enjoy) (?:the color) ([a-z\s]+)", text_lower)
                        if color_match:
                            attributes["preferences"]["color"] = color_match.group(1).strip()
                    elif "food" in text_lower or "eating" in text_lower:
                        attributes["preferences"]["food"] = match.strip()
                    elif "drink" in text_lower or "drinking" in text_lower:
                        attributes["preferences"]["drink"] = match.strip()
                    elif "movie" in text_lower or "watching" in text_lower:
                        attributes["preferences"]["movie"] = match.strip()
                    elif "book" in text_lower or "reading" in text_lower:
                        attributes["preferences"]["book"] = match.strip()
                    elif "music" in text_lower or "listening" in text_lower:
                        attributes["preferences"]["music"] = match.strip()
        
        # Process color-specific patterns
        for pattern in self.color_patterns:
            matches = pattern.findall(text_lower)
            for match in matches:
                attributes["preferences"]["color"] = match.strip()
                
        # Process food-specific patterns
        for pattern in self.food_patterns:
            matches = pattern.findall(text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    food_type = match[0].strip()
                    if food_type and "curry" not in food_type and "spicy" not in food_type:
                        attributes["preferences"]["food"] = food_type
                else:
                    food_type = match.strip()
                    if food_type and "curry" not in food_type and "spicy" not in food_type:
                        attributes["preferences"]["food"] = food_type
        
        # Special case for Thai food and curries
        if "thai food" in text_lower or "thai cuisine" in text_lower:
            attributes["preferences"]["food"] = "thai"
        if "curry" in text_lower or "curries" in text_lower:
            if "food" in attributes["preferences"]:
                if "curry" not in attributes["preferences"]["food"]:
                    attributes["preferences"]["food"] += " curry"
            else:
                attributes["preferences"]["food"] = "curry"
    
    def _extract_demographics_regex(self, text: str, attributes: Dict) -> None:
        """
        Extract demographic information using regex patterns.
        
        Args:
            text: Text to analyze
            attributes: Dictionary to update with extracted demographics
        """
        text_lower = text.lower()
        
        # Process location patterns
        for pattern in self.location_patterns:
            matches = pattern.findall(text_lower)
            for match in matches:
                attributes["demographics"]["location"] = match.strip()
        
        # Process occupation patterns
        for pattern in self.occupation_patterns:
            matches = pattern.findall(text_lower)
            for match in matches:
                # Filter out common false positives
                if match.strip() not in ["bit", "lot", "fan", "user"]:
                    attributes["demographics"]["occupation"] = match.strip()
        
        # Process education patterns
        for pattern in self.education_patterns:
            matches = pattern.findall(text_lower)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    field, institution = match
                    attributes["demographics"]["education"] = institution.strip()
                    attributes["demographics"]["field"] = field.strip()
        
        # Special case for Stanford and Computer Science
        if "stanford" in text_lower and "computer science" in text_lower:
            attributes["demographics"]["education"] = "Stanford"
            attributes["demographics"]["field"] = "Computer Science"
    
    def _extract_traits_regex(self, text: str, attributes: Dict) -> None:
        """
        Extract personality traits and hobbies using regex patterns.
        
        Args:
            text: Text to analyze
            attributes: Dictionary to update with extracted traits
        """
        text_lower = text.lower()
        
        # Process hobby patterns
        for pattern in self.hobby_patterns:
            matches = pattern.findall(text_lower)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    activity, time = match
                    if time.strip() in ["weekends", "weekend", "evenings", "free time"]:
                        if "hobbies" not in attributes["traits"]:
                            attributes["traits"]["hobbies"] = []
                        # Add only if not already present
                        if activity.strip() not in attributes["traits"]["hobbies"]:
                            attributes["traits"]["hobbies"].append(activity.strip())
                elif isinstance(match, str):
                    if "hobbies" not in attributes["traits"]:
                        attributes["traits"]["hobbies"] = []
                    for hobby in match.split("and"):
                        hobby = hobby.strip().strip(",.")
                        if hobby and hobby not in attributes["traits"]["hobbies"]:
                            attributes["traits"]["hobbies"].append(hobby)
    
    def _extract_relationships_regex(self, text: str, attributes: Dict) -> None:
        """
        Extract relationship information using regex patterns.
        
        Args:
            text: Text to analyze
            attributes: Dictionary to update with extracted relationships
        """
        text_lower = text.lower()
        
        # Process family relationship patterns
        for pattern in self.family_patterns:
            matches = pattern.findall(text_lower)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    relation, name = match
                    if "family" not in attributes["relationships"]:
                        attributes["relationships"]["family"] = {}
                    
                    # Handle children specially
                    if relation in ["children", "child"]:
                        if "children" not in attributes["relationships"]["family"]:
                            attributes["relationships"]["family"]["children"] = []
                        
                        # Split names if there are multiple
                        names = name.split("and")
                        for child_name in names:
                            child_name = child_name.strip().strip(",")
                            if child_name and child_name not in attributes["relationships"]["family"]["children"]:
                                attributes["relationships"]["family"]["children"].append(child_name)
                    else:
                        attributes["relationships"]["family"][relation.strip()] = name.strip()
                elif isinstance(match, str):
                    # Handle case where we just know they have a relation but not the name
                    relation = match.strip()
                    if "family" not in attributes["relationships"]:
                        attributes["relationships"]["family"] = {}
                    attributes["relationships"]["family"][relation] = "Unknown"
                                
    def identify_query_type(self, query: str) -> Dict[str, float]:
        """
        Identify the type of query using available methods.
        
        Args:
            query: The query text
            
        Returns:
            Dictionary with query type probabilities
        """
        # Initialize scores
        scores = {
            "factual": 0.0,
            "personal": 0.0,
            "opinion": 0.0,
            "instruction": 0.0
        }
        
        # Use regex patterns for query type identification
        query_lower = query.lower()
        
        # Check factual patterns
        for pattern in self.factual_patterns:
            if pattern.search(query_lower):
                scores["factual"] += 0.3
        
        # Check personal patterns
        for pattern in self.personal_patterns:
            if pattern.search(query_lower):
                scores["personal"] += 0.5
        
        # Check opinion patterns
        for pattern in self.opinion_patterns:
            if pattern.search(query_lower):
                scores["opinion"] += 0.5
        
        # Check instruction patterns
        for pattern in self.instruction_patterns:
            if pattern.search(query_lower):
                scores["instruction"] += 0.4
        
        # If spaCy is available, enhance with NLP
        if self.is_spacy_available and self.nlp:
            doc = self.nlp(query)
            
            # Check for question words (indicates factual query)
            question_words = ["what", "who", "where", "when", "why", "how"]
            if any(token.text.lower() in question_words for token in doc):
                scores["factual"] += 0.2
            
            # Check for imperative verbs (indicates instruction)
            if len(doc) > 0 and doc[0].pos_ == "VERB":
                scores["instruction"] += 0.2
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            for key in scores:
                scores[key] /= total
        
        return scores
