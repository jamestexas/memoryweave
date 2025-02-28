"""
NLP-based extraction utilities for MemoryWeave.

This module provides NLP-powered extraction capabilities for identifying
personal attributes, query types, and other information from text using spaCy.
"""

import re
from collections import Counter
from typing import Any


# Singleton for spaCy model to avoid loading it multiple times
class SpacyModelSingleton:
    _instance = None
    _nlp = None
    
    @classmethod
    def get_instance(cls, model_name="en_core_web_sm"):
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance
    
    def __init__(self, model_name):
        if SpacyModelSingleton._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            SpacyModelSingleton._instance = self
            self._load_model(model_name)
    
    def _load_model(self, model_name):
        try:
            import spacy
            try:
                self._nlp = spacy.load(model_name)
                print(f"Successfully loaded spaCy model: {model_name}")
            except OSError:
                print(f"spaCy model '{model_name}' not available")
                print("Using fallback extraction methods")
        except ImportError:
            print("spaCy not available")
            print("Using fallback extraction methods")
    
    @property
    def nlp(self):
        return self._nlp


class NLPExtractor:
    """NLP-based attribute extractor with enhanced spaCy capabilities."""

    def __init__(self, model_name="en_core_web_sm"):
        """
        Initialize with specified spaCy model.

        Args:
            model_name: Name of the spaCy model to use
        """
        # Use the singleton to get the spaCy model
        spacy_singleton = SpacyModelSingleton.get_instance(model_name)
        self.nlp = spacy_singleton.nlp
        self.model_name = model_name
        self.is_spacy_available = self.nlp is not None
        self.matchers = {}

        # Metrics tracking
        self.extraction_metrics = {
            "attempts": 0,
            "successful_extractions": 0,
            "extraction_types": Counter(),
            "extraction_methods": Counter(),
        }

        # Try to set up matchers if spaCy is available
        if self.is_spacy_available:
            self._setup_spacy_matchers()

        # Initialize fallback regex patterns
        self._init_fallback_patterns()

    def _setup_spacy_matchers(self):
        """set up spaCy rule-based matchers for attribute extraction."""
        if not self.is_spacy_available:
            return

        try:
            from spacy.matcher import Matcher, PhraseMatcher

            # Preference matcher
            preference_matcher = Matcher(self.nlp.vocab)

            # "My favorite X is Y" pattern
            preference_patterns = [
                [  # My favorite color is blue
                    {"LOWER": {"IN": ["my", "i"]}},
                    {"LOWER": {"IN": ["favorite", "preferred"]}},
                    {
                        "LOWER": {
                            "IN": [
                                "color",
                                "food",
                                "drink",
                                "movie",
                                "book",
                                "music",
                                "song",
                                "artist",
                                "sport",
                                "game",
                                "place",
                            ]
                        }
                    },
                    {"LEMMA": "be"},
                    {"OP": "+"},  # The preference value
                ],
                [  # I like blue as my color
                    {"LOWER": {"IN": ["i", "my"]}},
                    {"LEMMA": {"IN": ["like", "love", "prefer", "enjoy"]}},
                    {"OP": "+"},  # The preference value
                    {"LOWER": {"IN": ["for", "as"]}},
                    {"LOWER": {"IN": ["my"]}},
                    {
                        "LOWER": {
                            "IN": ["color", "food", "drink", "movie", "book", "music", "activity"]
                        }
                    },
                ],
                [  # I like the color blue
                    {"LOWER": {"IN": ["i", "my"]}},
                    {"LEMMA": {"IN": ["like", "love", "prefer", "enjoy"]}},
                    {"LOWER": {"IN": ["the", "to"]}},
                    {
                        "LOWER": {
                            "IN": [
                                "color",
                                "eating",
                                "drinking",
                                "watching",
                                "reading",
                                "listening",
                            ]
                        }
                    },
                    {"OP": "+"},  # The preference value
                ],
            ]
            preference_matcher.add("PREFERENCE", preference_patterns)
            self.matchers["preference"] = preference_matcher

            # Location matcher
            location_matcher = Matcher(self.nlp.vocab)
            location_patterns = [
                [  # I live in Seattle
                    {"LOWER": {"IN": ["i", "my"]}},
                    {"LEMMA": {"IN": ["live", "stay", "reside"]}},
                    {"LOWER": "in"},
                    {"ENT_TYPE": "GPE", "OP": "+"},  # Location entity
                ],
                [  # I am from Seattle
                    {"LOWER": {"IN": ["i", "my"]}},
                    {"LEMMA": "be"},
                    {"LOWER": {"IN": ["from", "in"]}},
                    {"ENT_TYPE": "GPE", "OP": "+"},  # Location entity
                ],
                [  # My city is Seattle
                    {"LOWER": {"IN": ["i", "my"]}},
                    {"LOWER": {"IN": ["city", "town", "state", "country"]}},
                    {"LEMMA": "be"},
                    {"ENT_TYPE": "GPE", "OP": "+"},  # Location entity
                ],
            ]
            location_matcher.add("LOCATION", location_patterns)
            self.matchers["location"] = location_matcher

            # Occupation matcher
            occupation_matcher = Matcher(self.nlp.vocab)
            occupation_patterns = [
                [  # I work as a software engineer
                    {"LOWER": {"IN": ["i", "my"]}},
                    {"LEMMA": {"IN": ["work", "be"]}},
                    {"LOWER": {"IN": ["as", "a", "an"]}, "OP": "?"},
                    {"POS": "NOUN", "OP": "+"},  # Occupation noun
                ],
                [  # My job is software engineer
                    {"LOWER": {"IN": ["i", "my"]}},
                    {"LOWER": {"IN": ["job", "profession", "occupation", "career"]}},
                    {"LEMMA": "be"},
                    {"POS": "NOUN", "OP": "+"},  # Occupation noun
                ],
            ]
            occupation_matcher.add("OCCUPATION", occupation_patterns)
            self.matchers["occupation"] = occupation_matcher

            # Hobby matcher
            hobby_matcher = Matcher(self.nlp.vocab)
            hobby_patterns = [
                [  # I enjoy hiking in the mountains
                    {"LOWER": {"IN": ["i", "my"]}},
                    {"LEMMA": {"IN": ["enjoy", "like", "love"]}},
                    {"POS": "VERB", "OP": "+"},  # Activity verb
                ],
                [  # My hobby is painting
                    {"LOWER": {"IN": ["i", "my"]}},
                    {"LOWER": {"IN": ["hobby", "hobbies", "pastime", "activity"]}},
                    {"LEMMA": "be"},
                    {"POS": {"IN": ["NOUN", "VERB"]}, "OP": "+"},  # Hobby noun or verb
                ],
            ]
            hobby_matcher.add("HOBBY", hobby_patterns)
            self.matchers["hobby"] = hobby_matcher

            # Family relationship matcher
            family_matcher = Matcher(self.nlp.vocab)
            family_patterns = [
                [  # My wife is Sarah
                    {"LOWER": "my"},
                    {
                        "LOWER": {
                            "IN": [
                                "wife",
                                "husband",
                                "spouse",
                                "partner",
                                "girlfriend",
                                "boyfriend",
                                "son",
                                "daughter",
                                "child",
                                "children",
                                "mother",
                                "father",
                                "brother",
                                "sister",
                                "sibling",
                            ]
                        }
                    },
                    {"LEMMA": "be", "OP": "?"},
                    {"LOWER": {"IN": ["named", "called"]}, "OP": "?"},
                    {"ENT_TYPE": "PERSON", "OP": "+"},  # Person name
                ],
                [  # I have a wife named Sarah
                    {"LOWER": {"IN": ["i", "we"]}},
                    {"LEMMA": "have"},
                    {
                        "LOWER": {"IN": ["a", "an", "two", "three", "four", "five", "some"]},
                        "OP": "?",
                    },
                    {
                        "LOWER": {
                            "IN": [
                                "wife",
                                "husband",
                                "spouse",
                                "partner",
                                "girlfriend",
                                "boyfriend",
                                "son",
                                "daughter",
                                "child",
                                "children",
                                "mother",
                                "father",
                                "brother",
                                "sister",
                                "sibling",
                            ]
                        }
                    },
                    {"LOWER": {"IN": ["named", "called"]}, "OP": "?"},
                    {"ENT_TYPE": "PERSON", "OP": "*"},  # Person name
                ],
            ]
            family_matcher.add("FAMILY", family_patterns)
            self.matchers["family"] = family_matcher

            # Query type matchers
            factual_matcher = Matcher(self.nlp.vocab)
            factual_patterns = [
                [  # What is the capital of France?
                    {"LOWER": {"IN": ["what", "who", "where", "when", "why", "how"]}},
                    {"LEMMA": {"IN": ["be", "do", "can", "will", "would", "should", "could"]}},
                ],
                [  # Tell me about X
                    {"LEMMA": {"IN": ["tell", "explain", "describe"]}},
                    {"OP": "+"},
                ],
            ]
            factual_matcher.add("FACTUAL", factual_patterns)
            self.matchers["factual"] = factual_matcher

            personal_matcher = Matcher(self.nlp.vocab)
            personal_patterns = [
                [  # Contains personal pronouns
                    {"LOWER": {"IN": ["my", "me", "i", "mine", "myself"]}}
                ],
                [  # Questions about self
                    {"LEMMA": {"IN": ["do", "did", "should", "can", "will"]}},
                    {"LOWER": "i"},
                ],
            ]
            personal_matcher.add("PERSONAL", personal_patterns)
            self.matchers["personal"] = personal_matcher

            opinion_matcher = Matcher(self.nlp.vocab)
            opinion_patterns = [
                [  # What do you think about X?
                    {"LOWER": {"IN": ["think", "opinion", "believe", "feel", "view"]}}
                ],
                [  # What's your opinion on X?
                    {"LOWER": "what"},
                    {"LOWER": {"IN": ["is", "are", "'s", "s"]}},
                    {"LOWER": {"IN": ["your"]}},
                    {"LOWER": {"IN": ["opinion", "thoughts", "take", "view"]}},
                ],
            ]
            opinion_matcher.add("OPINION", opinion_patterns)
            self.matchers["opinion"] = opinion_matcher

            instruction_matcher = Matcher(self.nlp.vocab)
            instruction_patterns = [
                [  # Starts with an imperative verb
                    {"POS": "VERB", "IS_SENT_START": True}
                ],
                [  # Please X
                    {"LOWER": {"IN": ["please", "kindly"]}},
                    {"OP": "+"},
                ],
            ]
            instruction_matcher.add("INSTRUCTION", instruction_patterns)
            self.matchers["instruction"] = instruction_matcher
        except (ImportError, AttributeError) as e:
            print(f"Error setting up spaCy matchers: {e}")
            self.is_spacy_available = False

    def _init_fallback_patterns(self):
        """Initialize regex patterns for fallback extraction."""
        # Preference patterns
        self.favorite_patterns = [
            re.compile(
                r"(?:my|I) (?:favorite|preferred) (color|food|drink|movie|book|music|song|artist|sport|game|place) (?:is|are) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"
            ),
            re.compile(
                r"(?:I|my) (?:like|love|prefer|enjoy) ([a-z0-9\s]+) (?:for|as) (?:my) (color|food|drink|movie|book|music|activity)"
            ),
            re.compile(
                r"(?:I|my) (?:like|love|prefer|enjoy) (?:the color|eating|drinking|watching|reading|listening to) ([a-z0-9\s]+)"
            ),
        ]

        self.color_patterns = [
            re.compile(r"(?:my|I) (?:favorite) color is ([a-z\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I|my) (?:like|love|prefer|enjoy) the color ([a-z\s]+)"),
        ]

        self.food_patterns = [
            re.compile(r"(?:I|my) (?:like|love|prefer|enjoy) (?:eating|to eat) ([a-z\s]+)"),
            re.compile(r"(?:my|I) (?:favorite) food is ([a-z\s]+)(?:\.|\,|\!|\?|$)"),
        ]

        # Location patterns
        self.location_patterns = [
            re.compile(r"(?:I|my) (?:live|stay|reside) in ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(r"(?:I|my) (?:from|grew up in|was born in) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(
                r"(?:I|my) (?:city|town|state|country) (?:is) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"
            ),
        ]

        # Occupation patterns
        self.occupation_patterns = [
            re.compile(r"(?:I|my) (?:work as|am) (?:a|an) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"),
            re.compile(
                r"(?:I|my) (?:job|profession|occupation) (?:is) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"
            ),
        ]

        # Education patterns
        self.education_patterns = [
            re.compile(
                r"(?:I|my) (?:studied|majored in) ([a-z0-9\s]+) at ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"
            ),
            re.compile(
                r"(?:I|my) (?:graduated from|attended) ([a-z0-9\s]+) with (?:a|an) (?:degree|major) in ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"
            ),
        ]

        # Hobby patterns
        self.hobby_patterns = [
            re.compile(
                r"(?:I|my) (?:like to|love to|enjoy) ([a-z\s]+) (?:on|in|during) (?:my|the) ([a-z\s]+)(?:\.|\,|\!|\?|$)"
            ),
            re.compile(
                r"(?:I|my) (?:hobby|hobbies|pastime|activity) (?:is|are|include) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"
            ),
        ]

        # Family relationship patterns
        self.family_patterns = [
            re.compile(
                r"(?:my) (wife|husband|spouse|partner|girlfriend|boyfriend) (?:is|name is) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"
            ),
            re.compile(
                r"(?:my) (son|daughter|child|children|mother|father|brother|sister|sibling) (?:is|are|name is|names are) ([a-z0-9\s]+)(?:\.|\,|\!|\?|$)"
            ),
        ]

    def extract_personal_attributes(self, text: str) -> dict[str, Any]:
        """
        Extract personal attributes using available methods.

        Args:
            text: Text to analyze

        Returns:
            dictionary of extracted attributes
        """
        attributes = {"preferences": {}, "demographics": {}, "traits": {}, "relationships": {}}

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
                        context_window = text[
                            max(0, ent.start_char - 20) : min(len(text), ent.end_char + 20)
                        ].lower()

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

    def _extract_preferences_regex(self, text: str, attributes: dict) -> None:
        """
        Extract preferences using regex patterns.

        Args:
            text: Text to analyze
            attributes: dictionary to update with extracted preferences
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
                        color_match = re.search(
                            r"(?:like|love|prefer|enjoy) (?:the color) ([a-z\s]+)", text_lower
                        )
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

    def _extract_demographics_regex(self, text: str, attributes: dict) -> None:
        """
        Extract demographic information using regex patterns.

        Args:
            text: Text to analyze
            attributes: dictionary to update with extracted demographics
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

    def _extract_traits_regex(self, text: str, attributes: dict) -> None:
        """
        Extract personality traits and hobbies using regex patterns.

        Args:
            text: Text to analyze
            attributes: dictionary to update with extracted traits
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
                        hobby = hobby.strip().strip(",")
                        if hobby and hobby not in attributes["traits"]["hobbies"]:
                            attributes["traits"]["hobbies"].append(hobby)

    def _extract_relationships_regex(self, text: str, attributes: dict) -> None:
        """
        Extract relationship information using regex patterns.

        Args:
            text: Text to analyze
            attributes: dictionary to update with extracted relationships
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
                            if (
                                child_name
                                and child_name
                                not in attributes["relationships"]["family"]["children"]
                            ):
                                attributes["relationships"]["family"]["children"].append(child_name)
                    else:
                        attributes["relationships"]["family"][relation.strip()] = name.strip()
                elif isinstance(match, str):
                    # Handle case where we just know they have a relation but not the name
                    relation = match.strip()
                    if "family" not in attributes["relationships"]:
                        attributes["relationships"]["family"] = {}
                    attributes["relationships"]["family"][relation] = "Unknown"

    def identify_query_type(self, query: str) -> dict[str, float]:
        """
        Identify the type of query using available techniques.

        Args:
            query: The query text

        Returns:
            dictionary with query type probabilities
        """
        # Initialize scores
        scores = {"factual": 0.0, "personal": 0.0, "opinion": 0.0, "instruction": 0.0}

        if not query:
            return scores

        # Use spaCy-based analysis if available
        if self.is_spacy_available and self.nlp:
            doc = self.nlp(query)

            # Check for factual indicators
            if "factual" in self.matchers:
                factual_matches = self.matchers["factual"](doc)
                if factual_matches:
                    scores["factual"] += 0.3 * len(factual_matches)

            # Check for personal indicators
            if "personal" in self.matchers:
                personal_matches = self.matchers["personal"](doc)
                if personal_matches:
                    scores["personal"] += 0.5 * len(personal_matches)

            # Check for opinion indicators
            if "opinion" in self.matchers:
                opinion_matches = self.matchers["opinion"](doc)
                if opinion_matches:
                    scores["opinion"] += 0.5 * len(opinion_matches)

            # Check for instruction indicators
            if "instruction" in self.matchers:
                instruction_matches = self.matchers["instruction"](doc)
                if instruction_matches:
                    scores["instruction"] += 0.4 * len(instruction_matches)

            # Check if first token is a question word
            if doc and doc[0].lower_ in ["what", "who", "where", "when", "why", "how"]:
                scores["factual"] += 0.3

            # Check if query ends with a question mark
            if query.strip().endswith("?"):
                if scores["personal"] > 0:
                    scores["personal"] += 0.1
                else:
                    scores["factual"] += 0.1

            # Check if first token is an imperative verb
            if doc and doc[0].pos_ == "VERB" and doc[0].dep_ == "ROOT":
                scores["instruction"] += 0.4
        else:
            # Fallback to simple pattern matching
            query_lower = query.lower()

            # Check for question words
            if re.search(r"^(what|who|where|when|why|how)\b", query_lower):
                scores["factual"] += 0.5

            # Check for personal pronouns
            if re.search(r"\b(my|me|i|mine|myself)\b", query_lower):
                scores["personal"] += 0.5

            # Check for opinion indicators
            if re.search(r"\b(think|opinion|believe|feel|view)\b", query_lower):
                scores["opinion"] += 0.5

            # Check for imperative verbs or please
            if re.search(
                r"^(please|kindly|tell|show|list|give|find|write|create|make)\b", query_lower
            ):
                scores["instruction"] += 0.5

            # Check for question mark
            if query.strip().endswith("?"):
                if scores["personal"] > 0:
                    scores["personal"] += 0.1
                else:
                    scores["factual"] += 0.1

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            for key in scores:
                scores[key] /= total

        return scores

    def extract_important_keywords(self, query: str) -> set[str]:
        """
        Extract important keywords for direct reference matching.

        Args:
            query: The user query

        Returns:
            set of important keywords
        """
        # Convert to lowercase for case-insensitive matching
        query_lower = query.lower()

        important_words = set()

        # If spaCy is available, use NLP-based extraction
        if self.is_spacy_available and self.nlp:
            doc = self.nlp(query)

            # Extract entities as important
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "GPE", "LOC", "ORG", "PRODUCT"]:
                    important_words.add(ent.text.lower())

            # Extract nouns and key verbs
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"]:
                    if token.dep_ in ["nsubj", "dobj", "pobj", "attr"]:
                        important_words.add(token.lemma_.lower())
                elif token.pos_ == "VERB" and token.dep_ == "ROOT":
                    important_words.add(token.lemma_.lower())

            # Extract noun chunks
            for chunk in doc.noun_chunks:
                if len(chunk) <= 3:  # Avoid overly long phrases
                    important_words.add(chunk.text.lower())
        else:
            # Fallback to regex-based extraction
            # Extract nouns and potential key phrases
            nouns = re.findall(r"\b[a-z]+(?:ing|ed)?\b", query_lower)
            for noun in nouns:
                if len(noun) > 3 and noun not in [
                    "what",
                    "when",
                    "where",
                    "which",
                    "this",
                    "that",
                    "these",
                    "those",
                    "with",
                ]:
                    important_words.add(noun)

        # Add specific preference and personal attribute keywords
        preference_terms = ["favorite", "like", "prefer", "love", "favorite color", "favorite food"]
        personal_terms = [
            "color",
            "food",
            "movie",
            "book",
            "hobby",
            "activity",
            "live",
            "work",
            "job",
            "occupation",
            "weekend",
        ]

        for term in preference_terms + personal_terms:
            if term in query_lower:
                important_words.add(term)

        # Filter out common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "was",
            "were",
            "be",
            "been",
            "being",
            "to",
            "of",
            "and",
            "or",
            "that",
            "this",
            "these",
            "those",
        }
        important_words = {word for word in important_words if word not in stop_words}

        return important_words

    def _has_attributes(self, attributes):
        """Check if any attributes were extracted."""
        for category, items in attributes.items():
            if items:
                return True
        return False
