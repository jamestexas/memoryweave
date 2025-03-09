# memoryweave/nlp/attributes.py
import re
from typing import Any

from pydantic import BaseModel, Field

from memoryweave.nlp.interfaces import AttributeExtractor


class Attribute(BaseModel):
    """A personal attribute extracted from text."""

    name: str
    value: str | list[str] | dict[str, Any]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    category: str | None = None

    class Config:
        frozen = True


class SpacyAttributeExtractor(AttributeExtractor):
    """Extract personal attributes using spaCy's linguistic features."""

    def __init__(self, model: str = "en_core_web_sm"):
        self._model_name = model
        self._nlp = None
        self._initialized = False

        # Try to initialize SpaCy
        self._initialize()

    def _initialize(self) -> None:
        """Initialize spaCy if not already initialized."""
        if self._initialized:
            return

        try:
            import spacy

            self._nlp = spacy.load(self._model_name)
            self._initialized = True
        except (ImportError, OSError):
            self._nlp = None
            self._initialized = False

    @property
    def available(self) -> bool:
        """Check if spaCy is available."""
        return self._initialized and self._nlp is not None

    def extract(self, text: str) -> list[Attribute]:
        """Extract attributes from text."""
        attributes_dict = self.extract_attributes(text)

        # Convert dictionary to list of Attribute objects
        result = []
        for name, value in attributes_dict.items():
            category = None

            # Determine category based on attribute name
            if name in ["job", "occupation", "profession", "role"]:
                category = "occupation"
            elif name in ["location", "address", "city", "country"]:
                category = "location"
            elif name in ["likes", "preferences", "favorites", "favorite"]:
                category = "preferences"

            attribute = Attribute(name=name, value=value, category=category)
            result.append(attribute)

        return result

    def extract_attributes(self, text: str) -> dict[str, Any]:
        """Extract personal attributes using linguistic patterns."""
        if not self.available or not text:
            return {}

        attributes = {}

        # Process the text with spaCy
        doc = self._nlp(text)

        # Process all sentences
        for sent in doc.sents:
            # Look for first-person subject patterns (I am/I work/I live)
            _subject = None
            verb = None

            for token in sent:
                # Find the main subject-verb structure
                if token.dep_ == "nsubj" and token.text.lower() in ["i", "me", "my", "we", "our"]:
                    _subject = token
                    # Find the associated verb
                    verb = token.head

                    # Process based on verb type
                    if verb.lemma_ == "be":  # "I am X"
                        for child in verb.children:
                            if child.dep_ == "attr":
                                # Extract identity/role information
                                if child.pos_ == "NOUN":
                                    attributes["occupation"] = child.text

                    elif verb.lemma_ == "live":  # "I live in X"
                        for child in verb.children:
                            if child.dep_ == "prep" and child.text.lower() == "in":
                                for loc in child.children:
                                    if loc.pos_ in ["PROPN", "NOUN"]:
                                        attributes["location"] = loc.text

                    elif verb.lemma_ in ["work", "do"]:  # "I work as X"
                        for child in verb.children:
                            if child.dep_ == "prep" and child.text.lower() in ["as", "for"]:
                                for obj in child.children:
                                    attributes["occupation"] = obj.text

                    # Look for possessive patterns
                    elif token.dep_ == "poss" and token.head.pos_ == "NOUN":
                        attribute_name = token.head.text.lower()

                        # Look for attribute value after "is"
                        if token.head.head.lemma_ == "be":
                            for sibling in token.head.head.children:
                                if sibling.dep_ == "attr":
                                    attributes[attribute_name] = sibling.text

        # Additional patterns for preferences
        for sent in doc.sents:
            if (
                "favorite" in sent.text.lower()
                or "like" in sent.text.lower()
                or "love" in sent.text.lower()
            ):
                for token in sent:
                    if token.text.lower() in ["favorite", "like", "love", "enjoy"]:
                        # Look for the object of preference
                        obj = None

                        # For "My favorite X is Y" pattern
                        if token.text.lower() == "favorite" and token.pos_ == "ADJ":
                            # Get the noun being modified
                            for sibling in token.head.children:
                                if sibling.dep_ == "attr":
                                    obj = sibling
                                    category = token.head.text.lower()
                                    attributes[f"favorite_{category}"] = obj.text

                        # For "I like/love X" pattern
                        elif token.pos_ == "VERB":
                            for child in token.children:
                                if child.dep_ in ["dobj", "attr"]:
                                    attributes["likes"] = child.text

        return attributes


class PatternAttributeExtractor(AttributeExtractor):
    """Extract personal attributes using flexible patterns."""

    def __init__(self):
        # Common attribute patterns that are general enough to be useful
        self._patterns = {
            "location": [
                (r"(?:i|we) (?:live|stay|reside) in\s+([A-Za-z\s,]+)", 0.8),
                (r"(?:i|we) am from\s+([A-Za-z\s,]+)", 0.8),
                (r"(?:my|our) (?:city|town|location) is\s+([A-Za-z\s,]+)", 0.8),
            ],
            "occupation": [
                (r"(?:i|we) (?:work|am employed) as a(?:n)?\s+([A-Za-z\s]+)", 0.8),
                (r"(?:i|we) am a(?:n)?\s+([A-Za-z\s]+)", 0.8),
                (r"(?:my|our) (?:job|occupation|profession) is\s+([A-Za-z\s]+)", 0.8),
            ],
            "preference": [
                (r"(?:i|we) (?:like|love|enjoy)\s+([A-Za-z\s]+)", 0.7),
                (r"(?:my|our) favorite\s+([A-Za-z\s]+) is\s+([A-Za-z\s]+)", 0.8),
            ],
        }

        # Compile patterns
        self._compiled_patterns = {}
        for attr_type, patterns in self._patterns.items():
            self._compiled_patterns[attr_type] = [
                (re.compile(pattern, re.IGNORECASE), confidence) for pattern, confidence in patterns
            ]

    @property
    def available(self) -> bool:
        """Always available since only using builtin modules."""
        return True

    def extract(self, text: str) -> list[Attribute]:
        """Extract attributes from text."""
        attributes_dict = self.extract_attributes(text)

        # Convert dictionary to list of Attribute objects
        result = []
        for name, value_data in attributes_dict.items():
            if isinstance(value_data, tuple):
                value, confidence = value_data
            else:
                value = value_data
                confidence = 0.7  # Default confidence

            attribute = Attribute(name=name, value=value, confidence=confidence)
            result.append(attribute)

        return result

    def extract_attributes(self, text: str) -> dict[str, Any]:
        """Extract personal attributes using pattern matching."""
        if not text:
            return {}

        attributes = {}

        # Apply each pattern group
        for attr_type, patterns in self._compiled_patterns.items():
            for pattern, confidence in patterns:
                matches = pattern.finditer(text)

                for match in matches:
                    groups = match.groups()
                    if not groups:
                        continue

                    # Handle different attribute types differently
                    if attr_type == "preference" and len(groups) == 2:
                        # For patterns like "my favorite X is Y"
                        pref_type, pref_value = groups
                        attributes[f"favorite_{pref_type.strip()}"] = (
                            pref_value.strip(),
                            confidence,
                        )
                    else:
                        # For other attributes, use the attribute type as the key
                        value = groups[0].strip()
                        attributes[attr_type] = (value, confidence)

        return attributes
