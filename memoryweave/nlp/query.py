# memoryweave/nlp/query.py
import re

from pydantic import BaseModel, Field

from memoryweave.nlp.interfaces import QueryTypeClassifier


class QueryType(BaseModel):
    """A query type with confidence score."""

    name: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    class Config:
        frozen = True


class PatternQueryClassifier(QueryTypeClassifier):
    """Classify queries using patterns."""

    def __init__(self):
        # Patterns for different query types
        self._patterns = {
            "factual": [
                r"^(what|who|where|when|why|how)\b",
                r"\b(explain|describe|tell me about)\b",
            ],
            "personal": [
                r"\b(my|me|i|mine|myself)\b",
                r"^(what's|what is) my\b",
            ],
            "opinion": [
                r"\b(think|opinion|believe|feel|view)\b",
                r"^(do you|what do you)\b",
            ],
            "instruction": [
                r"^(please|kindly|tell|show|find|write|create|make)\b",
                r"(list|summarize|analyze|review)\b",
            ],
        }

        # Compile patterns
        self._compiled_patterns = {
            query_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for query_type, patterns in self._patterns.items()
        }

    def classify(self, query: str) -> dict[str, float]:
        """
        Classify a query into different types.

        Args:
            query: The query text

        Returns:
            Dictionary mapping query types to confidence scores
        """
        if not query:
            return {"factual": 0.0, "personal": 0.0, "opinion": 0.0, "instruction": 0.0}

        # Initialize scores
        scores = {query_type: 0.0 for query_type in self._patterns.keys()}

        query_lower = query.lower()

        # Apply pattern matching
        for query_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    scores[query_type] += 0.3  # Add weight for each matching pattern

        # Apply special case handling for common patterns
        if "what is my" in query_lower or "what's my" in query_lower:
            scores["personal"] += 0.3

        if "where do i" in query_lower:
            scores["personal"] += 0.3

        if "tell me about" in query_lower and not any(
            term in query_lower for term in ["my", "me", "i"]
        ):
            scores["factual"] += 0.2

        # Check for question marks
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


class SpacyQueryClassifier(QueryTypeClassifier):
    """Classify queries using spaCy's linguistic features."""

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

    def classify(self, query: str) -> dict[str, float]:
        """
        Classify a query into different types using linguistic features.

        Args:
            query: The query text

        Returns:
            Dictionary mapping query types to confidence scores
        """
        if not query or not self._initialized:
            return {"factual": 0.0, "personal": 0.0, "opinion": 0.0, "instruction": 0.0}

        # Initialize scores
        scores = {"factual": 0.0, "personal": 0.0, "opinion": 0.0, "instruction": 0.0}

        # Process query with spaCy
        doc = self._nlp(query)

        # Check for question structure
        has_question_word = False
        has_question_mark = query.strip().endswith("?")

        for token in doc:
            # Check for question words
            if token.pos_ == "SCONJ" and token.text.lower() in [
                "what",
                "who",
                "where",
                "when",
                "why",
                "how",
            ]:
                has_question_word = True
                scores["factual"] += 0.4

            # Check for personal pronouns
            if token.pos_ == "PRON" and token.text.lower() in ["i", "me", "my", "mine", "myself"]:
                scores["personal"] += 0.3

            # Check for opinion indicators
            if token.lemma_ in ["think", "believe", "feel", "opinion"]:
                scores["opinion"] += 0.3

            # Check for imperative structure (verb at beginning)
            if token.i == 0 and token.pos_ == "VERB":
                scores["instruction"] += 0.4

            # Check for politeness markers
            if token.text.lower() in ["please", "kindly"]:
                scores["instruction"] += 0.3

        # If no strong signals, use question mark as factual indicator
        if has_question_mark and not has_question_word and max(scores.values()) < 0.3:
            scores["factual"] += 0.3

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            for key in scores:
                scores[key] /= total

        return scores
