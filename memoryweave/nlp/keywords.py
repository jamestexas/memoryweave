# memoryweave/nlp/keywords.py
import re
from collections import Counter

from pydantic import BaseModel, Field

from memoryweave.nlp.interfaces import KeywordExtractor


class Keyword(BaseModel):
    """A keyword extracted from text."""

    text: str
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)

    class Config:
        frozen = True


class DefaultStopwords:
    """Default English stopwords."""

    ENGLISH = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "because",
        "as",
        "what",
        "when",
        "where",
        "how",
        "who",
        "which",
        "this",
        "that",
        "these",
        "those",
        "then",
        "just",
        "so",
        "than",
        "such",
        "both",
        "through",
        "about",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "of",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
    }


class StatisticalKeywordExtractor(KeywordExtractor):
    """Extract keywords using statistical methods."""

    def __init__(self, stopwords: set[str] | None = None):
        self._stopwords = stopwords or DefaultStopwords.ENGLISH

    @property
    def available(self) -> bool:
        """Always available since only using builtin modules."""
        return True

    def extract(self, text: str, **kwargs) -> list[Keyword]:
        """Extract keywords from text."""
        # Convert to list of Keyword objects
        keywords = self.extract_keywords(
            text,
            min_length=kwargs.get("min_length", 3),
            max_keywords=kwargs.get("max_keywords", 10),
        )

        return [Keyword(text=kw, score=1.0) for kw in keywords]

    def extract_keywords(self, text: str, min_length: int = 3, max_keywords: int = 10) -> list[str]:
        """
        Extract keywords using statistical methods.

        Args:
            text: The text to extract keywords from
            min_length: Minimum length of keywords to include
            max_keywords: Maximum number of keywords to return

        Returns:
            List of extracted keywords
        """
        if not text:
            return []

        # Tokenize text
        words = self._tokenize(text)

        # Filter stopwords and short words
        filtered_words = [
            word
            for word in words
            if word.lower() not in self._stopwords and len(word) >= min_length
        ]

        # Count frequencies
        counter = Counter(filtered_words)

        # Get most common keywords
        most_common = counter.most_common(max_keywords)

        # Return just the keywords
        return [word for word, _ in most_common]

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        # Remove punctuation and split by whitespace
        words = re.findall(r"\b\w+\b", text.lower())
        return words


class YakeKeywordExtractor(KeywordExtractor):
    """Extract keywords using the YAKE algorithm."""

    def __init__(self):
        self._extractor = None
        self._initialized = False

        # Try to initialize YAKE
        self._initialize()

    def _initialize(self) -> None:
        """Initialize YAKE if not already initialized."""
        if self._initialized:
            return

        try:
            import yake

            language = "en"
            max_ngram_size = 2
            deduplication_threshold = 0.9

            self._extractor = yake.KeywordExtractor(
                lan=language, n=max_ngram_size, dedupLim=deduplication_threshold
            )
            self._initialized = True
        except ImportError:
            self._extractor = None
            self._initialized = False

    @property
    def available(self) -> bool:
        """Check if YAKE is available."""
        return self._initialized and self._extractor is not None

    def extract(self, text: str, **kwargs) -> list[Keyword]:
        """Extract keywords from text."""
        keywords = self.extract_keywords(text, max_keywords=kwargs.get("max_keywords", 10))

        # Keywords and scores from YAKE
        result = []
        for kw, score in keywords:
            # YAKE scores are inverse (lower is better)
            # Convert to a 0-1 scale where higher is better
            normalized_score = max(0, min(1, 1.0 - score))

            keyword = Keyword(text=kw, score=normalized_score)
            result.append(keyword)

        return result

    def extract_keywords(self, text: str, max_keywords: int = 10) -> list[tuple[str, float]]:
        """
        Extract keywords using YAKE.

        Args:
            text: The text to extract keywords from
            max_keywords: Maximum number of keywords to return

        Returns:
            List of (keyword, score) tuples
        """
        if not self.available or not text:
            return []

        # Extract keywords with YAKE
        keywords = self._extractor.extract_keywords(text)

        # Limit to max_keywords
        return keywords[:max_keywords]
