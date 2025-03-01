"""Keyword utilities for MemoryWeave.

This module provides utilities for keyword extraction and management,
including stopword filtering and keyword ranking.
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from memoryweave.nlp.patterns import STOPWORDS


def extract_keywords(
    text: str, stopwords: Optional[Set[str]] = None, min_length: int = 3, max_keywords: int = 10
) -> List[str]:
    """Extract keywords from text.

    Args:
        text: The text to extract keywords from
        stopwords: Optional set of stopwords to filter out
        min_length: Minimum length of keywords to include
        max_keywords: Maximum number of keywords to return

    Returns:
        List of extracted keywords
    """
    # Use default stopwords if none provided
    if stopwords is None:
        stopwords = STOPWORDS

    # Tokenize text
    tokens = tokenize(text)

    # Filter stopwords and short words
    filtered_tokens = [
        token for token in tokens if token.lower() not in stopwords and len(token) >= min_length
    ]

    # Count frequencies
    counter = Counter(filtered_tokens)

    # Get most common keywords
    most_common = counter.most_common(max_keywords)

    # Return just the keywords
    return [word for word, _ in most_common]


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    # Remove punctuation and split by whitespace
    words = re.findall(r"\b\w+\b", text.lower())
    return words


def rank_keywords(keywords: List[str], text: str) -> List[Tuple[str, float]]:
    """Rank keywords by importance in the text.

    Args:
        keywords: List of keywords to rank
        text: The text to analyze

    Returns:
        List of (keyword, score) tuples, sorted by descending score
    """
    # Count occurrences of each keyword
    keyword_counts = {}
    for keyword in keywords:
        # Use regex with word boundaries to count occurrences
        pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
        keyword_counts[keyword] = len(pattern.findall(text))

    # Calculate TF (term frequency)
    total_words = len(tokenize(text))
    if total_words == 0:
        total_words = 1  # Avoid division by zero

    term_frequencies = {keyword: count / total_words for keyword, count in keyword_counts.items()}

    # Calculate score based on TF and position in text
    scores = []
    for keyword in keywords:
        # Get TF
        tf = term_frequencies.get(keyword, 0)

        # Get first position
        pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
        match = pattern.search(text)
        position_score = 0
        if match:
            # Earlier positions get higher scores
            position = match.start() / len(text)
            position_score = 1.0 - position

        # Calculate final score (TF + position bonus)
        final_score = (tf * 0.7) + (position_score * 0.3)
        scores.append((keyword, final_score))

    # Sort by score (descending)
    return sorted(scores, key=lambda x: x[1], reverse=True)


def expand_keywords(
    keywords: List[str], word_relationships: Dict[str, List[str]], expansion_count: int = 3
) -> List[str]:
    """Expand keywords with related terms.

    Args:
        keywords: List of keywords to expand
        word_relationships: Dictionary mapping words to related words
        expansion_count: Number of related words to add per keyword

    Returns:
        List of original and expanded keywords
    """
    expanded = set(keywords)

    for keyword in keywords:
        if keyword in word_relationships:
            related = word_relationships[keyword]
            for word in related[:expansion_count]:
                expanded.add(word)

    return list(expanded)
