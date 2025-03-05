"""Pattern matching utilities for MemoryWeave.

This module provides utilities for pattern matching in text,
including regex-based matchers and more sophisticated matchers.
"""

import re
from dataclasses import dataclass
from re import Pattern
from typing import Any, Optional


@dataclass
class PatternMatch:
    """A match found by a pattern matcher."""

    pattern_id: str
    text: str
    start: int
    end: int
    groups: dict[str, str]
    score: float


class RegexMatcher:
    """Pattern matcher using regular expressions."""

    def __init__(self):
        """Initialize the regex matcher."""
        self._patterns: dict[str, Pattern] = {}
        self._pattern_configs: dict[str, dict[str, Any]] = {}

    def add_pattern(
        self, pattern_id: str, pattern: str, flags: int = re.IGNORECASE, score: float = 1.0
    ) -> None:
        """Add a pattern to the matcher."""
        self._patterns[pattern_id] = re.compile(pattern, flags)
        self._pattern_configs[pattern_id] = {"score": score}

    def add_patterns(self, patterns: dict[str, str]) -> None:
        """Add multiple patterns to the matcher."""
        for pattern_id, pattern in patterns.items():
            self.add_pattern(pattern_id, pattern)

    def find_matches(self, text: str) -> list[PatternMatch]:
        """Find all matches in the text."""
        results = []

        for pattern_id, pattern in self._patterns.items():
            for match in pattern.finditer(text):
                # Get match information
                start, end = match.span()
                match_text = match.group(0)

                # Extract named groups
                groups = {}
                for name, value in match.groupdict().items():
                    if value is not None:
                        groups[name] = value

                # Create pattern match
                result = PatternMatch(
                    pattern_id=pattern_id,
                    text=match_text,
                    start=start,
                    end=end,
                    groups=groups,
                    score=self._pattern_configs[pattern_id]["score"],
                )

                results.append(result)

        # Sort by position
        results.sort(key=lambda x: x.start)

        return results


class NamedEntityMatcher:
    """Pattern matcher for named entities."""

    def __init__(self):
        """Initialize the named entity matcher."""
        from memoryweave.nlp.patterns import ENTITY_PATTERNS

        # Create regex matcher with entity patterns
        self._matcher = RegexMatcher()
        for entity_type, pattern in ENTITY_PATTERNS.items():
            self._matcher.add_pattern(entity_type, pattern)

    def find_entities(self, text: str) -> list[PatternMatch]:
        """Find all entities in the text."""
        # Get initial matches
        matches = self._matcher.find_matches(text)

        # Filter out overlapping matches
        return self._filter_overlapping(matches)

    def _filter_overlapping(self, matches: list[PatternMatch]) -> list[PatternMatch]:
        """Filter out overlapping matches, keeping the highest scoring ones."""
        if not matches:
            return []

        # Sort by score (descending) to prefer higher scoring matches
        sorted_matches = sorted(matches, key=lambda m: m.score, reverse=True)

        # Keep track of used character positions
        used_positions = set()
        filtered_matches = []

        for match in sorted_matches:
            # Check if this match overlaps with already selected matches
            match_positions = set(range(match.start, match.end))
            if not match_positions.intersection(used_positions):
                # No overlap, add to filtered list
                filtered_matches.append(match)
                used_positions.update(match_positions)

        # Sort by position in text
        filtered_matches.sort(key=lambda m: m.start)

        return filtered_matches


class AttributeMatcher:
    """Pattern matcher for personal attributes."""

    def __init__(self):
        """Initialize the attribute matcher."""
        from memoryweave.nlp.patterns import PERSONAL_ATTRIBUTE_PATTERNS

        # Create a regex matcher for each attribute type
        self._matchers: dict[str, RegexMatcher] = {}

        for attr_type, patterns in PERSONAL_ATTRIBUTE_PATTERNS.items():
            matcher = RegexMatcher()
            for i, pattern in enumerate(patterns):
                pattern_id = f"{attr_type}_{i}"
                matcher.add_pattern(pattern_id, pattern)
            self._matchers[attr_type] = matcher

    def find_attributes(self, text: str) -> dict[str, list[PatternMatch]]:
        """Find all attributes in the text."""
        results = {}

        for attr_type, matcher in self._matchers.items():
            matches = matcher.find_matches(text)
            if matches:
                results[attr_type] = matches

        return results


class KeywordMatcher:
    """Matcher for keywords and key phrases."""

    def __init__(self, keywords: Optional[list[str]] = None):
        """Initialize the keyword matcher.

        Args:
            keywords: Optional list of keywords to initialize the matcher
        """
        self._keywords = set(keywords or [])
        self._keyword_patterns: dict[str, Pattern] = {}

        # Compile patterns for all keywords
        for keyword in self._keywords:
            self._compile_keyword(keyword)

    def add_keyword(self, keyword: str) -> None:
        """Add a keyword to the matcher."""
        if keyword not in self._keywords:
            self._keywords.add(keyword)
            self._compile_keyword(keyword)

    def add_keywords(self, keywords: list[str]) -> None:
        """Add multiple keywords to the matcher."""
        for keyword in keywords:
            self.add_keyword(keyword)

    def remove_keyword(self, keyword: str) -> None:
        """Remove a keyword from the matcher."""
        if keyword in self._keywords:
            self._keywords.remove(keyword)
            if keyword in self._keyword_patterns:
                del self._keyword_patterns[keyword]

    def find_keywords(self, text: str) -> list[PatternMatch]:
        """Find all keywords in the text."""
        results = []

        for keyword, pattern in self._keyword_patterns.items():
            for match in pattern.finditer(text):
                # Get match information
                start, end = match.span()
                match_text = match.group(0)

                # Create pattern match
                result = PatternMatch(
                    pattern_id=keyword, text=match_text, start=start, end=end, groups={}, score=1.0
                )

                results.append(result)

        # Sort by position
        results.sort(key=lambda x: x.start)

        return results

    def _compile_keyword(self, keyword: str) -> None:
        """Compile a regex pattern for a keyword."""
        # Escape special regex characters
        escaped = re.escape(keyword)

        # Create pattern that matches word boundaries
        pattern = rf"\b{escaped}\b"

        # Compile and store the pattern
        self._keyword_patterns[keyword] = re.compile(pattern, re.IGNORECASE)
