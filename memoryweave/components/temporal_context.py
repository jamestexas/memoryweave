# memoryweave/components/temporal_context.py
"""
Temporal context integration for MemoryWeave.

This module implements components for enhancing memories with temporal information
and temporal-aware retrieval. It helps the system understand and leverage time-based
relationships between memories.
"""

import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
from rich.logging import RichHandler
from scipy.cluster.hierarchy import fcluster, linkage

from memoryweave.components.base import Component, MemoryComponent
from memoryweave.components.component_names import ComponentName
from memoryweave.interfaces.memory import MemoryID
from memoryweave.storage.refactored.base_store import BaseMemoryStore

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(markup=True)])
logger = logging.getLogger(__name__)

# Assuming you have these defined elsewhere:
DATE_KEYWORDS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]

TIME_KEYWORDS = [
    "morning",
    "noon",
    "afternoon",
    "evening",
    "night",
    "midnight",
    "today",
    "yesterday",
    "tomorrow",
    "tonight",
    "now",
    "recent",
    "before",
    "after",
    "last",
    "next",
    "previous",
    "earlier",
    "later",
    "week",
    "month",
    "year",
    "decade",
    "century",
    "ago",
]

RELATIVE_TIME_KEYWORDS = [
    "days",
    "weeks",
    "months",
    "years",
    "hours",
    "minutes",
    "seconds",
    "day",
    "week",
    "month",
    "year",
    "hour",
    "minute",
    "second",
]

# Date patterns (example)
DATE_PATTERNS = [
    # Create a pattern for date keywords followed by a number
    re.compile(
        r"\b(?:{})\s+\d{{1,2}}\b".format("|".join(map(re.escape, DATE_KEYWORDS))),
        re.IGNORECASE,
    ),
    re.compile(r"\d{1,2}/\d{1,2}/\d{4}", re.IGNORECASE),
    re.compile(r"\d{1,2}-\d{1,2}-\d{4}", re.IGNORECASE),
    # Add more date patterns here
]

# Time patterns
TIME_PATTERNS = [
    re.compile(r"\b\d{1,2}:\d{2}\s?(am|pm)?\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s?(am|pm)\b", re.IGNORECASE),
    # Add more time patterns here
]

# Relative time patterns
RELATIVE_TIME_PATTERNS = [
    re.compile(
        r"\b(?:{})\s+ago\b".format("|".join(map(re.escape, RELATIVE_TIME_KEYWORDS))),
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:{})\s+(?:from|since)\s+now\b".format(
            "|".join(map(re.escape, RELATIVE_TIME_KEYWORDS))
        ),
        re.IGNORECASE,
    ),
]

# Month Day Pattern
MONTH_DAY_PATTERN = re.compile(
    r"\b(?:{})\s+(\d{{1,2}})\b".format("|".join(map(re.escape, DATE_KEYWORDS))),
    re.IGNORECASE,
)


class TemporalDecayComponent(MemoryComponent):
    """
    Component that applies temporal decay to memory activations.

    This component simulates the natural decay of memory activation over time,
    making older memories less accessible unless they are frequently accessed
    or have strong connections.
    """

    def __init__(self):
        """Initialize the temporal decay component."""
        self.short_term_half_life = 3600  # 1 hour
        self.long_term_half_life = 2592000  # 30 days
        self.activation_boost_on_access = 0.5
        self.minimum_activation = 0.1
        self.component_id = ComponentName.MEMORY_DECAY
        self.last_decay_time = time.time()

    def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the component with configuration.

        Args:
            config: Configuration dictionary with parameters:
                - short_term_half_life: Half-life for short-term decay (default: 3600s)
                - long_term_half_life: Half-life for long-term decay (default: 2592000s)
                - activation_boost_on_access: Activation increase when accessed (default: 0.5)
                - minimum_activation: Minimum activation level (default: 0.1)
        """
        self.short_term_half_life = config.get("short_term_half_life", 3600)
        self.long_term_half_life = config.get("long_term_half_life", 2592000)
        self.activation_boost_on_access = config.get("activation_boost_on_access", 0.5)
        self.minimum_activation = config.get("minimum_activation", 0.1)

    def process(self, data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        Process memory data to apply temporal decay.

        Args:
            data: Memory data to process
            context: Context information

        Returns:
            Updated memory data with decayed activation
        """
        # Create a deep copy of the data to avoid modifying the original
        result = dict(data)

        # Get current time
        current_time = context.get("current_time", time.time())

        # Get memory access time
        last_accessed = data.get("last_accessed", current_time)
        created_at = data.get("created_at", current_time)

        # Get current activation level
        activation = data.get("activation", 1.0)

        # Apply decay based on time since last access
        time_since_access = current_time - last_accessed
        time_since_creation = current_time - created_at

        # Calculate decay factor
        # Use short-term decay for recently created memories,
        # and long-term decay for older memories
        if time_since_creation < 7 * 86400:  # Less than 7 days old
            half_life = self.short_term_half_life
        else:
            half_life = self.long_term_half_life

        # Apply exponential decay
        decay_factor = 0.5 ** (time_since_access / half_life)

        # Apply decay to activation
        decayed_activation = max(self.minimum_activation, activation * decay_factor)

        # Check if memory is being accessed now
        is_accessed = context.get("is_accessed", False)

        # If accessed, boost activation
        if is_accessed:
            boosted_activation = min(1.0, decayed_activation + self.activation_boost_on_access)
            result["activation"] = boosted_activation
            result["last_accessed"] = current_time
        else:
            result["activation"] = decayed_activation

        # Add decay information
        result["decay_info"] = {
            "time_since_access": time_since_access,
            "decay_factor": decay_factor,
            "half_life": half_life,
        }

        return result

    def apply_decay_to_store(self, memory_store: BaseMemoryStore, current_time: float) -> None:
        """
        Apply decay to all memories in a store.

        Args:
            memory_store: The memory store to update
            current_time: Current timestamp for decay calculation
        """
        # Skip if no significant time has passed since last decay
        if current_time - self.last_decay_time < self.short_term_half_life / 10:
            return

        # Get all memories
        all_memories = memory_store.get_all()

        for memory in all_memories:
            # Get current activation
            activation = memory.metadata.get("activation", 1.0)

            # Get access and creation times
            last_accessed = memory.metadata.get(
                "last_accessed", memory.metadata.get("created_at", current_time)
            )
            created_at = memory.metadata.get("created_at", current_time)

            # Apply decay
            time_since_access = current_time - last_accessed
            time_since_creation = current_time - created_at

            # Use appropriate half-life
            if time_since_creation < 7 * 86400:  # Less than 7 days old
                half_life = self.short_term_half_life
            else:
                half_life = self.long_term_half_life

            # Apply exponential decay
            decay_factor = 0.5 ** (time_since_access / half_life)
            decayed_activation = max(self.minimum_activation, activation * decay_factor)

            # Update activation in store
            memory_store.update_metadata(memory.id, {"activation": decayed_activation})

        # Update last decay time
        self.last_decay_time = current_time


class TemporalContextBuilder(Component):
    """
    Component that extracts and integrates temporal information into memory contexts.

    This component:
    1. Extracts time references from queries
    2. Builds temporal representations for memories
    3. Groups memories into episodes based on temporal proximity
    4. Applies temporal decay to memory activations

    The temporal context helps the system understand when memories occurred
    and how they relate to each other in time.
    """

    def __init__(self, memory_store: Optional[BaseMemoryStore] = None):
        """
        Initialize the temporal context builder.

        Args:
            memory_store: Optional memory store to work with
        """
        self.memory_store = memory_store
        self.temporal_window = 3600  # Default 1-hour window for episodic clustering
        self.decay_half_life = 86400  # Default 1-day half-life for decay
        self.recency_boost_factor = 2.0  # Boost factor for recent memories
        self.recency_window = 86400  # Default 1-day window for recency boost
        self.max_temporal_clusters = 100  # Max number of episodes to maintain
        self.component_id = ComponentName.TEMPORAL_CONTEXT_BUILDER

        # Episodic clusters: {episode_id: {memory_ids, start_time, end_time, center_time}}
        self.episodes: dict[str, dict[str, Any]] = {}
        self.memory_to_episode: dict[MemoryID, str] = {}
        self.last_cluster_time = 0
        self.temporal_patterns = defaultdict(list)  # Track temporal patterns

    def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the component with configuration.

        Args:
            config: Configuration dictionary with parameters:
                - temporal_window: Time window for episodic clustering (default: 3600s)
                - decay_half_life: Half-life for temporal decay (default: 86400s)
                - recency_boost_factor: Boost factor for recent memories (default: 2.0)
                - recency_window: Time window for recency boosting (default: 86400s)
                - max_temporal_clusters: Maximum number of episodes (default: 100)
        """
        self.temporal_window = config.get("temporal_window", 3600)
        self.decay_half_life = config.get("decay_half_life", 86400)
        self.recency_boost_factor = config.get("recency_boost_factor", 2.0)
        self.recency_window = config.get("recency_window", 86400)
        self.max_temporal_clusters = config.get("max_temporal_clusters", 100)

        # set memory store if provided
        if "memory_store" in config:
            self.memory_store = config["memory_store"]

        # Build initial episodes if memory store is available
        if self.memory_store is not None:
            self._build_episodes()

    def extract_time_references(self, query: str) -> dict[str, Any]:
        """
        Extract time references from a query.

        Args:
            query: The query string

        Returns:
            dictionary of extracted temporal information
        """
        # Initialize result
        result = dict(
            has_temporal_reference=False,
            time_type=None,
            relative_time=None,
            absolute_time=None,
            time_expressions=[],
            time_keywords=[],
            debug_info={},
        )
        logger.debug(f"extract_time_references output for query '{query}': {result}")

        # No query, no temporal references
        if not query:
            return result

        # Convert to lowercase for keyword matching
        query_lower = query.lower()

        # Check for date patterns
        for pattern in DATE_PATTERNS:
            matches = pattern.findall(query)
            if matches:
                result["has_temporal_reference"] = True
                result["time_type"] = "absolute"

                string_matches = [m if isinstance(m, str) else " ".join(m) for m in matches]
                result["time_expressions"].extend(string_matches)

                for match in string_matches:
                    month_day_match = MONTH_DAY_PATTERN.search(match)
                    if month_day_match:
                        month_name, day = month_day_match.groups()
                        month_map = {
                            "january": 1,
                            "february": 2,
                            "march": 3,
                            "april": 4,
                            "may": 5,
                            "june": 6,
                            "july": 7,
                            "august": 8,
                            "september": 9,
                            "october": 10,
                            "november": 11,
                            "december": 12,
                        }
                        month_num = month_map.get(month_name.lower())
                        if month_num:
                            current_year = datetime.now().year
                            try:
                                date_obj = datetime(current_year, month_num, int(day))
                                result["absolute_time"] = date_obj.timestamp()
                                result["relative_time"] = date_obj.timestamp()
                                result["month_day_str"] = f"{month_name.lower()} {day}"
                                result["debug_info"]["parsed_date"] = date_obj.isoformat()
                                result["debug_info"]["month_day_str"] = result["month_day_str"]
                                break
                            except ValueError:
                                pass

        # Check for time patterns
        for pattern in TIME_PATTERNS:
            matches = pattern.findall(query)
            if matches:
                result["has_temporal_reference"] = True
                result["time_type"] = "absolute"
                result["time_expressions"].extend(matches)

        # Check for relative time patterns
        for pattern in RELATIVE_TIME_PATTERNS:
            matches = pattern.findall(query)
            if matches:
                result["has_temporal_reference"] = True
                result["time_type"] = "relative"
                result["time_expressions"].extend(
                    [m if isinstance(m, str) else " ".join(m) for m in matches]
                )

                for match in result["time_expressions"]:
                    parsed_time = self._parse_relative_time(match)
                    if parsed_time:
                        result["relative_time"] = parsed_time
                        result["debug_info"]["parsed_relative"] = datetime.fromtimestamp(
                            parsed_time
                        ).isoformat()
                        break

        # Check for time keywords
        found_keywords = []
        for keyword in TIME_KEYWORDS:
            if keyword in query_lower:
                found_keywords.append(keyword)
                result["has_temporal_reference"] = True

        result["time_keywords"] = found_keywords

        result["time_keywords"] = found_keywords

        # Fallback: if a temporal keyword was found but no relative_time was parsed, default to a sensible value.
        if result["time_keywords"] and result["relative_time"] is None:
            # For ambiguous terms, adjust default based on keyword content.
            # If the keyword contains "last week" or "a while ago", default to 7 days ago.
            if any(kw in result["time_keywords"] for kw in ["last week", "a while ago"]):
                default_time = time.time() - 604800  # 7 days ago
                default_label = "7 days ago"
            else:
                # Otherwise, default to the start of today.
                default_time = (
                    datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
                )
                default_label = "start of today"
            result["has_temporal_reference"] = True
            result["time_type"] = "relative"
            result["relative_time"] = default_time
            result["debug_info"]["default_relative_time"] = default_label

        return result

    def _parse_relative_time(self, time_expression: str) -> Optional[float]:
        """
        Parse a relative time expression into a timestamp.

        Args:
            time_expression: The time expression to parse

        Returns:
            Timestamp for the expression, or None if parsing fails
        """
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Handle common expressions
        if "yesterday" in time_expression:
            return (today - timedelta(days=1)).timestamp()
        elif "today" in time_expression:
            return today.timestamp()
        elif "tomorrow" in time_expression:
            return (today + timedelta(days=1)).timestamp()
        elif "next week" in time_expression:
            return (today + timedelta(days=7)).timestamp()
        elif "last week" in time_expression:
            return (today - timedelta(days=7)).timestamp()
        elif "next month" in time_expression:
            # Simplistic approximation
            return (today + timedelta(days=30)).timestamp()
        elif "last month" in time_expression:
            return (today - timedelta(days=30)).timestamp()
        elif "next year" in time_expression:
            return (today + timedelta(days=365)).timestamp()
        elif "last year" in time_expression:
            return (today - timedelta(days=365)).timestamp()

        # Handle "X days/weeks/months/years ago"
        ago_match = re.search(
            r"(\w+|\d+)\s+(day|week|month|year)s?\s+ago", time_expression, re.IGNORECASE
        )
        if ago_match:
            amount, unit = ago_match.groups()
            try:
                # Convert word numbers to digits
                word_to_number = {
                    "a": 1,
                    "one": 1,
                    "two": 2,
                    "three": 3,
                    "four": 4,
                    "five": 5,
                    "six": 6,
                    "seven": 7,
                    "eight": 8,
                    "nine": 9,
                    "ten": 10,
                }
                if amount.lower() in word_to_number:
                    amount = word_to_number[amount.lower()]
                else:
                    amount = int(amount)

                # Calculate delta
                if unit.lower() == "day":
                    delta = timedelta(days=amount)
                elif unit.lower() == "week":
                    delta = timedelta(weeks=amount)
                elif unit.lower() == "month":
                    delta = timedelta(days=30 * amount)  # Approximation
                elif unit.lower() == "year":
                    delta = timedelta(days=365 * amount)  # Approximation
                else:
                    return None

                return (today - delta).timestamp()
            except (ValueError, KeyError):
                return None

        # Handle "in X days/weeks/months/years"
        in_match = re.search(
            r"in\s+(\w+|\d+)\s+(day|week|month|year)s?", time_expression, re.IGNORECASE
        )
        if in_match:
            amount, unit = in_match.groups()
            try:
                # Convert word numbers to digits
                word_to_number = {
                    "a": 1,
                    "one": 1,
                    "two": 2,
                    "three": 3,
                    "four": 4,
                    "five": 5,
                    "six": 6,
                    "seven": 7,
                    "eight": 8,
                    "nine": 9,
                    "ten": 10,
                }
                if amount.lower() in word_to_number:
                    amount = word_to_number[amount.lower()]
                else:
                    amount = int(amount)

                # Calculate delta
                if unit.lower() == "day":
                    delta = timedelta(days=amount)
                elif unit.lower() == "week":
                    delta = timedelta(weeks=amount)
                elif unit.lower() == "month":
                    delta = timedelta(days=30 * amount)  # Approximation
                elif unit.lower() == "year":
                    delta = timedelta(days=365 * amount)  # Approximation
                else:
                    return None

                return (today + delta).timestamp()
            except (ValueError, KeyError):
                return None

        return None

    def _build_episodes(self) -> None:
        """
        Build episodic clusters from memory timestamps.

        This method groups memories into episodes based on temporal proximity.
        """
        if self.memory_store is None:
            return

        # Get all memories
        all_memories = self.memory_store.get_all()

        # Extract timestamps and IDs
        timestamps = []
        memory_ids = []
        metadata = []  # Store metadata to enable date-based indexing

        for memory in all_memories:
            creation_time = memory.metadata.get("created_at", 0)
            if creation_time > 0:
                timestamps.append(creation_time)
                memory_ids.append(memory.id)
                metadata.append(memory.metadata)

        # If no valid timestamps, return
        if not timestamps:
            return

        # Convert to numpy array and reshape for clustering
        timestamps_array = np.array(timestamps).reshape(-1, 1)

        # Perform hierarchical clustering
        if len(timestamps_array) > 1:
            # Calculate linkage matrix
            Z = linkage(timestamps_array, method="single")  # noqa: N806

            # Form flat clusters with distance threshold based on temporal window
            # Use absolute time difference instead of scaled value to better match query time references  # noqa: W505
            # Convert to seconds for consistent units
            threshold = self.temporal_window  # Time window in seconds (e.g., 3600 for 1 hour)
            clusters = fcluster(Z, threshold, criterion="distance")

            # Group memories by cluster
            cluster_memories = defaultdict(list)
            cluster_times = defaultdict(list)

            # Also create date-based indexing for efficient lookup
            date_to_episode = {}  # Maps date string (YYYY-MM-DD) to episode ID

            for i, cluster_id in enumerate(clusters):
                cluster_memories[cluster_id].append(memory_ids[i])
                cluster_times[cluster_id].append(timestamps[i])

            # Create episodes
            self.episodes = {}
            self.memory_to_episode = {}

            for cluster_id, mem_ids in cluster_memories.items():
                cluster_timestamps = cluster_times[cluster_id]
                episode_id = f"episode_{cluster_id}"

                # Get date representations for this episode
                start_date = datetime.fromtimestamp(min(cluster_timestamps))
                end_date = datetime.fromtimestamp(max(cluster_timestamps))
                center_time = sum(cluster_timestamps) / len(cluster_timestamps)

                # Calculate date representation for the entire episode
                episode_date = datetime.fromtimestamp(center_time)
                date_str = episode_date.strftime("%Y-%m-%d")
                month_day_str = episode_date.strftime("%B %d").lower()

                self.episodes[episode_id] = {
                    "memory_ids": set(mem_ids),
                    "start_time": min(cluster_timestamps),
                    "end_time": max(cluster_timestamps),
                    "center_time": center_time,
                    "date_str": date_str,
                    "month_day": month_day_str,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                }

                # Add to date index
                date_to_episode[date_str] = episode_id
                date_to_episode[month_day_str] = episode_id

                # Map memories to episodes
                for mem_id in mem_ids:
                    self.memory_to_episode[mem_id] = episode_id

            # Add date index to the object for easier lookup
            self.date_to_episode = date_to_episode

            # Limit to max episodes if needed
            if len(self.episodes) > self.max_temporal_clusters:
                # Sort by end time (most recent first)
                sorted_episodes = sorted(
                    self.episodes.items(), key=lambda x: x[1]["end_time"], reverse=True
                )

                # Keep only max_temporal_clusters
                kept_episodes = dict(sorted_episodes[: self.max_temporal_clusters])

                # Update mappings
                self.episodes = kept_episodes

                # Rebuild memory to episode mapping
                self.memory_to_episode = {}
                self.date_to_episode = {}
                for episode_id, episode in self.episodes.items():
                    # Rebuild memory mapping
                    for mem_id in episode["memory_ids"]:
                        self.memory_to_episode[mem_id] = episode_id

                    # Rebuild date mapping
                    self.date_to_episode[episode["date_str"]] = episode_id
                    self.date_to_episode[episode["month_day"]] = episode_id

        # Update last cluster time
        self.last_cluster_time = time.time()

    def apply_temporal_context(
        self, query: str, results: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Apply temporal context to retrieval results.

        Args:
            query: The query string
            results: The retrieval results to process
            context: Context information

        Returns:
            Updated results with temporal context applied
        """
        # Extract time references from query
        time_info = self.extract_time_references(query)

        # If no temporal reference, apply recency boost and return
        if not time_info["has_temporal_reference"]:
            return self._apply_recency_boost(results, context)

        # Get current time
        context.get("current_time", time.time())

        # If we have a specific time reference, use it
        target_time = None

        if time_info["relative_time"]:
            target_time = time_info["relative_time"]
        elif time_info["absolute_time"]:
            target_time = time_info["absolute_time"]

        # Apply temporal relevance scoring
        if target_time:
            return self._score_by_temporal_relevance(results, target_time, context)
        else:
            # Apply general temporal boosting for time-related queries
            return self._apply_temporal_keyword_boost(results, time_info, context)

    def _apply_recency_boost(
        self, results: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Apply recency boost to results.

        Args:
            results: The retrieval results to process
            context: Context information

        Returns:
            Updated results with recency boost applied
        """
        if not results:
            return results

        # Get current time
        current_time = context.get("current_time", time.time())

        # Create a copy of results to avoid modifying originals
        boosted_results = []

        for result in results:
            # Deep copy the result
            boosted_result = dict(result)

            # Get memory creation time
            creation_time = result.get("created_at", 0)

            if creation_time > 0:
                # Calculate time difference
                time_diff = current_time - creation_time

                # Apply recency boost if within window
                if time_diff <= self.recency_window:
                    # Calculate boost factor (linear decay)
                    boost_factor = 1.0 + (self.recency_boost_factor - 1.0) * (
                        1.0 - time_diff / self.recency_window
                    )

                    # Apply boost to relevance score
                    original_score = boosted_result.get("relevance_score", 0.0)
                    boosted_result["relevance_score"] = original_score * boost_factor
                    boosted_result["recency_boost"] = boost_factor

            boosted_results.append(boosted_result)

        # Sort by boosted relevance score
        boosted_results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

        return boosted_results

    def _score_by_temporal_relevance(
        self, results: list[dict[str, Any]], target_time: float, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Score results by temporal relevance to a target time.

        Args:
            results: The retrieval results to process
            target_time: The target timestamp to compare against
            context: Context information

        Returns:
            Updated results with temporal relevance applied
        """
        if not results:
            return results

        # set temporal scaling factor (how quickly relevance falls off with time difference)
        # Use a more flexible temporal scale to allow for episodic memory matches
        # Default is 1-day but use query-specific scale when available
        query = context.get("query", "")
        time_info = self.extract_time_references(query)

        # Use more flexible scale when searching for specific dates (e.g., "March 02")
        is_specific_date_query = time_info.get("time_type") == "absolute" and time_info.get(
            "time_expressions"
        )
        # Default 1-day scale
        default_scale = 86400

        # Use wider temporal window for specific date queries
        if is_specific_date_query:
            # Use a full-day window (86400 seconds) for date queries
            temporal_scale = 86400 * 2  # Two-day scale for date queries
        else:
            # Use context-provided or default scale
            temporal_scale = context.get("temporal_scale", default_scale)

        # First, check if we can directly match using date-based lookup
        # This handles queries like "What happened on March 02?"
        date_matched_results = []
        if is_specific_date_query and hasattr(self, "date_to_episode"):
            for expr in time_info["time_expressions"]:
                # Normalize to lowercase for matching
                expr_lower = expr.lower()
                if expr_lower in self.date_to_episode:
                    # Found an exact date match!
                    episode_id = self.date_to_episode[expr_lower]
                    episode = self.episodes.get(episode_id)
                    if episode:
                        # Get memories from this episode
                        episode_memories = episode["memory_ids"]
                        # Check if any results are in this episode
                        for result in results:
                            memory_id = str(result.get("memory_id"))
                            if memory_id in episode_memories:
                                # This is a direct match - give it a high score
                                result_copy = dict(result)
                                result_copy["relevance_score"] = 0.95  # Very high score
                                result_copy["temporal_relevance"] = 1.0
                                result_copy["direct_episode_match"] = True
                                result_copy["episode_id"] = episode_id
                                date_matched_results.append(result_copy)

        # If we found direct date matches, return them immediately
        if date_matched_results:
            # Sort by original relevance as a tiebreaker
            date_matched_results.sort(
                key=lambda x: x.get("original_relevance_score", 0.0), reverse=True
            )
            return date_matched_results

        # Otherwise, proceed with the standard temporal relevance approach
        # Create a copy of results to avoid modifying originals
        scored_results = []

        for result in results:
            # Deep copy the result
            scored_result = dict(result)

            # Store original score for reference
            original_score = scored_result.get("relevance_score", 0.0)
            scored_result["original_relevance_score"] = original_score

            # Get memory creation time
            creation_time = result.get("created_at", 0)

            if creation_time > 0:
                # Calculate time difference
                time_diff = abs(target_time - creation_time)

                # Calculate temporal relevance (Gaussian decay)
                temporal_relevance = np.exp(-(time_diff**2) / (2 * temporal_scale**2))

                # Check if memory is in an episode that matches the target time
                memory_id = str(result.get("memory_id"))
                episode_id = self.memory_to_episode.get(memory_id)

                # Episode boost - if the memory is part of a relevant episode
                episode_boost = 0.0
                if episode_id and episode_id in self.episodes:
                    episode = self.episodes[episode_id]
                    # If target time falls within the episode time range
                    if episode["start_time"] <= target_time <= episode["end_time"]:
                        episode_boost = 0.5  # Significant boost for being in the right episode
                        scored_result["in_target_episode"] = True

                # Enhanced temporal relevance with episode boost
                enhanced_relevance = min(1.0, temporal_relevance + episode_boost)

                # Get temporal weight from context or use default
                # Increase for specific date queries
                base_temporal_weight = 0.3
                if is_specific_date_query:
                    temporal_weight = 0.6  # Higher weight for date queries
                else:
                    temporal_weight = context.get("temporal_weight", base_temporal_weight)

                # Combine original score with temporal relevance
                combined_score = (
                    1.0 - temporal_weight
                ) * original_score + temporal_weight * enhanced_relevance

                # Update relevance score
                scored_result["relevance_score"] = combined_score
                scored_result["temporal_relevance"] = enhanced_relevance
                scored_result["time_diff"] = time_diff
                if episode_boost > 0:
                    scored_result["episode_boost"] = episode_boost

            scored_results.append(scored_result)

        # Sort by updated relevance score
        scored_results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

        return scored_results

    def _apply_temporal_keyword_boost(
        self, results: list[dict[str, Any]], time_info: dict[str, Any], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Apply boost for temporal keywords.

        Args:
            results: The retrieval results to process
            time_info: Temporal information extracted from query
            context: Context information

        Returns:
            Updated results with temporal keyword boost applied
        """
        if not results or not time_info["time_keywords"]:
            return results

        # Get current time
        current_time = context.get("current_time", time.time())

        # Create a copy of results to avoid modifying originals
        boosted_results = []

        # Keyword-specific handling
        recency_keywords = {"recent", "recently", "latest", "newest", "current"}
        oldness_keywords = {"old", "older", "ancient", "previously", "formerly"}

        # Detect keyword types
        is_recency_query = any(kw in time_info["time_keywords"] for kw in recency_keywords)
        is_oldness_query = any(kw in time_info["time_keywords"] for kw in oldness_keywords)

        for result in results:
            # Deep copy the result
            boosted_result = dict(result)

            # Get memory creation time
            creation_time = result.get("created_at", 0)

            if creation_time > 0:
                boost_factor = 1.0

                # Apply appropriate boost based on keyword type
                if is_recency_query:
                    # Boost recent memories
                    time_diff = current_time - creation_time
                    if time_diff <= self.recency_window:
                        boost_factor = 1.0 + (self.recency_boost_factor - 1.0) * (
                            1.0 - time_diff / self.recency_window
                        )

                elif is_oldness_query:
                    # Boost older memories
                    time_diff = current_time - creation_time
                    if time_diff > self.recency_window:
                        # Linear boost based on age (up to a limit)
                        max_age = 30 * 86400  # 30 days
                        normalized_age = min(time_diff, max_age) / max_age
                        boost_factor = 1.0 + 0.5 * normalized_age

                # Apply boost to relevance score
                original_score = boosted_result.get("relevance_score", 0.0)
                boosted_result["relevance_score"] = original_score * boost_factor
                boosted_result["temporal_keyword_boost"] = boost_factor

            boosted_results.append(boosted_result)

        # Sort by boosted relevance score
        boosted_results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

        return boosted_results

    def get_episode_for_memory(self, memory_id: MemoryID) -> Optional[str]:
        """
        Get the episode ID for a memory.

        Args:
            memory_id: The memory ID to check

        Returns:
            Episode ID if memory is part of an episode, None otherwise
        """
        return self.memory_to_episode.get(memory_id)

    def get_memories_in_episode(self, episode_id: str) -> set[MemoryID]:
        """
        Get all memories in an episode.

        Args:
            episode_id: The episode ID to check

        Returns:
            set of memory IDs in the episode
        """
        if episode_id in self.episodes:
            return set(self.episodes[episode_id]["memory_ids"])
        return set()

    def get_temporally_related_memories(
        self, memory_id: MemoryID, time_window: Optional[float] = None
    ) -> list[tuple[MemoryID, float]]:
        """
        Get memories that are temporally related to a given memory.

        Args:
            memory_id: The memory ID to find related memories for
            time_window: Custom time window (default: use instance temporal_window)

        Returns:
            list of (memory_id, temporal_proximity) tuples
        """
        if self.memory_store is None:
            return []

        # Get the memory
        try:
            memory = self.memory_store.get(memory_id)
        except KeyError:
            return []

        # Get creation time
        creation_time = memory.metadata.get("created_at", 0)
        if not creation_time:
            return []

        # Use instance window if not specified
        if time_window is None:
            time_window = self.temporal_window

        # Find temporally related memories
        related_memories = []

        for other_memory in self.memory_store.get_all():
            # Skip self
            if other_memory.id == memory_id:
                continue

            # Get other creation time
            other_time = other_memory.metadata.get("created_at", 0)
            if not other_time:
                continue

            # Check if within time window
            time_diff = abs(creation_time - other_time)
            if time_diff <= time_window:
                # Calculate temporal proximity (1 = identical time, 0 = at window edge)
                temporal_proximity = 1.0 - (time_diff / time_window)
                related_memories.append((other_memory.id, temporal_proximity))

        # Sort by proximity (highest first)
        related_memories.sort(key=lambda x: x[1], reverse=True)

        return related_memories

    def update_timestamp(self, memory_id: MemoryID, last_access_time: float) -> None:
        """
        Record that `memory_id` was just accessed at `last_access_time`.
        This stores the usage time in memory.metadata["last_accessed"].

        Args:
            memory_id: The ID of the memory
            last_access_time: The timestamp when it was accessed
        """
        if not self.memory_store:
            return
        try:
            memory = self.memory_store.get(memory_id)
            if memory is not None:
                memory.metadata["last_accessed"] = last_access_time
        except KeyError:
            # If memory_id not found, just ignore
            pass
