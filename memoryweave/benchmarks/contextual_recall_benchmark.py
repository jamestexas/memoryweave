#!/usr/bin/env python3
"""
Contextual Recall Benchmark

This script provides a focused benchmark for evaluating MemoryWeave's contextual fabric approach
against baseline retrieval methods, with specific emphasis on:

1. Recall performance in conversational contexts
2. Handling of temporal references
3. Implicit association retrieval
4. Contextual relevance in large memory sets

Features:
- Built-in timing instrumentation with minimal code changes
- Detailed metrics focused on recall and conversational coherence
- Visualization of results with statistical significance
- Realistic conversation scenarios that highlight contextual advantages
"""

import json
import logging
import os
import random
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from memoryweave.api.memory_weave import MemoryWeaveAPI
from memoryweave.benchmarks.utils.calc import (
    RetrievalMetrics,
    calculate_contextual_relevance,
    calculate_mrr,
    calculate_ndcg,
    calculate_recall_at_k,
    calculate_temporal_accuracy,
)
from memoryweave.benchmarks.utils.perf_timer import timer

# Configure rich logging
console = Console(highlight=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_path=False, rich_tracebacks=True)],
)
logger = logging.getLogger("contextual_recall")


# -----------------------------------------------------------------------------
# Benchmark Scenarios
# -----------------------------------------------------------------------------


class ScenarioType(str, Enum):
    """Types of benchmark scenarios."""

    CONVERSATIONAL = "conversational"
    TEMPORAL = "temporal"
    ASSOCIATIVE = "associative"
    LARGE_CONTEXT = "large_context"


@dataclass
class TestQuery:
    """Represents a test query with expected results."""

    id: str
    text: str
    relevant_ids: set[str]
    scenario_type: ScenarioType
    conversation_history: list[dict] = field(default_factory=list)
    temporal_context: Optional[dict] = None
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class Memory:
    """Represents a memory for testing."""

    id: str
    text: str
    metadata: dict
    embedding: Optional[np.ndarray] = None


@timer
class ConversationalScenario:
    """
    Tests ability to maintain context over multiple conversation turns.
    This scenario evaluates how well the system retrieves memories that are
    relevant to an ongoing conversation, not just the immediate query.
    """

    def __init__(self, embedding_model=None):
        self.name = "Conversational Continuity"
        self.description = "Tests ability to maintain context over multiple conversation turns"
        self.embedding_model = embedding_model
        self.memories = []
        self.queries = []

    @timer("generate_scenario")
    def generate(self, memory_limit=None) -> tuple[list[Memory], list[TestQuery]]:
        """Generate conversational scenario memories and queries."""
        # Create a multi-turn conversation about travel
        conversation = [
            {"role": "user", "text": "I'm planning a trip to Japan next month."},
            {
                "role": "assistant",
                "text": "That sounds exciting! Do you have specific cities in mind?",
            },
            {
                "role": "user",
                "text": "I want to visit Tokyo and Kyoto. I'm interested in seeing cherry blossoms.",
            },
            {
                "role": "assistant",
                "text": "Great choices! For cherry blossoms, late March to early April is usually best.",
            },
            {"role": "user", "text": "I'll be there from April 5-15. Is that good timing?"},
            {
                "role": "assistant",
                "text": "That should be good timing for cherry blossoms, especially in Kyoto.",
            },
            {
                "role": "user",
                "text": "I also want to try authentic Japanese food. Any recommendations?",
            },
            {
                "role": "assistant",
                "text": "Definitely try ramen, sushi, and tempura. Tokyo's Tsukiji market is great for fresh seafood.",
            },
            {"role": "user", "text": "My wife is vegetarian. Will that be difficult?"},
            {
                "role": "assistant",
                "text": "It can be challenging, but look for Buddhist temple cuisine (shojin ryori) in Kyoto.",
            },
        ]

        # Create memories from conversation
        memories = []
        timestamp_base = time.time() - 86400  # Start from yesterday

        for i, turn in enumerate(conversation):
            turn_timestamp = timestamp_base + i * 600  # 10-minute intervals

            memory_id = f"conv_{i}"
            memories.append(
                Memory(
                    id=memory_id,
                    text=turn["text"],
                    metadata={
                        "type": "conversation",
                        "role": turn["role"],
                        "timestamp": turn_timestamp,
                        "conversation_id": "japan_trip",
                        "turn_index": i,
                    },
                )
            )

        # Create additional memories about Japan
        japan_facts = [
            "Japan is an island country in East Asia with a population of about 126 million.",
            "Tokyo is the capital of Japan and one of the largest cities in the world.",
            "Kyoto was the capital of Japan for over a thousand years and has many historic temples.",
            "Cherry blossom (sakura) season is a major tourist attraction in Japan.",
            "Japanese cuisine is known for sushi, ramen, tempura, and many other dishes.",
            "The Shinkansen (bullet train) connects major cities in Japan at speeds up to 320 km/h.",
            "Mount Fuji is Japan's highest mountain and a UNESCO World Heritage site.",
            "Shojin ryori is traditional Buddhist vegetarian cuisine found in temples in Japan.",
            "Japan has four distinct seasons, with hot summers and cold winters in most regions.",
            "The Japanese yen (¥) is the currency of Japan.",
        ]

        for i, fact in enumerate(japan_facts):
            memory_id = f"japan_{i}"
            memories.append(
                Memory(
                    id=memory_id,
                    text=fact,
                    metadata={
                        "type": "factual",
                        "topic": "japan",
                        "timestamp": timestamp_base - (10 - i) * 86400,  # Spread over past 10 days
                    },
                )
            )

        # Add some unrelated memories as distractors
        distractors = [
            "Coffee was first discovered in Ethiopia and spread to the Middle East before reaching Europe.",
            "The Great Barrier Reef is the world's largest coral reef system, visible from space.",
            "Python is a popular programming language known for its readability and versatility.",
            "The Sahara Desert is the largest hot desert in the world, covering much of North Africa.",
            "Renewable energy sources include solar, wind, hydro, and geothermal power.",
        ]

        for i, distractor in enumerate(distractors):
            memory_id = f"distractor_{i}"
            memories.append(
                Memory(
                    id=memory_id,
                    text=distractor,
                    metadata={
                        "type": "factual",
                        "topic": f"distractor_{i}",
                        "timestamp": timestamp_base
                        - i * 86400,  # Recent but not part of conversation
                    },
                )
            )

        # Create test queries based on conversation
        queries = []

        # Query 1: Basic continuation (easy)
        history_1 = [conversation[i] for i in range(3)]
        queries.append(
            TestQuery(
                id="conv_q1",
                text="When is the best time to see cherry blossoms?",
                relevant_ids={"conv_3", "japan_3"},
                scenario_type=ScenarioType.CONVERSATIONAL,
                conversation_history=history_1,
                difficulty="easy",
            )
        )

        # Query 2: Later reference (medium)
        history_2 = [conversation[i] for i in range(6)]
        queries.append(
            TestQuery(
                id="conv_q2",
                text="What Japanese food should I try?",
                relevant_ids={"conv_7", "japan_4"},
                scenario_type=ScenarioType.CONVERSATIONAL,
                conversation_history=history_2,
                difficulty="medium",
            )
        )

        # Query 3: Complex reference (hard)
        history_3 = [conversation[i] for i in range(8)]
        queries.append(
            TestQuery(
                id="conv_q3",
                text="Will my dietary needs be accommodated?",
                relevant_ids={"conv_9", "japan_7"},
                scenario_type=ScenarioType.CONVERSATIONAL,
                conversation_history=history_3,
                difficulty="hard",
            )
        )

        # Query 4: Implicit continuation (very hard)
        history_4 = conversation
        queries.append(
            TestQuery(
                id="conv_q4",
                text="Are there any other places I should visit while I'm there?",
                relevant_ids={"japan_2", "japan_6", "conv_0", "conv_2"},
                scenario_type=ScenarioType.CONVERSATIONAL,
                conversation_history=history_4,
                difficulty="hard",
            )
        )

        return memories, queries


@timer
class TemporalScenario:
    """
    Tests ability to understand and handle temporal references correctly.
    This scenario evaluates how well the system retrieves memories based
    on time references like "yesterday," "last week," or "recently."
    """

    def __init__(self, embedding_model=None):
        self.name = "Temporal References"
        self.description = "Tests ability to understand and handle temporal references"
        self.embedding_model = embedding_model
        self.memories = []
        self.queries = []

    @timer("generate_scenario")
    def generate(self, memory_limit=None) -> tuple[list[Memory], list[TestQuery]]:
        """Generate temporal scenario memories and queries."""
        # Create memories with explicit timestamps spread across time
        memories = []
        now = time.time()

        # Yesterday's memories
        yesterday = now - 86400
        yesterday_date = datetime.fromtimestamp(yesterday).strftime("%Y-%m-%d")

        memories.append(
            Memory(
                id="yesterday_1",
                text=f"I went for a run in Central Park this morning on {yesterday_date}.",
                metadata={
                    "type": "activity",
                    "timestamp": yesterday + 28800,  # 8 AM yesterday
                    "temporal_label": "yesterday_morning",
                },
            )
        )

        memories.append(
            Memory(
                id="yesterday_2",
                text=f"I had sushi for lunch at Katsu restaurant on {yesterday_date}.",
                metadata={
                    "type": "meal",
                    "timestamp": yesterday + 43200,  # Noon yesterday
                    "temporal_label": "yesterday_noon",
                },
            )
        )

        memories.append(
            Memory(
                id="yesterday_3",
                text=f"I watched the movie 'Inception' in the evening on {yesterday_date}.",
                metadata={
                    "type": "entertainment",
                    "timestamp": yesterday + 72000,  # 8 PM yesterday
                    "temporal_label": "yesterday_evening",
                },
            )
        )

        # Last week's memories
        last_week = now - 7 * 86400
        last_week_date = datetime.fromtimestamp(last_week).strftime("%Y-%m-%d")

        memories.append(
            Memory(
                id="last_week_1",
                text=f"I started reading a book called 'Project Hail Mary' by Andy Weir on {last_week_date}.",
                metadata={
                    "type": "activity",
                    "timestamp": last_week + 50000,
                    "temporal_label": "last_week",
                },
            )
        )

        memories.append(
            Memory(
                id="last_week_2",
                text=f"I met with my friend Alex for coffee at Starbucks on {last_week_date}.",
                metadata={
                    "type": "social",
                    "timestamp": last_week + 150000,
                    "temporal_label": "last_week",
                },
            )
        )

        # This morning's memories
        this_morning = now - 28800  # 8 hours ago
        today_date = datetime.fromtimestamp(this_morning).strftime("%Y-%m-%d")

        memories.append(
            Memory(
                id="today_1",
                text=f"I had oatmeal with berries for breakfast this morning on {today_date}.",
                metadata={
                    "type": "meal",
                    "timestamp": this_morning,
                    "temporal_label": "today_morning",
                },
            )
        )

        # Last month memories
        last_month = now - 30 * 86400
        last_month_date = datetime.fromtimestamp(last_month).strftime("%Y-%m-%d")

        memories.append(
            Memory(
                id="last_month_1",
                text=f"I visited the Grand Canyon on vacation on {last_month_date}.",
                metadata={
                    "type": "travel",
                    "timestamp": last_month,
                    "temporal_label": "last_month",
                },
            )
        )

        memories.append(
            Memory(
                id="last_month_2",
                text=f"I started a new yoga practice on {last_month_date}.",
                metadata={
                    "type": "activity",
                    "timestamp": last_month + 86400 * 2,  # 2 days after start of last month
                    "temporal_label": "last_month",
                },
            )
        )

        # Create test queries with temporal references
        queries = []

        # Query 1: Explicit yesterday (easy)
        queries.append(
            TestQuery(
                id="temporal_q1",
                text="What did I do yesterday?",
                relevant_ids={"yesterday_1", "yesterday_2", "yesterday_3"},
                scenario_type=ScenarioType.TEMPORAL,
                temporal_context={"reference_time": now},
                difficulty="easy",
            )
        )

        # Query 2: Specific yesterday activity (medium)
        queries.append(
            TestQuery(
                id="temporal_q2",
                text="What movie did I watch yesterday?",
                relevant_ids={"yesterday_3"},
                scenario_type=ScenarioType.TEMPORAL,
                temporal_context={"reference_time": now},
                difficulty="medium",
            )
        )

        # Query 3: Last week reference (medium)
        queries.append(
            TestQuery(
                id="temporal_q3",
                text="What book did I start reading last week?",
                relevant_ids={"last_week_1"},
                scenario_type=ScenarioType.TEMPORAL,
                temporal_context={"reference_time": now},
                difficulty="medium",
            )
        )

        # Query 4: This morning reference (easy)
        queries.append(
            TestQuery(
                id="temporal_q4",
                text="What did I have for breakfast this morning?",
                relevant_ids={"today_1"},
                scenario_type=ScenarioType.TEMPORAL,
                temporal_context={"reference_time": now},
                difficulty="easy",
            )
        )

        # Query 5: Multiple temporal spans (hard)
        queries.append(
            TestQuery(
                id="temporal_q5",
                text="What activities have I started in the past month?",
                relevant_ids={"last_week_1", "last_month_2"},
                scenario_type=ScenarioType.TEMPORAL,
                temporal_context={"reference_time": now},
                difficulty="hard",
            )
        )

        return memories, queries


@timer
class AssociativeScenario:
    """
    Tests ability to find memories through associative connections.
    This scenario evaluates how well the system retrieves memories that
    are conceptually related but might not have direct keyword matches.
    """

    def __init__(self, embedding_model=None):
        self.name = "Associative Retrieval"
        self.description = "Tests ability to find memories through associative connections"
        self.embedding_model = embedding_model
        self.memories = []
        self.queries = []

    @timer("generate_scenario")
    def generate(self, memory_limit=None) -> tuple[list[Memory], list[TestQuery]]:
        """Generate associative scenario memories and queries."""
        memories = []
        now = time.time()

        # Create a set of memories about a topic with associative connections
        # Topic: Gardening
        gardening_memories = [
            {
                "id": "garden_1",
                "text": "I planted tomato and basil seeds in my garden last weekend.",
                "metadata": {
                    "type": "activity",
                    "topic": "gardening",
                    "subtopics": ["tomatoes", "basil", "planting"],
                    "timestamp": now - 7 * 86400,
                },
            },
            {
                "id": "garden_2",
                "text": "My tomato plants need to be watered daily during hot weather.",
                "metadata": {
                    "type": "knowledge",
                    "topic": "gardening",
                    "subtopics": ["tomatoes", "watering", "plant care"],
                    "timestamp": now - 6 * 86400,
                },
            },
            {
                "id": "garden_3",
                "text": "Basil grows well alongside tomatoes and helps repel pests naturally.",
                "metadata": {
                    "type": "knowledge",
                    "topic": "gardening",
                    "subtopics": ["basil", "tomatoes", "companion planting", "pest control"],
                    "timestamp": now - 5 * 86400,
                },
            },
            {
                "id": "garden_4",
                "text": "Organic fertilizers like compost improve soil health and plant growth.",
                "metadata": {
                    "type": "knowledge",
                    "topic": "gardening",
                    "subtopics": ["fertilizer", "compost", "soil health"],
                    "timestamp": now - 4 * 86400,
                },
            },
            {
                "id": "garden_5",
                "text": "I made a delicious pasta sauce with fresh tomatoes and basil from my garden.",
                "metadata": {
                    "type": "activity",
                    "topic": "cooking",
                    "subtopics": ["tomatoes", "basil", "pasta", "recipe"],
                    "timestamp": now - 3 * 86400,
                },
            },
            {
                "id": "garden_6",
                "text": "Growing your own vegetables reduces grocery expenses and environmental impact.",
                "metadata": {
                    "type": "opinion",
                    "topic": "gardening",
                    "subtopics": ["vegetables", "sustainability", "economics"],
                    "timestamp": now - 2 * 86400,
                },
            },
        ]

        # Topic: Programming
        programming_memories = [
            {
                "id": "code_1",
                "text": "I learned Python programming last year through an online course.",
                "metadata": {
                    "type": "experience",
                    "topic": "programming",
                    "subtopics": ["Python", "learning", "online course"],
                    "timestamp": now - 200 * 86400,
                },
            },
            {
                "id": "code_2",
                "text": "Python is great for data analysis with libraries like Pandas and NumPy.",
                "metadata": {
                    "type": "knowledge",
                    "topic": "programming",
                    "subtopics": ["Python", "data analysis", "Pandas", "NumPy"],
                    "timestamp": now - 150 * 86400,
                },
            },
            {
                "id": "code_3",
                "text": "I built a web scraper to collect gardening tips from various websites.",
                "metadata": {
                    "type": "activity",
                    "topic": "programming",
                    "subtopics": ["web scraping", "Python", "gardening", "project"],
                    "timestamp": now - 30 * 86400,
                },
            },
            {
                "id": "code_4",
                "text": "Version control systems like Git help track changes in coding projects.",
                "metadata": {
                    "type": "knowledge",
                    "topic": "programming",
                    "subtopics": ["Git", "version control", "software development"],
                    "timestamp": now - 100 * 86400,
                },
            },
            {
                "id": "code_5",
                "text": "I'm working on an app to track my garden's growth and automatically generate care reminders.",
                "metadata": {
                    "type": "activity",
                    "topic": "programming",
                    "subtopics": ["app development", "gardening", "project", "Python"],
                    "timestamp": now - 10 * 86400,
                },
            },
        ]

        # Convert to Memory objects
        for memory_data in gardening_memories + programming_memories:
            memories.append(
                Memory(
                    id=memory_data["id"],
                    text=memory_data["text"],
                    metadata=memory_data["metadata"],
                )
            )

        # Create some test queries that require associative recall
        queries = []

        # Query 1: Direct topic match (easy)
        queries.append(
            TestQuery(
                id="assoc_q1",
                text="What vegetables am I growing in my garden?",
                relevant_ids={"garden_1", "garden_2", "garden_6"},
                scenario_type=ScenarioType.ASSOCIATIVE,
                difficulty="easy",
            )
        )

        # Query 2: Cross-topic association (medium)
        queries.append(
            TestQuery(
                id="assoc_q2",
                text="How am I using technology for my gardening hobby?",
                relevant_ids={"code_3", "code_5"},
                scenario_type=ScenarioType.ASSOCIATIVE,
                difficulty="medium",
            )
        )

        # Query 3: Implicit association (hard)
        queries.append(
            TestQuery(
                id="assoc_q3",
                text="What are some sustainable practices I'm interested in?",
                relevant_ids={"garden_4", "garden_6"},
                scenario_type=ScenarioType.ASSOCIATIVE,
                difficulty="hard",
            )
        )

        # Query 4: Associative chain (very hard)
        queries.append(
            TestQuery(
                id="assoc_q4",
                text="How might I use my programming skills to improve my cooking?",
                relevant_ids={"code_5", "garden_5", "code_3"},
                scenario_type=ScenarioType.ASSOCIATIVE,
                difficulty="hard",
            )
        )

        return memories, queries


@timer
class LargeContextScenario:
    """
    Tests ability to retrieve relevant memories from a large context.
    This scenario evaluates how well the system handles a large number
    of memories and still retrieves the most contextually relevant ones.
    """

    def __init__(self, embedding_model=None):
        self.name = "Large Context Handling"
        self.description = "Tests ability to retrieve from a large set of memories"
        self.embedding_model = embedding_model
        self.memories = []
        self.queries = []

    @timer("generate_scenario")
    def generate(
        self, memory_count: int = 100, memory_limit: int | None = None
    ) -> tuple[list[Memory], list[TestQuery]]:
        """
        Generate large context scenario memories and queries.

        Args:
            memory_count: Number of memories to generate (default: 100)
        """
        memories = []
        now = time.time()

        # Create a set of "needle" memories that should be found
        needle_memories = [
            {
                "id": "needle_1",
                "text": "My best friend's wedding is scheduled for June 15th at the Botanical Gardens.",
                "metadata": {
                    "type": "event",
                    "importance": "high",
                    "timestamp": now - 10 * 86400,
                },
            },
            {
                "id": "needle_2",
                "text": "Dr. Smith prescribed 10mg of Lisinopril daily for my blood pressure.",
                "metadata": {
                    "type": "medical",
                    "importance": "high",
                    "timestamp": now - 20 * 86400,
                },
            },
            {
                "id": "needle_3",
                "text": "The password for my work account is stored in my password manager under 'Company Portal'.",
                "metadata": {
                    "type": "credential",
                    "importance": "medium",
                    "timestamp": now - 30 * 86400,
                },
            },
            {
                "id": "needle_4",
                "text": "I'm allergic to penicillin and it causes a severe rash if I take it.",
                "metadata": {
                    "type": "medical",
                    "importance": "high",
                    "timestamp": now - 180 * 86400,  # Older but important
                },
            },
            {
                "id": "needle_5",
                "text": "My flight to London is confirmed for July 3rd, departing at 9:45 PM from JFK Terminal 4.",
                "metadata": {
                    "type": "travel",
                    "importance": "high",
                    "timestamp": now - 5 * 86400,
                },
            },
        ]

        # Add needle memories
        for memory_data in needle_memories:
            memories.append(
                Memory(
                    id=memory_data["id"],
                    text=memory_data["text"],
                    metadata=memory_data["metadata"],
                )
            )

        # Generate distractor memories
        topics = ["weather", "sports", "news", "entertainment", "food", "technology", "science"]
        templates = [
            "I read an article about {topic} today.",
            "I watched a video about {topic} yesterday.",
            "My friend told me about an interesting {topic} fact.",
            "I learned something new about {topic} recently.",
            "I'm thinking about exploring {topic} more deeply.",
        ]

        for i in range(memory_count - len(needle_memories)):
            topic = random.choice(topics)
            template = random.choice(templates)
            text = template.format(topic=topic)

            # Add some specific content to make it less generic
            specifics = {
                "weather": ["rainy season", "heat wave", "storm patterns", "global climate"],
                "sports": [
                    "basketball finals",
                    "swimming competition",
                    "Olympic records",
                    "tennis tournament",
                ],
                "news": [
                    "economic policy",
                    "international relations",
                    "local elections",
                    "community event",
                ],
                "entertainment": [
                    "new movie release",
                    "TV series finale",
                    "celebrity interview",
                    "music festival",
                ],
                "food": [
                    "Italian cuisine",
                    "vegetarian recipes",
                    "baking techniques",
                    "restaurant review",
                ],
                "technology": [
                    "smartphone features",
                    "coding languages",
                    "AI development",
                    "gadget review",
                ],
                "science": [
                    "space exploration",
                    "medical breakthrough",
                    "physics theory",
                    "biology research",
                ],
            }

            specific = random.choice(specifics[topic])
            text = f"{text} The {specific} was particularly interesting."

            memories.append(
                Memory(
                    id=f"distractor_{i}",
                    text=text,
                    metadata={
                        "type": "general",
                        "topic": topic,
                        "importance": "low",
                        "timestamp": now - random.randint(1, 60) * 86400,
                    },
                )
            )

        # Create test queries for needle-in-haystack retrieval
        queries = []

        # Query 1: Explicit request (easy)
        queries.append(
            TestQuery(
                id="large_q1",
                text="When is my friend's wedding?",
                relevant_ids={"needle_1"},
                scenario_type=ScenarioType.LARGE_CONTEXT,
                difficulty="easy",
            )
        )

        # Query 2: Medical information (medium)
        queries.append(
            TestQuery(
                id="large_q2",
                text="What medications am I taking?",
                relevant_ids={"needle_2"},
                scenario_type=ScenarioType.LARGE_CONTEXT,
                difficulty="medium",
            )
        )

        # Query 3: Important but vague query (hard)
        queries.append(
            TestQuery(
                id="large_q3",
                text="What important health information do I need to know?",
                relevant_ids={"needle_2", "needle_4"},
                scenario_type=ScenarioType.LARGE_CONTEXT,
                difficulty="hard",
            )
        )

        # Query 4: Travel plans (medium)
        queries.append(
            TestQuery(
                id="large_q4",
                text="What are my upcoming travel plans?",
                relevant_ids={"needle_5"},
                scenario_type=ScenarioType.LARGE_CONTEXT,
                difficulty="medium",
            )
        )

        return memories, queries


# -----------------------------------------------------------------------------
# System Under Test Adapters
# -----------------------------------------------------------------------------


class SystemType(str, Enum):
    """Types of systems to benchmark."""

    MEMORYWEAVE_HYBRID = "memoryweave_hybrid"
    MEMORYWEAVE_STANDARD = "memoryweave_standard"
    MEMORYWEAVE_CHUNKED = "memoryweave_chunked"
    STANDARD_RAG = "standard_rag"


@timer
class SystemAdapter:
    """Base adapter for system under test."""

    def __init__(self, name, model_name=None, embedding_model=None):
        self.name = name
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.system = None

    @timer("initialize")
    def initialize(self):
        """Initialize the system."""
        raise NotImplementedError("Subclasses must implement initialize()")

    @timer("add_memory")
    def add_memory(self, memory: Memory):
        """Add a memory to the system."""
        raise NotImplementedError("Subclasses must implement add_memory()")

    @timer("retrieve")
    def retrieve(
        self,
        query: str,
        k: int = 10,
        conversation_history: list[dict] = None,
        temporal_context: dict = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories relevant to the query."""
        raise NotImplementedError("Subclasses must implement retrieve()")


@timer
class MemoryWeaveHybridAdapter(SystemAdapter):
    """Adapter for MemoryWeave hybrid system."""

    def __init__(self, model_name=None, embedding_model=None):
        super().__init__("MemoryWeave Hybrid", model_name, embedding_model)

    # Add this method to both adapter classes
    def _map_memory_id(self, original_id):
        """Map original memory IDs to the ones used internally."""
        # Extract just the base ID without any prefixes the system might have added
        if isinstance(original_id, str):
            # Handle IDs like "garden_1" or "conv_3"
            return original_id.split("_")[-1] if "_" in original_id else original_id
        return original_id

    @timer("initialize")
    def initialize(self):
        """Initialize the MemoryWeave system."""
        try:
            from memoryweave.api.hybrid_memory_weave import HybridMemoryWeaveAPI

            self.system = HybridMemoryWeaveAPI(
                model_name=self.model_name,
                debug=False,
            )

            # Replace embedding model with shared instance if provided
            if self.embedding_model:
                self.system.embedding_model = self.embedding_model

            return True
        except ImportError as e:
            logger.error(f"Failed to initialize MemoryWeave Hybrid: {e}")
            return False

    @timer("add_memory")
    def add_memory(self, memory: Memory):
        """Add a memory to the system."""
        if not self.system:
            logger.error("System not initialized")
            return False

        try:
            self.system.add_memory(memory.text, memory.metadata)
            return True
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return False

    @timer("retrieve")
    def retrieve(
        self,
        query: str,
        k: int = 10,
        conversation_history: list[dict] = None,
        temporal_context: dict = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories relevant to the query."""
        if not self.system:
            logger.error("System not initialized")
            return []

        try:
            # Prepare conversation history in correct format if provided
            if conversation_history:
                # Add conversation history to system
                for turn in conversation_history:
                    if turn.get("text"):
                        role = turn.get("role", "user")
                        self.system.add_memory(turn["text"], {"type": "conversation", "role": role})

            # Use the retrieve method to get raw results without LLM processing
            results = self.system.retrieve(query, top_k=k)

            # Format results as list of dictionaries
            formatted_results = []
            for item in results:
                # Get both the original ID and any mapped ID
                item_id = item.get("memory_id", "")
                # Create a list of possible ID variations to check
                id_variations = [
                    item_id,
                    self._map_memory_id(item_id),
                    f"conv_{item_id}" if not item_id.startswith("conv_") else item_id,
                    f"garden_{item_id}" if not item_id.startswith("garden_") else item_id,
                    f"code_{item_id}" if not item_id.startswith("code_") else item_id,
                    f"japan_{item_id}" if not item_id.startswith("japan_") else item_id,
                    f"yesterday_{item_id}" if not item_id.startswith("yesterday_") else item_id,
                ]

                formatted_results.append(
                    {
                        "id": item_id,
                        "text": item.get("content", ""),
                        "score": item.get("relevance_score", 0.5),
                        "metadata": item.get("metadata", {}),
                        "id_variations": id_variations,  # Store variations for comparison
                    }
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []


@timer
class MemoryWeaveStandardAdapter(SystemAdapter):
    """Adapter for standard MemoryWeave system."""

    def __init__(self, model_name=None, embedding_model=None):
        super().__init__("MemoryWeave Standard", model_name, embedding_model)

    @timer("initialize")
    def initialize(self):
        """Initialize the standard MemoryWeave system."""
        try:
            from memoryweave.api.memory_weave import MemoryWeaveAPI

            self.system = MemoryWeaveAPI(
                model_name=self.model_name,
                debug=False,
            )

            # Replace embedding model with shared instance if provided
            if self.embedding_model:
                self.system.embedding_model = self.embedding_model

            return True
        except ImportError as e:
            logger.error(f"Failed to initialize MemoryWeave Standard: {e}")
            return False

    @timer("add_memory")
    def add_memory(self, memory: Memory):
        """Add a memory to the system."""
        if not self.system:
            logger.error("System not initialized")
            return False

        try:
            self.system.add_memory(memory.text, memory.metadata)
            return True
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return False

    @timer("retrieve")
    def retrieve(
        self,
        query: str,
        k: int = 10,
        conversation_history: list[dict] = None,
        temporal_context: dict = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories relevant to the query."""
        if not self.system:
            logger.error("System not initialized")
            return []

        try:
            # Prepare conversation history in correct format if provided
            if conversation_history:
                # Add conversation history to system
                for turn in conversation_history:
                    if turn.get("text"):
                        role = turn.get("role", "user")
                        self.system.add_memory(turn["text"], {"type": "conversation", "role": role})

            # Use the retrieve method to get raw results without LLM processing
            results = self.system.retrieve(query, top_k=k)

            # Format results as list of dictionaries
            formatted_results = []
            for item in results:
                formatted_results.append(
                    {
                        "id": item.get("memory_id", ""),
                        "text": item.get("content", ""),
                        "score": item.get("relevance_score", 0.0),
                        "metadata": item.get("metadata", {}),
                    }
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []


@timer
class MemoryWeaveChunkedAdapter(SystemAdapter):
    """Adapter for chunked MemoryWeave system."""

    def __init__(self, model_name=None, embedding_model=None):
        super().__init__("MemoryWeave Chunked", model_name, embedding_model)

    @timer("initialize")
    def initialize(self):
        """Initialize the chunked MemoryWeave system."""
        try:
            from memoryweave.api.chunked_memory_weave import ChunkedMemoryWeaveAPI

            self.system = ChunkedMemoryWeaveAPI(
                model_name=self.model_name,
                debug=False,
            )

            # Replace embedding model with shared instance if provided
            if self.embedding_model:
                self.system.embedding_model = self.embedding_model

            return True
        except ImportError as e:
            logger.error(f"Failed to initialize MemoryWeave Chunked: {e}")
            return False

    @timer("add_memory")
    def add_memory(self, memory: Memory):
        """Add a memory to the system."""
        if not self.system:
            logger.error("System not initialized")
            return False

        try:
            self.system.add_memory(memory.text, memory.metadata)
            return True
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return False

    @timer("retrieve")
    def retrieve(
        self,
        query: str,
        k: int = 10,
        conversation_history: list[dict] = None,
        temporal_context: dict = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories relevant to the query."""
        if not self.system:
            logger.error("System not initialized")
            return []

        try:
            # Prepare conversation history in correct format if provided
            if conversation_history:
                # Add conversation history to system
                for turn in conversation_history:
                    if turn.get("text"):
                        role = turn.get("role", "user")
                        self.system.add_memory(turn["text"], {"type": "conversation", "role": role})

            # Use the retrieve method to get raw results without LLM processing
            results = self.system.retrieve(query, top_k=k)

            # Format results as list of dictionaries
            formatted_results = []
            for item in results:
                formatted_results.append(
                    {
                        "id": item.get("memory_id", ""),
                        "text": item.get("content", ""),
                        "score": item.get("relevance_score", 0.0),
                        "metadata": item.get("metadata", {}),
                    }
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []


@timer
class StandardRAGAdapter(SystemAdapter):
    """Adapter for standard RAG (retrieval augmented generation) system."""

    # Add this method to both adapter classes
    def _map_memory_id(self, original_id):
        """Map original memory IDs to the ones used internally."""
        # Extract just the base ID without any prefixes the system might have added
        if isinstance(original_id, str):
            # Handle IDs like "garden_1" or "conv_3"
            return original_id.split("_")[-1] if "_" in original_id else original_id
        return original_id

    def __init__(self, model_name=None, embedding_model=None):
        super().__init__("Standard RAG", model_name, embedding_model)

    @timer("initialize")
    def initialize(self):
        """Initialize the standard RAG system."""
        try:
            # Configure a MemoryWeave instance to behave like standard RAG
            # (similarity only, no advanced features)
            self.system = MemoryWeaveAPI(
                model_name=self.model_name,
                enable_category_management=False,
                enable_personal_attributes=False,
                enable_semantic_coherence=False,
                enable_dynamic_thresholds=False,
                debug=False,
            )

            # Configure the system to use only similarity retrieval
            self.system.strategy.initialize(
                {
                    "confidence_threshold": 0.1,
                    "similarity_weight": 1.0,  # Only use similarity
                    "associative_weight": 0.0,  # Disable associative retrieval
                    "temporal_weight": 0.0,  # Disable temporal relevance
                    "activation_weight": 0.0,  # Disable activation boosting
                }
            )

            # Replace embedding model with shared instance if provided
            if self.embedding_model:
                self.system.embedding_model = self.embedding_model

            return True
        except ImportError as e:
            logger.error(f"Failed to initialize Standard RAG: {e}")
            return False

    @timer("add_memory")
    def add_memory(self, memory: Memory):
        """Add a memory to the system."""
        if not self.system:
            logger.error("System not initialized")
            return False

        try:
            self.system.add_memory(memory.text, memory.metadata)
            return True
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return False

    @timer("retrieve")
    def retrieve(
        self,
        query: str,
        k: int = 10,
        conversation_history: list[dict] = None,
        temporal_context: dict = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories relevant to the query."""
        if not self.system:
            logger.error("System not initialized")
            return []

        try:
            # Standard RAG typically ignores conversation history,
            # but we'll add it for fair comparison
            if conversation_history:
                # Add conversation history to system
                for turn in conversation_history:
                    if turn.get("text"):
                        role = turn.get("role", "user")
                        self.system.add_memory(turn["text"], {"type": "conversation", "role": role})

            # Use the retrieve method to get raw results without LLM processing
            results = self.system.retrieve(query, top_k=k)

            # Format results as list of dictionaries
            formatted_results = []
            for item in results:
                # Get both the original ID and any mapped ID
                item_id = item.get("memory_id", "")
                # Create a list of possible ID variations to check
                id_variations = [
                    item_id,
                    self._map_memory_id(item_id),
                    f"conv_{item_id}" if not item_id.startswith("conv_") else item_id,
                    f"garden_{item_id}" if not item_id.startswith("garden_") else item_id,
                    f"code_{item_id}" if not item_id.startswith("code_") else item_id,
                    f"japan_{item_id}" if not item_id.startswith("japan_") else item_id,
                    f"yesterday_{item_id}" if not item_id.startswith("yesterday_") else item_id,
                ]

                formatted_results.append(
                    {
                        "id": item_id,
                        "text": item.get("content", ""),
                        "score": item.get("relevance_score", 0.5),
                        "metadata": item.get("metadata", {}),
                        "id_variations": id_variations,  # Store variations for comparison
                    }
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []


# -----------------------------------------------------------------------------
# Main Benchmark Class
# -----------------------------------------------------------------------------


@timer
class ContextualRecallBenchmark:
    """
    Benchmark for evaluating contextual recall capabilities of memory systems.
    This benchmark compares MemoryWeave's contextual fabric approach against
    baseline retrieval methods, focusing on recall performance in various scenarios.
    """

    def __init__(
        self,
        model_name: str = "unsloth/Llama-3.2-3B-Instruct",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dir: str = "./benchmark_results",
        systems_to_test: list[SystemType] = None,
        scenarios_to_run: list[ScenarioType] = None,
        max_memories: int = 500,
        debug: bool = False,
    ):
        """
        Initialize the benchmark.

        Args:
            model_name: Name of the model to use
            embedding_model_name: Name of the embedding model
            output_dir: Directory to save results
            systems_to_test: list of systems to test
            scenarios_to_run: list of scenarios to run
            max_memories: Maximum number of memories per scenario
            debug: Enable debug mode
        """
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.output_dir = output_dir
        self.max_memories = max_memories
        self.debug = debug

        # set default systems to test
        if systems_to_test is None:
            self.systems_to_test = [
                SystemType.MEMORYWEAVE_HYBRID,
                SystemType.MEMORYWEAVE_STANDARD,
                SystemType.STANDARD_RAG,
            ]
        else:
            self.systems_to_test = systems_to_test

        # set default scenarios to run
        if scenarios_to_run is None:
            self.scenarios_to_run = list(ScenarioType)
        else:
            self.scenarios_to_run = scenarios_to_run

        # Initialize results
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "embedding_model": embedding_model_name,
            "systems_tested": [s.value for s in self.systems_to_test],
            "scenarios_run": [s.value for s in self.scenarios_to_run],
            "system_metrics": {},
            "scenario_results": {},
        }

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize embedding model
        self.embedding_model = None
        self._initialize_embedding_model()

    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Initialized embedding model: {self.embedding_model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return False

    @timer("run_benchmark")
    def run_benchmark(self):
        """Run the benchmark for all systems and scenarios."""
        console.print(
            Panel.fit(
                f"[bold cyan]MemoryWeave Contextual Recall Benchmark[/bold cyan]\n\n"
                f"Model: [yellow]{self.model_name}[/yellow]\n"
                f"Embedding Model: [yellow]{self.embedding_model_name}[/yellow]\n"
                f"Systems: {', '.join([s.value for s in self.systems_to_test])}\n"
                f"Scenarios: {', '.join([s.value for s in self.scenarios_to_run])}\n"
                f"Max Memories: {self.max_memories}",
                border_style="cyan",
            )
        )

        # Track metrics for each system
        system_metrics = {system_type.value: {} for system_type in self.systems_to_test}
        scenario_results = {}

        # Create scenario generators
        scenario_generators = {
            ScenarioType.CONVERSATIONAL: ConversationalScenario(self.embedding_model),
            ScenarioType.TEMPORAL: TemporalScenario(self.embedding_model),
            ScenarioType.ASSOCIATIVE: AssociativeScenario(self.embedding_model),
            ScenarioType.LARGE_CONTEXT: LargeContextScenario(self.embedding_model),
        }

        # Create system adapters
        system_adapters = {}

        # Initialize systems - use a separate progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            init_task = progress.add_task(
                "Initializing systems...", total=len(self.systems_to_test)
            )

            for system_type in self.systems_to_test:
                if system_type == SystemType.MEMORYWEAVE_HYBRID:
                    adapter = MemoryWeaveHybridAdapter(
                        model_name=self.model_name, embedding_model=self.embedding_model
                    )
                elif system_type == SystemType.MEMORYWEAVE_CHUNKED:
                    adapter = MemoryWeaveChunkedAdapter(
                        model_name=self.model_name, embedding_model=self.embedding_model
                    )
                elif system_type == SystemType.MEMORYWEAVE_STANDARD:
                    adapter = MemoryWeaveStandardAdapter(
                        model_name=self.model_name, embedding_model=self.embedding_model
                    )
                elif system_type == SystemType.STANDARD_RAG:
                    adapter = StandardRAGAdapter(
                        model_name=self.model_name, embedding_model=self.embedding_model
                    )
                else:
                    logger.error(f"Unknown system type: {system_type}")
                    progress.advance(init_task)
                    continue

                # Initialize the system
                if adapter.initialize():
                    system_adapters[system_type] = adapter
                    console.print(f"  [green]✓[/green] {adapter.name} initialized")
                else:
                    console.print(f"  [red]✗[/red] Failed to initialize {adapter.name}")

                progress.advance(init_task)

        # Run benchmark for each scenario
        for scenario_type in self.scenarios_to_run:
            if scenario_type not in scenario_generators:
                logger.error(f"Unknown scenario type: {scenario_type}")
                continue

            # Generate scenario data
            scenario_generator = scenario_generators[scenario_type]
            console.print(f"\n[bold cyan]Running {scenario_generator.name} benchmark[/bold cyan]")
            console.print(f"[dim]{scenario_generator.description}[/dim]\n")

            # Generate memories and queries
            memory_limit = self.max_memories if scenario_type == ScenarioType.LARGE_CONTEXT else 50

            # Call generate with appropriate parameters depending on the scenario type
            if scenario_type == ScenarioType.LARGE_CONTEXT:
                memories, queries = scenario_generator.generate(memory_count=memory_limit)
            else:
                memories, queries = scenario_generator.generate(memory_limit=memory_limit)

            console.print(
                f"Generated [bold]{len(memories)}[/bold] memories and [bold]{len(queries)}[/bold] queries"
            )

            # Keep track of results for this scenario
            scenario_result = {}

            # For each system, run the benchmark
            for system_type, adapter in system_adapters.items():
                console.print(f"\n[bold]Testing {adapter.name}[/bold]")

                # Add memories to the system (in a separate progress bar)
                console.print(f"Adding {len(memories)} memories...")
                for i, memory in enumerate(memories):
                    adapter.add_memory(memory)
                    if (i + 1) % 10 == 0 or i + 1 == len(memories):
                        console.print(f"  Added {i + 1}/{len(memories)} memories")
                check_count = (
                    len(adapter.memory_store_adapter.get_all())
                    if hasattr(adapter, "memory_store_adapter")
                    else 0
                )
                console.print(f"  Verified [green]{check_count}[/green] memories in store")
                # Print the first memory ID for debugging
                if check_count > 0:
                    first_memory = adapter.memory_store_adapter.get_all()[0]
                    console.print(
                        f"  First memory ID: [cyan]{first_memory.id}[/cyan], expecting format like: [yellow]{list(queries[0].relevant_ids)[0] if queries[0].relevant_ids else 'unknown'}[/yellow]"
                    )

                # For each query, retrieve memories and calculate metrics
                system_scenario_metrics = RetrievalMetrics()
                console.print(f"Running {len(queries)} queries...")

                for query_idx, query in enumerate(queries):
                    # Time retrieval operation
                    adapter.timer.start("retrieval")
                    results = adapter.retrieve(
                        query=query.text,
                        k=10,
                        conversation_history=query.conversation_history,
                        temporal_context=query.temporal_context,
                    )
                    retrieval_time = adapter.timer.stop("retrieval")

                    # Extract memory IDs from results
                    retrieved_ids = [result["id"] for result in results]
                    id_variations = {
                        result["id"]: result.get("id_variations", []) for result in results
                    }
                    # Calculate metrics with ID variations
                    recall_at_1 = calculate_recall_at_k(
                        retrieved_ids, query.relevant_ids, 1, id_variations
                    )
                    # Calculate metrics
                    recall_at_1 = calculate_recall_at_k(retrieved_ids, query.relevant_ids, 1)
                    recall_at_3 = calculate_recall_at_k(retrieved_ids, query.relevant_ids, 3)
                    recall_at_5 = calculate_recall_at_k(retrieved_ids, query.relevant_ids, 5)
                    recall_at_10 = calculate_recall_at_k(retrieved_ids, query.relevant_ids, 10)
                    mrr = calculate_mrr(retrieved_ids, query.relevant_ids)
                    ndcg = calculate_ndcg(retrieved_ids, query.relevant_ids)

                    # Calculate temporal accuracy if applicable
                    temporal_accuracy = 0.0
                    if query.temporal_context:
                        # Create temporal relevance scores
                        temporal_relevance = {}
                        for memory in memories:
                            if "timestamp" in memory.metadata:
                                # Higher score for memories closer to reference time
                                reference_time = query.temporal_context.get(
                                    "reference_time", time.time()
                                )
                                time_diff = abs(memory.metadata["timestamp"] - reference_time)
                                # Exponential decay based on time difference
                                temporal_relevance[memory.id] = np.exp(
                                    -time_diff / (7 * 86400)
                                )  # 7-day half-life

                        temporal_accuracy = calculate_temporal_accuracy(
                            retrieved_ids, query.relevant_ids, temporal_relevance
                        )

                    # Calculate contextual relevance if applicable
                    contextual_relevance = 0.0
                    if query.conversation_history and self.embedding_model:
                        retrieved_texts = [result["text"] for result in results]
                        contextual_relevance = calculate_contextual_relevance(
                            query.text,
                            retrieved_texts,
                            query.conversation_history,
                            self.embedding_model,
                        )

                    # Store query results
                    system_scenario_metrics.query_count += 1
                    system_scenario_metrics.recall_at_1 += recall_at_1
                    system_scenario_metrics.recall_at_3 += recall_at_3
                    system_scenario_metrics.recall_at_5 += recall_at_5
                    system_scenario_metrics.recall_at_10 += recall_at_10
                    system_scenario_metrics.mrr += mrr
                    system_scenario_metrics.ndcg += ndcg
                    system_scenario_metrics.temporal_accuracy += temporal_accuracy
                    system_scenario_metrics.contextual_relevance += contextual_relevance
                    system_scenario_metrics.avg_retrieval_time += retrieval_time

                    # Store per-query results
                    system_scenario_metrics.per_query_results[query.id] = {
                        "query": query.text,
                        "recall_at_1": recall_at_1,
                        "recall_at_3": recall_at_3,
                        "recall_at_5": recall_at_5,
                        "recall_at_10": recall_at_10,
                        "mrr": mrr,
                        "ndcg": ndcg,
                        "temporal_accuracy": temporal_accuracy,
                        "contextual_relevance": contextual_relevance,
                        "retrieval_time": retrieval_time,
                        "difficulty": query.difficulty,
                        "retrieved_ids": retrieved_ids[:10],
                        "relevant_ids": list(query.relevant_ids),
                    }

                    console.print(f"  Completed query {query_idx + 1}/{len(queries)}")

                # Calculate averages
                if system_scenario_metrics.query_count > 0:
                    system_scenario_metrics.recall_at_1 /= system_scenario_metrics.query_count
                    system_scenario_metrics.recall_at_3 /= system_scenario_metrics.query_count
                    system_scenario_metrics.recall_at_5 /= system_scenario_metrics.query_count
                    system_scenario_metrics.recall_at_10 /= system_scenario_metrics.query_count
                    system_scenario_metrics.mrr /= system_scenario_metrics.query_count
                    system_scenario_metrics.ndcg /= system_scenario_metrics.query_count
                    system_scenario_metrics.temporal_accuracy /= system_scenario_metrics.query_count
                    system_scenario_metrics.contextual_relevance /= (
                        system_scenario_metrics.query_count
                    )
                    system_scenario_metrics.avg_retrieval_time /= (
                        system_scenario_metrics.query_count
                    )

                # Store scenario metrics for this system
                scenario_result[system_type.value] = system_scenario_metrics.to_dict()

                # Display scenario summary for this system
                console.print(
                    f"\n[bold]Results for {adapter.name} on {scenario_generator.name}:[/bold]"
                )
                console.print(f"  Recall@1: [cyan]{system_scenario_metrics.recall_at_1:.3f}[/cyan]")
                console.print(f"  Recall@5: [cyan]{system_scenario_metrics.recall_at_5:.3f}[/cyan]")
                console.print(f"  MRR: [cyan]{system_scenario_metrics.mrr:.3f}[/cyan]")
                if scenario_type == ScenarioType.TEMPORAL:
                    console.print(
                        f"  Temporal Accuracy: [cyan]{system_scenario_metrics.temporal_accuracy:.3f}[/cyan]"
                    )
                if scenario_type == ScenarioType.CONVERSATIONAL:
                    console.print(
                        f"  Contextual Relevance: [cyan]{system_scenario_metrics.contextual_relevance:.3f}[/cyan]"
                    )
                console.print(
                    f"  Avg Retrieval Time: [cyan]{system_scenario_metrics.avg_retrieval_time:.3f}s[/cyan]"
                )

            # Store results for this scenario
            scenario_results[scenario_type.value] = scenario_result

        # Calculate overall metrics for each system
        for system_type in self.systems_to_test:
            if system_type.value not in system_metrics:
                continue

            # Initialize metrics
            overall_metrics = {
                "avg_recall_at_1": 0.0,
                "avg_recall_at_3": 0.0,
                "avg_recall_at_5": 0.0,
                "avg_recall_at_10": 0.0,
                "avg_mrr": 0.0,
                "avg_ndcg": 0.0,
                "avg_temporal_accuracy": 0.0,
                "avg_contextual_relevance": 0.0,
                "avg_retrieval_time": 0.0,
                "scenario_count": 0,
            }

            # Calculate averages across scenarios
            for scenario_type in self.scenarios_to_run:
                if scenario_type.value not in scenario_results:
                    continue

                scenario_data = scenario_results[scenario_type.value]
                if system_type.value not in scenario_data:
                    continue

                system_data = scenario_data[system_type.value]
                overall_metrics["avg_recall_at_1"] += system_data["recall_at_1"]
                overall_metrics["avg_recall_at_3"] += system_data["recall_at_3"]
                overall_metrics["avg_recall_at_5"] += system_data["recall_at_5"]
                overall_metrics["avg_recall_at_10"] += system_data["recall_at_10"]
                overall_metrics["avg_mrr"] += system_data["mrr"]
                overall_metrics["avg_ndcg"] += system_data["ndcg"]
                overall_metrics["avg_temporal_accuracy"] += system_data["temporal_accuracy"]
                overall_metrics["avg_contextual_relevance"] += system_data["contextual_relevance"]
                overall_metrics["avg_retrieval_time"] += system_data["avg_retrieval_time"]
                overall_metrics["scenario_count"] += 1

            # Calculate final averages
            if overall_metrics["scenario_count"] > 0:
                for key in overall_metrics:
                    if key != "scenario_count":
                        overall_metrics[key] /= overall_metrics["scenario_count"]

            # Store overall metrics for this system
            system_metrics[system_type.value] = overall_metrics

        # Store all results
        self.results["system_metrics"] = system_metrics
        self.results["scenario_results"] = scenario_results

        # Display overall results
        self._display_results()

        # Save results
        self._save_results()

        # Create visualizations
        self._create_visualizations()

        return self.results

    def _display_results(self):
        """Display overall benchmark results."""
        console.print("\n[bold cyan]Overall Benchmark Results[/bold cyan]")

        # Create a table for overall system comparison
        table = Table(title="System Performance Comparison")

        # Add columns
        table.add_column("System", style="cyan")
        table.add_column("Recall@1", style="green")
        table.add_column("Recall@5", style="green")
        table.add_column("MRR", style="green")
        table.add_column("T-Acc", style="yellow")
        table.add_column("C-Rel", style="yellow")
        table.add_column("Time (s)", style="magenta")

        # Add rows for each system
        for system_type in self.systems_to_test:
            if system_type.value not in self.results["system_metrics"]:
                continue

            metrics = self.results["system_metrics"][system_type.value]

            table.add_row(
                system_type.value,
                f"{metrics['avg_recall_at_1']:.3f}",
                f"{metrics['avg_recall_at_5']:.3f}",
                f"{metrics['avg_mrr']:.3f}",
                f"{metrics['avg_temporal_accuracy']:.3f}",
                f"{metrics['avg_contextual_relevance']:.3f}",
                f"{metrics['avg_retrieval_time']:.3f}",
            )

        console.print(table)

        # Create tables for each scenario
        for scenario_type in self.scenarios_to_run:
            if scenario_type.value not in self.results["scenario_results"]:
                continue

            scenario_data = self.results["scenario_results"][scenario_type.value]

            console.print(f"\n[bold cyan]Results for {scenario_type.value} scenario[/bold cyan]")

            # Create a table for this scenario
            scenario_table = Table(title=f"{scenario_type.value} Performance Metrics")

            # Add columns
            scenario_table.add_column("System", style="cyan")
            scenario_table.add_column("Recall@1", style="green")
            scenario_table.add_column("Recall@5", style="green")
            scenario_table.add_column("MRR", style="green")

            # Add scenario-specific columns
            if scenario_type == ScenarioType.TEMPORAL:
                scenario_table.add_column("Temporal Acc", style="yellow")
            elif scenario_type == ScenarioType.CONVERSATIONAL:
                scenario_table.add_column("Contextual Rel", style="yellow")

            scenario_table.add_column("Time (s)", style="magenta")

            # Add rows for each system
            for system_type in self.systems_to_test:
                if system_type.value not in scenario_data:
                    continue

                system_data = scenario_data[system_type.value]

                # Build row based on scenario type
                row = [
                    system_type.value,
                    f"{system_data['recall_at_1']:.3f}",
                    f"{system_data['recall_at_5']:.3f}",
                    f"{system_data['mrr']:.3f}",
                ]

                # Add scenario-specific column
                if scenario_type == ScenarioType.TEMPORAL:
                    row.append(f"{system_data['temporal_accuracy']:.3f}")
                elif scenario_type == ScenarioType.CONVERSATIONAL:
                    row.append(f"{system_data['contextual_relevance']:.3f}")

                # Add time column
                row.append(f"{system_data['avg_retrieval_time']:.3f}")

                scenario_table.add_row(*row)

            console.print(scenario_table)

    def _convert_numpy_types(self, obj):
        """Recursively convert NumPy types to Python native types."""
        import numpy as np

        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (np.integer, np.floating)):
                    obj[key] = float(value) if isinstance(value, np.floating) else int(value)
                elif isinstance(value, np.ndarray):
                    obj[key] = value.tolist()
                elif isinstance(value, (dict, list)):
                    self._convert_numpy_types(value)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                if isinstance(value, (np.integer, np.floating)):
                    obj[i] = float(value) if isinstance(value, np.floating) else int(value)
                elif isinstance(value, np.ndarray):
                    obj[i] = value.tolist()
                elif isinstance(value, (dict, list)):
                    self._convert_numpy_types(value)

    def _save_results(self):
        """Save benchmark results to a JSON file with NumPy type handling."""

        # Create a custom JSON encoder that handles NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        # Add version information and execution details
        filename = os.path.join(
            self.output_dir,
            f"contextual_recall_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        # Convert any NumPy values in the results dictionary
        self._convert_numpy_types(self.results)

        try:
            with open(filename, "w") as f:
                json.dump(self.results, f, indent=2, cls=NumpyEncoder)
            console.print(f"\n[bold green]Results saved to:[/bold green] {filename}")
        except Exception as e:
            console.print(f"[yellow]Failed to save results: {e}[/yellow]")
            if self.debug:
                import traceback

                console.print(traceback.format_exc())

        # Also create visualizations where possible
        try:
            self._create_visualizations()
        except Exception as e:
            console.print(f"[yellow]Failed to create visualizations: {e}[/yellow]")
            if self.debug:
                import traceback

                console.print(traceback.format_exc())

    def _create_visualizations(self):
        """Create visualizations of benchmark results."""
        try:
            # Create directory for visualizations
            viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)

            # Extract data
            systems = [
                system_type.value
                for system_type in self.systems_to_test
                if system_type.value in self.results["system_metrics"]
            ]

            # Skip if no data
            if not systems:
                logger.warning("No data available for visualizations")
                return

            # Extract metrics
            recall_1 = [
                self.results["system_metrics"][system]["avg_recall_at_1"] for system in systems
            ]
            recall_5 = [
                self.results["system_metrics"][system]["avg_recall_at_5"] for system in systems
            ]
            mrr = [self.results["system_metrics"][system]["avg_mrr"] for system in systems]
            temporal = [
                self.results["system_metrics"][system]["avg_temporal_accuracy"]
                for system in systems
            ]
            contextual = [
                self.results["system_metrics"][system]["avg_contextual_relevance"]
                for system in systems
            ]
            times = [
                self.results["system_metrics"][system]["avg_retrieval_time"] for system in systems
            ]

            # Create recall comparison chart
            self._create_bar_chart(
                systems,
                [recall_1, recall_5],
                ["Recall@1", "Recall@5"],
                "Recall Performance Comparison",
                "Recall Score",
                os.path.join(viz_dir, "recall_comparison.png"),
            )

            # Create MRR comparison chart
            self._create_bar_chart(
                systems,
                [mrr],
                ["MRR"],
                "Mean Reciprocal Rank Comparison",
                "MRR Score",
                os.path.join(viz_dir, "mrr_comparison.png"),
            )

            # Create specialized metrics chart
            self._create_bar_chart(
                systems,
                [temporal, contextual],
                ["Temporal Accuracy", "Contextual Relevance"],
                "Specialized Metrics Comparison",
                "Score",
                os.path.join(viz_dir, "specialized_metrics.png"),
            )

            # Create efficiency chart
            self._create_bar_chart(
                systems,
                [times],
                ["Retrieval Time"],
                "Retrieval Efficiency Comparison",
                "Time (seconds)",
                os.path.join(viz_dir, "efficiency_comparison.png"),
                invert_colors=True,
            )

            # Create per-scenario comparison charts
            for scenario_type in self.scenarios_to_run:
                if scenario_type.value not in self.results["scenario_results"]:
                    continue

                scenario_data = self.results["scenario_results"][scenario_type.value]
                scenario_systems = [system for system in systems if system in scenario_data]

                if not scenario_systems:
                    continue

                # Extract scenario metrics
                scenario_recall_1 = [
                    scenario_data[system]["recall_at_1"] for system in scenario_systems
                ]
                scenario_recall_5 = [
                    scenario_data[system]["recall_at_5"] for system in scenario_systems
                ]
                scenario_mrr = [scenario_data[system]["mrr"] for system in scenario_systems]

                # Create scenario recall chart
                self._create_bar_chart(
                    scenario_systems,
                    [scenario_recall_1, scenario_recall_5],
                    ["Recall@1", "Recall@5"],
                    f"{scenario_type.value} Scenario: Recall Performance",
                    "Recall Score",
                    os.path.join(viz_dir, f"{scenario_type.value}_recall.png"),
                )

                # Create scenario MRR chart
                self._create_bar_chart(
                    scenario_systems,
                    [scenario_mrr],
                    ["MRR"],
                    f"{scenario_type.value} Scenario: MRR Performance",
                    "MRR Score",
                    os.path.join(viz_dir, f"{scenario_type.value}_mrr.png"),
                )

            # Create radar chart comparing all systems
            self._create_radar_chart(
                systems=systems,
                metrics=self.results["system_metrics"],
                output_file=os.path.join(viz_dir, "system_comparison_radar.png"),
            )

            console.print(f"[bold green]Visualizations saved to:[/bold green] {viz_dir}")

        except Exception as e:
            console.print(f"[yellow]Failed to create visualizations: {str(e)}[/yellow]")
            if self.debug:
                console.print(traceback.format_exc())

    def _create_bar_chart(
        self, labels, data_series, series_names, title, ylabel, output_file, invert_colors=False
    ):
        """Create a grouped bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(labels))  # Label locations
        width = 0.8 / len(data_series)  # Width of the bars

        # Colors
        colors = (
            ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
            if not invert_colors
            else ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
        )

        # Plot each data series
        for i, (data, name) in enumerate(zip(data_series, series_names)):
            offset = width * (i - len(data_series) / 2 + 0.5)
            bars = ax.bar(x + offset, data, width, label=name, color=colors[i % len(colors)])

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Add labels and legend
        ax.set_xlabel("System")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _create_radar_chart(self, systems, metrics, output_file):
        """Create a radar chart for system comparison."""
        # Categories for radar chart
        categories = [
            "Recall@1",
            "Recall@5",
            "MRR",
            "NDCG",
            "Temporal Accuracy",
            "Contextual Relevance",
        ]

        # Number of categories
        N = len(categories)

        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Initialize the figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], categories, size=12)

        # Plot each system
        for i, system in enumerate(systems):
            # Extract values for this system
            values = [
                metrics[system]["avg_recall_at_1"],
                metrics[system]["avg_recall_at_5"],
                metrics[system]["avg_mrr"],
                metrics[system]["avg_ndcg"],
                metrics[system]["avg_temporal_accuracy"],
                metrics[system]["avg_contextual_relevance"],
            ]

            # Close the loop
            values += values[:1]

            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle="solid", label=system)
            ax.fill(angles, values, alpha=0.1)

        # Add legend
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        # set title
        plt.title("System Comparison Across Metrics", size=15, y=1.1)

        # Save the chart
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)


# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MemoryWeave Contextual Recall Benchmark")
    parser.add_argument(
        "--model", default="unsloth/Llama-3.2-3B-Instruct", help="Name of the model to use"
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name of the embedding model to use",
    )
    parser.add_argument(
        "--output-dir", default="./benchmark_results", help="Directory to save results"
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=[s.value for s in SystemType],
        default=None,
        help="Systems to test (default: all)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=[s.value for s in ScenarioType],
        default=None,
        help="Scenarios to run (default: all)",
    )
    parser.add_argument(
        "--max-memories", type=int, default=500, help="Maximum number of memories per scenario"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Convert string types to enum types
    systems_to_test = [SystemType(s) for s in args.systems] if args.systems else None
    scenarios_to_run = [ScenarioType(s) for s in args.scenarios] if args.scenarios else None

    # Run benchmark
    benchmark = ContextualRecallBenchmark(
        model_name=args.model,
        embedding_model_name=args.embedding_model,
        output_dir=args.output_dir,
        systems_to_test=systems_to_test,
        scenarios_to_run=scenarios_to_run,
        max_memories=args.max_memories,
        debug=args.debug,
    )

    benchmark.run_benchmark()
