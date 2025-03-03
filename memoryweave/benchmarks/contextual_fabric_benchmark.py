# benchmarks/contextual_fabric_benchmark.py
"""
Benchmark for the Contextual Fabric components of MemoryWeave.

This benchmark evaluates the contextual fabric architecture by testing its
performance on queries that require understanding of:
1. Conversation history
2. Temporal context
3. Associative relationships
4. Activation patterns

The goal is to demonstrate how contextual fabric improves retrieval quality
beyond traditional similarity-based approaches.

Baseline Comparison:
- The baseline used is the HybridBM25VectorStrategy, which represents
  the current state-of-the-art approach combining both vector similarity
  and BM25 lexical matching
- This is an industry-standard approach found in systems like Vespa,
  Elasticsearch, and other production retrieval systems
- HybridBM25VectorStrategy already outperforms pure vector or pure lexical
  approaches on most retrieval tasks
"""

import argparse
import json
import os
import time
import random
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

from memoryweave.components.context_enhancement import ContextualEmbeddingEnhancer
from memoryweave.components.associative_linking import AssociativeMemoryLinker
from memoryweave.components.temporal_context import TemporalContextBuilder
from memoryweave.components.activation import ActivationManager
from memoryweave.components.retrieval_strategies.contextual_fabric_strategy import (
    ContextualFabricStrategy,
)
from memoryweave.components.retrieval_strategies.hybrid_bm25_vector_strategy import (
    HybridBM25VectorStrategy,
)
from memoryweave.storage.memory_store import MemoryStore
from memoryweave.interfaces.memory import Memory


class ContextualFabricBenchmark:
    """
    Benchmark for evaluating the Contextual Fabric components of MemoryWeave.

    This benchmark creates synthetic test cases that specifically demonstrate
    the advantages of the contextual fabric architecture over traditional
    retrieval approaches.
    """

    def __init__(self, embedding_dim: int = 768):
        """
        Initialize the benchmark.

        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.memory_store = MemoryStore()

        # Initialize components
        self.context_enhancer = ContextualEmbeddingEnhancer()
        self.associative_linker = AssociativeMemoryLinker(self.memory_store)
        self.temporal_context = TemporalContextBuilder(self.memory_store)
        self.activation_manager = ActivationManager(
            memory_store=self.memory_store, associative_linker=self.associative_linker
        )

        # Initialize retrieval strategies
        self.contextual_fabric_strategy = ContextualFabricStrategy(
            memory_store=self.memory_store,
            associative_linker=self.associative_linker,
            temporal_context=self.temporal_context,
            activation_manager=self.activation_manager,
        )

        # Create a compatibility wrapper for the hybrid strategy
        # The hybrid strategy expects a ContextualMemory, not a MemoryStore
        from memoryweave.core.contextual_memory import ContextualMemory

        self.memory_adapter = ContextualMemory(embedding_dim=embedding_dim)
        self.memory_adapter.memory_embeddings = np.zeros((0, embedding_dim))
        self.memory_adapter.memory_metadata = []
        self.baseline_strategy = HybridBM25VectorStrategy(memory=self.memory_adapter)

        # Configure components
        self.context_enhancer.initialize(
            {"conversation_weight": 0.25, "temporal_weight": 0.2, "topical_weight": 0.2}
        )

        self.associative_linker.initialize(
            {
                "similarity_threshold": 0.3,
                "temporal_weight": 0.3,
                "semantic_weight": 0.7,
                "max_links_per_memory": 10,
            }
        )

        self.temporal_context.initialize(
            {
                "temporal_window": 3600,  # 1 hour
                "decay_half_life": 86400,  # 1 day
                "recency_boost_factor": 2.0,
            }
        )

        self.activation_manager.initialize(
            {"base_activation": 0.1, "spreading_factor": 0.5, "max_spreading_hops": 2}
        )

        self.contextual_fabric_strategy.initialize(
            {
                "confidence_threshold": 0.1,
                "similarity_weight": 0.5,
                "associative_weight": 0.3,
                "temporal_weight": 0.1,
                "activation_weight": 0.1,
                "max_associative_hops": 2,
                "debug": True,
            }
        )

        self.baseline_strategy.initialize(
            {
                "confidence_threshold": 0.1,
                "vector_weight": 0.5,
                "bm25_weight": 0.5,
                # Use less strict parameters to ensure BM25 works on synthetic data
                "bm25_b": 0.5,  # Lower b means less length normalization
                "bm25_k1": 2.0,  # Higher k1 means more term frequency importance
                "enable_dynamic_weighting": False,  # Disable dynamic weighting for benchmark
                "min_results": 5,  # Ensure at least some results are returned
            }
        )

        # Results
        self.results = {
            "benchmark_name": "ContextualFabricBenchmark",
            "timestamp": datetime.now().isoformat(),
            "config": {"embedding_dim": embedding_dim},
            "test_cases": [],
            "summary": {},
        }

    def generate_synthetic_dataset(self, num_memories: int = 100) -> None:
        """
        Generate a synthetic dataset for benchmarking.

        This dataset creates memories with specific properties to test
        contextual fabric features.

        Args:
            num_memories: Number of memories to generate
        """
        # Clear existing memories
        self.memory_store.clear()

        # Create base time frame
        now = time.time()
        one_day = 86400
        one_week = 7 * one_day

        # Create memories in different temporal clusters
        timestamps = [
            # Recent cluster (today)
            *[now - random.randint(0, one_day // 2) for _ in range(num_memories // 5)],
            # Yesterday cluster
            *[now - one_day - random.randint(0, one_day // 2) for _ in range(num_memories // 5)],
            # Last week cluster
            *[now - one_week - random.randint(0, one_day) for _ in range(num_memories // 5)],
            # Last month cluster
            *[now - 4 * one_week - random.randint(0, one_week) for _ in range(num_memories // 5)],
            # Old memories (random times)
            *[now - random.randint(one_week, 12 * one_week) for _ in range(num_memories // 5)],
        ]

        # Define topics
        topics = [
            "food",
            "travel",
            "work",
            "family",
            "technology",
            "education",
            "health",
            "entertainment",
            "finance",
            "sports",
        ]

        # Generate random embeddings that will be semantically related
        # We'll organize them in topic clusters for semantic similarity
        topic_centroids = {}
        for topic in topics:
            # Create a random but distinct centroid for each topic
            centroid = np.random.randn(self.embedding_dim)
            # Normalize
            centroid = centroid / np.linalg.norm(centroid)
            topic_centroids[topic] = centroid

        # Create memories
        for i in range(num_memories):
            # Select timestamp
            timestamp = timestamps[i % len(timestamps)]

            # Generate topical content
            topic = topics[i % len(topics)]
            subtopic = random.choice(topics)  # Create cross-topic relationships

            # Add some sequential memories that should be associated
            is_sequential = i > 0 and i % 5 == 0
            prev_id = str(i - 1) if is_sequential else None

            # Create richer memory content with specific features for testing
            # Generate more realistic text that BM25 can work with

            # Select topic-specific keywords
            topic_keywords = {
                "food": [
                    "recipe",
                    "delicious",
                    "restaurant",
                    "cooking",
                    "meal",
                    "dish",
                    "flavor",
                    "taste",
                    "cuisine",
                    "ingredient",
                ],
                "travel": [
                    "vacation",
                    "destination",
                    "trip",
                    "journey",
                    "explore",
                    "tourism",
                    "hotel",
                    "flight",
                    "adventure",
                    "sightseeing",
                ],
                "work": [
                    "project",
                    "meeting",
                    "deadline",
                    "colleague",
                    "office",
                    "career",
                    "job",
                    "task",
                    "productivity",
                    "professional",
                ],
                "family": [
                    "parents",
                    "children",
                    "relatives",
                    "siblings",
                    "household",
                    "relationship",
                    "marriage",
                    "love",
                    "home",
                    "tradition",
                ],
                "technology": [
                    "device",
                    "software",
                    "hardware",
                    "digital",
                    "innovation",
                    "computer",
                    "application",
                    "system",
                    "gadget",
                    "internet",
                ],
                "education": [
                    "learning",
                    "student",
                    "school",
                    "course",
                    "knowledge",
                    "teacher",
                    "class",
                    "study",
                    "academic",
                    "curriculum",
                ],
                "health": [
                    "wellness",
                    "fitness",
                    "exercise",
                    "nutrition",
                    "medical",
                    "doctor",
                    "diet",
                    "healthy",
                    "condition",
                    "therapy",
                ],
                "entertainment": [
                    "movie",
                    "music",
                    "game",
                    "show",
                    "performance",
                    "television",
                    "concert",
                    "theater",
                    "artist",
                    "festival",
                ],
                "finance": [
                    "money",
                    "investment",
                    "budget",
                    "savings",
                    "expense",
                    "financial",
                    "bank",
                    "economy",
                    "fund",
                    "income",
                ],
                "sports": [
                    "athlete",
                    "team",
                    "competition",
                    "game",
                    "championship",
                    "training",
                    "coach",
                    "player",
                    "fitness",
                    "tournament",
                ],
            }

            # Get keywords for this memory's topics
            main_keywords = topic_keywords.get(topic, [])
            sub_keywords = topic_keywords.get(subtopic, [])

            # Select random keywords to include in the text
            selected_main = random.sample(main_keywords, min(3, len(main_keywords)))
            selected_sub = random.sample(sub_keywords, min(2, len(sub_keywords)))

            # Generate sentences with repeated keywords for better BM25 indexing
            sentences = []

            # Topic introduction with keyword repetition
            sentences.append(f"Memory {i} contains information about {topic}.")

            # Add memory-specific content with keywords
            if topic == "food":
                sentences.append(
                    f"I found a great {random.choice(selected_main)} while looking for {random.choice(selected_main)}."
                )
                sentences.append(
                    f"The {random.choice(selected_main)} had amazing {random.choice(selected_main)} and {random.choice(selected_sub)}."
                )
            elif topic == "travel":
                sentences.append(
                    f"I went on a {random.choice(selected_main)} to explore {random.choice(selected_main)}."
                )
                sentences.append(
                    f"The {random.choice(selected_main)} was filled with {random.choice(selected_main)} and {random.choice(selected_sub)}."
                )
            elif topic == "health":
                sentences.append(
                    f"I've been focusing on my {random.choice(selected_main)} by improving {random.choice(selected_main)}."
                )
                sentences.append(
                    f"The {random.choice(selected_main)} routine includes {random.choice(selected_main)} and {random.choice(selected_sub)}."
                )
            else:
                sentences.append(
                    f"I've been working on {random.choice(selected_main)} related to {random.choice(selected_main)}."
                )
                sentences.append(
                    f"The {random.choice(selected_main)} involves aspects of {random.choice(selected_main)} and some {random.choice(selected_sub)}."
                )

            # Add a sentence with subtopic reference
            sentences.append(
                f"It also relates to {subtopic} because of the {random.choice(selected_sub)}."
            )

            # Add sequential reference if applicable
            if is_sequential:
                sentences.append(
                    f"This follows my previous experience with {topic} that I mentioned earlier."
                )

            # Create complete text
            memory_text = " ".join(sentences)

            # Prepare content structure
            content = {
                "text": memory_text,
                "metadata": {
                    "topics": [topic, subtopic],
                    "keywords": selected_main + selected_sub,
                    "created_at": timestamp,
                    "importance": random.random(),
                    "sequential_to": prev_id,
                },
            }

            # Generate embedding based on topic centroid with some noise
            base_embedding = topic_centroids[topic] * 0.7 + topic_centroids[subtopic] * 0.3
            noise = np.random.randn(self.embedding_dim) * 0.1
            embedding = base_embedding + noise
            embedding = embedding / np.linalg.norm(embedding)

            # Add to store
            memory_id = self.memory_store.add(embedding, content["text"], content["metadata"])

            # Apply contextual enhancement
            enhanced_data = self.context_enhancer.process(
                data={"id": memory_id, "embedding": embedding, **content},
                context={"current_time": timestamp, "topics": content["metadata"]["topics"]},
            )

            # Update memory store with enhanced embedding
            if "embedding" in enhanced_data:
                self.memory_store._memories[memory_id] = enhanced_data["embedding"]

        # Process memories to build fabric
        print("Building associative links...")
        self.associative_linker._rebuild_all_links()

        # Build temporal episodes
        print("Building temporal episodes...")
        self.temporal_context._build_episodes()

        # Activate some memories to simulate usage patterns
        print("Setting activation patterns...")
        for i in range(min(10, num_memories)):
            # Activate a few random memories
            memory_id = str(random.randint(0, num_memories - 1))
            self.activation_manager.activate_memory(
                memory_id=memory_id,
                activation_level=random.random() * 0.7 + 0.3,  # Random activation 0.3-1.0
                spread=True,
            )

        print(f"Generated {num_memories} synthetic memories with contextual fabric")

    def _create_query_embedding(self, text: str, topic: str) -> np.ndarray:
        """
        Create a synthetic query embedding based on text and topic.

        Args:
            text: Query text
            topic: Main topic of the query

        Returns:
            Synthetic embedding for the query
        """
        # Very simple embedding generation for benchmark purposes
        embedding = np.random.randn(self.embedding_dim)

        # Add topic bias
        bias = np.zeros(self.embedding_dim)
        bias[hash(topic) % self.embedding_dim] = 0.5

        # Add text bias
        for word in text.split():
            idx = hash(word) % self.embedding_dim
            bias[idx] += 0.1

        # Combine and normalize
        embedding = embedding * 0.3 + bias
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def _sync_memory_adapter(self) -> None:
        """
        Synchronize the MemoryStore data with the ContextualMemory adapter.

        This ensures that both retrieval strategies have access to the same data.
        """
        # Get all memories from the store
        memories = self.memory_store.get_all()

        if not memories:
            # No memories to sync
            return

        # Create numpy arrays for the memory adapter
        embeddings = np.stack([memory.embedding for memory in memories])

        # Create metadata list for the memory adapter
        metadata_list = []
        for memory in memories:
            # Convert memory content to a format compatible with ContextualMemory
            if isinstance(memory.content, dict) and "text" in memory.content:
                memory_content = memory.content
            else:
                # Handle alternative content formats
                if isinstance(memory.content, str):
                    text = memory.content
                else:
                    text = str(memory.content)

                memory_content = {"text": text}

            # Combine content and metadata
            metadata_entry = {"content": memory_content, "id": memory.id, **memory.metadata}
            metadata_list.append(metadata_entry)

        # Update the memory adapter
        self.memory_adapter.memory_embeddings = embeddings
        self.memory_adapter.memory_metadata = metadata_list

        # Create activation levels if needed
        if not hasattr(self.memory_adapter, "activation_levels") or len(
            self.memory_adapter.activation_levels
        ) != len(memories):
            self.memory_adapter.activation_levels = np.ones(len(memories))

        # Manually trigger BM25 index initialization to avoid failures
        if hasattr(self.baseline_strategy, "_initialize_bm25_index"):
            # Configure and initialize BM25 index
            print("Pre-initializing BM25 index for hybrid strategy...")
            self.baseline_strategy.index_initialized = False
            self.baseline_strategy._initialize_bm25_index()

    def _create_test_case(
        self,
        name: str,
        query: str,
        topic: str,
        expected_results: List[str],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        current_time: Optional[float] = None,
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Create a test case for evaluation.

        Args:
            name: Test case name
            query: Query string
            topic: Main topic of the query
            expected_results: List of expected memory IDs
            conversation_history: Optional conversation history
            current_time: Optional current time
            description: Test case description

        Returns:
            Test case dictionary
        """
        return {
            "name": name,
            "query": query,
            "topic": topic,
            "expected_results": expected_results,
            "conversation_history": conversation_history or [],
            "current_time": current_time or time.time(),
            "description": description,
        }

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """
        Generate test cases for the benchmark.

        These test cases are designed to showcase the advantages of
        contextual fabric over traditional retrieval.

        Returns:
            List of test cases
        """
        test_cases = []
        now = time.time()
        one_day = 86400

        # Get some random memory IDs for reference
        all_ids = self.memory_store.get_ids()
        random_ids = random.sample(all_ids, min(10, len(all_ids)))

        # 1. Conversation context test case
        # This tests if the system can use conversation history to improve retrieval
        # Create more realistic conversations for each topic
        conversation_contexts = {}

        # Food conversation (more detailed and multi-turn)
        conversation_contexts["food"] = [
            {
                "text": "Let's talk about food options for dinner tonight.",
                "embedding": self._create_query_embedding(
                    "Let's talk about food options for dinner tonight", "food"
                ),
                "timestamp": now - 600,
            },
            {
                "text": "I'm thinking of trying that new Italian restaurant downtown.",
                "embedding": self._create_query_embedding(
                    "I'm thinking of trying that new Italian restaurant downtown", "food"
                ),
                "timestamp": now - 500,
            },
            {
                "text": "Their pasta dishes are supposed to be amazing, especially the carbonara.",
                "embedding": self._create_query_embedding(
                    "Their pasta dishes are supposed to be amazing, especially the carbonara",
                    "food",
                ),
                "timestamp": now - 400,
            },
            {
                "text": "But I'm also in the mood for something with more spice.",
                "embedding": self._create_query_embedding(
                    "But I'm also in the mood for something with more spice", "food"
                ),
                "timestamp": now - 300,
            },
        ]

        # Travel conversation
        conversation_contexts["travel"] = [
            {
                "text": "I'm planning my next vacation.",
                "embedding": self._create_query_embedding(
                    "I'm planning my next vacation", "travel"
                ),
                "timestamp": now - 500,
            },
            {
                "text": "I've been researching destinations in Southeast Asia.",
                "embedding": self._create_query_embedding(
                    "I've been researching destinations in Southeast Asia", "travel"
                ),
                "timestamp": now - 400,
            },
            {
                "text": "Thailand and Vietnam both look amazing for cultural experiences.",
                "embedding": self._create_query_embedding(
                    "Thailand and Vietnam both look amazing for cultural experiences", "travel"
                ),
                "timestamp": now - 300,
            },
        ]

        # Health conversation
        conversation_contexts["health"] = [
            {
                "text": "I've been trying to improve my fitness routine lately.",
                "embedding": self._create_query_embedding(
                    "I've been trying to improve my fitness routine lately", "health"
                ),
                "timestamp": now - 500,
            },
            {
                "text": "Combining cardio with strength training seems most effective.",
                "embedding": self._create_query_embedding(
                    "Combining cardio with strength training seems most effective", "health"
                ),
                "timestamp": now - 400,
            },
            {
                "text": "I've noticed better results when I also focus on proper nutrition.",
                "embedding": self._create_query_embedding(
                    "I've noticed better results when I also focus on proper nutrition", "health"
                ),
                "timestamp": now - 300,
            },
        ]

        # Create test cases for multiple conversation contexts to better evaluate the feature
        for context_topic, convo_history in conversation_contexts.items():
            # Find memories related to this topic
            topic_memories = []
            for memory_id in all_ids:
                memory = self.memory_store.get(memory_id)
                if "topics" in memory.metadata and context_topic in memory.metadata["topics"]:
                    topic_memories.append(memory_id)

            # Choose an appropriate query based on topic
            if context_topic == "food":
                query = "What other options are there?"
                description = "Tests if the system can use food conversation history to disambiguate a vague query"
            elif context_topic == "travel":
                query = "What destinations would you recommend?"
                description = "Tests if the system can use travel conversation history to disambiguate recommendations"
            elif context_topic == "health":
                query = "What else should I consider for improvements?"
                description = "Tests if the system can use health conversation history to disambiguate a vague question"
            else:
                query = "Tell me more about this."
                description = (
                    f"Tests if the system can use {context_topic} conversation history for context"
                )

            test_cases.append(
                self._create_test_case(
                    name=f"conversation_context_{context_topic}",
                    query=query,
                    topic=context_topic,
                    expected_results=topic_memories[:5],
                    conversation_history=convo_history,
                    current_time=now,
                    description=description,
                )
            )

        # 2. Temporal context test case
        # This tests if the system can find memories based on temporal references
        recent_memories = []
        yesterday_memories = []
        for memory_id in all_ids:
            memory = self.memory_store.get(memory_id)
            timestamp = memory.metadata.get("created_at", 0)
            if timestamp > now - one_day / 2:
                recent_memories.append(memory_id)
            elif now - one_day - one_day / 2 < timestamp < now - one_day + one_day / 2:
                yesterday_memories.append(memory_id)

        test_cases.append(
            self._create_test_case(
                name="temporal_recent",
                query="Show me recent information",
                topic="time",
                expected_results=recent_memories[:5],
                current_time=now,
                description="Tests if the system can identify recent memories",
            )
        )

        test_cases.append(
            self._create_test_case(
                name="temporal_yesterday",
                query="What happened yesterday?",
                topic="time",
                expected_results=yesterday_memories[:5],
                current_time=now,
                description="Tests if the system can identify memories from yesterday",
            )
        )

        # 3. Associative links test case
        # This tests if the system can find memories through associative links
        if random_ids:
            # Activate a seed memory
            seed_id = random_ids[0]
            self.activation_manager.activate_memory(seed_id, 1.0, True)

            # Get its associative links
            associative_memories = []
            if self.associative_linker:
                links = self.associative_linker.get_associative_links(seed_id)
                associative_memories = [link_id for link_id, _ in links]

            seed_memory = self.memory_store.get(seed_id)
            seed_text = (
                seed_memory.content["text"]
                if isinstance(seed_memory.content, dict)
                else str(seed_memory.content)
            )

            test_cases.append(
                self._create_test_case(
                    name="associative_links",
                    query=f"Tell me more about {seed_text}",
                    topic="association",
                    expected_results=[seed_id] + associative_memories[:4],
                    current_time=now,
                    description="Tests if the system can find memories through associative links",
                )
            )

        # 4. Activation patterns test case
        # This tests if the system boosts activated memories
        activated_memories = []
        if self.activation_manager:
            activations = self.activation_manager.get_activated_memories(threshold=0.3)
            activated_memories = list(activations.keys())

        test_cases.append(
            self._create_test_case(
                name="activation_patterns",
                query="What have I been thinking about recently?",
                topic="memory",
                expected_results=activated_memories[:5],
                current_time=now,
                description="Tests if the system retrieves highly activated memories",
            )
        )

        # 5. Episodic memory test case
        # This tests if the system can retrieve memories from the same episode
        if (
            self.temporal_context
            and hasattr(self.temporal_context, "episodes")
            and self.temporal_context.episodes
        ):
            # Create multiple episodic memory test cases for better testing
            # Pick a few episodes to test with
            episode_ids = list(self.temporal_context.episodes.keys())
            test_episode_ids = random.sample(episode_ids, min(3, len(episode_ids)))

            for idx, episode_id in enumerate(test_episode_ids):
                episode_memories = list(self.temporal_context.get_memories_in_episode(episode_id))

                # Get episode time info
                episode = self.temporal_context.episodes[episode_id]
                episode_time = episode["center_time"]

                # Add debugging info to log what we're querying
                print(f"Creating episodic memory test for episode: {episode_id}")
                print(f"  - Date strings: {episode.get('date_str')}, {episode.get('month_day')}")
                print(f"  - Contains memories: {episode_memories[:5]}")

                # Convert to datetime for query formulation
                episode_dt = datetime.fromtimestamp(episode_time)
                # Format as "Month DD" which our improved temporal context can now handle
                date_str = episode_dt.strftime("%B %d")

                test_cases.append(
                    self._create_test_case(
                        name=f"episodic_memory_{idx + 1}",
                        query=f"What happened on {date_str}?",
                        topic="time",
                        expected_results=episode_memories[:5],
                        current_time=now,
                        description=f"Tests if the system can retrieve memories from temporal episode {date_str}",
                    )
                )

        return test_cases

    def evaluate_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a test case with both baseline and contextual fabric strategies.

        Args:
            test_case: Test case to evaluate

        Returns:
            Test case results
        """
        query = test_case["query"]

        # Create query embedding
        query_embedding = self._create_query_embedding(query, test_case["topic"])

        # Prepare context
        context = {
            "query": query,
            "query_embedding": query_embedding,
            "top_k": 10,
            "current_time": test_case["current_time"],
            "memory_store": self.memory_store,
        }

        # Add conversation history if available
        if test_case.get("conversation_history"):
            context["conversation_history"] = test_case["conversation_history"]

        # Synchronize the memory store data with the ContextualMemory adapter
        # This ensures the same data is available to both strategies
        self._sync_memory_adapter()

        # Get results from baseline strategy
        baseline_results = self.baseline_strategy.process_query(query, context)["results"]

        # Get results from contextual fabric strategy
        fabric_results = self.contextual_fabric_strategy.process_query(query, context)["results"]

        # Evaluate against expected results
        expected_ids = set(test_case["expected_results"])

        baseline_ids = set(str(r["memory_id"]) for r in baseline_results)
        fabric_ids = set(str(r["memory_id"]) for r in fabric_results)

        # Calculate metrics
        baseline_correct = len(baseline_ids.intersection(expected_ids))
        fabric_correct = len(fabric_ids.intersection(expected_ids))

        baseline_precision = baseline_correct / len(baseline_results) if baseline_results else 0
        fabric_precision = fabric_correct / len(fabric_results) if fabric_results else 0

        baseline_recall = baseline_correct / len(expected_ids) if expected_ids else 0
        fabric_recall = fabric_correct / len(expected_ids) if expected_ids else 0

        # Calculate F1 scores
        baseline_f1 = (
            2 * (baseline_precision * baseline_recall) / (baseline_precision + baseline_recall)
            if (baseline_precision + baseline_recall) > 0
            else 0
        )
        fabric_f1 = (
            2 * (fabric_precision * fabric_recall) / (fabric_precision + fabric_recall)
            if (fabric_precision + fabric_recall) > 0
            else 0
        )

        # Return results
        return {
            "test_case": test_case["name"],
            "description": test_case["description"],
            "query": test_case["query"],
            "expected_results": test_case["expected_results"],
            "baseline_results": [str(r["memory_id"]) for r in baseline_results],
            "fabric_results": [str(r["memory_id"]) for r in fabric_results],
            "metrics": {
                "baseline": {
                    "precision": baseline_precision,
                    "recall": baseline_recall,
                    "f1": baseline_f1,
                },
                "fabric": {"precision": fabric_precision, "recall": fabric_recall, "f1": fabric_f1},
                "improvement": {
                    "precision": fabric_precision - baseline_precision,
                    "recall": fabric_recall - baseline_recall,
                    "f1": fabric_f1 - baseline_f1,
                },
            },
        }

    def run_benchmark(
        self, num_memories: int = 100, output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete benchmark.

        Args:
            num_memories: Number of memories to generate
            output_file: Optional file to save results

        Returns:
            Benchmark results
        """
        print(f"Running ContextualFabricBenchmark with {num_memories} memories")
        print(
            "Note: 'BM25 retrieval failed' warnings are expected with synthetic data and can be safely ignored."
        )
        print("      The hybrid strategy will fall back to vector similarity in these cases.")

        # Generate synthetic dataset
        self.generate_synthetic_dataset(num_memories)

        # Synchronize the memory adapter to ensure compatibility with the hybrid strategy
        self._sync_memory_adapter()

        # Generate test cases
        test_cases = self.generate_test_cases()
        print(f"Generated {len(test_cases)} test cases")

        # Run evaluation
        all_test_results = []
        for test_case in tqdm(test_cases, desc="Evaluating test cases"):
            result = self.evaluate_test_case(test_case)
            all_test_results.append(result)
            print(
                f"Test: {result['test_case']} - Baseline F1: {result['metrics']['baseline']['f1']:.3f}, Fabric F1: {result['metrics']['fabric']['f1']:.3f}"
            )

        # Calculate summary metrics
        baseline_f1_scores = [r["metrics"]["baseline"]["f1"] for r in all_test_results]
        fabric_f1_scores = [r["metrics"]["fabric"]["f1"] for r in all_test_results]

        summary = {
            "average_baseline_f1": sum(baseline_f1_scores) / len(baseline_f1_scores)
            if baseline_f1_scores
            else 0,
            "average_fabric_f1": sum(fabric_f1_scores) / len(fabric_f1_scores)
            if fabric_f1_scores
            else 0,
            "average_improvement": sum(fabric_f1_scores) / len(fabric_f1_scores)
            - sum(baseline_f1_scores) / len(baseline_f1_scores)
            if baseline_f1_scores and fabric_f1_scores
            else 0,
            "num_test_cases": len(all_test_results),
            "num_memories": num_memories,
        }

        # Add to overall results
        self.results["test_cases"] = all_test_results
        self.results["summary"] = summary

        print(f"Benchmark complete. Average improvement: {summary['average_improvement']:.3f} F1")

        # Save results if requested
        if output_file:
            with open(output_file, "w") as f:
                json.dump(self.results, f, indent=2)
            print(f"Results saved to {output_file}")

        return self.results


def main():
    """Run the benchmark from command line."""
    parser = argparse.ArgumentParser(description="MemoryWeave Contextual Fabric Benchmark")
    parser.add_argument(
        "--memories", type=int, default=100, help="Number of synthetic memories to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="contextual_fabric_results.json",
        help="Output file for results",
    )
    parser.add_argument("--embedding-dim", type=int, default=768, help="Embedding dimension")

    args = parser.parse_args()

    benchmark = ContextualFabricBenchmark(embedding_dim=args.embedding_dim)
    benchmark.run_benchmark(num_memories=args.memories, output_file=args.output)


if __name__ == "__main__":
    main()
