# memoryweave/evaluation/synthetic/semantic_generator.py
"""
Advanced semantic query and memory generation for MemoryWeave evaluation.

This module provides enhanced synthetic data generation with semantic
relationships, memory chains, contradictions, and temporal structures.
"""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class MemoryRelationship:
    """Represents a semantic relationship between memories."""

    type: str  # "supports", "contradicts", "elaborates", "temporal_before", "temporal_after"
    source_idx: int
    target_idx: int
    strength: float = 1.0  # 0.0 to 1.0, how strong the relationship is


@dataclass
class MemorySeries:
    """Represents a series of related memories forming a coherent narrative."""

    name: str
    description: str
    memory_indices: list[int]
    temporal_order: bool = True  # Whether the series has a temporal ordering


@dataclass
class SemanticMemory:
    """Enhanced memory representation with semantic relationships."""

    text: str
    metadata: dict[str, Any]
    embedding: np.ndarray
    timestamp: datetime
    related_memories: list[MemoryRelationship] = field(default_factory=list)
    importance: float = 0.5  # 0.0 to 1.0


class SemanticDataGenerator:
    """
    Generates semantically rich synthetic data for evaluation.

    This generator creates memories with semantic relationships, temporal
    structure, and varying importance. It also generates queries with
    specific characteristics to test different aspects of retrieval.
    """

    def __init__(
        self,
        embedding_model: Any = None,
        random_seed: int = 42,
        complexity: str = "medium",  # "simple", "medium", "complex"
    ):
        """
        Initialize the semantic data generator.

        Args:
            embedding_model: Model to use for generating embeddings
            random_seed: Seed for random number generation
            complexity: Complexity level of generated data
        """
        self.random_seed = random_seed
        self.complexity = complexity
        random.seed(random_seed)
        np.random.seed(random_seed)

        # set up embedding model
        self.embedding_model = embedding_model
        if self.embedding_model is None and HAS_SENTENCE_TRANSFORMERS:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize storage
        self.memories: list[SemanticMemory] = []
        self.memory_series: list[MemorySeries] = []
        self.query_templates = self._load_query_templates()

    def _load_query_templates(self) -> dict[str, Any]:
        """Load query templates for different query types."""
        return {
            "factual": [
                "What is {subject}?",
                "Tell me about {subject}.",
                "Describe {subject}.",
                "What do you know about {subject}?",
                "Give me information about {subject}.",
            ],
            "personal": [
                "What is my {attribute}?",
                "Do you know my {attribute}?",
                "What's my {attribute}?",
                "Tell me my {attribute}.",
                "Can you remind me what my {attribute} is?",
            ],
            "temporal": [
                "What happened {timeframe}?",
                "What did I do {timeframe}?",
                "Tell me about what occurred {timeframe}.",
                "What events took place {timeframe}?",
                "What do you remember from {timeframe}?",
            ],
            "causal": [
                "Why did {event} happen?",
                "What caused {event}?",
                "What led to {event}?",
                "What was the reason for {event}?",
                "Explain why {event} occurred.",
            ],
            "comparative": [
                "Which is better, {option1} or {option2}?",
                "Compare {option1} and {option2}.",
                "What's the difference between {option1} and {option2}?",
                "How does {option1} compare to {option2}?",
                "Is {option1} better than {option2}?",
            ],
            "conditional": [
                "What would happen if {condition}?",
                "If {condition}, then what?",
                "What's the outcome if {condition}?",
                "Assuming {condition}, what would follow?",
                "In the case that {condition}, what would occur?",
            ],
            "multi_hop": [
                "Who {relation} the person who {attribute}?",
                "What is the {attribute} of the {entity} that {relation}?",
                "When did the event that caused {outcome} take place?",
                "Where is the {entity} that {relation} the {other_entity}?",
                "Why did the person who {attribute} decide to {action}?",
            ],
        }

    def _generate_timestamp(self, start_date=None, end_date=None) -> datetime:
        """Generate a random timestamp within a given range."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        time_diff = (end_date - start_date).total_seconds()
        random_seconds = random.randint(0, int(time_diff))
        return start_date + timedelta(seconds=random_seconds)

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if not self.embedding_model:
            # Return random similarity if no embedding model
            return random.uniform(0.1, 0.9)

        try:
            # Get embeddings
            embedding1 = self.embedding_model.encode(
                text1,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            embedding2 = self.embedding_model.encode(
                text2,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

            # Compute cosine similarity
            similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
            return float(similarity)
        except Exception:
            # Fall back to random similarity if any error occurs
            return random.uniform(0.1, 0.9)

    def _generate_contradiction(self, original_text: str) -> str:
        """Generate a contradictory version of the given text."""
        # Simple contradiction generation rules
        if "is" in original_text:
            return original_text.replace("is", "is not")
        elif "am" in original_text:
            return original_text.replace("am", "am not")
        elif "are" in original_text:
            return original_text.replace("are", "are not")
        elif "was" in original_text:
            return original_text.replace("was", "was not")
        elif "were" in original_text:
            return original_text.replace("were", "were not")
        elif "will" in original_text:
            return original_text.replace("will", "will not")
        elif "can" in original_text:
            return original_text.replace("can", "cannot")
        elif "has" in original_text:
            return original_text.replace("has", "does not have")
        elif "have" in original_text:
            return original_text.replace("have", "do not have")
        elif "do" in original_text:
            return original_text.replace("do", "do not")
        elif "does" in original_text:
            return original_text.replace("does", "does not")
        elif "did" in original_text:
            return original_text.replace("did", "did not")
        elif "like" in original_text:
            return original_text.replace("like", "dislike")
        elif "love" in original_text:
            return original_text.replace("love", "hate")
        elif "enjoy" in original_text:
            return original_text.replace("enjoy", "dislike")
        elif "good" in original_text:
            return original_text.replace("good", "bad")
        elif "best" in original_text:
            return original_text.replace("best", "worst")
        elif "great" in original_text:
            return original_text.replace("great", "terrible")
        elif "yes" in original_text:
            return original_text.replace("yes", "no")

        # If no simple rule applies, try to negate the whole statement
        if not original_text.startswith("not "):
            return "It is not true that " + original_text

        return "The opposite is true: " + original_text.replace("not ", "")

    def generate_memory_series(self, num_series: int) -> list[MemorySeries]:
        """Generate series of related memories."""
        if not self.memories:
            raise ValueError("No memories generated yet. Call generate_memories first.")

        # Clear existing series
        self.memory_series = []

        # Define potential series themes
        themes = [
            "learning a new skill",
            "planning a trip",
            "working on a project",
            "developing a relationship",
            "solving a problem",
            "making a decision",
            "experiencing a life change",
            "pursuing a goal",
            "overcoming a challenge",
            "building a career",
        ]

        for i in range(min(num_series, len(themes))):
            # Select a theme and create a description
            theme = themes[i]
            description = f"A series of memories related to {theme}."

            # Determine number of memories in this series
            series_size = random.randint(3, min(5, len(self.memories) // 2))

            # Select random memories for this series
            memory_indices = random.sample(range(len(self.memories)), series_size)

            # Create the series
            series = MemorySeries(
                name=f"Series {i + 1}: {theme.title()}",
                description=description,
                memory_indices=memory_indices,
                temporal_order=random.random() > 0.2,  # 80% chance of temporal ordering
            )

            # Add relationships between memories in the series
            for j in range(len(memory_indices) - 1):
                source_idx = memory_indices[j]
                target_idx = memory_indices[j + 1]

                if series.temporal_order:
                    # Update timestamps to ensure temporal order
                    if self.memories[target_idx].timestamp <= self.memories[source_idx].timestamp:
                        new_timestamp = self.memories[source_idx].timestamp + timedelta(
                            days=random.randint(1, 30)
                        )
                        self.memories[target_idx].timestamp = new_timestamp

                # Add relationship
                relationship = MemoryRelationship(
                    type="elaborates" if random.random() > 0.3 else "supports",
                    source_idx=source_idx,
                    target_idx=target_idx,
                    strength=random.uniform(0.7, 1.0),
                )

                self.memories[source_idx].related_memories.append(relationship)

            self.memory_series.append(series)

        return self.memory_series

    def generate_contradictions(self, contradiction_rate: float = 0.1) -> int:
        """Generate contradictory memories for a subset of existing memories."""
        if not self.memories:
            raise ValueError("No memories generated yet. Call generate_memories first.")

        # Determine how many contradictions to generate
        num_contradictions = int(len(self.memories) * contradiction_rate)

        # Track generated contradictions
        contradiction_count = 0

        # Generate contradictions
        for _ in range(num_contradictions):
            # Select a random memory to contradict
            source_idx = random.randint(0, len(self.memories) - 1)
            source_memory = self.memories[source_idx]

            # Generate contradictory text
            contradictory_text = self._generate_contradiction(source_memory.text)

            # Create metadata
            metadata = {
                **source_memory.metadata,
                "type": "contradiction",
                "contradicts": source_idx,
            }

            # Create embedding
            if self.embedding_model:
                embedding = self.embedding_model.encode(contradictory_text)
            else:
                # Generate random embedding with similar dimension
                embedding = np.random.randn(source_memory.embedding.shape[0])
                embedding = embedding / np.linalg.norm(embedding)  # Normalize

            # Create timestamp (usually later than the original)
            timestamp = source_memory.timestamp + timedelta(days=random.randint(1, 60))

            # Create the contradictory memory
            contradiction_memory = SemanticMemory(
                text=contradictory_text,
                metadata=metadata,
                embedding=embedding,
                timestamp=timestamp,
                importance=random.uniform(0.4, 0.8),
            )

            # Add relationship
            relationship = MemoryRelationship(
                type="contradicts",
                source_idx=len(self.memories),  # This will be the index of the new memory
                target_idx=source_idx,
                strength=random.uniform(0.7, 1.0),
            )

            contradiction_memory.related_memories.append(relationship)

            # Add to memories
            self.memories.append(contradiction_memory)
            contradiction_count += 1

        return contradiction_count

    def generate_memories(
        self, num_memories: int = 100, categories: Optional[list[str]] = None
    ) -> list[SemanticMemory]:
        """Generate semantically structured memories."""
        # Define default categories if not provided
        if categories is None:
            categories = [
                "personal_fact",
                "preference",
                "event",
                "opinion",
                "knowledge",
                "goal",
                "reflection",
            ]

        # Clear existing memories
        self.memories = []

        # Generate start and end dates for timestamp range
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()

        # Generate memories
        for i in range(num_memories):
            # Select a category
            category = random.choice(categories)

            # Generate text based on category
            if category == "personal_fact":
                attributes = ["name", "age", "hometown", "occupation", "birthday"]
                attribute = random.choice(attributes)
                values = {
                    "name": ["Alex", "Jordan", "Taylor", "Sam", "Morgan"],
                    "age": ["25", "30", "35", "40", "45"],
                    "hometown": ["Austin", "Seattle", "Chicago", "New York", "San Francisco"],
                    "occupation": [
                        "software engineer",
                        "data scientist",
                        "teacher",
                        "doctor",
                        "designer",
                    ],
                    "birthday": ["January 15", "March 22", "July 8", "October 30", "December 5"],
                }
                value = random.choice(values.get(attribute, ["unknown"]))
                text = f"My {attribute} is {value}."

            elif category == "preference":
                pref_types = ["food", "color", "movie", "book", "music", "hobby"]
                pref_type = random.choice(pref_types)
                pref_values = {
                    "food": ["pizza", "sushi", "pasta", "tacos", "curry"],
                    "color": ["blue", "green", "purple", "red", "yellow"],
                    "movie": [
                        "Inception",
                        "The Matrix",
                        "Star Wars",
                        "The Godfather",
                        "Pulp Fiction",
                    ],
                    "book": [
                        "1984",
                        "To Kill a Mockingbird",
                        "Harry Potter",
                        "Dune",
                        "The Great Gatsby",
                    ],
                    "music": ["rock", "jazz", "pop", "classical", "hip-hop"],
                    "hobby": ["hiking", "reading", "gaming", "cooking", "painting"],
                }
                value = random.choice(pref_values.get(pref_type, ["unknown"]))
                text = f"My favorite {pref_type} is {value}."

            elif category == "event":
                events = [
                    "attended a conference",
                    "went on a vacation",
                    "had a dinner party",
                    "completed a project",
                    "learned a new skill",
                    "met with friends",
                    "visited a museum",
                    "watched a movie",
                    "read a book",
                    "went hiking",
                ]
                locations = ["at home", "in the city", "at work", "in the park", "online"]
                event = random.choice(events)
                location = random.choice(locations)
                text = f"I {event} {location}."

            elif category == "opinion":
                topics = ["AI", "climate change", "remote work", "social media", "technology"]
                opinions = [
                    "is beneficial",
                    "is concerning",
                    "has potential",
                    "is overrated",
                    "is important",
                ]
                topic = random.choice(topics)
                opinion = random.choice(opinions)
                text = f"I think {topic} {opinion}."

            elif category == "knowledge":
                facts = [
                    "Python is a programming language known for its readability.",
                    "The capital of France is Paris.",
                    "Water boils at 100 degrees Celsius at sea level.",
                    "The human body has 206 bones.",
                    "The Earth orbits around the Sun.",
                    "Photosynthesis is how plants make their food.",
                    "The Great Wall of China is visible from space.",
                    "Vincent van Gogh painted 'The Starry Night'.",
                    "The speed of light is approximately 300,000 km/s.",
                    "DNA stands for deoxyribonucleic acid.",
                ]
                text = random.choice(facts)

            elif category == "goal":
                goals = [
                    "learn a new language",
                    "travel to Japan",
                    "start a business",
                    "improve my health",
                    "read more books",
                    "learn to play an instrument",
                    "write a novel",
                    "run a marathon",
                    "save for retirement",
                    "reduce my carbon footprint",
                ]
                timeframes = [
                    "this year",
                    "next month",
                    "eventually",
                    "by next summer",
                    "within 5 years",
                ]
                goal = random.choice(goals)
                timeframe = random.choice(timeframes)
                text = f"My goal is to {goal} {timeframe}."

            elif category == "reflection":
                reflections = [
                    "I've made significant progress in my career.",
                    "I've been spending too much time online lately.",
                    "I need to focus more on my health and wellbeing.",
                    "I'm proud of what I've accomplished this year.",
                    "I should reach out to old friends more often.",
                    "I've been feeling more balanced lately.",
                    "I need to be more patient with myself.",
                    "I've learned a lot from my recent challenges.",
                    "I appreciate the support from my friends and family.",
                    "I should take more time to enjoy the simple things.",
                ]
                text = random.choice(reflections)

            else:
                text = f"This is memory {i} of category {category}."

            # Generate metadata
            metadata = {
                "category": category,
                "content": text,
                "type": category,
                "importance": random.uniform(0.1, 1.0),
            }

            # Generate embedding
            if self.embedding_model:
                embedding = self.embedding_model.encode(text)
            else:
                # Generate random embedding
                embedding = np.random.randn(768)  # Default dimension
                embedding = embedding / np.linalg.norm(embedding)  # Normalize

            # Generate timestamp
            timestamp = self._generate_timestamp(start_date, end_date)

            # Create the memory
            memory = SemanticMemory(
                text=text,
                metadata=metadata,
                embedding=embedding,
                timestamp=timestamp,
                importance=metadata["importance"],
            )

            # Add to memories
            self.memories.append(memory)

        # After creating all memories, add some relationships
        self._add_memory_relationships()

        return self.memories

    def _add_memory_relationships(self, relationship_density: float = 0.1):
        """Add semantic relationships between memories."""
        # Determine number of relationships to add
        num_memories = len(self.memories)
        num_relationships = int(num_memories * relationship_density * num_memories)

        # Add random relationships
        for _ in range(num_relationships):
            source_idx = random.randint(0, num_memories - 1)
            target_idx = random.randint(0, num_memories - 1)

            # Avoid self-relationships
            if source_idx == target_idx:
                continue

            # Generate relationship type
            rel_types = ["supports", "elaborates", "references", "temporal_after"]
            rel_type = random.choice(rel_types)

            # If temporal relationship, ensure timestamps are consistent
            if rel_type == "temporal_after":
                if self.memories[source_idx].timestamp > self.memories[target_idx].timestamp:
                    # Swap indices to maintain temporal order
                    source_idx, target_idx = target_idx, source_idx

            # Create relationship
            relationship = MemoryRelationship(
                type=rel_type,
                source_idx=source_idx,
                target_idx=target_idx,
                strength=random.uniform(0.3, 0.9),
            )

            # Add to source memory
            self.memories[source_idx].related_memories.append(relationship)

    def generate_queries(
        self, num_queries: int = 20, query_types: Optional[list[str]] = None
    ) -> list[dict[str, Any]]:
        """Generate semantically meaningful queries with known relevant memories."""
        if not self.memories:
            raise ValueError("No memories generated yet. Call generate_memories first.")

        # Define default query types if not provided
        if query_types is None:
            query_types = list(self.query_templates.keys())

        # Initialize result list
        queries = []

        # Track used memory indices to ensure diversity
        used_memory_indices = set()

        # Generate queries
        for i in range(num_queries):
            # Select query type
            query_type = random.choice(query_types)

            # Select template
            template = random.choice(self.query_templates[query_type])

            # Select relevant memories based on query type
            relevant_indices = self._select_relevant_memories(query_type, used_memory_indices)

            # If no relevant memories found, try another query type
            if not relevant_indices:
                continue

            # Add to used indices
            used_memory_indices.update(relevant_indices)

            # Generate query text and fill template
            query_text, template_data = self._fill_query_template(
                template, query_type, relevant_indices
            )

            # Generate embedding
            if self.embedding_model:
                embedding = self.embedding_model.encode(query_text)
            else:
                # Generate random embedding
                embedding = np.random.randn(768)  # Default dimension
                embedding = embedding / np.linalg.norm(embedding)  # Normalize

            # Create query object
            query = {
                "query": query_text,
                "type": query_type,
                "relevant_indices": relevant_indices,
                "template_data": template_data,
                "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                "expected_answer": self._generate_expected_answer(query_text, relevant_indices),
            }

            queries.append(query)

        return queries

    def _select_relevant_memories(self, query_type: str, used_indices: set[int]) -> list[int]:
        """Select relevant memories for a query based on type."""
        # Initialize result
        relevant_indices = []

        if query_type == "factual":
            # Find knowledge memories
            knowledge_indices = [
                i
                for i, mem in enumerate(self.memories)
                if mem.metadata.get("category") == "knowledge" and i not in used_indices
            ]

            if knowledge_indices:
                # Select one primary knowledge memory
                primary_idx = random.choice(knowledge_indices)
                relevant_indices.append(primary_idx)

                # Look for related memories
                for relationship in self.memories[primary_idx].related_memories:
                    if relationship.type in ["supports", "elaborates"]:
                        relevant_indices.append(relationship.target_idx)

        elif query_type == "personal":
            # Find personal facts or preferences
            personal_indices = [
                i
                for i, mem in enumerate(self.memories)
                if mem.metadata.get("category") in ["personal_fact", "preference"]
                and i not in used_indices
            ]

            if personal_indices:
                # Select one personal memory
                relevant_indices.append(random.choice(personal_indices))

        elif query_type == "temporal":
            # Find events within a similar timeframe
            if not self.memories:
                return []

            # Select random timeframe
            timeframes = [
                "yesterday",
                "last week",
                "last month",
                "last year",
                "this morning",
                "recently",
                "a few days ago",
            ]
            timeframe = random.choice(timeframes)

            # Find memories with appropriate timestamps
            if "yesterday" in timeframe:
                cutoff = datetime.now() - timedelta(days=1)
            elif "week" in timeframe:
                cutoff = datetime.now() - timedelta(days=7)
            elif "month" in timeframe:
                cutoff = datetime.now() - timedelta(days=30)
            elif "year" in timeframe:
                cutoff = datetime.now() - timedelta(days=365)
            elif "morning" in timeframe:
                cutoff = datetime.now() - timedelta(hours=12)
            else:
                cutoff = datetime.now() - timedelta(days=random.randint(3, 10))

            # Find events after cutoff
            event_indices = [
                i
                for i, mem in enumerate(self.memories)
                if mem.timestamp >= cutoff
                and mem.metadata.get("category") == "event"
                and i not in used_indices
            ]

            # Select random events
            num_events = min(len(event_indices), random.randint(1, 3))
            if event_indices and num_events > 0:
                relevant_indices.extend(random.sample(event_indices, num_events))

        elif query_type == "causal":
            # Find events with causal relationships
            for i, memory in enumerate(self.memories):
                if i in used_indices:
                    continue

                # Check if this memory has causal relationships
                causal_relations = [
                    rel
                    for rel in memory.related_memories
                    if rel.type in ["temporal_after"] and rel.strength > 0.5
                ]

                if causal_relations:
                    # Select this memory and one of its causal relations
                    relevant_indices.append(i)
                    relation = random.choice(causal_relations)
                    relevant_indices.append(relation.target_idx)
                    break

        elif query_type == "comparative":
            # Find memories of the same category for comparison
            categories = {}
            for i, memory in enumerate(self.memories):
                if i in used_indices:
                    continue

                category = memory.metadata.get("category")
                if category:
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(i)

            # Find categories with at least 2 memories
            comparable_categories = [
                cat for cat, indices in categories.items() if len(indices) >= 2
            ]

            if comparable_categories:
                category = random.choice(comparable_categories)
                indices = categories[category]
                # Select 2 random memories from this category
                selected = random.sample(indices, 2)
                relevant_indices.extend(selected)

        elif query_type == "conditional":
            # Find memories that could have conditions
            for i, memory in enumerate(self.memories):
                if i in used_indices:
                    continue

                # Events or goals work well for conditional queries
                if memory.metadata.get("category") in ["event", "goal"]:
                    relevant_indices.append(i)
                    break

        elif query_type == "multi_hop":
            # For multi-hop, find a chain of related memories
            # Start with a random memory
            start_indices = [i for i in range(len(self.memories)) if i not in used_indices]

            if start_indices:
                start_idx = random.choice(start_indices)
                chain = [start_idx]

                # Try to build a chain
                current_idx = start_idx
                for _ in range(2):  # Try to get a 3-hop chain
                    relations = self.memories[current_idx].related_memories
                    if relations:
                        # Find an unused relation
                        unused_relations = [
                            rel
                            for rel in relations
                            if rel.target_idx not in chain and rel.target_idx not in used_indices
                        ]
                        if unused_relations:
                            next_rel = random.choice(unused_relations)
                            chain.append(next_rel.target_idx)
                            current_idx = next_rel.target_idx

                if len(chain) >= 2:  # At least 2 hops
                    relevant_indices.extend(chain)

        # If still no relevant memories, select random ones based on importance
        if not relevant_indices:
            available_indices = [i for i in range(len(self.memories)) if i not in used_indices]
            if available_indices:
                # Select memories with higher importance
                weighted_indices = [(i, self.memories[i].importance) for i in available_indices]
                weighted_indices.sort(key=lambda x: x[1], reverse=True)

                # Take top 20% of important memories
                top_indices = [
                    idx for idx, _ in weighted_indices[: max(1, len(weighted_indices) // 5)]
                ]

                # Select 1-3 random memories from top indices
                num_selected = min(len(top_indices), random.randint(1, 3))
                if num_selected > 0:
                    relevant_indices.extend(random.sample(top_indices, num_selected))

        return relevant_indices

    def _fill_query_template(
        self, template: str, query_type: str, relevant_indices: list[int]
    ) -> tuple[str, dict[str, Any]]:
        """Fill a query template with data from relevant memories."""
        template_data = {}

        if query_type == "factual":
            if not relevant_indices:
                return "", {}

            # Get subject from relevant memory
            memory = self.memories[relevant_indices[0]]
            text = memory.text

            # Extract subject
            if ":" in text:
                subject = text.split(":")[0].strip()
            else:
                words = text.split()
                if len(words) > 2:
                    subject = " ".join(words[1:3])
                else:
                    subject = words[0] if words else "this topic"

            template_data["subject"] = subject

        elif query_type == "personal":
            if not relevant_indices:
                return "", {}

            # Get attribute from relevant memory
            memory = self.memories[relevant_indices[0]]
            text = memory.text

            # Extract attribute
            if "My " in text and " is " in text:
                attribute = text.split("My ")[1].split(" is ")[0].strip()
            else:
                attribute = memory.metadata.get("category", "information")

            template_data["attribute"] = attribute

        elif query_type == "temporal":
            timeframes = [
                "yesterday",
                "last week",
                "last month",
                "last year",
                "this morning",
                "recently",
                "a few days ago",
            ]
            template_data["timeframe"] = random.choice(timeframes)

        elif query_type == "causal":
            if not relevant_indices:
                return "", {}

            # Get event from relevant memory
            memory = self.memories[relevant_indices[0]]
            text = memory.text

            # Extract event
            if "I " in text:
                event = text.split("I ")[1].split(".")[0].strip()
            else:
                event = text.split(".")[0].strip()

            template_data["event"] = event

        elif query_type == "comparative":
            if len(relevant_indices) < 2:
                return "", {}

            # Get options from relevant memories
            memory1 = self.memories[relevant_indices[0]]
            memory2 = self.memories[relevant_indices[1]]

            # Extract options
            if memory1.metadata.get("category") == "preference" and "favorite" in memory1.text:
                option1 = memory1.text.split("favorite ")[1].split(" is ")[0].strip()
                if "favorite" in memory2.text:
                    option2 = memory2.text.split("favorite ")[1].split(" is ")[0].strip()
                else:
                    option2 = memory2.text.split(".")[0].strip()
            else:
                option1 = memory1.text.split(".")[0].strip()
                option2 = memory2.text.split(".")[0].strip()

            template_data["option1"] = option1
            template_data["option2"] = option2

        elif query_type == "conditional":
            if not relevant_indices:
                return "", {}

            # Get condition from relevant memory
            memory = self.memories[relevant_indices[0]]
            text = memory.text

            # Extract condition
            if "goal" in memory.metadata.get("category", ""):
                if "My goal is to " in text:
                    condition = text.split("My goal is to ")[1].split(".")[0].strip()
                else:
                    condition = text.split(".")[0].strip()
            elif "I " in text:
                condition = text.split("I ")[1].split(".")[0].strip()
            else:
                condition = text.split(".")[0].strip()

            template_data["condition"] = condition

        elif query_type == "multi_hop":
            if len(relevant_indices) < 2:
                return "", {}

            # Get attributes from relevant memories
            memory1 = self.memories[relevant_indices[0]]
            memory2 = self.memories[relevant_indices[1]]

            # Generate relation terms
            relations = ["knows", "works with", "is friends with", "is related to", "met"]
            attributes = ["likes", "studies", "lives in", "works on", "visited"]
            entities = ["person", "place", "book", "movie", "project"]

            template_data["relation"] = random.choice(relations)
            template_data["attribute"] = random.choice(attributes)
            template_data["entity"] = random.choice(entities)
            template_data["other_entity"] = random.choice(entities)
            template_data["action"] = random.choice(["go", "stay", "leave", "return"])

            # Extract specific attribute if available
            if "My " in memory1.text and " is " in memory1.text:
                attribute = memory1.text.split("My ")[1].split(" is ")[0].strip()
                template_data["attribute"] = attribute

            if "I " in memory2.text:
                action = memory2.text.split("I ")[1].split(".")[0].strip()
                template_data["action"] = action

        # Fill template with data
        filled_template = template
        for key, value in template_data.items():
            placeholder = "{" + key + "}"
            if placeholder in filled_template:
                filled_template = filled_template.replace(placeholder, value)

        return filled_template, template_data

    def _generate_expected_answer(self, query: str, relevant_indices: list[int]) -> str:
        """Generate expected answer for a query based on relevant memories."""
        if not relevant_indices:
            return "No relevant information available."

        # Combine information from relevant memories
        memory_texts = [self.memories[idx].text for idx in relevant_indices]

        # Generate simple answer based on query type
        if "What is" in query or "Tell me about" in query:
            if len(memory_texts) == 1:
                return memory_texts[0]
            else:
                return " ".join(memory_texts)

        elif "Do you know" in query or "What's my" in query:
            for text in memory_texts:
                if "My " in text and " is " in text:
                    attribute = text.split("My ")[1].split(" is ")[0].strip()
                    value = text.split(" is ")[1].strip(".")
                    if attribute in query:
                        return f"Your {attribute} is {value}."
            return "I don't have that information."

        elif "When" in query or "What happened" in query:
            event_texts = [text for text in memory_texts if "I " in text]
            if event_texts:
                return " ".join(event_texts)
            return "No events found for that timeframe."

        elif "Why" in query or "What caused" in query:
            return memory_texts[0] if memory_texts else "I don't know the cause."

        elif "Which is better" in query or "Compare" in query:
            if len(memory_texts) >= 2:
                return f"Based on your memories: {memory_texts[0]} However, {memory_texts[1]}"
            return "I don't have enough information to compare."

        elif "what would happen" in query.lower() or "if" in query.lower():
            return (
                "Based on your past experiences: " + memory_texts[0]
                if memory_texts
                else "I can't predict that outcome."
            )

        # Default answer
        return " ".join(memory_texts)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save generated data to a file."""
        if not self.memories:
            raise ValueError("No data to save. Generate memories first.")

        # Prepare data for serialization
        serializable_data = {
            "memories": [
                {
                    "text": memory.text,
                    "metadata": memory.metadata,
                    "embedding": memory.embedding.tolist()
                    if isinstance(memory.embedding, np.ndarray)
                    else memory.embedding,
                    "timestamp": memory.timestamp.isoformat(),
                    "importance": memory.importance,
                    "related_memories": [
                        {
                            "type": rel.type,
                            "source_idx": rel.source_idx,
                            "target_idx": rel.target_idx,
                            "strength": rel.strength,
                        }
                        for rel in memory.related_memories
                    ],
                }
                for memory in self.memories
            ],
            "series": [
                {
                    "name": series.name,
                    "description": series.description,
                    "memory_indices": series.memory_indices,
                    "temporal_order": series.temporal_order,
                }
                for series in self.memory_series
            ],
        }

        # Add queries if available
        queries = self.generate_queries()
        if queries:
            serializable_data["queries"] = queries

        # Write to file
        with open(file_path, "w") as f:
            json.dump(serializable_data, f, indent=2)

    @classmethod
    def load_from_file(
        cls, file_path: Union[str, Path], embedding_model=None
    ) -> "SemanticDataGenerator":
        """Load data from a file into a new generator instance."""
        with open(file_path) as f:
            data = json.load(f)

        # Create new instance
        generator = cls(embedding_model=embedding_model)

        # Load memories
        generator.memories = []
        for mem_data in data.get("memories", []):
            embedding = np.array(mem_data["embedding"])
            timestamp = datetime.fromisoformat(mem_data["timestamp"])

            memory = SemanticMemory(
                text=mem_data["text"],
                metadata=mem_data["metadata"],
                embedding=embedding,
                timestamp=timestamp,
                importance=mem_data.get("importance", 0.5),
            )

            # Add relationships later (need all memories loaded first)
            generator.memories.append(memory)

        # Add relationships
        for i, mem_data in enumerate(data.get("memories", [])):
            for rel_data in mem_data.get("related_memories", []):
                relationship = MemoryRelationship(
                    type=rel_data["type"],
                    source_idx=rel_data["source_idx"],
                    target_idx=rel_data["target_idx"],
                    strength=rel_data.get("strength", 1.0),
                )
                generator.memories[i].related_memories.append(relationship)

        # Load series
        generator.memory_series = []
        for series_data in data.get("series", []):
            series = MemorySeries(
                name=series_data["name"],
                description=series_data["description"],
                memory_indices=series_data["memory_indices"],
                temporal_order=series_data.get("temporal_order", True),
            )
            generator.memory_series.append(series)

        return generator


def generate_semantic_dataset(
    output_path: Union[str, Path],
    num_memories: int = 100,
    num_queries: int = 20,
    num_series: int = 5,
    contradiction_rate: float = 0.1,
    complexity: str = "medium",
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Generate a complete semantic dataset for evaluation.

    Args:
        output_path: Path to save the dataset
        num_memories: Number of base memories to generate
        num_queries: Number of queries to generate
        num_series: Number of memory series to generate
        contradiction_rate: Rate of contradictory memories to generate
        complexity: Complexity level of the dataset ("simple", "medium", "complex")
        random_seed: Random seed for reproducibility

    Returns:
        dictionary containing the generated dataset
    """
    # Create generator
    generator = SemanticDataGenerator(random_seed=random_seed, complexity=complexity)

    # Generate base memories
    generator.generate_memories(num_memories=num_memories)

    # Generate memory series
    generator.generate_memory_series(num_series=num_series)

    # Generate contradictions
    generator.generate_contradictions(contradiction_rate=contradiction_rate)

    # Generate queries
    queries = generator.generate_queries(num_queries=num_queries)

    # Save to file
    generator.save_to_file(output_path)

    # Return dataset
    return {"memories": generator.memories, "series": generator.memory_series, "queries": queries}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate semantic dataset for MemoryWeave evaluation"
    )
    parser.add_argument(
        "--output", type=str, default="datasets/semantic_dataset.json", help="Output file path"
    )
    parser.add_argument("--memories", type=int, default=100, help="Number of memories to generate")
    parser.add_argument("--queries", type=int, default=20, help="Number of queries to generate")
    parser.add_argument("--series", type=int, default=5, help="Number of memory series to generate")
    parser.add_argument(
        "--contradictions", type=float, default=0.1, help="Rate of contradictory memories"
    )
    parser.add_argument(
        "--complexity",
        type=str,
        default="medium",
        help="Dataset complexity (simple, medium, complex)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print(
        f"Generating semantic dataset with {args.memories} memories and {args.queries} queries..."
    )

    dataset = generate_semantic_dataset(
        output_path=args.output,
        num_memories=args.memories,
        num_queries=args.queries,
        num_series=args.series,
        contradiction_rate=args.contradictions,
        complexity=args.complexity,
        random_seed=args.seed,
    )

    print(
        f"Generated {len(dataset['memories'])} memories, {len(dataset['series'])} series, and {len(dataset['queries'])} queries."
    )
    print(f"Dataset saved to {args.output}")
