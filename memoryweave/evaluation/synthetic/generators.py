# memoryweave/evaluation/synthetic/generators.py
"""
Synthetic test data generators for MemoryWeave evaluation.

This module provides classes for generating synthetic memories and queries
for testing and benchmarking. This helps provide unbiased, comprehensive
evaluation data that can test diverse memory retrieval scenarios.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class MockEmbeddingModel:
    """Mock embedding model for use when sentence_transformers is not available."""

    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim
        self.call_count = 0

    def encode(self, text, batch_size=32, **kwargs):
        """Create a deterministic but unique embedding for any text."""
        self.call_count += 1
        if isinstance(text, list):
            return np.array([self._encode_single(t) for t in text])
        return self._encode_single(text)

    def _encode_single(self, text):
        """Create a single embedding."""
        # Use hash for deterministic but unique embeddings
        hash_val = hash(text) % 1000000
        np.random.seed(hash_val)
        embedding = np.random.randn(self.embedding_dim)
        return embedding / np.linalg.norm(embedding)  # Normalize


@dataclass
class MemoryCategory:
    """Represents a category of memories with related attributes."""

    name: str
    attributes: list[str]
    templates: list[str]
    related_categories: list[str] = field(default_factory=list)


@dataclass
class QueryTemplate:
    """Template for generating synthetic queries."""

    template: str
    category: str
    expected_memory_types: list[str]
    complexity: str = "simple"  # simple, moderate, complex

    def generate(self, data: dict[str, Any]) -> tuple[str, list[str]]:
        """
        Generate a query from this template.

        Args:
            data: dictionary of data to fill template slots

        Returns:
            tuple of (query_text, list_of_relevant_memory_types)
        """
        # Format the template with the provided data
        query = self.template.format(**data)
        return query, self.expected_memory_types


class SyntheticMemoryGenerator:
    """Generates synthetic memories for testing and benchmarking."""

    # Default categories for memory generation
    DEFAULT_CATEGORIES = [
        MemoryCategory(
            name="personal_attributes",
            attributes=["name", "age", "hometown", "occupation", "education"],
            templates=[
                "My name is {name}.",
                "I am {age} years old.",
                "I'm from {hometown}.",
                "I work as a {occupation}.",
                "I studied {education}.",
            ],
        ),
        MemoryCategory(
            name="preferences",
            attributes=[
                "favorite_food",
                "favorite_color",
                "favorite_movie",
                "favorite_book",
                "favorite_music",
            ],
            templates=[
                "My favorite food is {favorite_food}.",
                "My favorite color is {favorite_color}.",
                "My favorite movie is {favorite_movie}.",
                "My favorite book is {favorite_book}.",
                "I enjoy listening to {favorite_music}.",
            ],
            related_categories=["personal_attributes"],
        ),
        MemoryCategory(
            name="relationships",
            attributes=["family_members", "friends", "colleagues", "pets"],
            templates=[
                "My {relation_type} is named {name}.",
                "I have a {pet_type} named {pet_name}.",
                "My friend {friend_name} and I enjoy {activity} together.",
                "I work with {colleague_name} on {project}.",
            ],
            related_categories=["personal_attributes"],
        ),
        MemoryCategory(
            name="events",
            attributes=["event_type", "location", "date", "people_involved"],
            templates=[
                "I {event_action} at {location} on {date}.",
                "I remember when we went to {location} for {occasion}.",
                "I attended {event_name} with {people_involved}.",
                "Last {timeframe}, I {event_action}.",
            ],
            related_categories=["relationships"],
        ),
        MemoryCategory(
            name="factual_knowledge",
            attributes=["topic", "fact", "source", "date_learned"],
            templates=[
                "{topic}: {fact}",
                "I learned that {fact} from {source}.",
                "An interesting fact about {topic} is that {fact}.",
                "According to {source}, {fact}.",
            ],
        ),
        MemoryCategory(
            name="opinions",
            attributes=["topic", "opinion", "strength", "reasoning"],
            templates=[
                "I {strength} believe that {topic} is {opinion}.",
                "My view on {topic} is that {opinion} because {reasoning}.",
                "I think {topic} is {opinion}.",
                "In my opinion, {opinion} regarding {topic}.",
            ],
            related_categories=["factual_knowledge"],
        ),
        MemoryCategory(
            name="skills",
            attributes=["skill_name", "proficiency", "experience", "context"],
            templates=[
                "I am {proficiency} at {skill_name}.",
                "I have {experience} years of experience with {skill_name}.",
                "I learned {skill_name} while {context}.",
                "My {skill_name} skills are {proficiency}.",
            ],
        ),
        MemoryCategory(
            name="goals",
            attributes=["goal", "timeframe", "motivation", "progress"],
            templates=[
                "My goal is to {goal} within {timeframe}.",
                "I want to {goal} because {motivation}.",
                "I'm working towards {goal} and have made {progress} progress.",
                "One of my aspirations is to {goal}.",
            ],
        ),
    ]

    def __init__(
        self,
        categories: Optional[list[MemoryCategory]] = None,
        embedding_model: Any = None,
        embedding_dim: int = 768,
        random_seed: int = 42,
    ):
        """
        Initialize the memory generator.

        Args:
            categories: list of memory categories to use (defaults to DEFAULT_CATEGORIES)
            embedding_model: Model to use for generating embeddings
            embedding_dim: Dimension of embeddings if using mock model
            random_seed: Seed for random number generation
        """
        self.categories = categories or self.DEFAULT_CATEGORIES
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # set up embedding model
        if embedding_model is None:
            if HAS_SENTENCE_TRANSFORMERS:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            else:
                self.embedding_model = MockEmbeddingModel(embedding_dim=embedding_dim)
        else:
            self.embedding_model = embedding_model

    def generate_memory_value(self, attribute: str) -> str:
        """Generate a realistic value for a memory attribute."""
        # dictionary of example values for different attributes
        attribute_values = {
            "name": ["Alex", "Taylor", "Jordan", "Casey", "Morgan", "Jamie", "Riley"],
            "age": ["25", "30", "35", "40", "45"],
            "hometown": ["Austin", "Seattle", "Chicago", "Boston", "San Francisco"],
            "occupation": [
                "Software Engineer",
                "Data Scientist",
                "Product Manager",
                "Designer",
                "Marketing Specialist",
            ],
            "education": ["Computer Science", "Engineering", "Business", "Psychology", "Biology"],
            "favorite_food": ["pizza", "sushi", "tacos", "pasta", "burgers"],
            "favorite_color": ["blue", "green", "purple", "red", "yellow"],
            "favorite_movie": [
                "Inception",
                "The Matrix",
                "Star Wars",
                "The Godfather",
                "Pulp Fiction",
            ],
            "favorite_book": [
                "1984",
                "To Kill a Mockingbird",
                "Harry Potter",
                "Dune",
                "The Great Gatsby",
            ],
            "favorite_music": ["rock", "pop", "jazz", "hip hop", "classical"],
            "relation_type": ["brother", "sister", "mother", "father", "cousin"],
            "pet_type": ["dog", "cat", "bird", "fish", "hamster"],
            "pet_name": ["Buddy", "Luna", "Max", "Bella", "Charlie"],
            "friend_name": ["Sam", "Alex", "Chris", "Pat", "Taylor"],
            "colleague_name": ["Jamie", "Robin", "Morgan", "Casey", "Jordan"],
            "activity": ["hiking", "movies", "gaming", "cooking", "traveling"],
            "project": [
                "data analysis",
                "website redesign",
                "app development",
                "marketing campaign",
            ],
            "event_action": [
                "went hiking",
                "attended a conference",
                "had a dinner party",
                "visited a museum",
            ],
            "location": ["New York", "Paris", "Tokyo", "London", "San Francisco"],
            "date": ["January 15", "last summer", "two weeks ago", "yesterday", "last year"],
            "event_name": ["Tech Conference", "Wedding", "Birthday Party", "Concert", "Graduation"],
            "people_involved": ["friends", "family", "colleagues", "classmates"],
            "occasion": ["a birthday", "a wedding", "vacation", "a business trip"],
            "timeframe": ["week", "month", "year", "summer", "winter"],
            "topic": ["AI", "climate change", "space exploration", "history", "technology"],
            "fact": [
                "is advancing rapidly",
                "has significant impacts",
                "began in the 20th century",
                "is transforming society",
                "presents both opportunities and challenges",
            ],
            "source": [
                "a research paper",
                "a news article",
                "a documentary",
                "a book",
                "an expert",
            ],
            "date_learned": ["recently", "last year", "in college", "a few months ago"],
            "opinion": ["beneficial", "concerning", "fascinating", "overrated", "important"],
            "strength": ["strongly", "somewhat", "partly", "completely"],
            "reasoning": [
                "it has proven benefits",
                "evidence suggests",
                "based on experience",
                "multiple sources confirm",
                "experts agree",
            ],
            "skill_name": [
                "Python programming",
                "data analysis",
                "public speaking",
                "project management",
                "graphic design",
            ],
            "proficiency": ["beginner", "intermediate", "advanced", "expert"],
            "experience": ["1-2", "3-5", "5-10", "10+"],
            "context": ["working on a project", "taking a course", "on the job", "as a hobby"],
            "goal": [
                "learn a new language",
                "travel to Japan",
                "start a business",
                "improve my health",
                "finish my degree",
            ],
            "motivation": [
                "personal growth",
                "career advancement",
                "it's been a dream",
                "to challenge myself",
                "to help others",
            ],
            "progress": ["significant", "some", "limited", "substantial"],
        }

        # Return a random value from the attribute values, or a generic one if not found
        if attribute in attribute_values:
            return random.choice(attribute_values[attribute])
        return f"example {attribute}"

    def generate_memory_data(
        self, num_values_per_category: int = 3
    ) -> dict[str, dict[str, list[str]]]:
        """
        Generate a dataset of memory values for each attribute in each category.

        Args:
            num_values_per_category: Number of values to generate for each attribute

        Returns:
            dictionary of category -> attribute -> list of values
        """
        data = {}

        for category in self.categories:
            category_data = {}
            for attribute in category.attributes:
                values = [
                    self.generate_memory_value(attribute) for _ in range(num_values_per_category)
                ]
                category_data[attribute] = values
            data[category.name] = category_data

        return data

    def generate_memories(
        self,
        num_memories: int = 100,
        memory_data: Optional[dict[str, dict[str, list[str]]]] = None,
        unique_memories: bool = True,
    ) -> list[tuple[str, dict[str, Any], np.ndarray]]:
        """
        Generate synthetic memories.

        Args:
            num_memories: Number of memories to generate
            memory_data: Pre-generated memory data (will be generated if None)
            unique_memories: Whether to ensure all memories are unique

        Returns:
            list of (memory_text, metadata, embedding) tuples
        """
        if memory_data is None:
            memory_data = self.generate_memory_data()

        memories = []
        generated_texts = set()
        attempts = 0
        max_attempts = num_memories * 3  # Avoid infinite loop

        while len(memories) < num_memories and attempts < max_attempts:
            attempts += 1

            # Select a random category and template
            category = random.choice(self.categories)
            template = random.choice(category.templates)

            # Fill in the template with data
            data = {}
            for slot in self._extract_slots(template):
                # Find all categories that might have this attribute
                potential_categories = [category.name]
                for c in self.categories:
                    if c.name in category.related_categories and slot in c.attributes:
                        potential_categories.append(c.name)

                # Try to find the slot in any of these categories
                found = False
                for cat_name in potential_categories:
                    if cat_name in memory_data and slot in memory_data[cat_name]:
                        data[slot] = random.choice(memory_data[cat_name][slot])
                        found = True
                        break

                # If not found, generate a value
                if not found:
                    data[slot] = self.generate_memory_value(slot)

            # Generate the memory text
            try:
                memory_text = template.format(**data)

                # Skip if we're ensuring uniqueness and this memory already exists
                if unique_memories and memory_text in generated_texts:
                    continue

                # Create metadata
                metadata = {
                    "category": category.name,
                    "content": memory_text,
                    "type": category.name,
                    **data,
                }

                # Generate embedding
                embedding = self.embedding_model.encode(memory_text, show_progress_bar=False)

                # Add to memories
                memories.append((memory_text, metadata, embedding))
                generated_texts.add(memory_text)

            except KeyError:
                # Template had a slot we couldn't fill, try again
                continue

        # Shuffle memories
        random.shuffle(memories)
        return memories[:num_memories]

    def save_memories_to_file(
        self, memories: list[tuple[str, dict[str, Any], np.ndarray]], file_path: Union[str, Path]
    ) -> None:
        """
        Save generated memories to a file.

        Args:
            memories: list of memory tuples
            file_path: Path to save the memories
        """
        # Convert to a serializable format
        serializable_memories = []
        for text, metadata, embedding in memories:
            serializable_memories.append({
                "text": text,
                "metadata": metadata,
                "embedding": embedding.tolist(),
            })

        # Write to file
        with open(file_path, "w") as f:
            json.dump(serializable_memories, f, indent=2)

    def _extract_slots(self, template: str) -> set[str]:
        """Extract slot names from a template string."""
        slots = set()
        start = 0
        while True:
            start = template.find("{", start)
            if start == -1:
                break
            end = template.find("}", start)
            if end == -1:
                break
            slot = template[start + 1 : end]
            slots.add(slot)
            start = end
        return slots


class SyntheticQueryGenerator:
    """Generates synthetic queries for testing and benchmarking."""

    # Default query templates
    DEFAULT_QUERY_TEMPLATES = [
        # Personal attribute queries
        QueryTemplate(
            template="What is my {attribute}?",
            category="personal_attributes",
            expected_memory_types=["personal_attributes"],
        ),
        QueryTemplate(
            template="Tell me about my {attribute}.",
            category="personal_attributes",
            expected_memory_types=["personal_attributes"],
        ),
        QueryTemplate(
            template="What is my {family_relation}'s name?",
            category="relationships",
            expected_memory_types=["relationships"],
        ),
        # Preference queries
        QueryTemplate(
            template="What is my favorite {preference_type}?",
            category="preferences",
            expected_memory_types=["preferences"],
        ),
        QueryTemplate(
            template="Do I like {item}?",
            category="preferences",
            expected_memory_types=["preferences"],
        ),
        # Event queries
        QueryTemplate(
            template="When did I last {activity}?",
            category="events",
            expected_memory_types=["events"],
        ),
        QueryTemplate(
            template="Did I ever visit {location}?",
            category="events",
            expected_memory_types=["events"],
        ),
        QueryTemplate(
            template="What happened when I {activity}?",
            category="events",
            expected_memory_types=["events"],
        ),
        # Factual knowledge queries
        QueryTemplate(
            template="What do I know about {topic}?",
            category="factual_knowledge",
            expected_memory_types=["factual_knowledge"],
        ),
        QueryTemplate(
            template="Tell me about {topic}.",
            category="factual_knowledge",
            expected_memory_types=["factual_knowledge"],
        ),
        # Opinion queries
        QueryTemplate(
            template="What do I think about {topic}?",
            category="opinions",
            expected_memory_types=["opinions"],
        ),
        QueryTemplate(
            template="How do I feel about {topic}?",
            category="opinions",
            expected_memory_types=["opinions"],
        ),
        # Skill queries
        QueryTemplate(
            template="How skilled am I at {skill}?",
            category="skills",
            expected_memory_types=["skills"],
        ),
        QueryTemplate(
            template="What skills do I have related to {domain}?",
            category="skills",
            expected_memory_types=["skills"],
        ),
        # Goal queries
        QueryTemplate(
            template="What are my goals for {timeframe}?",
            category="goals",
            expected_memory_types=["goals"],
        ),
        QueryTemplate(
            template="What am I working towards?", category="goals", expected_memory_types=["goals"]
        ),
        # Multi-category queries (complex)
        QueryTemplate(
            template="Tell me about my interest in {topic} and any goals related to it.",
            category="mixed",
            expected_memory_types=["preferences", "goals", "opinions"],
            complexity="complex",
        ),
        QueryTemplate(
            template="What do I know about {topic} and what's my opinion on it?",
            category="mixed",
            expected_memory_types=["factual_knowledge", "opinions"],
            complexity="complex",
        ),
        QueryTemplate(
            template="What skills do I have that relate to my work as a {occupation}?",
            category="mixed",
            expected_memory_types=["skills", "personal_attributes"],
            complexity="complex",
        ),
        QueryTemplate(
            template="What events have I attended with my {relation_type} {name}?",
            category="mixed",
            expected_memory_types=["events", "relationships"],
            complexity="complex",
        ),
    ]

    def __init__(
        self,
        templates: Optional[list[QueryTemplate]] = None,
        embedding_model: Any = None,
        embedding_dim: int = 768,
        random_seed: int = 42,
    ):
        """
        Initialize the query generator.

        Args:
            templates: Query templates to use (defaults to DEFAULT_QUERY_TEMPLATES)
            embedding_model: Model to use for generating embeddings
            embedding_dim: Dimension of embeddings if using mock model
            random_seed: Seed for random number generation
        """
        self.templates = templates or self.DEFAULT_QUERY_TEMPLATES
        self.random_seed = random_seed
        random.seed(random_seed)

        # set up embedding model
        if embedding_model is None:
            if HAS_SENTENCE_TRANSFORMERS:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            else:
                self.embedding_model = MockEmbeddingModel(embedding_dim=embedding_dim)
        else:
            self.embedding_model = embedding_model

    def generate_query_data(
        self, memories: list[tuple[str, dict[str, Any], np.ndarray]]
    ) -> dict[str, dict[str, list[str]]]:
        """
        Extract data from memories that can be used for generating queries.

        Args:
            memories: list of memory tuples (text, metadata, embedding)

        Returns:
            dictionary of category -> attribute -> list of values
        """
        data = {}

        # Organize memory metadata by category
        for _, metadata, _ in memories:
            category = metadata.get("category", "unknown")
            if category not in data:
                data[category] = {}

            # Add each metadata field to the appropriate category
            for key, value in metadata.items():
                if key not in ["category", "content", "type"]:
                    if key not in data[category]:
                        data[category][key] = []
                    if value not in data[category][key]:
                        data[category][key].append(value)

        # Additional derived fields that might be useful for queries
        if "factual_knowledge" in data:
            if "topic" in data["factual_knowledge"]:
                # Ensure we have these fields for queries
                if "domain" not in data.get("skills", {}):
                    if "skills" not in data:
                        data["skills"] = {}
                    data["skills"]["domain"] = data["factual_knowledge"]["topic"]

        # Add some generics
        data["generic"] = {
            "family_relation": ["mother", "father", "brother", "sister", "cousin"],
            "preference_type": ["food", "color", "movie", "book", "music"],
            "activity": ["traveling", "hiking", "reading", "cooking", "gaming"],
            "item": ["coffee", "tea", "chocolate", "spicy food", "classical music"],
            "topic": ["technology", "science", "history", "art", "politics"],
            "timeframe": ["this year", "the future", "next month", "this week"],
            "skill": ["programming", "cooking", "writing", "public speaking", "design"],
            "attribute": ["name", "age", "hometown", "job", "education"],
            "location": ["New York", "Paris", "Tokyo", "London", "San Francisco"],
            "relation_type": ["friend", "colleague", "relative", "neighbor"],
        }

        return data

    def generate_queries(
        self,
        memories: list[tuple[str, dict[str, Any], np.ndarray]],
        num_queries: int = 50,
        query_data: Optional[dict[str, dict[str, list[str]]]] = None,
        complexity_distribution: Optional[dict[str, float]] = None,
    ) -> list[tuple[str, list[int], np.ndarray]]:
        """
        Generate synthetic queries with ground truth relevant memory indices.

        Args:
            memories: list of memory tuples (text, metadata, embedding)
            num_queries: Number of queries to generate
            query_data: Pre-generated query data (will be generated if None)
            complexity_distribution: Distribution of query complexity (simple, moderate, complex)
                                    e.g. {"simple": 0.6, "moderate": 0.3, "complex": 0.1}

        Returns:
            list of (query_text, relevant_memory_indices, embedding) tuples
        """
        if query_data is None:
            query_data = self.generate_query_data(memories)

        if complexity_distribution is None:
            complexity_distribution = {"simple": 0.7, "moderate": 0.2, "complex": 0.1}

        # Filter templates by complexity based on the distribution
        templates_by_complexity = {}
        for complexity in ["simple", "moderate", "complex"]:
            templates_by_complexity[complexity] = [
                t for t in self.templates if t.complexity == complexity
            ]

            # If no templates of this complexity, use simple ones
            if not templates_by_complexity[complexity]:
                templates_by_complexity[complexity] = [
                    t for t in self.templates if t.complexity == "simple"
                ]

        queries = []
        for _ in range(num_queries):
            # Select complexity based on distribution
            complexity = random.choices(
                list(complexity_distribution.keys()), weights=list(complexity_distribution.values())
            )[0]

            # Select template
            template = random.choice(templates_by_complexity[complexity])

            # Try to generate a query using this template
            query_text, relevant_indices = self._generate_query_from_template(
                template, memories, query_data
            )

            if query_text and relevant_indices:
                # Generate embedding
                embedding = self.embedding_model.encode(query_text, show_progress_bar=False)
                queries.append((query_text, relevant_indices, embedding))

        return queries[:num_queries]

    def _generate_query_from_template(
        self,
        template: QueryTemplate,
        memories: list[tuple[str, dict[str, Any], np.ndarray]],
        query_data: dict[str, dict[str, list[str]]],
    ) -> tuple[str, list[int]]:
        """
        Generate a query from a template and find relevant memories.

        Args:
            template: Query template
            memories: list of memory tuples
            query_data: Data for filling templates

        Returns:
            tuple of (query_text, list_of_relevant_memory_indices)
        """
        # Extract slots from template
        slots = self._extract_slots(template.template)

        # Get data for this template's category
        category_data = query_data.get(template.category, {})

        # If this is a "mixed" category, combine data from expected memory types
        if template.category == "mixed":
            for memory_type in template.expected_memory_types:
                if memory_type in query_data:
                    for key, values in query_data[memory_type].items():
                        if key not in category_data:
                            category_data[key] = []
                        category_data[key].extend(values)

        # Fill slots
        slot_values = {}
        for slot in slots:
            # Try to get from category data first
            if slot in category_data and category_data[slot]:
                slot_values[slot] = random.choice(category_data[slot])
            # Try generic data next
            elif slot in query_data.get("generic", {}):
                slot_values[slot] = random.choice(query_data["generic"][slot])
            # If still not found, try any category
            else:
                for cat, attrs in query_data.items():
                    if slot in attrs and attrs[slot]:
                        slot_values[slot] = random.choice(attrs[slot])
                        break

            # If still not filled, use a placeholder
            if slot not in slot_values:
                return None, []  # Skip this template

        # Generate query text
        try:
            query_text = template.template.format(**slot_values)
        except KeyError:
            return None, []  # Skip this template

        # Find relevant memories
        relevant_indices = []
        for i, (_, metadata, _) in enumerate(memories):
            # Check if memory type matches expected types
            memory_type = metadata.get("category", "unknown")
            if memory_type in template.expected_memory_types:
                # Check if specific slot values are found in this memory
                for slot, value in slot_values.items():
                    if slot in metadata and metadata[slot] == value:
                        relevant_indices.append(i)
                        break

                # For some templates, also match on content
                if not relevant_indices and memory_type in template.expected_memory_types:
                    content = metadata.get("content", "").lower()
                    for value in slot_values.values():
                        if isinstance(value, str) and value.lower() in content:
                            relevant_indices.append(i)
                            break

        # If no relevant memories found, this query isn't useful
        if not relevant_indices:
            return None, []

        return query_text, relevant_indices

    def save_queries_to_file(
        self, queries: list[tuple[str, list[int], np.ndarray]], file_path: Union[str, Path]
    ) -> None:
        """
        Save generated queries to a file.

        Args:
            queries: list of query tuples
            file_path: Path to save the queries
        """
        serializable_queries = []
        for text, indices, embedding in queries:
            serializable_queries.append({
                "query": text,
                "relevant_indices": indices,
                "embedding": embedding.tolist(),
            })

        with open(file_path, "w") as f:
            json.dump(serializable_queries, f, indent=2)

    def generate_evaluation_dataset(
        self,
        memories: list[tuple[str, dict[str, Any], np.ndarray]],
        num_queries: int = 50,
        file_path: Union[str, Path] = None,
    ) -> dict[str, Any]:
        """
        Generate a complete evaluation dataset with memories and queries.

        Args:
            memories: list of memory tuples
            num_queries: Number of queries to generate
            file_path: Path to save the dataset (optional)

        Returns:
            dictionary with memories and queries
        """
        # Generate queries
        queries = self.generate_queries(memories, num_queries)

        # Format as an evaluation dataset
        dataset = {
            "memories": [
                {"text": text, "metadata": metadata, "embedding": embedding.tolist()}
                for text, metadata, embedding in memories
            ],
            "queries": [
                {"query": text, "relevant_indices": indices, "embedding": embedding.tolist()}
                for text, indices, embedding in queries
            ],
        }

        # Save to file if specified
        if file_path:
            with open(file_path, "w") as f:
                json.dump(dataset, f, indent=2)

        return dataset

    def _extract_slots(self, template: str) -> set[str]:
        """Extract slot names from a template string."""
        slots = set()
        start = 0
        while True:
            start = template.find("{", start)
            if start == -1:
                break
            end = template.find("}", start)
            if end == -1:
                break
            slot = template[start + 1 : end]
            slots.add(slot)
            start = end
        return slots
