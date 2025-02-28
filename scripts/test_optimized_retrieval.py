"""
Test optimized retrieval strategies for better scaling performance.

This script implements and evaluates optimized retrieval strategies to maintain
high F1 scores even with large memory sizes.
"""

import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from memoryweave.core import ContextualMemory, ContextualRetriever, MemoryEncoder

# Create output directory
os.makedirs("test_output/optimized", exist_ok=True)

# Global spaCy model for reuse
global_nlp = None


# Helper class for sentence embedding
class EmbeddingModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = {}  # Simple cache for embeddings

    def encode(self, text):
        # Check cache first
        if text in self.cache:
            return self.cache[text]

        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling
        attention_mask = inputs["attention_mask"]
        embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts

        # Cache the result
        embedding = mean_pooled.numpy()[0]
        self.cache[text] = embedding
        return embedding


class OptimizedContextualRetriever(ContextualRetriever):
    """
    Optimized version of ContextualRetriever with improved scaling performance.
    """

    def __init__(self, memory, embedding_model, **kwargs):
        """Initialize with parent class constructor."""
        # Extract our custom parameters before passing to parent
        self.use_clustering = kwargs.pop("use_clustering", False)
        self.cluster_count = kwargs.pop("cluster_count", 10)
        self.use_prefiltering = kwargs.pop("use_prefiltering", False)
        self.prefilter_method = kwargs.pop("prefilter_method", "keyword")

        # Initialize parent class
        super().__init__(memory, embedding_model, **kwargs)

        # Initialize clustering variables
        self.clusters = None
        self.cluster_centers = None
        self.memory_to_cluster = {}

        # Use global spaCy model if available
        global global_nlp
        self.nlp = global_nlp
        if self.use_prefiltering and self.prefilter_method == "keyword" and self.nlp is None:
            try:
                import spacy

                if global_nlp is None:
                    print("Loading spaCy model for keyword extraction (first time)")
                    global_nlp = spacy.load("en_core_web_sm")
                self.nlp = global_nlp
            except:
                print("Could not load spaCy model, falling back to basic keyword extraction")

    def build_clusters(self):
        """Build clusters for faster retrieval with large memory sets."""
        if not self.use_clustering or len(self.memory.memory_embeddings) < 100:
            return

        print("Building memory clusters for optimized retrieval...")

        # Use k-means clustering
        from sklearn.cluster import KMeans

        # Determine number of clusters based on memory size
        n_clusters = min(self.cluster_count, len(self.memory.memory_embeddings) // 10)
        n_clusters = max(5, n_clusters)  # At least 5 clusters

        # Fit k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.memory_to_cluster = kmeans.fit_predict(self.memory.memory_embeddings)
        self.cluster_centers = kmeans.cluster_centers_

        # Create cluster lookup
        self.clusters = {}
        for i, cluster_id in enumerate(self.memory_to_cluster):
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(i)

        print(f"Created {n_clusters} memory clusters")

    def _prefilter_memories(self, query: str, query_embedding: np.ndarray) -> List[int]:
        """Prefilter memories to reduce the search space."""
        if not self.use_prefiltering:
            return list(range(len(self.memory.memory_embeddings)))

        if self.prefilter_method == "cluster" and self.clusters is not None:
            # Find closest clusters
            cluster_similarities = np.dot(self.cluster_centers, query_embedding)
            top_clusters = np.argsort(-cluster_similarities)[:3]  # Get top 3 clusters

            # Collect memory indices from top clusters
            candidate_indices = []
            for cluster_id in top_clusters:
                candidate_indices.extend(self.clusters.get(cluster_id, []))

            return candidate_indices

        elif self.prefilter_method == "keyword":
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            if not keywords:
                return list(range(len(self.memory.memory_embeddings)))

            # Find memories containing these keywords
            candidate_indices = []
            for i, metadata in enumerate(self.memory.memory_metadata):
                memory_text = ""
                for field in ["text", "content", "description"]:
                    if field in metadata:
                        memory_text += " " + str(metadata[field]).lower()

                # Check if any keyword is in the memory text
                if any(keyword in memory_text for keyword in keywords):
                    candidate_indices.append(i)

            # If too few candidates, return all memories
            if len(candidate_indices) < 10:
                return list(range(len(self.memory.memory_embeddings)))

            return candidate_indices

        # Default: return all memories
        return list(range(len(self.memory.memory_embeddings)))

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract important keywords from text."""
        if self.nlp:
            # Use spaCy for better keyword extraction
            doc = self.nlp(text)
            keywords = set()

            # Extract named entities
            for ent in doc.ents:
                keywords.add(ent.text.lower())

            # Extract noun phrases and important nouns
            for chunk in doc.noun_chunks:
                keywords.add(chunk.text.lower())

            # Extract important verbs
            for token in doc:
                if token.pos_ == "VERB" and token.is_alpha and len(token.text) > 3:
                    keywords.add(token.lemma_.lower())

            # Filter out stop words and short words
            keywords = {k for k in keywords if len(k) > 3 and not k.startswith("the ")}
            return keywords
        else:
            # Fallback to basic keyword extraction
            words = text.lower().split()
            return {
                w
                for w in words
                if len(w) > 3
                and w
                not in ["what", "when", "where", "which", "this", "that", "these", "those", "with"]
            }

    def _improved_two_stage_retrieval(
        self,
        query_embedding: np.ndarray,
        query_type: str,
        important_keywords: Set[str],
        top_k: int,
        params: Dict[str, Any],
        prefiltered_indices: List[int],
    ) -> List[Dict]:
        """
        Improved two-stage retrieval with prefiltering.

        Args:
            query_embedding: Query embedding
            query_type: Type of query (personal, factual, etc.)
            important_keywords: Important keywords from the query
            top_k: Number of results to return
            params: Adjusted parameters for this query type
            prefiltered_indices: Indices of memories to consider

        Returns:
            List of retrieved memories
        """
        # First stage: Retrieve a larger set of candidates with lower threshold
        first_stage_threshold = max(
            0.05, params["confidence_threshold"] * params.get("first_stage_threshold_factor", 0.7)
        )

        # Expand keywords for factual queries if enabled
        expanded_keywords = important_keywords
        if self.enable_keyword_expansion and query_type == "factual":
            expanded_keywords = self._expand_keywords(important_keywords)

        # Get memory embeddings for prefiltered indices
        if prefiltered_indices:
            memory_embeddings = self.memory.memory_embeddings[prefiltered_indices]

            # Calculate similarities for prefiltered memories
            similarities = np.dot(memory_embeddings, query_embedding)

            # Apply threshold
            valid_indices = np.where(similarities >= first_stage_threshold)[0]
            if len(valid_indices) == 0:
                return []

            # Get top candidates
            top_count = min(self.first_stage_k, len(valid_indices))
            # Fix: Ensure top_count is less than the array size for argpartition
            if top_count >= len(valid_indices):
                # If we have fewer valid indices than top_count, just sort them all
                top_indices = np.argsort(-similarities[valid_indices])
            else:
                top_indices = np.argpartition(-similarities[valid_indices], top_count)[:top_count]

            # Map back to original indices
            candidate_indices = [prefiltered_indices[valid_indices[i]] for i in top_indices]

            # Create candidate memories
            candidates = []
            for idx in candidate_indices:
                # Find the index in prefiltered_indices
                prefiltered_idx = prefiltered_indices.index(idx)
                # Get the similarity score
                score = float(similarities[prefiltered_idx])
                metadata = self.memory.memory_metadata[idx]

                # Apply keyword boosting
                boost = 1.0
                if expanded_keywords:
                    boost = self._calculate_keyword_boost(metadata, expanded_keywords)

                candidates.append(
                    {
                        "memory_id": int(idx),
                        "relevance_score": score * boost,
                        "original_score": score,
                        "keyword_boost": boost,
                        **metadata,
                    }
                )
        else:
            # Fallback to original retrieval if no prefiltering
            if self.retrieval_strategy == "similarity":
                candidates = self._retrieve_by_similarity(
                    query_embedding, self.first_stage_k, expanded_keywords, first_stage_threshold
                )
            elif self.retrieval_strategy == "temporal":
                candidates = self._retrieve_by_recency(self.first_stage_k)
            else:  # hybrid approach
                candidates = self._retrieve_hybrid(
                    query_embedding, self.first_stage_k, expanded_keywords, first_stage_threshold
                )

        # Second stage: Re-rank candidates
        # Sort by relevance score
        candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Apply semantic coherence check if enabled
        if self.semantic_coherence_check and len(candidates) > 1:
            # Convert to proper format for memory's _apply_coherence_check
            candidate_tuples = [
                (
                    c["memory_id"],
                    c["relevance_score"],
                    {k: v for k, v in c.items() if k not in ["memory_id", "relevance_score"]},
                )
                for c in candidates
            ]

            coherent_tuples = self.memory._apply_coherence_check(candidate_tuples, query_embedding)

            # Convert back to our format
            coherent_candidates = []
            for memory_id, score, metadata in coherent_tuples:
                coherent_candidates.append(
                    {
                        "memory_id": memory_id,
                        "relevance_score": score,
                        **metadata,
                    }
                )

            candidates = coherent_candidates

        # Apply adaptive k selection if enabled
        if self.adaptive_retrieval and len(candidates) > 1:
            # Use the adjusted adaptive_k_factor
            adaptive_k_factor = params.get("adaptive_k_factor", self.adaptive_k_factor)
            scores = np.array([c["relevance_score"] for c in candidates])
            diffs = np.diff(scores)

            # Find significant drops
            significance_threshold = adaptive_k_factor * scores[0]
            significant_drops = np.where((-diffs) > significance_threshold)[0]

            if len(significant_drops) > 0:
                # Use the first significant drop as the cut point
                cut_idx = significant_drops[0] + 1
                candidates = candidates[:cut_idx]

        # Take top_k from the re-ranked candidates
        return candidates[:top_k]

    def retrieve_for_context(
        self,
        current_input: str,
        conversation_history: Optional[List[Dict]] = None,
        top_k: int = 5,
        confidence_threshold: float = None,
    ) -> List[Dict]:
        """
        Optimized version of retrieve_for_context.

        Args:
            current_input: The current user input
            conversation_history: Recent conversation history
            top_k: Number of memories to retrieve
            confidence_threshold: Minimum similarity score for memory inclusion

        Returns:
            List of relevant memory entries with metadata
        """
        # Build clusters if needed and not already built
        if self.use_clustering and self.clusters is None:
            self.build_clusters()

        # Update conversation state
        self._update_conversation_state(current_input, conversation_history)

        # Encode the query context
        query_context = self._build_query_context(current_input, conversation_history)
        query_embedding = self.embedding_model.encode(query_context)

        # Extract important keywords for direct reference matching
        important_keywords = self.nlp_extractor.extract_important_keywords(current_input)

        # Extract and update personal attributes if present in the input or response
        if conversation_history:
            for turn in conversation_history[-3:]:  # Look at recent turns
                message = turn.get("message", "")
                response = turn.get("response", "")
                extracted_attributes = self.nlp_extractor.extract_personal_attributes(message)
                self._update_personal_attributes(extracted_attributes)

                extracted_attributes = self.nlp_extractor.extract_personal_attributes(response)
                self._update_personal_attributes(extracted_attributes)

        # Also check current input for personal attributes
        extracted_attributes = self.nlp_extractor.extract_personal_attributes(current_input)
        self._update_personal_attributes(extracted_attributes)

        # If no confidence threshold is provided, use the default
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        # Determine query type and adjust parameters accordingly
        query_type, adjusted_params = self._adapt_to_query_type(current_input, confidence_threshold)

        # Prefilter memories to reduce search space
        prefiltered_indices = self._prefilter_memories(current_input, query_embedding)

        # Use improved two-stage retrieval
        if self.use_two_stage_retrieval:
            memories = self._improved_two_stage_retrieval(
                query_embedding,
                query_type,
                important_keywords,
                top_k,
                adjusted_params,
                prefiltered_indices,
            )
        else:
            # Use the original retrieval methods with adjusted parameters
            memories = self._single_stage_retrieval(
                query_embedding, query_type, important_keywords, top_k, adjusted_params
            )

        # Enhance results with personal attributes relevant to the query
        enhanced_memories = self._enhance_with_personal_attributes(memories, current_input)

        # Apply memory decay if enabled
        if self.memory_decay_enabled:
            self._apply_memory_decay()

        # Track retrieval metrics for dynamic threshold adjustment
        if self.dynamic_threshold_adjustment:
            self._track_retrieval_metrics(query_embedding, enhanced_memories)

        # Ensure we have at least min_results_guarantee results
        if len(enhanced_memories) < self.min_results_guarantee:
            # If we don't have enough results, try again with a lower threshold
            if len(memories) < self.min_results_guarantee:
                fallback_threshold = max(0.05, adjusted_params["confidence_threshold"] * 0.5)
                fallback_memories = self._retrieve_by_similarity(
                    query_embedding,
                    self.min_results_guarantee,
                    important_keywords,
                    fallback_threshold,
                )

                # Add any new memories that weren't already retrieved
                existing_ids = {
                    m.get("memory_id")
                    for m in enhanced_memories
                    if isinstance(m.get("memory_id"), int)
                }

                for memory in fallback_memories:
                    if memory["memory_id"] not in existing_ids:
                        enhanced_memories.append(memory)
                        if len(enhanced_memories) >= self.min_results_guarantee:
                            break

        return enhanced_memories


def generate_synthetic_memories(count, categories):
    """Generate synthetic memories for testing."""
    memories = []

    # Templates for different categories
    templates = {
        "personal": [
            "My name is {name}",
            "I live in {city}, {country}",
            "My favorite color is {color}",
            "I work as a {occupation}",
            "I have a {pet_type} named {pet_name}",
            "My hobby is {hobby}",
        ],
        "factual": [
            "The capital of {country} is {city}",
            "{person} was born in {year}",
            "The {animal} is native to {region}",
            "The chemical formula for {compound} is {formula}",
            "{book} was written by {author}",
            "The {landmark} is located in {location}",
        ],
        "technical": [
            "{language} is a programming language used for {application}",
            "The {algorithm} algorithm is used for {purpose}",
            "{framework} is a framework for {technology}",
            "The {component} is a part of {system}",
            "{protocol} is a protocol used in {field}",
            "The {term} refers to {definition}",
        ],
    }

    # Sample data for templates
    data = {
        "name": ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn"],
        "city": ["Seattle", "Tokyo", "Paris", "London", "New York", "Berlin", "Sydney", "Toronto"],
        "country": ["USA", "Japan", "France", "UK", "Canada", "Germany", "Australia", "Italy"],
        "color": ["blue", "red", "green", "purple", "yellow", "orange", "black", "white"],
        "occupation": [
            "engineer",
            "doctor",
            "teacher",
            "artist",
            "scientist",
            "writer",
            "designer",
            "programmer",
        ],
        "pet_type": ["dog", "cat", "bird", "fish", "hamster", "rabbit", "turtle", "ferret"],
        "pet_name": ["Max", "Luna", "Charlie", "Bella", "Oliver", "Lucy", "Leo", "Daisy"],
        "hobby": [
            "hiking",
            "painting",
            "reading",
            "cooking",
            "gardening",
            "photography",
            "music",
            "traveling",
        ],
        "person": [
            "Einstein",
            "Newton",
            "Curie",
            "Darwin",
            "Tesla",
            "Turing",
            "Hawking",
            "Goodall",
        ],
        "year": ["1879", "1643", "1867", "1809", "1856", "1912", "1942", "1934"],
        "animal": [
            "tiger",
            "panda",
            "koala",
            "penguin",
            "eagle",
            "dolphin",
            "elephant",
            "kangaroo",
        ],
        "region": [
            "Asia",
            "Africa",
            "Australia",
            "Antarctica",
            "Europe",
            "North America",
            "South America",
            "Arctic",
        ],
        "compound": [
            "water",
            "carbon dioxide",
            "glucose",
            "methane",
            "ammonia",
            "sodium chloride",
            "oxygen",
            "hydrogen",
        ],
        "formula": ["H2O", "CO2", "C6H12O6", "CH4", "NH3", "NaCl", "O2", "H2"],
        "book": [
            "1984",
            "Moby Dick",
            "Pride and Prejudice",
            "The Great Gatsby",
            "To Kill a Mockingbird",
            "War and Peace",
            "Hamlet",
            "Don Quixote",
        ],
        "author": [
            "Orwell",
            "Melville",
            "Austen",
            "Fitzgerald",
            "Lee",
            "Tolstoy",
            "Shakespeare",
            "Cervantes",
        ],
        "landmark": [
            "Eiffel Tower",
            "Statue of Liberty",
            "Great Wall",
            "Taj Mahal",
            "Colosseum",
            "Pyramids",
            "Big Ben",
            "Mount Rushmore",
        ],
        "location": [
            "Paris",
            "New York",
            "China",
            "India",
            "Rome",
            "Egypt",
            "London",
            "South Dakota",
        ],
        "language": ["Python", "JavaScript", "Java", "C++", "Rust", "Go", "Swift", "Kotlin"],
        "application": [
            "web development",
            "data science",
            "mobile apps",
            "game development",
            "systems programming",
            "cloud computing",
            "machine learning",
            "IoT",
        ],
        "algorithm": [
            "sorting",
            "search",
            "clustering",
            "hashing",
            "encryption",
            "compression",
            "pathfinding",
            "optimization",
        ],
        "purpose": [
            "organizing data",
            "finding information",
            "grouping similar items",
            "fast lookups",
            "securing data",
            "reducing size",
            "navigation",
            "finding the best solution",
        ],
        "framework": [
            "React",
            "TensorFlow",
            "Django",
            "Spring Boot",
            "Angular",
            "Flutter",
            "Scikit-learn",
            "Express",
        ],
        "technology": [
            "frontend",
            "machine learning",
            "web backends",
            "enterprise applications",
            "SPAs",
            "mobile development",
            "data science",
            "Node.js applications",
        ],
        "component": [
            "CPU",
            "GPU",
            "RAM",
            "SSD",
            "motherboard",
            "power supply",
            "network card",
            "cooling system",
        ],
        "system": [
            "computer",
            "game console",
            "smartphone",
            "server",
            "IoT device",
            "embedded system",
            "smart home",
            "vehicle",
        ],
        "protocol": ["HTTP", "TCP/IP", "SMTP", "FTP", "SSH", "Bluetooth", "WiFi", "MQTT"],
        "field": [
            "web",
            "networking",
            "email",
            "file transfer",
            "secure connections",
            "short-range wireless",
            "wireless networking",
            "IoT communication",
        ],
        "term": [
            "algorithm",
            "data structure",
            "API",
            "IDE",
            "compiler",
            "database",
            "repository",
            "framework",
        ],
        "definition": [
            "step-by-step procedure",
            "organized data container",
            "interface for software",
            "development environment",
            "code translator",
            "data storage system",
            "code storage",
            "reusable codebase",
        ],
    }

    # Generate memories
    for i in range(count):
        category = np.random.choice(categories)
        template = np.random.choice(templates[category])

        # Fill in the template with random data
        memory_text = template
        for key in data:
            if "{" + key + "}" in memory_text:
                memory_text = memory_text.replace("{" + key + "}", np.random.choice(data[key]))

        memories.append({"text": memory_text, "category": category})

    return memories


def generate_test_queries(memory_data, query_count=10):
    """Generate test queries based on memories."""
    queries = []

    # Sample memories to create queries from
    sampled_indices = np.random.choice(
        len(memory_data), min(query_count, len(memory_data)), replace=False
    )

    for idx in sampled_indices:
        memory = memory_data[idx]
        text = memory["text"]
        category = memory["category"]

        if category == "personal":
            # For personal memories, create "what is my X" type queries
            if "name is" in text:
                queries.append(
                    {"query": "What is my name?", "expected": text, "category": category}
                )
            elif "live in" in text:
                queries.append(
                    {"query": "Where do I live?", "expected": text, "category": category}
                )
            elif "favorite color" in text:
                queries.append(
                    {"query": "What is my favorite color?", "expected": text, "category": category}
                )
            elif "work as" in text:
                queries.append({"query": "What is my job?", "expected": text, "category": category})
            elif "have a" in text and "named" in text:
                pet_type = text.split("have a ")[1].split(" named")[0]
                queries.append(
                    {"query": f"Do I have a {pet_type}?", "expected": text, "category": category}
                )
            elif "hobby is" in text:
                queries.append(
                    {"query": "What are my hobbies?", "expected": text, "category": category}
                )
        elif category == "factual":
            # For factual memories, create fact-based queries
            if "capital of" in text:
                country = text.split("capital of ")[1].split(" is")[0]
                queries.append(
                    {
                        "query": f"What is the capital of {country}?",
                        "expected": text,
                        "category": category,
                    }
                )
            elif "born in" in text:
                person = text.split(" was born")[0]
                queries.append(
                    {"query": f"When was {person} born?", "expected": text, "category": category}
                )
            elif "native to" in text:
                animal = text.split("The ")[1].split(" is native")[0]
                queries.append(
                    {
                        "query": f"Where is the {animal} native to?",
                        "expected": text,
                        "category": category,
                    }
                )
            elif "chemical formula" in text:
                compound = text.split("formula for ")[1].split(" is")[0]
                queries.append(
                    {
                        "query": f"What is the chemical formula for {compound}?",
                        "expected": text,
                        "category": category,
                    }
                )
            elif "written by" in text:
                book = text.split(" was written")[0]
                queries.append(
                    {"query": f"Who wrote {book}?", "expected": text, "category": category}
                )
            elif "located in" in text:
                landmark = text.split("The ")[1].split(" is located")[0]
                queries.append(
                    {
                        "query": f"Where is the {landmark} located?",
                        "expected": text,
                        "category": category,
                    }
                )
        else:  # technical
            # For technical memories, create technical queries
            if "programming language" in text:
                language = text.split(" is a programming")[0]
                queries.append(
                    {
                        "query": f"What is {language} used for?",
                        "expected": text,
                        "category": category,
                    }
                )
            elif "algorithm is used" in text:
                algorithm = text.split("The ")[1].split(" algorithm")[0]
                queries.append(
                    {
                        "query": f"What is the {algorithm} algorithm used for?",
                        "expected": text,
                        "category": category,
                    }
                )
            elif "framework for" in text:
                framework = text.split(" is a framework")[0]
                queries.append(
                    {
                        "query": f"What is {framework} a framework for?",
                        "expected": text,
                        "category": category,
                    }
                )
            elif "part of" in text:
                component = text.split("The ")[1].split(" is a part")[0]
                queries.append(
                    {
                        "query": f"What system is the {component} a part of?",
                        "expected": text,
                        "category": category,
                    }
                )
            elif "protocol used" in text:
                protocol = text.split(" is a protocol")[0]
                queries.append(
                    {
                        "query": f"What is {protocol} used for?",
                        "expected": text,
                        "category": category,
                    }
                )
            elif "refers to" in text:
                term = text.split("The ")[1].split(" refers")[0]
                queries.append(
                    {
                        "query": f"What does the term {term} refer to?",
                        "expected": text,
                        "category": category,
                    }
                )

    return queries[:query_count]  # Ensure we return at most query_count queries


def populate_memory(memory, encoder, memory_data):
    """Populate memory with data."""
    print(f"Populating memory with {len(memory_data)} items...")
    for item in tqdm(memory_data):
        embedding, metadata = encoder.encode_concept(
            concept=item["category"], description=item["text"], related_concepts=[item["category"]]
        )
        memory.add_memory(embedding, item["text"], metadata)


def test_retrieval_performance(retriever, queries):
    """Test retrieval performance on a set of queries."""
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    retrieval_times = []

    for query in tqdm(queries):
        # Time the retrieval operation
        start_time = time.time()
        retrieved = retriever.retrieve_for_context(query["query"], top_k=5)
        retrieval_time = time.time() - start_time
        retrieval_times.append(retrieval_time)

        # Extract retrieved texts
        retrieved_texts = [
            item.get("text", "") or item.get("content", "") or item.get("description", "")
            for item in retrieved
        ]

        # Check if expected answer is in retrieved texts
        expected = query["expected"]
        found = any(expected in text for text in retrieved_texts)

        # Calculate precision and recall
        precision = 1 / len(retrieved) if found else 0
        recall = 1 if found else 0

        # Calculate F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    # Calculate averages
    query_count = len(queries)
    avg_precision = total_precision / query_count if query_count else 0
    avg_recall = total_recall / query_count if query_count else 0
    avg_f1 = total_f1 / query_count if query_count else 0
    avg_retrieval_time = sum(retrieval_times) / query_count if query_count else 0

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "avg_retrieval_time": avg_retrieval_time,
        "retrieval_times": retrieval_times,
    }


def compare_retrieval_methods(embedding_model, memory_sizes=[100, 200, 500, 1000, 5000]):
    """Compare original and optimized retrieval methods."""
    categories = ["personal", "factual", "technical"]
    results = {}

    print(f"Comparing retrieval methods with memory sizes: {memory_sizes}")

    # Initialize spaCy once for all tests
    try:
        import spacy

        global global_nlp
        print("Loading spaCy model once for all tests")
        global_nlp = spacy.load("en_core_web_sm")
    except:
        print("Could not load spaCy model, will use fallback methods")

    for size in memory_sizes:
        print(f"\nTesting with {size} memories...")

        # Generate synthetic memories and queries
        memory_data = generate_synthetic_memories(size, categories)
        queries = generate_test_queries(memory_data, query_count=20)

        # Initialize memory and encoder
        memory_dim = embedding_model.encode("test").shape[0]
        memory = ContextualMemory(embedding_dim=memory_dim)
        encoder = MemoryEncoder(embedding_model)

        # Populate memory
        populate_memory(memory, encoder, memory_data)

        # Test original retriever
        print("Testing original retriever...")
        original_retriever = ContextualRetriever(
            memory=memory,
            embedding_model=embedding_model,
            use_two_stage_retrieval=True,
            query_type_adaptation=True,
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            personal_query_threshold=0.5,
            factual_query_threshold=0.2,
            adaptive_k_factor=0.15,
        )

        original_performance = test_retrieval_performance(original_retriever, queries)

        # Test optimized retriever with clustering
        print("Testing optimized retriever with clustering...")
        optimized_retriever_clustering = OptimizedContextualRetriever(
            memory=memory,
            embedding_model=embedding_model,
            use_two_stage_retrieval=True,
            query_type_adaptation=True,
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            personal_query_threshold=0.5,
            factual_query_threshold=0.2,
            adaptive_k_factor=0.15,
            use_clustering=True,
            cluster_count=min(50, size // 20),
        )

        clustering_performance = test_retrieval_performance(optimized_retriever_clustering, queries)

        # Test optimized retriever with prefiltering
        print("Testing optimized retriever with prefiltering...")
        optimized_retriever_prefiltering = OptimizedContextualRetriever(
            memory=memory,
            embedding_model=embedding_model,
            use_two_stage_retrieval=True,
            query_type_adaptation=True,
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            personal_query_threshold=0.5,
            factual_query_threshold=0.2,
            adaptive_k_factor=0.15,
            use_prefiltering=True,
            prefilter_method="keyword",
        )

        prefiltering_performance = test_retrieval_performance(
            optimized_retriever_prefiltering, queries
        )

        # Test optimized retriever with both optimizations
        print("Testing optimized retriever with all optimizations...")
        optimized_retriever_combined = OptimizedContextualRetriever(
            memory=memory,
            embedding_model=embedding_model,
            use_two_stage_retrieval=True,
            query_type_adaptation=True,
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            personal_query_threshold=0.5,
            factual_query_threshold=0.2,
            adaptive_k_factor=0.15,
            use_clustering=True,
            cluster_count=min(50, size // 20),
            use_prefiltering=True,
            prefilter_method="keyword",
        )

        combined_performance = test_retrieval_performance(optimized_retriever_combined, queries)

        # Store results - use integer keys for consistency
        results[size] = {
            "original": original_performance,
            "clustering": clustering_performance,
            "prefiltering": prefiltering_performance,
            "combined": combined_performance,
            "memory_size": size,
            "query_count": len(queries),
        }

        # Print summary
        print(f"\nResults for {size} memories:")
        print(
            f"  Original:     Precision={original_performance['precision']:.3f}, Recall={original_performance['recall']:.3f}, F1={original_performance['f1']:.3f}, Time={original_performance['avg_retrieval_time']:.3f}s"
        )
        print(
            f"  Clustering:   Precision={clustering_performance['precision']:.3f}, Recall={clustering_performance['recall']:.3f}, F1={clustering_performance['f1']:.3f}, Time={clustering_performance['avg_retrieval_time']:.3f}s"
        )
        print(
            f"  Prefiltering: Precision={prefiltering_performance['precision']:.3f}, Recall={prefiltering_performance['recall']:.3f}, F1={prefiltering_performance['f1']:.3f}, Time={prefiltering_performance['avg_retrieval_time']:.3f}s"
        )
        print(
            f"  Combined:     Precision={combined_performance['precision']:.3f}, Recall={combined_performance['recall']:.3f}, F1={combined_performance['f1']:.3f}, Time={combined_performance['avg_retrieval_time']:.3f}s"
        )

        # Clear memory to avoid OOM
        gc.collect()

    # Save results
    with open("test_output/optimized/comparison_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    return results


def plot_comparison_results(results):
    """Plot the comparison results."""
    sizes = sorted([int(size) for size in results.keys()])
    methods = ["original", "clustering", "prefiltering", "combined"]
    metrics = ["precision", "recall", "f1", "avg_retrieval_time"]
    metric_labels = ["Precision", "Recall", "F1 Score", "Avg. Retrieval Time (s)"]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Colors for different methods
    colors = ["blue", "green", "orange", "red"]

    # Plot each metric
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]

        for j, method in enumerate(methods):
            values = [results[size][method][metric] for size in sizes]
            ax.plot(sizes, values, "o-", color=colors[j], label=method.capitalize())

        ax.set_xlabel("Memory Size")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs. Memory Size")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig("test_output/optimized/comparison_results.png")
    # Don't show the plot interactively to avoid timeout
    # plt.show()


def main():
    """Run the optimized retrieval test."""
    print("MemoryWeave Optimized Retrieval Test")
    print("===================================")

    # Load embedding model
    print("Loading embedding model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embedding_model = EmbeddingModelWrapper(model, tokenizer)

    # Get memory sizes from command line or use defaults
    import argparse

    parser = argparse.ArgumentParser(description="Test optimized retrieval performance.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 200, 500, 1000, 5000],
        help="Memory sizes to test",
    )
    args = parser.parse_args()

    # Run comparison
    results = compare_retrieval_methods(embedding_model, args.sizes)

    # Plot results
    plot_comparison_results(results)

    print("\nOptimized retrieval test completed. Results saved to test_output/optimized directory.")


if __name__ == "__main__":
    main()
