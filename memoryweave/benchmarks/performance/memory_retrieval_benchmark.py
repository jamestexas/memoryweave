# File: memoryweave/benchmarks/performance/memory_retrieval_benchmark.py

import gc
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from memoryweave.benchmarks.base import Benchmark, BenchmarkConfig, BenchmarkResult
from memoryweave.benchmarks.utils.visualization import create_bar_chart, create_radar_chart
from memoryweave.components.retriever import Retriever
from memoryweave.core.contextual_memory import ContextualMemory

# Import embedding model (simplified for brevity)
try:
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
except ImportError:
    # Create mock embedding model
    embedding_dim = 384

    class MockEmbeddingModel:
        def encode(self, text, show_progress_bar=False):
            if isinstance(text, list):
                return np.array([self._encode_single(t) for t in text])
            return self._encode_single(text)

        def _encode_single(self, text):
            hash_val = hash(text) % 1000000
            np.random.seed(hash_val)
            embedding = np.random.randn(embedding_dim)
            return embedding / np.linalg.norm(embedding)

    embedding_model = MockEmbeddingModel()


@dataclass
class MemoryRetrievalConfig(BenchmarkConfig):
    """Configuration for memory retrieval benchmark."""

    retriever_type: str = "legacy"  # "legacy", "components", etc.
    embedding_dim: int = embedding_dim
    max_memories: int = 1000
    memories_to_test: int = 500  # Number of memories to generate
    queries_to_test: int = 50  # Number of queries to run
    use_art_clustering: bool = False
    confidence_threshold: float = 0.3
    semantic_coherence_check: bool = False
    adaptive_retrieval: bool = False
    use_two_stage_retrieval: bool = False
    query_type_adaptation: bool = False
    dynamic_threshold_adjustment: bool = False
    memory_decay_enabled: bool = False
    use_ann: bool = False


class MemoryRetrievalBenchmark(Benchmark):
    """Benchmark for different memory retrieval configurations."""

    def __init__(self, configs: List[MemoryRetrievalConfig]):
        super().__init__(configs)
        self.test_data = None

    def generate_test_data(self, num_memories: int = 500, num_queries: int = 50):
        """Generate synthetic test data for benchmarking."""
        print(f"Generating test data: {num_memories} memories, {num_queries} queries")

        # Enhanced synthetic data generation
        memory_texts = []
        memory_types = ["personal", "factual", "opinion", "event"]

        # Generate more realistic test data with controlled properties
        # Personal facts with specific entities and attributes
        for i in range(num_memories // 4):
            entities = ["I", "my friend", "my family", "my dog", "my cat", "my house"]
            attributes = ["like", "enjoy", "prefer", "visited", "bought", "own"]
            objects = [
                "pizza",
                "coffee",
                "movies",
                "books",
                "running",
                "hiking",
                "Paris",
                "New York",
            ]

            entity = np.random.choice(entities)
            attribute = np.random.choice(attributes)
            obj = np.random.choice(objects)

            fact = f"Personal fact {i}: {entity} {attribute} {obj}."
            memory_texts.append((fact, "personal"))

        # Domain-specific factual information
        domains = {
            "technology": [
                "Python",
                "JavaScript",
                "machine learning",
                "cloud computing",
                "cybersecurity",
            ],
            "science": ["biology", "physics", "chemistry", "astronomy", "geology"],
            "history": [
                "World War II",
                "Roman Empire",
                "Industrial Revolution",
                "Renaissance",
                "Cold War",
            ],
            "arts": ["painting", "music", "literature", "film", "theater"],
        }

        for i in range(num_memories // 4):
            domain = np.random.choice(list(domains.keys()))
            topic = np.random.choice(domains[domain])

            templates = [
                f"Factual info {i}: {topic} is a key concept in {domain}.",
                f"Factual info {i}: {topic} developed during the 20th century.",
                f"Factual info {i}: {topic} has significantly influenced modern {domain}.",
                f"Factual info {i}: {topic} involves complex principles and techniques.",
            ]

            fact = np.random.choice(templates)
            memory_texts.append((fact, "factual"))

        # Add varied opinion memories
        for i in range(num_memories // 4):
            subjects = [
                "AI",
                "social media",
                "remote work",
                "electric vehicles",
                "renewable energy",
            ]
            opinions = ["promising", "concerning", "revolutionary", "overrated", "essential"]

            subject = np.random.choice(subjects)
            opinion = np.random.choice(opinions)

            fact = (
                f"Opinion {i}: I believe {subject} is {opinion} because of its impact on society."
            )
            memory_texts.append((fact, "opinion"))

        # Add event memories
        for i in range(num_memories // 4):
            actions = ["attended", "organized", "participated in", "witnessed", "spoke at"]
            events = ["conference", "meeting", "workshop", "demonstration", "exhibition", "concert"]
            locations = ["San Francisco", "Berlin", "Tokyo", "Toronto", "London", "Sydney"]

            action = np.random.choice(actions)
            event = np.random.choice(events)
            location = np.random.choice(locations)

            fact = f"Event {i}: {action} a {event} in {location} last month."
            memory_texts.append((fact, "event"))

        # Shuffle and truncate to requested size
        np.random.shuffle(memory_texts)
        memory_texts = memory_texts[:num_memories]

        # Generate embeddings
        memory_embeddings = []
        for text, _ in memory_texts:
            embedding = embedding_model.encode(text, show_progress_bar=False)
            memory_embeddings.append(embedding)

        # Generate test queries with relevant memories
        queries = []
        for i in range(num_queries):
            # Create different query types
            if i % 5 == 0:  # Personal fact query
                query = "What do I enjoy doing?"
            elif i % 5 == 1:  # Factual knowledge query
                domain = np.random.choice(list(domains.keys()))
                topic = np.random.choice(domains[domain])
                query = f"Tell me about {topic}"
            elif i % 5 == 2:  # Opinion query
                subject = np.random.choice(subjects)
                query = f"What do I think about {subject}?"
            elif i % 5 == 3:  # Event query
                query = f"What happened in {np.random.choice(locations)}?"
            else:  # Generic query
                query = "What information do you have about my preferences?"

            # Select a few random memories as expected answers
            num_relevant = np.random.randint(1, 5)
            relevant_indices = np.random.choice(len(memory_texts), size=num_relevant, replace=False)

            queries.append((query, list(relevant_indices)))

        self.test_data = {
            "memory_texts": memory_texts,
            "memory_embeddings": memory_embeddings,
            "queries": queries,
        }

        return self.test_data

    def setup(self, config: MemoryRetrievalConfig) -> Dict[str, Any]:
        """Set up the benchmark environment for a specific configuration."""
        if not self.test_data:
            self.generate_test_data(
                num_memories=config.memories_to_test, num_queries=config.queries_to_test
            )

        # Create memory system
        memory = ContextualMemory(
            embedding_dim=config.embedding_dim,
            max_memories=config.max_memories,
            use_art_clustering=config.use_art_clustering,
            use_ann=config.use_ann,
        )

        # Add memories
        for i, ((text, memory_type), embedding) in enumerate(
            zip(self.test_data["memory_texts"], self.test_data["memory_embeddings"])
        ):
            metadata = {"type": memory_type, "index": i, "content": text}
            memory.add_memory(embedding, text, metadata)

        # Create retriever based on configuration
        retriever = Retriever(memory=memory, embedding_model=embedding_model)
        retriever.minimum_relevance = config.confidence_threshold

        # Configure features based on config
        if config.semantic_coherence_check and hasattr(retriever, "configure_semantic_coherence"):
            retriever.configure_semantic_coherence(enable=True)

        if config.query_type_adaptation and hasattr(retriever, "configure_query_type_adaptation"):
            retriever.configure_query_type_adaptation(enable=True)

        if config.use_two_stage_retrieval and hasattr(retriever, "configure_two_stage_retrieval"):
            retriever.configure_two_stage_retrieval(enable=True)

        if config.adaptive_retrieval and hasattr(retriever, "enable_dynamic_threshold_adjustment"):
            retriever.enable_dynamic_threshold_adjustment(enable=True)

        # Initialize components
        if hasattr(retriever, "initialize_components"):
            retriever.initialize_components()

        return {
            "memory": memory,
            "retriever": retriever,
        }

    def run_single_benchmark(
        self, config: MemoryRetrievalConfig, setup_data: Dict[str, Any]
    ) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        memory = setup_data["memory"]
        retriever = setup_data["retriever"]

        print(f"Running benchmark for: {config.name}")
        start_time = time.time()

        # Run queries
        query_times = []
        retrieval_counts = []
        all_expected = []
        all_retrieved = []

        for query, expected_indices in self.test_data["queries"]:
            # Clear caches
            gc.collect()

            # Time the query
            query_start = time.time()
            results = retriever.retrieve(query, top_k=10)
            query_time = time.time() - query_start
            query_times.append(query_time)

            # Extract results
            retrieved_indices = []
            for r in results:
                if isinstance(r.get("memory_id"), int):
                    retrieved_indices.append(r.get("memory_id"))

            retrieval_counts.append(len(retrieved_indices))

            # Store for precision/recall calculation
            all_expected.append(set(expected_indices))
            all_retrieved.append(set(retrieved_indices))

        # Calculate metrics
        precisions = []
        recalls = []
        f1_scores = []

        for expected, retrieved in zip(all_expected, all_retrieved):
            precision = len(expected.intersection(retrieved)) / len(retrieved) if retrieved else 0.0
            recall = len(expected.intersection(retrieved)) / len(expected) if expected else 1.0

            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        end_time = time.time()

        # Create metrics
        metrics = {
            "avg_query_time": np.mean(query_times),
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1_score": np.mean(f1_scores),
            "avg_retrieval_count": np.mean(retrieval_counts),
        }

        # Additional data for detailed analysis
        additional_data = {
            "query_times": query_times,
            "retrieval_counts": retrieval_counts,
            "individual_precisions": precisions,
            "individual_recalls": recalls,
            "individual_f1_scores": f1_scores,
        }

        return BenchmarkResult(
            config_name=config.name,
            metrics=metrics,
            start_time=start_time,
            end_time=end_time,
            additional_data=additional_data,
        )

    def visualize_results(self) -> None:
        """Generate visualizations for benchmark results."""
        if not self.results:
            print("No results to visualize")
            return

        # Extract data for plotting
        config_names = [r.config_name for r in self.results]

        # Create comparison charts for each metric
        metrics = {
            "avg_query_time": {"title": "Average Query Time", "ylabel": "Time (s)"},
            "precision": {"title": "Precision", "ylabel": "Score"},
            "recall": {"title": "Recall", "ylabel": "Score"},
            "f1_score": {"title": "F1 Score", "ylabel": "Score"},
            "avg_retrieval_count": {"title": "Average Results Count", "ylabel": "Count"},
        }

        for metric, info in metrics.items():
            data = {r.config_name: r.metrics[metric] for r in self.results}
            create_bar_chart(
                data=data,
                title=info["title"],
                ylabel=info["ylabel"],
                output_file=f"benchmark_results_{metric}.png",
            )

        # Create radar chart for comparing all metrics
        normalized_metrics = {}

        for r in self.results:
            # Normalize metrics (lower is better for time and count)
            normalized = {}

            # Invert time so lower is better for visualization
            max_time = max(res.metrics["avg_query_time"] for res in self.results)
            normalized["speed"] = 1 - (r.metrics["avg_query_time"] / max_time)

            # Higher is better for precision, recall, f1
            normalized["precision"] = r.metrics["precision"]
            normalized["recall"] = r.metrics["recall"]
            normalized["f1_score"] = r.metrics["f1_score"]

            # Normalize result count (assume optimal is middle range)
            counts = [res.metrics["avg_retrieval_count"] for res in self.results]
            avg_count = np.mean(counts)
            count_distance = abs(r.metrics["avg_retrieval_count"] - avg_count)
            max_distance = max(abs(count - avg_count) for count in counts)
            normalized["result_efficiency"] = 1 - (
                count_distance / max_distance if max_distance > 0 else 0
            )

            normalized_metrics[r.config_name] = normalized

        create_radar_chart(
            data=normalized_metrics,
            metrics=["speed", "precision", "recall", "f1_score", "result_efficiency"],
            title="Overall Performance Comparison",
            output_file="benchmark_results_radar.png",
        )

        print("Visualizations saved to benchmark_results_*.png")
