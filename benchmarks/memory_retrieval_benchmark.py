#!/usr/bin/env python
"""
Memory Retrieval Benchmark

This script benchmarks different configurations of the memory retrieval system.
It measures performance metrics including time, memory usage, precision, recall, and F1 scores.
"""

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from memoryweave.components.retriever import Retriever
from memoryweave.core.contextual_memory import ContextualMemory
from memoryweave.core.refactored_retrieval import RefactoredRetriever

# Try to import sentence_transformers; use a mock if not available
try:
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    print("SentenceTransformer not available, using mock embedding model")

    class MockEmbeddingModel:
        def __init__(self, embedding_dim=768):
            self.embedding_dim = embedding_dim
            self.call_count = 0

        def encode(self, text, batch_size=32):
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

    embedding_model = MockEmbeddingModel()


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    retriever_type: str  # "legacy", "refactored", or "components"
    embedding_dim: int = 768
    max_memories: int = 1000
    use_art_clustering: bool = False
    confidence_threshold: float = 0.3
    semantic_coherence_check: bool = False
    adaptive_retrieval: bool = False
    use_two_stage_retrieval: bool = False
    query_type_adaptation: bool = False
    dynamic_threshold_adjustment: bool = False
    memory_decay_enabled: bool = False


@dataclass
class BenchmarkResults:
    """Results from a benchmark run."""

    config: BenchmarkConfig
    setup_time: float
    query_times: List[float]
    avg_query_time: float
    precision: float
    recall: float
    f1_score: float
    memory_usage: float
    retrieval_counts: List[int]
    avg_retrieval_count: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary."""
        return {
            "config": {
                "name": self.config.name,
                "retriever_type": self.config.retriever_type,
                "confidence_threshold": self.config.confidence_threshold,
                "use_art_clustering": self.config.use_art_clustering,
                "semantic_coherence_check": self.config.semantic_coherence_check,
                "adaptive_retrieval": self.config.adaptive_retrieval,
                "use_two_stage_retrieval": self.config.use_two_stage_retrieval,
                "query_type_adaptation": self.config.query_type_adaptation,
            },
            "setup_time": self.setup_time,
            "avg_query_time": self.avg_query_time,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "memory_usage": self.memory_usage,
            "avg_retrieval_count": self.avg_retrieval_count,
        }


class MemoryRetrievalBenchmark:
    """Benchmark for memory retrieval performance."""

    def __init__(self, configs: List[BenchmarkConfig]):
        """Initialize the benchmark with configurations to test."""
        self.configs = configs
        self.test_data = []
        self.results = []

    def generate_test_data(self, num_memories: int = 500, num_queries: int = 50):
        """Generate synthetic test data."""
        print(f"Generating test data: {num_memories} memories, {num_queries} queries")

        # Generate synthetic memories with ground truth relationships
        memory_texts = []
        memory_types = ["personal", "factual", "opinion", "event"]

        # Personal facts
        for i in range(100):
            personal_fact = f"Personal fact {i}: I {np.random.choice(['like', 'love', 'enjoy', 'prefer'])} {np.random.choice(['pizza', 'sushi', 'steak', 'ice cream', 'coffee', 'tea'])}"
            memory_texts.append((personal_fact, "personal"))

        # Factual information
        topics = ["Python", "Machine Learning", "AI", "Databases", "Cloud Computing"]
        for i in range(100):
            topic = np.random.choice(topics)
            factual_info = f"Factual info {i}: {topic} is a technology used for {np.random.choice(['data analysis', 'automation', 'development', 'scaling', 'optimization'])}"
            memory_texts.append((factual_info, "factual"))

        # Opinions
        subjects = ["AI", "Technology", "Programming", "Data Science", "Cloud"]
        for i in range(100):
            subject = np.random.choice(subjects)
            opinion = f"Opinion {i}: I think {subject} is {np.random.choice(['amazing', 'interesting', 'challenging', 'important', 'evolving'])}"
            memory_texts.append((opinion, "opinion"))

        # Events
        for i in range(100):
            event = f"Event {i}: Attended {np.random.choice(['meeting', 'conference', 'workshop', 'lecture', 'seminar'])} about {np.random.choice(topics)}"
            memory_texts.append((event, "event"))

        # Add random memories to fill up to num_memories
        while len(memory_texts) < num_memories:
            memory_type = np.random.choice(memory_types)
            memory = f"Random {memory_type} {len(memory_texts)}: Some random content about {np.random.choice(topics)}"
            memory_texts.append((memory, memory_type))

        # Shuffle and truncate to num_memories
        np.random.shuffle(memory_texts)
        memory_texts = memory_texts[:num_memories]

        # Generate embeddings
        memory_embeddings = []
        for text, _ in memory_texts:
            embedding = embedding_model.encode(text)
            memory_embeddings.append(embedding)

        # Generate test queries with ground truth relevant memories
        queries = []
        for i in range(num_queries):
            # Randomly select a subset of memories that should be relevant
            num_relevant = np.random.randint(1, 5)
            relevant_indices = np.random.choice(len(memory_texts), size=num_relevant, replace=False)

            # Create a query related to these memories
            if i % 4 == 0:  # Personal query
                query = f"What do I {np.random.choice(['like', 'love', 'enjoy', 'prefer'])}?"
            elif i % 4 == 1:  # Factual query
                topic = np.random.choice(topics)
                query = f"Tell me about {topic}"
            elif i % 4 == 2:  # Opinion query
                subject = np.random.choice(subjects)
                query = f"What do I think about {subject}?"
            else:  # Event query
                query = "What events have I attended recently?"

            queries.append((query, list(relevant_indices)))

        self.test_data = {
            "memory_texts": memory_texts,
            "memory_embeddings": memory_embeddings,
            "queries": queries,
        }

        return self.test_data

    def prepare_memory(self, config: BenchmarkConfig) -> Tuple[Any, Any]:
        """Prepare memory and retriever based on the configuration."""
        print(f"Preparing memory for config: {config.name}")

        # Create memory
        memory = ContextualMemory(
            embedding_dim=config.embedding_dim,
            max_memories=config.max_memories,
            use_art_clustering=config.use_art_clustering,
        )

        # Load test data
        for i, ((text, memory_type), embedding) in enumerate(
            zip(self.test_data["memory_texts"], self.test_data["memory_embeddings"])
        ):
            metadata = {"type": memory_type, "index": i}
            memory.add_memory(embedding, text, metadata)

        # Create retriever based on type
        if config.retriever_type == "legacy":
            # Use RefactoredRetriever instead of ContextualRetriever
            retriever = RefactoredRetriever(
                memory=memory,
                embedding_model=embedding_model,
                retrieval_strategy="hybrid",
                confidence_threshold=config.confidence_threshold,
                semantic_coherence_check=config.semantic_coherence_check,
                adaptive_retrieval=config.adaptive_retrieval,
                use_two_stage_retrieval=config.use_two_stage_retrieval,
                query_type_adaptation=config.query_type_adaptation,
            )
        elif config.retriever_type == "components":
            retriever = Retriever(memory=memory, embedding_model=embedding_model)
            retriever.minimum_relevance = config.confidence_threshold

            if config.use_two_stage_retrieval:
                retriever.configure_two_stage_retrieval(
                    enable=True,
                    first_stage_k=20,
                    first_stage_threshold_factor=0.7,
                )

            if config.query_type_adaptation:
                retriever.configure_query_type_adaptation(
                    enable=True,
                    adaptation_strength=1.0,
                )

            if config.dynamic_threshold_adjustment:
                retriever.enable_dynamic_threshold_adjustment(
                    enable=True,
                    window_size=5,
                )
        else:
            # For now, default to refactored retriever
            retriever = RefactoredRetriever(
                memory=memory,
                embedding_model=embedding_model,
                confidence_threshold=config.confidence_threshold,
            )

        return memory, retriever

    def run_benchmark(self, save_path: str = None):
        """Run the benchmark for all configurations."""
        if not self.test_data:
            self.generate_test_data()

        for config in self.configs:
            print(f"\nRunning benchmark for: {config.name}")

            # Time setup
            setup_start = time.time()
            memory, retriever = self.prepare_memory(config)
            setup_time = time.time() - setup_start

            # Run queries
            query_times = []
            retrieval_counts = []
            all_expected = []
            all_retrieved = []

            for query, expected_indices in tqdm(self.test_data["queries"], desc="Queries"):
                # Clear any caches before each query
                gc.collect()

                # Time the query
                start_time = time.time()

                if config.retriever_type == "components":
                    results = retriever.retrieve(query, top_k=10)
                else:
                    results = retriever.retrieve_for_context(query, top_k=10)

                query_time = time.time() - start_time
                query_times.append(query_time)

                # Extract retrieved indices
                if config.retriever_type == "components":
                    retrieved_indices = [
                        r.get("memory_id") for r in results if isinstance(r.get("memory_id"), int)
                    ]
                else:
                    retrieved_indices = [
                        r.get("memory_id") for r in results if isinstance(r.get("memory_id"), int)
                    ]

                retrieval_counts.append(len(retrieved_indices))

                # Store for precision/recall calculation
                all_expected.append(set(expected_indices))
                all_retrieved.append(set(retrieved_indices))

            # Calculate precision, recall, F1
            precisions = []
            recalls = []
            f1_scores = []

            for expected, retrieved in zip(all_expected, all_retrieved):
                if retrieved:
                    precision = len(expected.intersection(retrieved)) / len(retrieved)
                else:
                    precision = 0.0

                if expected:
                    recall = len(expected.intersection(retrieved)) / len(expected)
                else:
                    recall = 1.0

                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)

            # Create results object
            result = BenchmarkResults(
                config=config,
                setup_time=setup_time,
                query_times=query_times,
                avg_query_time=np.mean(query_times),
                precision=np.mean(precisions),
                recall=np.mean(recalls),
                f1_score=np.mean(f1_scores),
                memory_usage=0.0,  # Would need additional tooling to measure
                retrieval_counts=retrieval_counts,
                avg_retrieval_count=np.mean(retrieval_counts),
            )

            self.results.append(result)

            print(f"Results for {config.name}:")
            print(f"  Avg Query Time: {result.avg_query_time:.4f}s")
            print(f"  Precision: {result.precision:.4f}")
            print(f"  Recall: {result.recall:.4f}")
            print(f"  F1 Score: {result.f1_score:.4f}")
            print(f"  Avg Results: {result.avg_retrieval_count:.2f}")

        # Save results if path provided
        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            results_dict = {r.config.name: r.to_dict() for r in self.results}
            with open(save_path, "w") as f:
                json.dump(results_dict, f, indent=2)

        # Generate and show visualizations
        self.visualize_results()

        return self.results

    def visualize_results(self):
        """Visualize benchmark results."""
        if not self.results:
            print("No results to visualize")
            return

        # Extract data for plotting
        names = [r.config.name for r in self.results]
        avg_times = [r.avg_query_time for r in self.results]
        precisions = [r.precision for r in self.results]
        recalls = [r.recall for r in self.results]
        f1_scores = [r.f1_score for r in self.results]
        avg_counts = [r.avg_retrieval_count for r in self.results]

        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Query time
        axs[0, 0].bar(names, avg_times)
        axs[0, 0].set_title("Average Query Time (s)")
        axs[0, 0].set_ylabel("Time (s)")
        axs[0, 0].tick_params(axis="x", rotation=45)

        # Precision, recall, F1
        x = np.arange(len(names))
        width = 0.25
        axs[0, 1].bar(x - width, precisions, width, label="Precision")
        axs[0, 1].bar(x, recalls, width, label="Recall")
        axs[0, 1].bar(x + width, f1_scores, width, label="F1")
        axs[0, 1].set_title("Retrieval Quality Metrics")
        axs[0, 1].set_xticks(x)
        axs[0, 1].set_xticklabels(names)
        axs[0, 1].tick_params(axis="x", rotation=45)
        axs[0, 1].legend()

        # Average retrieval count
        axs[1, 0].bar(names, avg_counts)
        axs[1, 0].set_title("Average Number of Results")
        axs[1, 0].set_ylabel("Count")
        axs[1, 0].tick_params(axis="x", rotation=45)

        # Time vs F1 scatter
        axs[1, 1].scatter(avg_times, f1_scores, s=100)
        for i, name in enumerate(names):
            axs[1, 1].annotate(name, (avg_times[i], f1_scores[i]))
        axs[1, 1].set_title("Time vs F1 Score")
        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].set_ylabel("F1 Score")

        # Layout and display
        plt.tight_layout()
        plt.savefig("benchmark_results.png")
        plt.close()

        print("Visualizations saved to benchmark_results.png")


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="Memory Retrieval Benchmark")
    parser.add_argument(
        "--num-memories", type=int, default=500, help="Number of memories to generate"
    )
    parser.add_argument("--num-queries", type=int, default=50, help="Number of test queries to run")
    parser.add_argument(
        "--save-path", type=str, default="benchmark_results.json", help="Path to save results"
    )
    args = parser.parse_args()

    # Define configurations to benchmark
    configs = [
        BenchmarkConfig(
            name="Legacy-Basic",
            retriever_type="legacy",
            confidence_threshold=0.3,
            semantic_coherence_check=False,
            adaptive_retrieval=False,
            use_two_stage_retrieval=False,
            query_type_adaptation=False,
        ),
        BenchmarkConfig(
            name="Legacy-Advanced",
            retriever_type="legacy",
            confidence_threshold=0.3,
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            use_two_stage_retrieval=True,
            query_type_adaptation=True,
        ),
        BenchmarkConfig(
            name="Components-Basic",
            retriever_type="components",
            confidence_threshold=0.3,
            semantic_coherence_check=False,
            adaptive_retrieval=False,
            use_two_stage_retrieval=False,
            query_type_adaptation=False,
        ),
        BenchmarkConfig(
            name="Components-Advanced",
            retriever_type="components",
            confidence_threshold=0.3,
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            use_two_stage_retrieval=True,
            query_type_adaptation=True,
        ),
        BenchmarkConfig(
            name="ART-Clustering",
            retriever_type="legacy",
            use_art_clustering=True,
            confidence_threshold=0.3,
            semantic_coherence_check=True,
            adaptive_retrieval=True,
        ),
    ]

    # Run the benchmark
    benchmark = MemoryRetrievalBenchmark(configs)
    benchmark.generate_test_data(num_memories=args.num_memories, num_queries=args.num_queries)
    benchmark.run_benchmark(save_path=args.save_path)


if __name__ == "__main__":
    main()
