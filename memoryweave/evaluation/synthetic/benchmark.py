# memoryweave/evaluation/synthetic/benchmark.py
"""
Synthetic test benchmark for MemoryWeave evaluation.

This module provides a benchmark runner that uses synthetic data to evaluate
different retrieval strategies and configurations in a controlled environment.
The benchmark produces metrics like precision, recall, and F1 score to help
compare different approaches.
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from memoryweave.components.retriever import Retriever
from memoryweave.core.contextual_memory import ContextualMemory
from memoryweave.evaluation.synthetic.generators import (
    SyntheticMemoryGenerator,
    SyntheticQueryGenerator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("synthetic_benchmark.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    retriever_type: str  # "components" or other custom type
    confidence_threshold: float = 0.3
    use_art_clustering: bool = False
    semantic_coherence_check: bool = False
    adaptive_retrieval: bool = False
    use_two_stage_retrieval: bool = False
    query_type_adaptation: bool = False
    dynamic_threshold_adjustment: bool = False
    memory_decay_enabled: bool = False
    evaluation_mode: bool = True  # Whether to run in evaluation mode (no special case handling)
    top_k: int = 5


@dataclass
class BenchmarkResults:
    """Results from a benchmark run."""

    config: BenchmarkConfig
    precision: float
    recall: float
    f1_score: float
    avg_query_time: float
    avg_retrieval_count: float

    def to_dict(self) -> dict[str, Any]:
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
                "top_k": self.config.top_k,
            },
            "metrics": {
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "avg_query_time": self.avg_query_time,
                "avg_retrieval_count": self.avg_retrieval_count,
            },
        }


class SyntheticBenchmark:
    """Benchmark that uses synthetic data to evaluate retrieval strategies."""

    def __init__(
        self,
        configs: list[BenchmarkConfig],
        embedding_model: Any = None,
        dataset_path: Optional[Union[str, Path]] = None,
        random_seed: int = 42,
    ):
        """
        Initialize the benchmark.

        Args:
            configs: list of configurations to evaluate
            embedding_model: Model to use for embeddings (optional)
            dataset_path: Path to existing dataset file (optional)
            random_seed: Random seed for reproducibility
        """
        self.configs = configs
        self.embedding_model = embedding_model
        self.dataset_path = dataset_path
        self.random_seed = random_seed
        self.results = []
        self.dataset = None

        # Set random seed
        np.random.seed(random_seed)

        # Set up logging
        self.logger = logging.getLogger(__name__)

    def generate_dataset(
        self,
        num_memories: int = 200,
        num_queries: int = 50,
        save_path: Optional[Union[str, Path]] = None,
    ) -> dict[str, Any]:
        """
        Generate a synthetic dataset for benchmarking.

        Args:
            num_memories: Number of memories to generate
            num_queries: Number of queries to generate
            save_path: Path to save the dataset (optional)

        Returns:
            dictionary with memories and queries
        """
        logger.info(
            f"Generating synthetic dataset with {num_memories} memories and {num_queries} queries"
        )

        # Create generators
        memory_generator = SyntheticMemoryGenerator(
            embedding_model=self.embedding_model, random_seed=self.random_seed
        )

        query_generator = SyntheticQueryGenerator(
            embedding_model=self.embedding_model, random_seed=self.random_seed
        )

        # Generate memories
        logger.info("Generating synthetic memories")
        memories = memory_generator.generate_memories(num_memories=num_memories)

        # Generate queries
        logger.info("Generating synthetic queries")
        dataset = query_generator.generate_evaluation_dataset(
            memories=memories, num_queries=num_queries, file_path=save_path
        )

        self.dataset = dataset
        logger.info(
            f"Generated dataset with {len(dataset['memories'])} memories and {len(dataset['queries'])} queries"
        )

        if save_path:
            self.dataset_path = save_path
            logger.info(f"Saved dataset to {save_path}")

        return dataset

    def load_dataset(self, path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
        """
        Load a dataset from a file.

        Args:
            path: Path to the dataset file (defaults to self.dataset_path)

        Returns:
            dictionary with memories and queries
        """
        path = path or self.dataset_path
        if not path:
            raise ValueError("No dataset path provided")

        logger.info(f"Loading dataset from {path}")
        with open(path) as f:
            self.dataset = json.load(f)

        logger.info(
            f"Loaded dataset with {len(self.dataset['memories'])} memories and {len(self.dataset['queries'])} queries"
        )
        return self.dataset

    def prepare_memory(self, config: BenchmarkConfig) -> tuple[ContextualMemory, Retriever]:
        """
        Prepare memory and retriever based on the configuration.

        Args:
            config: Benchmark configuration

        Returns:
            tuple of (memory, retriever)
        """
        if not self.dataset:
            if self.dataset_path:
                self.load_dataset()
            else:
                raise ValueError("No dataset available. Generate or load a dataset first.")

        logger.info(f"Preparing memory and retriever for config: {config.name}")

        # Determine embedding dimension from the first memory in the dataset
        if self.dataset["memories"]:
            first_embedding = np.array(self.dataset["memories"][0]["embedding"])
            embedding_dim = first_embedding.shape[0]
        else:
            embedding_dim = 768  # Default dimension

        logger.info(f"Using embedding dimension: {embedding_dim}")

        # Create memory
        memory = ContextualMemory(
            embedding_dim=embedding_dim,
            max_memories=len(self.dataset["memories"]),
            use_art_clustering=config.use_art_clustering,
        )

        # Load memories
        for i, mem in enumerate(self.dataset["memories"]):
            embedding = np.array(mem["embedding"])

            # Check that embedding has the correct dimension
            if embedding.shape[0] != embedding_dim:
                logger.warning(
                    f"Embedding dimension mismatch: expected {embedding_dim}, got {embedding.shape[0]}. Skipping memory."
                )
                continue

            text = mem["text"]
            metadata = {**mem["metadata"], "index": i}  # Add original index to metadata
            memory.add_memory(embedding, text, metadata)

        # Create retriever
        retriever = Retriever(memory=memory, embedding_model=self.embedding_model)
        retriever.minimum_relevance = config.confidence_threshold
        retriever.top_k = config.top_k

        # First initialize components to ensure everything is registered
        retriever.initialize_components()

        # Then configure features
        if config.semantic_coherence_check:
            retriever.configure_semantic_coherence(enable=True)

        if config.query_type_adaptation:
            retriever.configure_query_type_adaptation(enable=True)

        if config.use_two_stage_retrieval:
            retriever.configure_two_stage_retrieval(enable=True)

        if config.dynamic_threshold_adjustment:
            retriever.enable_dynamic_threshold_adjustment(enable=True)

        if config.memory_decay_enabled:
            # Enable memory decay if the component is available
            try:
                retriever.enable_memory_decay(enable=True)
            except AttributeError:
                logger.warning("Memory decay not available in this version of the retriever")

        # Rebuild the pipeline with updated configurations
        retriever._build_default_pipeline()

        return memory, retriever

    def run_benchmark(self, save_path: Optional[Union[str, Path]] = None) -> list[BenchmarkResults]:
        """
        Run the benchmark for all configurations.

        Args:
            save_path: Path to save the results (optional)

        Returns:
            list of benchmark results
        """
        if not self.dataset:
            if self.dataset_path:
                self.load_dataset()
            else:
                raise ValueError("No dataset available. Generate or load a dataset first.")

        self.results = []

        for config in self.configs:
            logger.info(f"\nRunning benchmark for: {config.name}")

            # Prepare memory and retriever
            memory, retriever = self.prepare_memory(config)

            # Run queries
            query_times = []
            retrieval_counts = []
            all_expected = []
            all_retrieved = []

            logger.info(f"Running {len(self.dataset['queries'])} queries")
            for query_item in tqdm(self.dataset["queries"], desc="Queries"):
                query = query_item["query"]
                expected_indices = query_item["relevant_indices"]
                query_embedding = np.array(query_item["embedding"])

                # Set evaluation mode to prevent special case handling
                context = {"in_evaluation": config.evaluation_mode}
                
                # Set logging level to INFO during evaluation to capture details
                import logging
                evaluation_logger = logging.getLogger()
                original_level = evaluation_logger.level
                evaluation_logger.setLevel(logging.INFO)
                
                # Detailed logging about the configuration
                evaluation_logger.info(f"Benchmark: Retrieving with config={config.name}, " +
                           f"evaluation_mode={config.evaluation_mode}, " +
                           f"confidence_threshold={config.confidence_threshold}, " +
                           f"semantic_coherence_check={config.semantic_coherence_check}, " +
                           f"use_two_stage_retrieval={config.use_two_stage_retrieval}, " +
                           f"query_type_adaptation={config.query_type_adaptation}")

                # Time the query
                start_time = time.time()

                # Check query embedding dimension
                memory_dim = getattr(memory, "embedding_dim", 768)
                if query_embedding.shape[0] != memory_dim:
                    logger.warning(
                        f"Query embedding dimension mismatch: memory expects {memory_dim}, got {query_embedding.shape[0]}. "
                        f"Using dummy embedding for this query."
                    )
                    # Create a compatible embedding
                    query_embedding = np.ones(memory_dim) / np.sqrt(memory_dim)

                # Set up retriever with the query embedding
                if hasattr(retriever, "embedding_model") and not retriever.embedding_model:
                    # Temporarily set the embedding model to a mock one that returns our query embedding
                    original_embedding_model = retriever.embedding_model

                    class MockEmbeddingModelForBenchmark:
                        def encode(self, query_text):
                            return query_embedding

                    retriever.embedding_model = MockEmbeddingModelForBenchmark()

                # Retrieve memories relevant to the query with evaluation flag
                # We'll use a custom attribute on the memory object to flag that we're in evaluation mode
                if hasattr(memory, "in_evaluation"):
                    original_in_evaluation = memory.in_evaluation
                    memory.in_evaluation = config.evaluation_mode
                else:
                    # If the memory doesn't have this attribute, add it temporarily
                    memory.in_evaluation = config.evaluation_mode
                    original_in_evaluation = None

                # Update the memory_manager components working context to indicate we're in evaluation
                if hasattr(retriever, "memory_manager"):
                    retriever.memory_manager.working_context = {
                        "in_evaluation": config.evaluation_mode
                    }

                # Retrieve results
                results = retriever.retrieve(
                    query, top_k=config.top_k, minimum_relevance=config.confidence_threshold
                )

                # Restore original settings
                if hasattr(retriever, "embedding_model") and original_embedding_model is not None:
                    retriever.embedding_model = original_embedding_model

                if original_in_evaluation is not None:
                    memory.in_evaluation = original_in_evaluation
                elif hasattr(memory, "in_evaluation"):
                    # Remove the attribute if it wasn't there before
                    delattr(memory, "in_evaluation")

                # Restore logging level
                evaluation_logger.setLevel(original_level)

                query_time = time.time() - start_time
                query_times.append(query_time)

                # Extract retrieved indices
                retrieved_indices = []
                for r in results:
                    # Get the original memory index from metadata
                    if "index" in r:
                        retrieved_indices.append(r["index"])
                    elif "memory_id" in r:
                        # For compatibility with different result formats
                        memory_id = r["memory_id"]
                        if isinstance(memory_id, int):
                            metadata = memory.memory_metadata[memory_id]
                            if "index" in metadata:
                                retrieved_indices.append(metadata["index"])

                    # Try to get from metadata if it exists
                    if "metadata" in r and "index" in r["metadata"]:
                        retrieved_indices.append(r["metadata"]["index"])

                retrieval_counts.append(len(retrieved_indices))

                # Store for precision/recall calculation
                all_expected.append(set(expected_indices))
                all_retrieved.append(set(retrieved_indices))

                # Log result for debugging
                logger.debug(f"Query: {query}")
                logger.debug(f"Expected: {expected_indices}")
                logger.debug(f"Retrieved: {retrieved_indices}")

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
                precision=np.mean(precisions),
                recall=np.mean(recalls),
                f1_score=np.mean(f1_scores),
                avg_query_time=np.mean(query_times),
                avg_retrieval_count=np.mean(retrieval_counts),
            )

            self.results.append(result)

            logger.info(f"Results for {config.name}:")
            logger.info(f"  Precision: {result.precision:.4f}")
            logger.info(f"  Recall: {result.recall:.4f}")
            logger.info(f"  F1 Score: {result.f1_score:.4f}")
            logger.info(f"  Avg Query Time: {result.avg_query_time:.4f}s")
            logger.info(f"  Avg Results: {result.avg_retrieval_count:.2f}")

        # Save results if path provided
        if save_path:
            logger.info(f"Saving results to {save_path}")
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            results_dict = {r.config.name: r.to_dict() for r in self.results}
            with open(save_path, "w") as f:
                json.dump(results_dict, f, indent=2)

        # Generate visualizations
        self.visualize_results()

        return self.results

    def visualize_results(self, save_path: str = "synthetic_benchmark_results.png"):
        """
        Visualize benchmark results.

        Args:
            save_path: Path to save the visualization
        """
        if not self.results:
            logger.warning("No results to visualize")
            return

        # Extract data for plotting
        names = [r.config.name for r in self.results]
        precisions = [r.precision for r in self.results]
        recalls = [r.recall for r in self.results]
        f1_scores = [r.f1_score for r in self.results]
        avg_times = [r.avg_query_time for r in self.results]
        avg_counts = [r.avg_retrieval_count for r in self.results]

        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Precision, recall, F1
        x = np.arange(len(names))
        width = 0.25
        axs[0, 0].bar(x - width, precisions, width, label="Precision")
        axs[0, 0].bar(x, recalls, width, label="Recall")
        axs[0, 0].bar(x + width, f1_scores, width, label="F1")
        axs[0, 0].set_title("Retrieval Quality Metrics")
        axs[0, 0].set_xticks(x)
        axs[0, 0].set_xticklabels(names, rotation=45, ha="right")
        axs[0, 0].legend()
        axs[0, 0].set_ylim(0, 1.0)

        # Query time
        axs[0, 1].bar(names, avg_times)
        axs[0, 1].set_title("Average Query Time (s)")
        axs[0, 1].set_ylabel("Time (s)")
        axs[0, 1].tick_params(axis="x", rotation=45)
        plt.setp(axs[0, 1].get_xticklabels(), ha="right")

        # Average retrieval count
        axs[1, 0].bar(names, avg_counts)
        axs[1, 0].set_title("Average Number of Results")
        axs[1, 0].set_ylabel("Count")
        axs[1, 0].tick_params(axis="x", rotation=45)
        plt.setp(axs[1, 0].get_xticklabels(), ha="right")

        # Time vs F1 scatter
        axs[1, 1].scatter(avg_times, f1_scores, s=100)
        for i, name in enumerate(names):
            axs[1, 1].annotate(name, (avg_times[i], f1_scores[i]))
        axs[1, 1].set_title("Time vs F1 Score")
        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].set_ylabel("F1 Score")

        # Layout and display
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Visualizations saved to {save_path}")


def main():
    """Main function to run the benchmark from the command line."""
    parser = argparse.ArgumentParser(description="Synthetic Memory Retrieval Benchmark")
    parser.add_argument(
        "--num-memories", type=int, default=200, help="Number of memories to generate"
    )
    parser.add_argument("--num-queries", type=int, default=50, help="Number of test queries to run")
    parser.add_argument("--dataset", type=str, default=None, help="Path to existing dataset file")
    parser.add_argument(
        "--save-dataset",
        type=str,
        default="synthetic_dataset.json",
        help="Path to save generated dataset",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default="synthetic_benchmark_results.json",
        help="Path to save results",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Define configurations to benchmark
    configs = [
        BenchmarkConfig(
            name="Basic",
            retriever_type="basic",
            confidence_threshold=0.3,
            top_k=5,
            semantic_coherence_check=False,
            adaptive_retrieval=False,
            use_two_stage_retrieval=False,
            query_type_adaptation=False,
            evaluation_mode=True,  # No special case handling
        ),
        BenchmarkConfig(
            name="With-Semantic-Coherence",
            retriever_type="components",
            confidence_threshold=0.3,
            top_k=5,
            semantic_coherence_check=True,
            adaptive_retrieval=False,
            use_two_stage_retrieval=False,
            query_type_adaptation=False,
            evaluation_mode=True,
        ),
        BenchmarkConfig(
            name="With-Query-Adaptation",
            retriever_type="components",
            confidence_threshold=0.3,
            top_k=5,
            semantic_coherence_check=False,
            adaptive_retrieval=False,
            use_two_stage_retrieval=False,
            query_type_adaptation=True,
            evaluation_mode=True,
        ),
        BenchmarkConfig(
            name="With-Two-Stage",
            retriever_type="components",
            confidence_threshold=0.3,
            top_k=5,
            semantic_coherence_check=False,
            adaptive_retrieval=False,
            use_two_stage_retrieval=True,
            query_type_adaptation=False,
            evaluation_mode=True,
        ),
        BenchmarkConfig(
            name="Full-Advanced",
            retriever_type="components",
            confidence_threshold=0.3,
            top_k=5,
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            use_two_stage_retrieval=True,
            query_type_adaptation=True,
            dynamic_threshold_adjustment=True,
            evaluation_mode=True,
        ),
    ]

    # Add variants with different top_k values
    configs.extend(
        [
            BenchmarkConfig(
                name="Precision-Focused",
                retriever_type="components",
                confidence_threshold=0.5,  # Higher threshold for better precision
                top_k=3,  # Fewer results
                semantic_coherence_check=True,
                adaptive_retrieval=True,
                use_two_stage_retrieval=True,
                query_type_adaptation=True,
                evaluation_mode=True,
            ),
            BenchmarkConfig(
                name="Recall-Focused",
                retriever_type="components",
                confidence_threshold=0.2,  # Lower threshold for better recall
                top_k=10,  # More results
                semantic_coherence_check=True,
                adaptive_retrieval=True,
                use_two_stage_retrieval=True,
                query_type_adaptation=True,
                evaluation_mode=True,
            ),
        ]
    )

    # Initialize benchmark
    benchmark = SyntheticBenchmark(
        configs=configs, dataset_path=args.dataset, random_seed=args.random_seed
    )

    # Generate or load dataset
    if args.dataset:
        benchmark.load_dataset(args.dataset)
    else:
        benchmark.generate_dataset(
            num_memories=args.num_memories,
            num_queries=args.num_queries,
            save_path=args.save_dataset,
        )

    # Run benchmark
    benchmark.run_benchmark(save_path=args.save_results)


if __name__ == "__main__":
    main()
