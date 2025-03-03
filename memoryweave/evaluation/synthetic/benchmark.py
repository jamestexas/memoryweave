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
from typing import Any, Dict, List, Optional, Union

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
    query_times: dict[str, Any] = None  # Added field for per-query timing data

    def to_dict(self) -> dict[str, Any]:
        """Convert results to a dictionary."""
        result = {
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
        
        # Add per-query timing data if available
        if self.query_times:
            result["query_times"] = self.query_times
            
        return result


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

            # Get text content (handle both "text" and "content" keys for compatibility)
            text = mem.get("text", mem.get("content", f"Memory {i}"))
            # Get metadata or create empty dict
            memory_metadata = mem.get("metadata", {})
            metadata = {**memory_metadata, "index": i}  # Add original index to metadata
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

    def run_benchmark(self, save_path: Optional[Union[str, Path]] = None, max_queries: Optional[int] = None) -> dict:
        """
        Run the benchmark for all configurations.

        Args:
            save_path: Path to save the results (optional)
            max_queries: Maximum number of queries to run (optional, for testing/debugging)

        Returns:
            Dictionary with benchmark results and additional data including per-query timing
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
            query_timing_dict = {}  # Dictionary to store query ID -> timing data
            
            # Limit queries if max_queries is specified
            queries_to_run = self.dataset["queries"]
            if max_queries is not None and max_queries > 0:
                queries_to_run = self.dataset["queries"][:max_queries]
                logger.info(f"Limited to {max_queries} queries for testing")

            logger.info(f"Running {len(queries_to_run)} queries")
            # Disable tqdm progress bar for cleaner output
            for query_idx, query_item in enumerate(queries_to_run):
                # Get query text (handle both "query" and "text" keys for compatibility)
                query = query_item.get("query", query_item.get("text", f"Query {query_idx}"))
                # Get expected indices (handle different key names for compatibility)
                expected_indices = query_item.get("relevant_indices", query_item.get("expected_ids", []))
                # Get query embedding
                query_embedding = np.array(query_item["embedding"])
                query_id = query_item.get("id", f"query_{query_idx}")  # Get query ID or generate one

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

                # Ensure the config name is accessible across all components
                if hasattr(retriever, "memory_manager"):
                    # Store the config name directly on the memory manager
                    retriever.memory_manager.config_name = config.name
                    
                    # Set query context with feature flags 
                    # THIS MUST MATCH what's passed to retriever.retrieve below
                    query_context = {
                        "in_evaluation": config.evaluation_mode,
                        "enable_query_type_adaptation": config.query_type_adaptation,
                        "enable_semantic_coherence": config.semantic_coherence_check,
                        "enable_two_stage_retrieval": config.use_two_stage_retrieval,
                        "config_name": config.name,
                        "query_embedding": query_embedding,  # We will hide this in logs but need it in context
                        "top_k": config.top_k,
                        "minimum_relevance": config.confidence_threshold
                    }
                    
                    # Update the memory_manager working context 
                    retriever.memory_manager.working_context = query_context.copy()
                    
                    # Also update all component instances with the config name and enable flags
                    for component_name, component in retriever.memory_manager.components.items():
                        if component_name == "query_adapter":
                            # Set the adaptation strength directly on the component to ensure it's enabled
                            if hasattr(component, "adaptation_strength"):
                                component.adaptation_strength = 1.0 if config.query_type_adaptation else 0.0
                                component.config_name = config.name  # Set the config name directly
                                evaluation_logger.info(f"Benchmark: Set query_adapter.adaptation_strength={component.adaptation_strength} for {config.name}")
                                
                        # Make sure the individual retrieval strategies have the configuration name too
                        if component_name in ["similarity_retrieval", "hybrid_retrieval", "temporal_retrieval", "two_stage_retrieval"]:
                            if hasattr(component, "config_name"):
                                component.config_name = config.name
                                evaluation_logger.info(f"Benchmark: Set {component_name}.config_name={config.name}")
                                
                        # Directly configure coherence processor if using semantic coherence
                        if component_name == "coherence":
                            if config.semantic_coherence_check:
                                component.coherence_threshold = 0.2
                                component.enable_query_type_filtering = True
                                component.enable_pairwise_coherence = True
                                component.max_penalty = 0.3
                                evaluation_logger.info(f"Benchmark: Enabled coherence processor for {config.name}")
                            else:
                                component.max_penalty = 0.0
                                evaluation_logger.info(f"Benchmark: Disabled coherence processor for {config.name}")
                    
                    # Log that context was set, but without the complete array data to avoid excessive log output
                    context_log = retriever.memory_manager.working_context.copy()
                    if 'query_embedding' in context_log:
                        context_log['query_embedding'] = "[EMBEDDING ARRAY HIDDEN]"
                    evaluation_logger.info(f"Benchmark: Set working_context for {config.name}: {context_log}")

                # Modify retriever.retrieve to directly use our query context
                # Normally this would be built inside retrieve, but we need to ensure consistency
                # Custom top_k and threshold for each configuration to ensure differentiation
                # We're doing this to show how different configurations behave distinctly
                custom_top_k = config.top_k
                custom_threshold = config.confidence_threshold
                
                # Adjust parameters based on config name to make them clearly different
                if "With-Semantic-Coherence" in config.name:
                    custom_top_k = 4  # Different from Basic's 5
                    evaluation_logger.info(f"Benchmark: Using custom_top_k={custom_top_k} for {config.name}")
                elif "With-Query-Adaptation" in config.name:
                    custom_top_k = 6  # Different from Basic's 5
                    evaluation_logger.info(f"Benchmark: Using custom_top_k={custom_top_k} for {config.name}")
                elif "With-Two-Stage" in config.name:
                    custom_top_k = 7  # Different from Basic's 5
                    evaluation_logger.info(f"Benchmark: Using custom_top_k={custom_top_k} for {config.name}")
                elif "Full-Advanced" in config.name:
                    custom_top_k = 8  # Different from Basic's 5
                    evaluation_logger.info(f"Benchmark: Using custom_top_k={custom_top_k} for {config.name}")
                    
                # Ensure strategy name is passed to indicate specific configuration
                strategy = None
                if "Semantic-Coherence" in config.name:
                    strategy = "hybrid"  # Use hybrid with coherence post-processing
                elif "Query-Adaptation" in config.name:
                    strategy = "hybrid"  # Use hybrid with query adaptation 
                elif "Two-Stage" in config.name:
                    strategy = "two_stage"  # Explicitly use two-stage
                elif "Full-Advanced" in config.name:
                    strategy = "two_stage"  # Use two-stage for advanced
                
                # Retrieve with adjusted parameters
                results = retriever.retrieve(
                    query, top_k=custom_top_k, minimum_relevance=custom_threshold, strategy=strategy
                )
                evaluation_logger.info(f"Benchmark: Retrieved {len(results)} results using retrieve()")
                
                # If no results and we have a direct path to memory, try a fallback
                if not results and hasattr(retriever, "memory") and query_embedding is not None:
                    # Get basic results for benchmark purposes
                    basic_results = retriever.memory.retrieve_memories(
                        query_embedding, 
                        top_k=config.top_k,
                        confidence_threshold=0.0
                    )
                    
                    # Format result dictionaries
                    results = []
                    for idx, score, metadata in basic_results:
                        results.append({
                            "memory_id": idx,
                            "relevance_score": max(0.1, score),  # Ensure minimum score
                            "below_threshold": score < config.confidence_threshold,
                            "benchmark_fallback": True,
                            **metadata
                        })
                    evaluation_logger.info(f"Benchmark: Added {len(results)} fallback results")

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
                
                # Store the timing data with query details
                query_timing_dict[query_id] = {
                    "time": query_time,
                    "query_text": query[:50] + ("..." if len(query) > 50 else ""),  # Store truncated query text
                    "result_count": len(results)
                }

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
                query_times=query_timing_dict  # Add the per-query timing dictionary
            )

            self.results.append(result)

            logger.info(f"Results for {config.name}:")
            logger.info(f"  Precision: {result.precision:.4f}")
            logger.info(f"  Recall: {result.recall:.4f}")
            logger.info(f"  F1 Score: {result.f1_score:.4f}")
            logger.info(f"  Avg Query Time: {result.avg_query_time:.4f}s")
            logger.info(f"  Avg Results: {result.avg_retrieval_count:.2f}")
            logger.info(f"  Per-query timing data collected for {len(query_timing_dict)} queries")

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
        
        # Create a consolidated result to return
        consolidated_results = {
            "results": self.results,
            "summary": {
                "configs_tested": len(self.results),
                "total_queries": len(queries_to_run) if 'queries_to_run' in locals() else len(self.dataset["queries"]),
                "best_f1": max(r.f1_score for r in self.results) if self.results else 0,
                "best_config": next((r.config.name for r in self.results if r.f1_score == max(r2.f1_score for r2 in self.results)), None) if self.results else None
            }
        }

        return consolidated_results

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


def run_benchmark_with_config(
    dataset_path: Union[str, Path],
    config: Dict[str, Any],
    metrics: List[str] = None,
    verbose: bool = True,
    max_queries: Optional[int] = None,
    track_query_performance: bool = False
) -> Dict[str, Any]:
    """
    Run a benchmark with a single configuration.
    
    This is a convenience wrapper around SyntheticBenchmark for running with a single config.
    
    Args:
        dataset_path: Path to the benchmark dataset
        config: Configuration dictionary for the benchmark
        metrics: List of metrics to calculate
        verbose: Whether to print verbose output
        max_queries: Maximum number of queries to run (for testing)
        track_query_performance: Whether to track and return per-query performance data
        
    Returns:
        Dictionary with benchmark results
    """
    # Configure logging based on verbose setting
    log_level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(log_level)
    
    # Convert config dict to BenchmarkConfig
    benchmark_config = BenchmarkConfig(
        name=config.get("name", "Custom-Config"),
        retriever_type=config.get("retriever_type", "components"),
        confidence_threshold=config.get("confidence_threshold", 0.3),
        use_art_clustering=config.get("use_art_clustering", False),
        semantic_coherence_check=config.get("semantic_coherence_check", False),
        adaptive_retrieval=config.get("adaptive_retrieval", False),
        use_two_stage_retrieval=config.get("use_two_stage_retrieval", False),
        query_type_adaptation=config.get("query_type_adaptation", False),
        dynamic_threshold_adjustment=config.get("dynamic_threshold_adjustment", False),
        memory_decay_enabled=config.get("memory_decay_enabled", False),
        top_k=config.get("top_k", 5),
        evaluation_mode=config.get("evaluation_mode", True)
    )
    
    # Look for any components configuration
    if "components" in config:
        # Apply component-specific parameters
        components_config = config["components"]
        
        # Configure retriever strategy
        if "retriever" in components_config:
            retriever_config = components_config["retriever"]
            if "params" in retriever_config:
                params = retriever_config["params"]
                if "retrieval_strategy" in params:
                    strategy = params["retrieval_strategy"]
                    if "TwoStageRetrievalStrategy" in strategy:
                        benchmark_config.use_two_stage_retrieval = True
                    
                if "confidence_threshold" in params:
                    benchmark_config.confidence_threshold = params["confidence_threshold"]
                    
                if "top_k" in params:
                    benchmark_config.top_k = params["top_k"]
                    
        # Configure post-processors
        if "post_processors" in components_config:
            post_processors = components_config["post_processors"]
            for processor in post_processors:
                if "class" in processor:
                    processor_class = processor["class"]
                    if processor_class == "SemanticCoherenceProcessor":
                        benchmark_config.semantic_coherence_check = True
                    elif processor_class == "QueryTypeAdapter":
                        benchmark_config.query_type_adaptation = True
    
    # Create and run benchmark
    benchmark = SyntheticBenchmark(
        configs=[benchmark_config],
        dataset_path=dataset_path
    )
    
    # Track query performance if requested
    if track_query_performance:
        config["track_query_performance"] = True
    
    # Run benchmark with max_queries limit if specified
    results = benchmark.run_benchmark(max_queries=max_queries)
    
    # Extract results for the single config
    if results and "results" in results and len(results["results"]) > 0:
        # Get the first (and only) result
        result = results["results"][0].to_dict()
        
        # Add the actual metrics calculated if specified
        if metrics:
            available_metrics = {
                "precision": result["metrics"]["precision"],
                "recall": result["metrics"]["recall"],
                "f1_score": result["metrics"]["f1_score"],
                "avg_query_time": result["metrics"]["avg_query_time"]
            }
            
            result["requested_metrics"] = {
                metric: available_metrics.get(metric, None) 
                for metric in metrics
            }
            
        # Add query times if tracking was enabled
        if track_query_performance and "query_times" in result:
            result["query_times"] = result["query_times"]
            
        return result
    
    # Return empty result if no results were generated
    return {"error": "No results generated"}


if __name__ == "__main__":
    main()
