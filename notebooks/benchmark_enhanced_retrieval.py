"""
Benchmark script for MemoryWeave enhanced retrieval mechanisms.

This script benchmarks different configurations of the enhanced retrieval mechanisms,
focusing on the balance between precision and recall for different query types.
"""

import json
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from memoryweave.core import ContextualMemory, ContextualRetriever, MemoryEncoder
from memoryweave.utils.analysis import analyze_query_similarities, analyze_retrieval_performance

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)


# Helper class for sentence embedding
class EmbeddingModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, text):
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

        return mean_pooled.numpy()[0]


class EnhancedRetrievalBenchmark:
    """Benchmark different configurations of the enhanced retrieval mechanisms."""

    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the benchmark.

        Args:
            embedding_model_name: Name of the embedding model to use
        """
        print(f"Loading embedding model: {embedding_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        model = AutoModel.from_pretrained(embedding_model_name)
        self.embedding_model = EmbeddingModelWrapper(model, tokenizer)
        self.embedding_dim = model.config.hidden_size

        # Datasets
        self.memory_data = None
        self.test_queries = None
        self.relevance_judgments = None

        # Results
        self.results = defaultdict(list)

    def load_test_data(self, file_path="datasets/evaluation_queries.json"):
        """
        Load test data from the evaluation queries file.

        Args:
            file_path: Path to the evaluation queries JSON file
        """
        with open(file_path) as f:
            data = json.load(f)

        # Prepare memory data and test queries
        memory_data = []
        test_queries = []
        relevance_judgments = {}

        for idx, item in enumerate(data):
            query = item["query"]
            expected = item["expected_answer"]
            category = item["category"]

            # Add to memory data
            memory_data.append((expected, category, query.split()[0]))

            # Add to test queries
            test_queries.append((query, category))

            # Add relevance judgment (each query is relevant to its own expected answer)
            relevance_judgments[(len(test_queries) - 1, len(memory_data) - 1)] = 1.0

        self.memory_data = memory_data
        self.test_queries = test_queries
        self.relevance_judgments = relevance_judgments

        print(f"Loaded {len(self.memory_data)} memories and {len(self.test_queries)} test queries")

    def benchmark_configuration(
        self,
        config_name: str,
        use_art_clustering: bool = False,
        confidence_threshold: float = 0.0,
        semantic_coherence_check: bool = False,
        adaptive_retrieval: bool = False,
        adaptive_k_factor: float = 0.3,
        use_two_stage_retrieval: bool = False,
        first_stage_k: int = 20,
        query_type_adaptation: bool = False,
        dynamic_threshold_adjustment: bool = False,
        memory_decay_enabled: bool = False,
    ):
        """
        Benchmark a specific configuration of the enhanced retrieval mechanisms.

        Args:
            config_name: Name to identify this configuration in results
            use_art_clustering: Whether to use ART-inspired clustering
            confidence_threshold: Minimum similarity score for retrieval
            semantic_coherence_check: Whether to check semantic coherence
            adaptive_retrieval: Whether to adaptively select k
            adaptive_k_factor: Factor to control how conservative adaptive K selection is
            use_two_stage_retrieval: Whether to use two-stage retrieval pipeline
            first_stage_k: Number of candidates to retrieve in first stage
            query_type_adaptation: Whether to adapt retrieval based on query type
            dynamic_threshold_adjustment: Whether to dynamically adjust thresholds
            memory_decay_enabled: Whether to apply memory decay
        """
        print(f"\nBenchmarking configuration: {config_name}")

        # Initialize memory system
        memory = ContextualMemory(
            embedding_dim=self.embedding_dim,
            use_art_clustering=use_art_clustering,
            default_confidence_threshold=confidence_threshold,
            semantic_coherence_check=semantic_coherence_check,
            adaptive_retrieval=adaptive_retrieval,
        )

        encoder = MemoryEncoder(self.embedding_model)

        retriever = ContextualRetriever(
            memory=memory,
            embedding_model=self.embedding_model,
            retrieval_strategy="hybrid",
            confidence_threshold=confidence_threshold,
            semantic_coherence_check=semantic_coherence_check,
            adaptive_retrieval=adaptive_retrieval,
            adaptive_k_factor=adaptive_k_factor,
            use_two_stage_retrieval=use_two_stage_retrieval,
            first_stage_k=first_stage_k,
            query_type_adaptation=query_type_adaptation,
            dynamic_threshold_adjustment=dynamic_threshold_adjustment,
            memory_decay_enabled=memory_decay_enabled,
        )

        # Populate memory
        memory_start_time = time.time()
        for mem_idx, (text, category, subcategory) in enumerate(self.memory_data):
            # Encode memory
            embedding, metadata = encoder.encode_concept(
                concept=category, description=text, related_concepts=[subcategory]
            )

            # Add to memory
            memory.add_memory(embedding, text, metadata)

        memory_time = time.time() - memory_start_time
        print(f"Memory population time: {memory_time:.2f} seconds")

        # Benchmark retrieval
        retrieval_start_time = time.time()
        precision_at_k = []
        recall_at_k = []
        f1_at_k = []
        retrieval_times = []

        # Track metrics by query category
        category_metrics = defaultdict(lambda: {"precision": [], "recall": [], "f1": []})

        for query_idx, (query_text, query_category) in enumerate(self.test_queries):
            # Time retrieval
            query_start_time = time.time()
            retrieved = retriever.retrieve_for_context(query_text, top_k=5)
            query_time = time.time() - query_start_time
            retrieval_times.append(query_time)

            # Calculate precision and recall
            retrieved_indices = [item.get("memory_id") for item in retrieved]
            retrieved_indices = [idx for idx in retrieved_indices if isinstance(idx, int)]

            # Get relevant memories for this query
            relevant_indices = [
                memory_idx
                for memory_idx in range(len(self.memory_data))
                if self.relevance_judgments.get((query_idx, memory_idx), 0.0) > 0.5
            ]

            # Precision: how many retrieved items are relevant
            if len(retrieved_indices) > 0:
                precision = sum(1 for idx in retrieved_indices if idx in relevant_indices) / len(
                    retrieved_indices
                )
            else:
                precision = 0.0

            # Recall: how many relevant items were retrieved
            if len(relevant_indices) > 0:
                recall = sum(1 for idx in retrieved_indices if idx in relevant_indices) / len(
                    relevant_indices
                )
            else:
                recall = 1.0  # No relevant items, so perfect recall

            # Calculate F1 score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            precision_at_k.append(precision)
            recall_at_k.append(recall)
            f1_at_k.append(f1)

            # Add to category metrics
            category_metrics[query_category]["precision"].append(precision)
            category_metrics[query_category]["recall"].append(recall)
            category_metrics[query_category]["f1"].append(f1)

            # Print some results
            if query_idx < 3:  # Only print the first few queries to avoid clutter
                print(f"\nQuery: {query_text}")
                print(f"Retrieved {len(retrieved)} memories in {query_time:.3f} seconds")
                print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

                # Print top 3 retrieved memories
                for i, mem in enumerate(retrieved[:3]):
                    relevance = "✓" if mem.get("memory_id") in relevant_indices else "✗"
                    print(
                        f"  {i + 1}. [{relevance}] {mem.get('text', '')[:50]}... (Score: {mem.get('relevance_score', 0):.3f})"
                    )

        retrieval_time = time.time() - retrieval_start_time
        print(f"Total retrieval time: {retrieval_time:.2f} seconds")

        # Calculate average metrics
        avg_precision = np.mean(precision_at_k)
        avg_recall = np.mean(recall_at_k)
        avg_f1 = np.mean(f1_at_k)
        avg_retrieval_time = np.mean(retrieval_times)

        print(f"Average precision: {avg_precision:.3f}")
        print(f"Average recall: {avg_recall:.3f}")
        print(f"Average F1 score: {avg_f1:.3f}")
        print(f"Average retrieval time: {avg_retrieval_time:.3f} seconds per query")

        # Calculate category-specific metrics
        category_avg_metrics = {}
        for category, metrics in category_metrics.items():
            category_avg_metrics[category] = {
                "precision": float(np.mean(metrics["precision"])),
                "recall": float(np.mean(metrics["recall"])),
                "f1": float(np.mean(metrics["f1"])),
            }
            print(f"\n{category.capitalize()} queries:")
            print(f"  Precision: {category_avg_metrics[category]['precision']:.3f}")
            print(f"  Recall: {category_avg_metrics[category]['recall']:.3f}")
            print(f"  F1: {category_avg_metrics[category]['f1']:.3f}")

        # Store results
        self.results["configuration"].append(config_name)
        self.results["avg_precision"].append(avg_precision)
        self.results["avg_recall"].append(avg_recall)
        self.results["avg_f1"].append(avg_f1)
        self.results["avg_retrieval_time"].append(avg_retrieval_time)
        self.results["memory_time"].append(memory_time)
        self.results["use_art_clustering"].append(use_art_clustering)
        self.results["confidence_threshold"].append(confidence_threshold)
        self.results["semantic_coherence_check"].append(semantic_coherence_check)
        self.results["adaptive_retrieval"].append(adaptive_retrieval)
        self.results["adaptive_k_factor"].append(adaptive_k_factor if adaptive_retrieval else None)
        self.results["use_two_stage_retrieval"].append(use_two_stage_retrieval)
        self.results["first_stage_k"].append(first_stage_k if use_two_stage_retrieval else None)
        self.results["query_type_adaptation"].append(query_type_adaptation)
        self.results["dynamic_threshold_adjustment"].append(dynamic_threshold_adjustment)
        self.results["memory_decay_enabled"].append(memory_decay_enabled)
        self.results["category_metrics"].append(category_avg_metrics)

    def generate_report(self, output_file="enhanced_retrieval_benchmark"):
        """
        Generate a report of the benchmark results.

        Args:
            output_file: File to save the report to
        """
        if not self.results:
            print("No results to report. Run benchmark_configuration first.")
            return

        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)

        # Print summary table
        print("\n=== Benchmark Results ===")

        # Sort by F1 score
        sorted_df = results_df.sort_values("avg_f1", ascending=False)
        print(
            sorted_df[
                ["configuration", "avg_precision", "avg_recall", "avg_f1", "avg_retrieval_time"]
            ].to_string(index=False)
        )

        # Create precision vs. recall plot with F1 contours
        plt.figure(figsize=(12, 8))

        # Create F1 score contours
        x = np.linspace(0.01, 1, 100)
        y = np.linspace(0.01, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = 2 * X * Y / (X + Y)

        # Plot F1 contours
        CS = plt.contour(
            X,
            Y,
            Z,
            levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            alpha=0.3,
            linestyles="dashed",
            colors="gray",
        )
        plt.clabel(CS, inline=True, fontsize=9)

        # Plot data points
        scatter = plt.scatter(
            results_df["avg_recall"],
            results_df["avg_precision"],
            s=150,
            alpha=0.8,
            c=results_df["avg_f1"],
            cmap="viridis",
        )

        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("F1 Score")

        # Add labels to each point
        for i, config in enumerate(results_df["configuration"]):
            plt.annotate(
                config,
                (results_df["avg_recall"][i], results_df["avg_precision"][i]),
                xytext=(7, 0),
                textcoords="offset points",
                fontsize=9,
            )

        plt.xlabel("Average Recall")
        plt.ylabel("Average Precision")
        plt.title("Precision vs. Recall for Different Configurations")
        plt.grid(True, alpha=0.3)

        # Set axis limits with some padding
        plt.xlim(
            max(0, min(results_df["avg_recall"]) - 0.05),
            min(1, max(results_df["avg_recall"]) + 0.05),
        )
        plt.ylim(
            max(0, min(results_df["avg_precision"]) - 0.05),
            min(1, max(results_df["avg_precision"]) + 0.05),
        )

        # Save the plot
        output_path = f"output/{output_file}_precision_recall.png"
        plt.savefig(output_path)
        print(f"Saved precision-recall plot to {output_path}")

        # Create category-specific performance plot
        plt.figure(figsize=(15, 10))

        # Get unique categories
        categories = list(self.results["category_metrics"][0].keys())
        
        # Create subplots for each category
        for i, category in enumerate(categories):
            plt.subplot(len(categories), 1, i + 1)
            
            # Extract category metrics for each configuration
            precisions = [metrics[category]["precision"] for metrics in self.results["category_metrics"]]
            recalls = [metrics[category]["recall"] for metrics in self.results["category_metrics"]]
            f1s = [metrics[category]["f1"] for metrics in self.results["category_metrics"]]
            
            # Create bar positions
            x = np.arange(len(self.results["configuration"]))
            width = 0.25
            
            # Plot bars
            plt.bar(x - width, precisions, width, label="Precision")
            plt.bar(x, recalls, width, label="Recall")
            plt.bar(x + width, f1s, width, label="F1")
            
            # Add labels and title
            plt.xlabel("Configuration")
            plt.ylabel("Score")
            plt.title(f"{category.capitalize()} Query Performance")
            plt.xticks(x, self.results["configuration"], rotation=45, ha="right")
            plt.legend()
            plt.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
        
        # Save the plot
        output_path = f"output/{output_file}_category_performance.png"
        plt.savefig(output_path)
        print(f"Saved category performance plot to {output_path}")

        # Save results to CSV
        csv_path = f"output/{output_file}_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")

        # Save detailed results to JSON
        json_path = f"output/{output_file}_detailed_results.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved detailed results to {json_path}")

        return results_df


def run_benchmark():
    """
    Run the benchmark with various configurations of the enhanced retrieval mechanisms.
    """
    benchmark = EnhancedRetrievalBenchmark()
    benchmark.load_test_data()

    # Test baseline configuration
    benchmark.benchmark_configuration(
        config_name="Baseline",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        use_two_stage_retrieval=False,
        query_type_adaptation=False,
    )

    # Test two-stage retrieval
    benchmark.benchmark_configuration(
        config_name="Two-Stage Retrieval",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        use_two_stage_retrieval=True,
        first_stage_k=20,
        query_type_adaptation=False,
    )

    # Test query type adaptation
    benchmark.benchmark_configuration(
        config_name="Query Type Adaptation",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        use_two_stage_retrieval=False,
        query_type_adaptation=True,
    )

    # Test adaptive K selection
    benchmark.benchmark_configuration(
        config_name="Adaptive K (Conservative)",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=True,
        adaptive_k_factor=0.3,
        use_two_stage_retrieval=False,
        query_type_adaptation=False,
    )

    # Test adaptive K selection (less conservative)
    benchmark.benchmark_configuration(
        config_name="Adaptive K (Balanced)",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=True,
        adaptive_k_factor=0.15,
        use_two_stage_retrieval=False,
        query_type_adaptation=False,
    )

    # Test semantic coherence check
    benchmark.benchmark_configuration(
        config_name="Semantic Coherence",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=True,
        adaptive_retrieval=False,
        use_two_stage_retrieval=False,
        query_type_adaptation=False,
    )

    # Test dynamic threshold adjustment
    benchmark.benchmark_configuration(
        config_name="Dynamic Threshold",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        use_two_stage_retrieval=False,
        query_type_adaptation=False,
        dynamic_threshold_adjustment=True,
    )

    # Test memory decay
    benchmark.benchmark_configuration(
        config_name="Memory Decay",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        use_two_stage_retrieval=False,
        query_type_adaptation=False,
        memory_decay_enabled=True,
    )

    # Test combined approach (two-stage + query adaptation)
    benchmark.benchmark_configuration(
        config_name="Combined Approach",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=True,
        adaptive_retrieval=True,
        adaptive_k_factor=0.15,
        use_two_stage_retrieval=True,
        first_stage_k=20,
        query_type_adaptation=True,
    )

    # Test optimized combined approach
    benchmark.benchmark_configuration(
        config_name="Optimized Combined",
        use_art_clustering=True,
        confidence_threshold=0.2,
        semantic_coherence_check=True,
        adaptive_retrieval=True,
        adaptive_k_factor=0.15,
        use_two_stage_retrieval=True,
        first_stage_k=30,
        query_type_adaptation=True,
        dynamic_threshold_adjustment=True,
        memory_decay_enabled=True,
    )

    # Generate report
    benchmark.generate_report()


if __name__ == "__main__":
    run_benchmark()
