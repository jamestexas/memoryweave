"""
Benchmark script for MemoryWeave retrievers.

This script benchmarks different configurations of the MemoryWeave system,
focusing on retrieval quality and performance metrics.
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


class MemoryBenchmark:
    """Benchmark different configurations of the MemoryWeave system."""

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

        # Error analysis tracking
        self.query_details = defaultdict(list)

    def load_test_data(
        self,
        memory_data=None,
        test_queries=None,
        relevance_judgments=None,
        queries_file="datasets/evaluation_queries.json",
    ):
        """
        Load test data for benchmarking.

        If not provided, generate synthetic test data.

        Args:
            memory_data: list of (text, category, subcategory) tuples for memory items
            test_queries: list of (query_text, category) tuples for test queries
            relevance_judgments: dict mapping (query_idx, memory_idx) to relevance score
            queries_file: Path to the JSON file containing evaluation queries
        """
        if memory_data is None:
            # Load memory data from queries file
            print("Loading memory data from queries file...")
            self.memory_data = self._load_memories_from_queries(queries_file)
        else:
            self.memory_data = memory_data

        if test_queries is None:
            # Load test queries from queries file
            print("Loading test queries from queries file...")
            self.test_queries = self._load_queries_from_json(queries_file)
        else:
            self.test_queries = test_queries

        if relevance_judgments is None:
            # Generate synthetic relevance judgments
            print("Generating synthetic relevance judgments...")
            self.relevance_judgments = self._generate_synthetic_relevance()
        else:
            self.relevance_judgments = relevance_judgments

        print(f"Loaded {len(self.memory_data)} memories and {len(self.test_queries)} test queries")

    def _load_memories_from_queries(self, queries_file):
        """Load memory data from a JSON file."""
        with open(queries_file) as f:
            data = json.load(f)

        memories = []
        for item in data:
            memories.append((
                item["expected_answer"],
                item["category"],
                "general",
            ))  # Adjust as needed

        return memories

    def _load_queries_from_json(self, queries_file):
        """Load test queries from a JSON file."""
        with open(queries_file) as f:
            data = json.load(f)

        queries = []
        for item in data:
            queries.append((item["query"], item["category"]))

        return queries

    def _generate_synthetic_memories(self) -> list[tuple[str, str, str]]:
        """Generate synthetic memory data for testing."""
        # Categories and subcategories for synthetic data
        categories = {
            "technology": ["programming", "ai", "hardware", "software", "gadgets"],
            "science": ["physics", "biology", "chemistry", "astronomy", "medicine"],
            "arts": ["music", "painting", "literature", "film", "photography"],
            "history": ["ancient", "medieval", "modern", "war", "politics"],
            "personal": ["preferences", "experiences", "opinions", "relationships", "habits"],
        }

        memories = []

        # Generate memories for each category and subcategory
        for category, subcategories in categories.items():
            for subcategory in subcategories:
                # Generate multiple memories per subcategory
                for i in range(5):  # 5 memories per subcategory
                    text = f"This is a {subcategory} memory about {category} (#{i + 1})"
                    memories.append((text, category, subcategory))

        return memories

    def _generate_synthetic_queries(self) -> list[tuple[str, str]]:
        """Generate synthetic test queries."""
        # Extract unique categories from memory data
        categories = set(category for _, category, _ in self.memory_data)

        queries = []

        # Generate queries for each category
        for category in categories:
            # Direct query
            queries.append((f"Tell me about {category}", category))
            # Question query
            queries.append((f"What do you know about {category}?", category))
            # Specific query
            queries.append((f"I'm interested in learning more about {category}", category))

        # Add some cross-category queries
        if len(categories) >= 2:
            categories_list = list(categories)
            queries.append((
                f"Compare {categories_list[0]} and {categories_list[1]}",
                "cross-category",
            ))

        return queries

    def _generate_synthetic_relevance(self) -> dict[tuple[int, int], float]:
        """Generate synthetic relevance judgments."""
        relevance = {}

        # For each query
        for query_idx, (query_text, query_category) in enumerate(self.test_queries):
            # For each memory
            for memory_idx, (_memory_text, memory_category, _memory_subcategory) in enumerate(
                self.memory_data
            ):
                # Calculate relevance based on category match
                if query_category == memory_category:
                    relevance[(query_idx, memory_idx)] = 1.0  # Highly relevant
                elif query_category == "cross-category" and memory_category in query_text:
                    relevance[(query_idx, memory_idx)] = 0.8  # Relevant for cross-category queries
                else:
                    relevance[(query_idx, memory_idx)] = 0.0  # Not relevant

        return relevance

    def benchmark_configuration(
        self,
        config_name: str,
        use_art_clustering: bool = False,
        confidence_threshold: float = 0.0,
        semantic_coherence_check: bool = False,
        adaptive_retrieval: bool = False,
        enable_category_consolidation: bool = False,
        retrieval_strategy: str = "hybrid",
        adaptive_k_factor: float = 0.3,  # Added parameter to control adaptive K conservativeness
        use_two_stage_retrieval: bool = False,  # New parameter for two-stage retrieval
        query_type_adaptation: bool = False,  # New parameter for query type adaptation
    ):
        """
        Benchmark a specific configuration of the memory system.

        Args:
            config_name: Name to identify this configuration in results
            use_art_clustering: Whether to use ART-inspired clustering
            confidence_threshold: Minimum similarity score for retrieval
            semantic_coherence_check: Whether to check semantic coherence
            adaptive_retrieval: Whether to adaptively select k
            enable_category_consolidation: Whether to enable category consolidation
            retrieval_strategy: Retrieval strategy to use
            adaptive_k_factor: Factor to control how conservative adaptive K selection is (lower = more results)
            use_two_stage_retrieval: Whether to use a two-stage retrieval pipeline
            query_type_adaptation: Whether to adapt query type based on context
            use_two_stage_retrieval: Whether to use two-stage retrieval pipeline
            query_type_adaptation: Whether to adapt retrieval based on query type
        """
        print(f"\nBenchmarking configuration: {config_name}")

        # Initialize memory system
        memory = ContextualMemory(
            embedding_dim=self.embedding_dim,
            use_art_clustering=use_art_clustering,
            default_confidence_threshold=confidence_threshold,
            semantic_coherence_check=semantic_coherence_check,
            adaptive_retrieval=adaptive_retrieval,
            enable_category_consolidation=enable_category_consolidation,
        )

        encoder = MemoryEncoder(self.embedding_model)

        retriever = ContextualRetriever(
            memory=memory,
            embedding_model=self.embedding_model,
            retrieval_strategy=retrieval_strategy,
            confidence_threshold=confidence_threshold,
            semantic_coherence_check=semantic_coherence_check,
            adaptive_retrieval=adaptive_retrieval,
            adaptive_k_factor=adaptive_k_factor,  # Pass the adaptive k factor to the retriever
            use_two_stage_retrieval=use_two_stage_retrieval,  # Pass the two-stage retrieval flag
            query_type_adaptation=query_type_adaptation,  # Pass the query type adaptation flag
        )

        # Populate memory
        memory_start_time = time.time()
        for _mem_idx, (text, category, subcategory) in enumerate(self.memory_data):
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
        f1_at_k = []  # Added F1 score tracking
        retrieval_times = []
        query_results = []  # Track detailed results for each query

        for query_idx, (query_text, _query_category) in enumerate(self.test_queries):
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
            f1_at_k.append(f1)  # Store F1 score

            # Detailed tracking for error analysis
            correct_retrieved = [idx for idx in retrieved_indices if idx in relevant_indices]
            missed_relevant = [idx for idx in relevant_indices if idx not in retrieved_indices]
            false_positives = [idx for idx in retrieved_indices if idx not in relevant_indices]

            query_details = {
                "query": query_text,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "retrieved_count": len(retrieved_indices),
                "relevant_count": len(relevant_indices),
                "correct_retrieved": correct_retrieved,
                "missed_relevant": missed_relevant,
                "false_positives": false_positives,
                "retrieval_time": query_time,
            }
            query_results.append(query_details)

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
        avg_f1 = np.mean(f1_at_k)  # Calculate average F1
        avg_retrieval_time = np.mean(retrieval_times)

        print(f"Average precision: {avg_precision:.3f}")
        print(f"Average recall: {avg_recall:.3f}")
        print(f"Average F1 score: {avg_f1:.3f}")
        print(f"Average retrieval time: {avg_retrieval_time:.3f} seconds per query")

        # Find problematic queries (low F1 score)
        if len(query_results) > 0:
            print("\nTop 3 Problematic Queries:")
            sorted_queries = sorted(query_results, key=lambda x: x["f1"])
            for idx, q in enumerate(sorted_queries[:3]):
                print(f'  {idx + 1}. Query: "{q["query"]}"')
                print(
                    f"     Precision: {q['precision']:.2f}, Recall: {q['recall']:.2f}, F1: {q['f1']:.2f}"
                )
                print(f"     Retrieved: {q['retrieved_count']}, Relevant: {q['relevant_count']}")
                if q["missed_relevant"]:
                    missed_texts = [
                        self.memory_data[idx][0][:40] + "..." for idx in q["missed_relevant"][:2]
                    ]
                    print(
                        f"     Missed relevant items: {len(q['missed_relevant'])}, e.g.: {', '.join(missed_texts)}"
                    )

        # Store results
        self.results["configuration"].append(config_name)
        self.results["avg_precision"].append(avg_precision)
        self.results["avg_recall"].append(avg_recall)
        self.results["avg_f1"].append(avg_f1)  # Store average F1
        self.results["avg_retrieval_time"].append(avg_retrieval_time)
        self.results["memory_time"].append(memory_time)
        self.results["use_art_clustering"].append(use_art_clustering)
        self.results["confidence_threshold"].append(confidence_threshold)
        self.results["semantic_coherence_check"].append(semantic_coherence_check)
        self.results["adaptive_retrieval"].append(adaptive_retrieval)
        self.results["enable_category_consolidation"].append(enable_category_consolidation)
        self.results["retrieval_strategy"].append(retrieval_strategy)
        self.results["adaptive_k_factor"].append(adaptive_k_factor if adaptive_retrieval else None)
        self.results["use_two_stage_retrieval"].append(use_two_stage_retrieval)
        self.results["query_type_adaptation"].append(query_type_adaptation)

        # Store detailed query results for later analysis
        self.query_details[config_name] = query_results

    def generate_report(self, output_file=None):
        """
        Generate a report of the benchmark results.

        Args:
            output_file: Optional file to save the report to
        """
        if not self.results:
            print("No results to report. Run benchmark_configuration first.")
            return

        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)

        # Print summary table
        print("\n=== Benchmark Results ===")

        # Sort by F1 score (combining precision and recall)
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
        X, Y = np.meshgrid(x, y)  # noqa: N806
        Z = 2 * X * Y / (X + Y)  # noqa: N806

        # Plot F1 contours
        CS = plt.contour(  # noqa: N806
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
            c=results_df["avg_f1"],  # Color by F1 score
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

        # Save or show the plot
        if output_file:
            output_path = f"output/{output_file}_precision_recall.png"
            plt.savefig(output_path)
            print(f"Saved precision-recall plot to {output_path}")

            # Also save the results data
            csv_path = f"output/{output_file}_results.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")
        else:
            output_path = "output/benchmark_precision_recall.png"
            plt.savefig(output_path)
            print(f"Saved precision-recall plot to {output_path}")

            # Also save the results data
            csv_path = "output/benchmark_results.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")

        # Create a bar chart comparing retrieval times
        plt.figure(figsize=(12, 6))
        bar_positions = np.arange(len(results_df["configuration"]))

        plt.bar(bar_positions, results_df["avg_retrieval_time"], alpha=0.7)
        plt.xticks(bar_positions, results_df["configuration"], rotation=45, ha="right")
        plt.ylabel("Average Retrieval Time (seconds)")
        plt.title("Retrieval Performance for Different Configurations")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        # Save or show the plot
        if output_file:
            output_path = f"output/{output_file}_retrieval_time.png"
            plt.savefig(output_path)
            print(f"Saved retrieval time plot to {output_path}")
        else:
            output_path = "output/benchmark_retrieval_time.png"
            plt.savefig(output_path)
            print(f"Saved retrieval time plot to {output_path}")

        # Generate a detailed error analysis report for problematic queries
        self._generate_error_analysis(output_file)

        return results_df

    def _generate_error_analysis(self, output_file=None):
        """Generate detailed error analysis for problematic queries."""
        if not self.query_details:
            return

        f1_by_query = defaultdict(dict)
        missing_by_query = defaultdict(dict)

        # Collect F1 scores and missing items by query across configurations
        for config_name, query_results in self.query_details.items():
            for query_result in query_results:
                query = query_result["query"]
                f1_by_query[query][config_name] = query_result["f1"]
                missing_by_query[query][config_name] = len(query_result["missed_relevant"])

        # Find the most problematic queries across all configurations
        avg_f1_by_query = {
            query: np.mean(list(configs.values())) for query, configs in f1_by_query.items()
        }
        problematic_queries = sorted(avg_f1_by_query.items(), key=lambda x: x[1])[:10]

        # Create error analysis report
        plt.figure(figsize=(14, 8))

        # Plot F1 scores for problematic queries across configurations
        query_names = [q[0][:30] + "..." if len(q[0]) > 30 else q[0] for q in problematic_queries]
        config_names = list(self.results["configuration"])

        # Create data for heatmap
        heatmap_data = np.zeros((len(query_names), len(config_names)))
        for i, (query, _) in enumerate(problematic_queries):
            for j, config in enumerate(config_names):
                heatmap_data[i, j] = f1_by_query[query].get(config, 0)

        # Plot heatmap
        plt.imshow(heatmap_data, cmap="YlGnBu", aspect="auto")
        plt.colorbar(label="F1 Score")

        # Add labels
        plt.xticks(np.arange(len(config_names)), config_names, rotation=45, ha="right")
        plt.yticks(np.arange(len(query_names)), query_names)

        plt.title("F1 Scores for Problematic Queries Across Configurations")
        plt.tight_layout()

        # Save or show the plot
        if output_file:
            output_path = f"output/{output_file}_error_analysis.png"
            plt.savefig(output_path)
            print(f"Saved error analysis to {output_path}")

            # Save detailed error data
            error_data = {
                "problematic_queries": {q: {"avg_f1": f} for q, f in problematic_queries},
                "f1_by_config": f1_by_query,
                "missing_items_by_config": missing_by_query,
            }

            json_path = f"output/{output_file}_error_analysis.json"
            with open(json_path, "w") as f:
                json.dump(error_data, f, indent=2)
            print(f"Saved detailed error analysis to {json_path}")
        else:
            output_path = "output/benchmark_error_analysis.png"
            plt.savefig(output_path)
            print(f"Saved error analysis to {output_path}")


def run_benchmark():
    """
    Run the benchmark with various configurations.
    """
    benchmark = MemoryBenchmark()
    benchmark.load_test_data()

    # Test baseline configuration
    benchmark.benchmark_configuration(
        config_name="Baseline",
        use_art_clustering=False,
        confidence_threshold=0.0,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
    )

    # Test confidence thresholding
    benchmark.benchmark_configuration(
        config_name="Confidence Threshold (0.3)",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
    )

    # Test confidence thresholding + semantic coherence
    benchmark.benchmark_configuration(
        config_name="Conf + Semantic Coherence",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=True,
        adaptive_retrieval=False,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
    )

    # Test adaptive retrieval with less conservative setting
    benchmark.benchmark_configuration(
        config_name="Adaptive Retrieval (Conservative)",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=True,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.3,  # Original, more conservative setting
    )

    # Test adaptive retrieval with less aggressive filtering
    benchmark.benchmark_configuration(
        config_name="Adaptive Retrieval (Balanced)",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=True,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.15,  # Less conservative setting
    )

    # Test adaptive retrieval with more aggressive filtering
    benchmark.benchmark_configuration(
        config_name="Adaptive Retrieval (Liberal)",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=True,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.05,  # Least conservative setting
    )

    # Test ART clustering
    benchmark.benchmark_configuration(
        config_name="ART Clustering",
        use_art_clustering=True,
        confidence_threshold=0.0,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
    )

    # Test ART clustering + consolidation
    benchmark.benchmark_configuration(
        config_name="ART + Consolidation",
        use_art_clustering=True,
        confidence_threshold=0.0,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        enable_category_consolidation=True,
        retrieval_strategy="hybrid",
    )

    # Test original full featured configuration
    benchmark.benchmark_configuration(
        config_name="Full Features (Conservative)",
        use_art_clustering=True,
        confidence_threshold=0.3,
        semantic_coherence_check=True,
        adaptive_retrieval=True,
        enable_category_consolidation=True,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.3,  # Original conservative setting
    )

    # Test full featured with balanced settings
    benchmark.benchmark_configuration(
        config_name="Full Features (Balanced)",
        use_art_clustering=True,
        confidence_threshold=0.2,  # Lower threshold
        semantic_coherence_check=True,
        adaptive_retrieval=True,
        enable_category_consolidation=True,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.15,  # More balanced setting
    )

    # Test full featured with more recall-oriented settings
    benchmark.benchmark_configuration(
        config_name="Full Features (Recall-Focused)",
        use_art_clustering=True,
        confidence_threshold=0.15,  # Lower threshold
        semantic_coherence_check=True,
        adaptive_retrieval=True,
        enable_category_consolidation=True,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.05,  # Setting for more recall
    )
    
    # NEW: Test two-stage retrieval pipeline
    benchmark.benchmark_configuration(
        config_name="Two-Stage Retrieval",
        use_art_clustering=True,
        confidence_threshold=0.15,  # Lower initial threshold
        semantic_coherence_check=True,
        adaptive_retrieval=True,
        enable_category_consolidation=True,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.1,  # Balanced setting
        use_two_stage_retrieval=True,  # Enable two-stage retrieval
        query_type_adaptation=False,
    )
    
    # NEW: Test query type adaptation
    benchmark.benchmark_configuration(
        config_name="Query Type Adaptation",
        use_art_clustering=True,
        confidence_threshold=0.25,  # Base threshold
        semantic_coherence_check=True,
        adaptive_retrieval=True,
        enable_category_consolidation=True,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.2,  # Balanced setting
        use_two_stage_retrieval=False,
        query_type_adaptation=True,  # Enable query type adaptation
    )
    
    # NEW: Test combined approach (two-stage + query adaptation)
    benchmark.benchmark_configuration(
        config_name="Combined Approach",
        use_art_clustering=True,
        confidence_threshold=0.2,  # Moderate threshold
        semantic_coherence_check=True,
        adaptive_retrieval=True,
        enable_category_consolidation=True,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.15,  # Balanced setting
        use_two_stage_retrieval=True,  # Enable two-stage retrieval
        query_type_adaptation=True,  # Enable query type adaptation
    )

    # NEW: Test two-stage retrieval pipeline
    benchmark.benchmark_configuration(
        config_name="Two-Stage Retrieval",
        use_art_clustering=True,
        confidence_threshold=0.15,  # Lower initial threshold
        semantic_coherence_check=True,
        adaptive_retrieval=True,
        enable_category_consolidation=True,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.1,  # Balanced setting
    )
    
    # NEW: Test query type adaptation
    benchmark.benchmark_configuration(
        config_name="Query Type Adaptation",
        use_art_clustering=True,
        confidence_threshold=0.25,  # Base threshold
        semantic_coherence_check=True,
        adaptive_retrieval=True,
        enable_category_consolidation=True,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.2,  # Balanced setting
    )

    # Generate report
    benchmark.generate_report(output_file="memory_benchmark")


if __name__ == "__main__":
    """
    Run the benchmark with:
    python notebooks/benchmark_memory.py
    """
    run_benchmark()
