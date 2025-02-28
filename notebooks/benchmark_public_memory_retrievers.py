# path/to/benchmark_memory_retrievers.py


import json
import os
import time
from collections import defaultdict
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from transformers import AutoModel, AutoTokenizer

from memoryweave.core import ContextualMemory, ContextualRetriever, MemoryEncoder

console = Console()
# Optionally, to load a public dataset via Hugging Face datasets
try:
    from datasets import load_dataset
except ImportError:
    console.log(
        "[yellow]datasets library not found; public dataset functions will be disabled.[/yellow]"
    )


# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)


# Helper class for sentence embedding
class EmbeddingModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, text: str) -> np.ndarray:
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
        summed = torch.sum(masked_embeddings, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts
        return mean_pooled.numpy()[0]


class MemoryBenchmark:
    """Benchmark different configurations of the MemoryWeave system."""

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the benchmark.

        Args:
            embedding_model_name: Name of the embedding model to use
        """
        console.log(f"Loading embedding model: [bold]{embedding_model_name}[/bold]")
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        model = AutoModel.from_pretrained(embedding_model_name)
        self.embedding_model = EmbeddingModelWrapper(model, tokenizer)
        self.embedding_dim = model.config.hidden_size

        # Datasets
        self.memory_data: list[tuple[str, str, str]] = []
        self.test_queries: list[tuple[str, str]] = []
        self.relevance_judgments: dict[tuple[int, int], float] = {}

        # Results
        self.results = defaultdict(list)

        # Error analysis tracking
        self.query_details = defaultdict(list)

    def load_test_data(
        self,
        memory_data: list[tuple[str, str, str]] = None,
        test_queries: list[tuple[str, str]] = None,
        relevance_judgments: dict[tuple[int, int], float] = None,
        queries_file: str = "datasets/evaluation_queries.json",
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
            console.log("Loading memory data from queries file...")
            self.memory_data = self._load_memories_from_queries(queries_file)
        else:
            self.memory_data = memory_data

        if test_queries is None:
            console.log("Loading test queries from queries file...")
            self.test_queries = self._load_queries_from_json(queries_file)
        else:
            self.test_queries = test_queries

        if relevance_judgments is None:
            console.log("Generating synthetic relevance judgments...")
            self.relevance_judgments = self._generate_synthetic_relevance()
        else:
            self.relevance_judgments = relevance_judgments

        console.log(
            f"Loaded [bold]{len(self.memory_data)}[/bold] memories and [bold]{len(self.test_queries)}[/bold] test queries"
        )

    def _load_memories_from_queries(self, queries_file: str) -> list[tuple[str, str, str]]:
        """Load memory data from a JSON file."""
        with open(queries_file) as f:
            data = json.load(f)

        memories = []
        for item in data:
            memories.append(
                (
                    item["expected_answer"],
                    item["category"],
                    "general",
                )
            )
        return memories

    def _load_queries_from_json(self, queries_file: str) -> list[tuple[str, str]]:
        """Load test queries from a JSON file."""
        with open(queries_file) as f:
            data = json.load(f)

        queries = []
        for item in data:
            queries.append((item["query"], item["category"]))
        return queries

    def _generate_synthetic_memories(self) -> list[tuple[str, str, str]]:
        """Generate synthetic memory data for testing."""
        categories = {
            "technology": ["programming", "ai", "hardware", "software", "gadgets"],
            "science": ["physics", "biology", "chemistry", "astronomy", "medicine"],
            "arts": ["music", "painting", "literature", "film", "photography"],
            "history": ["ancient", "medieval", "modern", "war", "politics"],
            "personal": ["preferences", "experiences", "opinions", "relationships", "habits"],
        }
        memories = []
        for category, subcategories in categories.items():
            for subcategory in subcategories:
                for i in range(5):  # 5 memories per subcategory
                    text = f"This is a {subcategory} memory about {category} (#{i + 1})"
                    memories.append((text, category, subcategory))
        return memories

    def _generate_synthetic_queries(self) -> list[tuple[str, str]]:
        """Generate synthetic test queries."""
        categories = set(category for _, category, _ in self.memory_data)
        queries = []
        for category in categories:
            queries.append((f"Tell me about {category}", category))
            queries.append((f"What do you know about {category}?", category))
            queries.append((f"I'm interested in learning more about {category}", category))
        if len(categories) >= 2:
            categories_list = list(categories)
            queries.append(
                (
                    f"Compare {categories_list[0]} and {categories_list[1]}",
                    "cross-category",
                )
            )
        return queries

    def _generate_synthetic_relevance(self) -> dict[tuple[int, int], float]:
        """Generate synthetic relevance judgments."""
        relevance = {}
        for query_idx, (_query_text, query_category) in enumerate(self.test_queries):
            for memory_idx, (_memory_text, memory_category, _memory_subcategory) in enumerate(
                self.memory_data
            ):
                if query_category == memory_category:
                    relevance[(query_idx, memory_idx)] = 1.0
                elif query_category == "cross-category" and memory_category in _query_text:
                    relevance[(query_idx, memory_idx)] = 0.8
                else:
                    relevance[(query_idx, memory_idx)] = 0.0
        return relevance

    # --- Public dataset functions (optional) ---
    def load_public_dataset(self, sample_size: int = 1000):
        """
        Load a sample of the Natural Questions dataset.
        """
        try:
            console.log("Loading Natural Questions dataset...", style="bold blue")
            dataset = load_dataset("natural_questions", split="train", streaming=True)
            console.log("Shuffling and sampling dataset...", style="bold blue")
            dataset = dataset.shuffle(seed=42)
            # dataset = list(islice(dataset, sample_size))  # take the first sample_size elements
            _dataset_iter = islice(dataset, sample_size)
            questions = [(item["question"]["text"], "general", "general") for item in _dataset_iter]
            self.memory_data = questions
            console.log(f"Loaded [bold]{len(self.memory_data)}[/bold] memories.")
        except Exception as e:
            console.log(f"[red]Error loading public dataset: {e}[/red]")

    def create_public_queries(self, num_queries: int = 100):
        """
        Use the questions from the Natural Questions dataset as test queries.
        """
        try:
            console.log("Creating test queries from Natural Questions...")
            dataset = load_dataset("natural_questions", split="train", streaming=True)
            console.log("Shuffling and sampling dataset...", style="bold blue")
            dataset = dataset.shuffle(seed=42)
            dataset = list(islice(dataset, num_queries))  # take the first num_queries elements

            queries = []
            for item in dataset:
                question = item["question"]["text"]
                queries.append((question, "general"))
            self.test_queries = queries
            console.log(f"Created [bold]{len(self.test_queries)}[/bold] test queries.")
        except Exception as e:
            console.log(f"[red]Error creating public queries: {e}[/red]")

    def benchmark_configuration(
        self,
        config_name: str,
        use_art_clustering: bool = False,
        confidence_threshold: float = 0.0,
        semantic_coherence_check: bool = False,
        adaptive_retrieval: bool = False,
        enable_category_consolidation: bool = False,
        retrieval_strategy: str = "hybrid",
        adaptive_k_factor: float = 0.3,
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
        """  # noqa: W505
        console.log(f"\n[bold blue]Benchmarking configuration: {config_name}[/bold blue]")

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
            adaptive_k_factor=adaptive_k_factor,
        )

        memory_start_time = time.time()
        for _mem_idx, (text, category, subcategory) in enumerate(self.memory_data):
            embedding, metadata = encoder.encode_concept(
                concept=category, description=text, related_concepts=[subcategory]
            )
            memory.add_memory(embedding, text, metadata)
        memory_time = time.time() - memory_start_time
        console.log(f"Memory population time: [bold]{memory_time:.2f}[/bold] seconds")

        retrieval_start_time = time.time()
        precision_at_k = []
        recall_at_k = []
        f1_at_k = []
        retrieval_times = []
        query_results = []

        for query_idx, (query_text, _query_category) in enumerate(self.test_queries):
            query_start_time = time.time()
            retrieved = retriever.retrieve_for_context(query_text, top_k=5)
            query_time = time.time() - query_start_time
            retrieval_times.append(query_time)

            retrieved_indices = [
                item.get("memory_id")
                for item in retrieved
                if isinstance(item.get("memory_id"), int)
            ]
            relevant_indices = [
                memory_idx
                for memory_idx in range(len(self.memory_data))
                if self.relevance_judgments.get((query_idx, memory_idx), 0.0) > 0.5
            ]

            if len(retrieved_indices) > 0:
                precision = sum(1 for idx in retrieved_indices if idx in relevant_indices) / len(
                    retrieved_indices
                )
            else:
                precision = 0.0

            if len(relevant_indices) > 0:
                recall = sum(1 for idx in retrieved_indices if idx in relevant_indices) / len(
                    relevant_indices
                )
            else:
                recall = 1.0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            precision_at_k.append(precision)
            recall_at_k.append(recall)
            f1_at_k.append(f1)

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

            if query_idx < 3:
                console.log(f"\n[bold]Query:[/bold] {query_text}")
                console.log(
                    f"Retrieved [bold]{len(retrieved)}[/bold] memories in {query_time:.3f} seconds"
                )
                console.log(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
                for i, mem in enumerate(retrieved[:3]):
                    relevance_mark = "✓" if mem.get("memory_id") in relevant_indices else "✗"
                    console.log(
                        f"  {i + 1}. [{relevance_mark}] {mem.get('text', '')[:50]}... (Score: {mem.get('relevance_score', 0):.3f})"
                    )

        retrieval_time = time.time() - retrieval_start_time
        console.log(f"Total retrieval time: [bold]{retrieval_time:.2f}[/bold] seconds")

        avg_precision = np.mean(precision_at_k)
        avg_recall = np.mean(recall_at_k)
        avg_f1 = np.mean(f1_at_k)
        avg_retrieval_time = np.mean(retrieval_times)

        console.log(f"Average precision: {avg_precision:.3f}")
        console.log(f"Average recall: {avg_recall:.3f}")
        console.log(f"Average F1 score: {avg_f1:.3f}")
        console.log(f"Average retrieval time: {avg_retrieval_time:.3f} seconds per query")

        if query_results:
            console.log("\nTop 3 Problematic Queries:")
            sorted_queries = sorted(query_results, key=lambda x: x["f1"])
            for idx, q in enumerate(sorted_queries[:3]):
                console.log(f'  {idx + 1}. Query: "{q["query"]}"')
                console.log(
                    f"     Precision: {q['precision']:.2f}, Recall: {q['recall']:.2f}, F1: {q['f1']:.2f}"
                )
                console.log(
                    f"     Retrieved: {q['retrieved_count']}, Relevant: {q['relevant_count']}"
                )
                if q["missed_relevant"]:
                    missed_texts = [
                        self.memory_data[idx][0][:40] + "..." for idx in q["missed_relevant"][:2]
                    ]
                    console.log(
                        f"     Missed relevant items: {len(q['missed_relevant'])}, e.g.: {', '.join(missed_texts)}"
                    )

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
        self.results["enable_category_consolidation"].append(enable_category_consolidation)
        self.results["retrieval_strategy"].append(retrieval_strategy)
        self.results["adaptive_k_factor"].append(adaptive_k_factor if adaptive_retrieval else None)

        self.query_details[config_name] = query_results

    def generate_report(self, output_file: str = None) -> pd.DataFrame:
        """
        Generate a report of the benchmark results.

        Args:
            output_file: Optional file prefix to save the report to
        """
        if not self.results:
            console.log("[red]No results to report. Run benchmark_configuration first.[/red]")
            return pd.DataFrame()

        results_df = pd.DataFrame(self.results)
        console.log("\n=== Benchmark Results ===")
        sorted_df = results_df.sort_values("avg_f1", ascending=False)
        console.log(
            sorted_df[
                ["configuration", "avg_precision", "avg_recall", "avg_f1", "avg_retrieval_time"]
            ].to_string(index=False)
        )

        plt.figure(figsize=(12, 8))
        x = np.linspace(0.01, 1, 100)
        y = np.linspace(0.01, 1, 100)
        X, Y = np.meshgrid(x, y)  # noqa: N806
        Z = 2 * X * Y / (X + Y)  # noqa: N806
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
        scatter = plt.scatter(
            results_df["avg_recall"],
            results_df["avg_precision"],
            s=150,
            alpha=0.8,
            c=results_df["avg_f1"],
            cmap="viridis",
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("F1 Score")
        for i, config in enumerate(results_df["configuration"]):
            plt.annotate(
                config,
                (results_df["avg_recall"].iloc[i], results_df["avg_precision"].iloc[i]),
                xytext=(7, 0),
                textcoords="offset points",
                fontsize=9,
            )
        plt.xlabel("Average Recall")
        plt.ylabel("Average Precision")
        plt.title("Precision vs. Recall for Different Configurations")
        plt.grid(True, alpha=0.3)
        plt.xlim(
            max(0, min(results_df["avg_recall"]) - 0.05),
            min(1, max(results_df["avg_recall"]) + 0.05),
        )
        plt.ylim(
            max(0, min(results_df["avg_precision"]) - 0.05),
            min(1, max(results_df["avg_precision"]) + 0.05),
        )

        if output_file:
            output_path = f"output/{output_file}_precision_recall.png"
            plt.savefig(output_path)
            console.log(f"Saved precision-recall plot to {output_path}")
            csv_path = f"output/{output_file}_results.csv"
            results_df.to_csv(csv_path, index=False)
            console.log(f"Saved results to {csv_path}")
        else:
            output_path = "output/benchmark_precision_recall.png"
            plt.savefig(output_path)
            console.log(f"Saved precision-recall plot to {output_path}")
            csv_path = "output/benchmark_results.csv"
            results_df.to_csv(csv_path, index=False)
            console.log(f"Saved results to {csv_path}")

        plt.figure(figsize=(12, 6))
        bar_positions = np.arange(len(results_df["configuration"]))
        plt.bar(bar_positions, results_df["avg_retrieval_time"], alpha=0.7)
        plt.xticks(bar_positions, results_df["configuration"], rotation=45, ha="right")
        plt.ylabel("Average Retrieval Time (seconds)")
        plt.title("Retrieval Performance for Different Configurations")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        if output_file:
            output_path = f"output/{output_file}_retrieval_time.png"
            plt.savefig(output_path)
            console.log(f"Saved retrieval time plot to {output_path}")
        else:
            output_path = "output/benchmark_retrieval_time.png"
            plt.savefig(output_path)
            console.log(f"Saved retrieval time plot to {output_path}")

        self._generate_error_analysis(output_file)
        return results_df

    def _generate_error_analysis(self, output_file: str = None):
        """Generate detailed error analysis for problematic queries."""
        if not self.query_details:
            return

        f1_by_query = defaultdict(dict)
        missing_by_query = defaultdict(dict)

        for config_name, query_results in self.query_details.items():
            for query_result in query_results:
                query = query_result["query"]
                f1_by_query[query][config_name] = query_result["f1"]
                missing_by_query[query][config_name] = len(query_result["missed_relevant"])

        avg_f1_by_query = {
            query: np.mean(list(configs.values())) for query, configs in f1_by_query.items()
        }
        problematic_queries = sorted(avg_f1_by_query.items(), key=lambda x: x[1])[:10]

        plt.figure(figsize=(14, 8))
        query_names = [q[0][:30] + "..." if len(q[0]) > 30 else q[0] for q in problematic_queries]
        config_names = list(self.results["configuration"])
        heatmap_data = np.zeros((len(query_names), len(config_names)))
        for i, (query, _) in enumerate(problematic_queries):
            for j, config in enumerate(config_names):
                heatmap_data[i, j] = f1_by_query[query].get(config, 0)
        plt.imshow(heatmap_data, cmap="YlGnBu", aspect="auto")
        plt.colorbar(label="F1 Score")
        plt.xticks(np.arange(len(config_names)), config_names, rotation=45, ha="right")
        plt.yticks(np.arange(len(query_names)), query_names)
        plt.title("F1 Scores for Problematic Queries Across Configurations")
        plt.tight_layout()

        if output_file:
            output_path = f"output/{output_file}_error_analysis.png"
            plt.savefig(output_path)
            console.log(f"Saved error analysis to {output_path}")
            error_data = {
                "problematic_queries": {q: {"avg_f1": f} for q, f in problematic_queries},
                "f1_by_config": f1_by_query,
                "missing_items_by_config": missing_by_query,
            }
            json_path = f"output/{output_file}_error_analysis.json"
            with open(json_path, "w") as f:
                json.dump(error_data, f, indent=2)
            console.log(f"Saved detailed error analysis to {json_path}")
        else:
            output_path = "output/benchmark_error_analysis.png"
            plt.savefig(output_path)
            console.log(f"Saved error analysis to {output_path}")


def run_benchmark():
    """
    Run the benchmark with various configurations.
    """
    benchmark = MemoryBenchmark()
    benchmark.load_test_data()

    # Uncomment below to use public dataset functions:
    benchmark.load_public_dataset(sample_size=1000)
    benchmark.create_public_queries(num_queries=100)

    benchmark.benchmark_configuration(
        config_name="Baseline",
        use_art_clustering=False,
        confidence_threshold=0.0,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
    )

    benchmark.benchmark_configuration(
        config_name="Confidence Threshold (0.3)",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
    )

    benchmark.benchmark_configuration(
        config_name="Conf + Semantic Coherence",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=True,
        adaptive_retrieval=False,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
    )

    benchmark.benchmark_configuration(
        config_name="Adaptive Retrieval (Conservative)",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=True,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.3,
    )

    benchmark.benchmark_configuration(
        config_name="Adaptive Retrieval (Balanced)",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=True,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.15,
    )

    benchmark.benchmark_configuration(
        config_name="Adaptive Retrieval (Liberal)",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=True,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.05,
    )

    benchmark.benchmark_configuration(
        config_name="ART Clustering",
        use_art_clustering=True,
        confidence_threshold=0.0,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
    )

    benchmark.benchmark_configuration(
        config_name="ART + Consolidation",
        use_art_clustering=True,
        confidence_threshold=0.0,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        enable_category_consolidation=True,
        retrieval_strategy="hybrid",
    )

    benchmark.benchmark_configuration(
        config_name="Full Features (Conservative)",
        use_art_clustering=True,
        confidence_threshold=0.3,
        semantic_coherence_check=True,
        adaptive_retrieval=True,
        enable_category_consolidation=True,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.3,
    )

    benchmark.benchmark_configuration(
        config_name="Full Features (Balanced)",
        use_art_clustering=True,
        confidence_threshold=0.2,
        semantic_coherence_check=True,
        adaptive_retrieval=True,
        enable_category_consolidation=True,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.15,
    )

    benchmark.benchmark_configuration(
        config_name="Full Features (Recall-Focused)",
        use_art_clustering=True,
        confidence_threshold=0.15,
        semantic_coherence_check=True,
        adaptive_retrieval=True,
        enable_category_consolidation=True,
        retrieval_strategy="hybrid",
        adaptive_k_factor=0.05,
    )

    benchmark.generate_report(output_file="memory_benchmark")


if __name__ == "__main__":
    """
    Run the benchmark with:
    python notebooks/benchmark_memory.py
    """
    run_benchmark()
