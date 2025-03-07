#!/usr/bin/env python3
"""
Context Retrieval Strategies Benchmark

This script compares different context retrieval strategies:
- MemoryWeave's RAM (Retrieval-Augmented Memory)
- Traditional RAG (Retrieval-Augmented Generation)
- BM25 (keyword-based retrieval)
- Pure vector similarity

It focuses on retrieval performance only (no LLM inference) to compare:
- Retrieval accuracy
- Retrieval speed
- Memory usage
- Quality with different types of queries
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from memoryweave.api.hybrid_memory_weave import HybridMemoryWeaveAPI

# Import MemoryWeave components
from memoryweave.api.llm_provider import LLMProvider
from memoryweave.api.memory_weave import MemoryWeaveAPI
from memoryweave.components.retriever import _get_embedder

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Try to import optional dependencies
try:
    from rank_bm25 import BM25Okapi

    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("BM25 not available - install with: pip install rank-bm25")

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("psutil not available - install with: pip install psutil")

# Setup logging and console
console = Console()
logger = logging.getLogger("retrieval_benchmark")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Number of test iterations for reliable timing
DEFAULT_ITERATIONS = 5
DEFAULT_TOP_K = 10
OUTPUT_DIR = "./benchmark_results"


class BM25Retriever:
    """Simple BM25 retrieval system implementation."""

    def __init__(self, tokenizer=None):
        self.documents = []
        self.document_ids = []
        self.bm25 = None
        self.tokenizer = tokenizer or (lambda x: x.lower().split())

    def add_memory(self, text, metadata=None):
        """Add a document to the BM25 index."""
        doc_id = len(self.documents)
        self.documents.append(text)
        self.document_ids.append(doc_id)

        # Rebuild index
        tokenized_corpus = [self.tokenizer(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        return doc_id

    def retrieve(self, query, top_k=10):
        """Retrieve documents using BM25 scoring."""
        if not self.bm25:
            return []

        tokenized_query = self.tokenizer(query)
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k documents
        top_indices = np.argsort(doc_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0:
                results.append(
                    {
                        "id": self.document_ids[idx],
                        "text": self.documents[idx],
                        "score": float(doc_scores[idx]),
                    }
                )

        return results


class VectorSimilarityRetriever:
    """Pure vector similarity retrieval implementation."""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.documents = []
        self.document_ids = []
        self.embeddings = []

    def add_memory(self, text, metadata=None):
        """Add a document and its embedding."""
        doc_id = len(self.documents)
        self.documents.append(text)
        self.document_ids.append(doc_id)

        # Generate embedding
        embedding = self.embedding_model.encode(text)
        self.embeddings.append(embedding)

        return doc_id

    def retrieve(self, query, top_k=10):
        """Retrieve documents using cosine similarity."""
        if not self.embeddings:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)

        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            # Normalize vectors
            norm_query = np.linalg.norm(query_embedding)
            norm_doc = np.linalg.norm(doc_embedding)

            if norm_query > 0 and norm_doc > 0:
                # Cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (norm_query * norm_doc)
                similarities.append((i, float(similarity)))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        results = []
        for idx, score in similarities[:top_k]:
            results.append(
                {
                    "id": self.document_ids[idx],
                    "text": self.documents[idx],
                    "score": score,
                }
            )

        return results


class MemoryWeaveRAM:
    """Wrapper for MemoryWeave's RAM retrieval."""

    def __init__(
        self,
        embedding_model,
        llm_provider: LLMProvider | None = None,
        use_hybrid=True,
        device="auto",
        two_stage_retrieval: bool = True,
    ):
        """Initialize MemoryWeave RAM retriever."""
        if use_hybrid:
            self.system = HybridMemoryWeaveAPI(
                model_name=DEFAULT_MODEL,  # No LLM needed
                embedding_model_name=EMBEDDING_MODEL,  # Will set manually
                debug=False,
                device=device,
                llm_provider=llm_provider,
                two_stage_retrieval=two_stage_retrieval,
            )

            # Configure for benchmark performance
            self.system.configure_chunking(
                adaptive_chunk_threshold=1000,  # Higher threshold to reduce chunking
                max_chunks_per_memory=2,  # Fewer chunks for benchmarking
                importance_threshold=0.7,  # Higher threshold for importance
            )

            # Set lower thresholds for better recall
            self.system.strategy.confidence_threshold = 0.05

        else:
            self.system = MemoryWeaveAPI(
                model_name=DEFAULT_MODEL,  # No LLM needed
                embedding_model_name=EMBEDDING_MODEL,  # Will set manually
                debug=False,
                device=device,
            )

        # Set embedding model
        self.system.embedding_model = embedding_model

        # For tracking added documents
        self.documents = []
        self.document_ids = []

    def add_memory(self, text: str, metadata: dict[str, Any] = None) -> str:
        """Add a document to MemoryWeave."""
        if metadata is None:
            metadata = {"type": "benchmark", "created_at": time.time()}

        # Use integer index as document ID for easier comparison
        doc_id = len(self.documents)
        metadata["doc_index"] = doc_id  # Store original index for retrieval comparison

        memory_id = self.system.add_memory(text, metadata)
        self.documents.append(text)
        self.document_ids.append(memory_id)

        return doc_id  # Return the document index, not the memory ID

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retrieve documents using MemoryWeave RAM."""
        results = self.system.retrieve(query, top_k=top_k)

        # Format results - IMPORTANT: Map to benchmark doc_index
        formatted_results = []
        for item in results:
            metadata = item.get("metadata", {})
            doc_index = metadata.get("doc_index")

            formatted_results.append(
                {
                    "id": doc_index if doc_index is not None else -1,
                    "memory_id": item.get("memory_id", ""),
                    "text": item.get("content", ""),
                    "score": item.get("relevance_score", 0.0),
                }
            )
        console.print(
            f"[dim]Retrieved IDs and metadata:[/dim] {formatted_results[:3]}"
        )  # Check first few results

        return formatted_results


class StandardRAG:
    """Implementation of standard RAG (vector similarity only)."""

    def __init__(self, embedding_model):
        """Initialize standard RAG retriever."""
        self.system = MemoryWeaveAPI(
            model_name=DEFAULT_MODEL,  # No LLM needed
            embedding_model_name=EMBEDDING_MODEL,  # Will set manually
            debug=False,
            # Disable contextual features
            enable_category_management=False,
            enable_personal_attributes=False,
            enable_semantic_coherence=False,
            enable_dynamic_thresholds=False,
        )

        # Set embedding model
        self.system.embedding_model = embedding_model

        # Configure strategy to use pure vector similarity
        if hasattr(self.system, "strategy"):
            self.system.strategy.initialize(
                {
                    "confidence_threshold": 0.1,
                    "similarity_weight": 1.0,
                    "associative_weight": 0.0,
                    "temporal_weight": 0.0,
                    "activation_weight": 0.0,
                }
            )

        # For tracking added documents
        self.documents = []
        self.document_ids = []

    def add_memory(self, text, metadata=None):
        """Add a document to standard RAG."""
        if metadata is None:
            metadata = {"type": "benchmark", "created_at": time.time()}

        doc_id = self.system.add_memory(text, metadata)
        self.documents.append(text)
        self.document_ids.append(doc_id)

        return doc_id

    def retrieve(self, query, top_k=10):
        """Retrieve documents using standard RAG."""
        results = self.system.retrieve(query, top_k=top_k)

        # Format results
        formatted_results = []
        for item in results:
            formatted_results.append(
                {
                    "id": item.get("memory_id", ""),
                    "text": item.get("content", ""),
                    "score": item.get("relevance_score", 0.0),
                }
            )

        return formatted_results


def get_memory_usage():
    """Get current process memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


def load_test_data(test_type="general"):
    """Load test data for benchmarking."""
    # Define general test data
    general_documents = [
        "Paris is the capital of France and one of the most visited cities in the world.",
        "Machine learning algorithms include decision trees, neural networks, and support vector machines.",
        "Python uses reference counting and a garbage collector for memory management.",
        "Web development best practices include responsive design, semantic HTML, and accessibility.",
        "Quantum computing uses quantum bits or qubits which can exist in multiple states simultaneously.",
        "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.",
        "Natural language processing (NLP) is a field of AI that focuses on interactions between computers and human language.",
        "The Great Wall of China is over 13,000 miles long and was built to protect against invasions.",
        "JavaScript is a programming language used for web development to create interactive elements.",
        "Climate change refers to long-term shifts in temperatures and weather patterns on Earth.",
        "The theory of relativity was developed by Albert Einstein in the early 20th century.",
        "Renewable energy sources include solar, wind, hydroelectric, and geothermal power.",
        "Artificial intelligence aims to create systems that can perform tasks requiring human intelligence.",
        "DNA (deoxyribonucleic acid) is a molecule that carries genetic information in all living organisms.",
        "The United Nations was established after World War II to promote international cooperation.",
        "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
        "Cloud computing delivers computer services over the internet, including storage and processing power.",
        "The Great Barrier Reef is the world's largest coral reef system, located off the coast of Australia.",
        "Bitcoin is a decentralized digital currency that operates without a central bank.",
        "The Olympic Games are held every four years and include summer and winter sports competitions.",
    ]

    # Define test queries with expected relevant document indices
    general_queries = [
        {
            "text": "What is the capital of France?",
            "relevant_indices": [0, 5],  # Paris docs
            "type": "factual",
        },
        {
            "text": "Tell me about machine learning algorithms",
            "relevant_indices": [1, 12],  # ML and AI docs
            "type": "conceptual",
        },
        {
            "text": "How does Python handle memory?",
            "relevant_indices": [2],  # Python memory management
            "type": "technical",
        },
        {
            "text": "What energy sources don't contribute to climate change?",
            "relevant_indices": [11, 9],  # Renewable energy and climate change
            "type": "inferential",
        },
        {
            "text": "Tell me about famous landmarks in Europe",
            "relevant_indices": [5, 0],  # Eiffel Tower and Paris
            "type": "general",
        },
    ]

    if test_type == "general":
        return general_documents, general_queries

    # You can add more specialized test sets here
    return general_documents, general_queries


def benchmark_retrieval_system(system_name, system, documents, queries, top_k=10, iterations=3):
    """Benchmark a retrieval system's performance."""

    # Results storage
    recall_scores = []
    precision_scores = []
    query_times = []
    memory_usage_before = get_memory_usage()

    # Load documents
    console.print(f"\n[bold cyan]Loading documents into {system_name}...[/bold cyan]")
    load_start = time.time()

    for doc in documents:
        system.add_memory(doc)

    load_time = time.time() - load_start
    console.print(f"Loaded {len(documents)} documents in {load_time:.3f}s")

    # Memory usage after loading
    memory_usage_after = get_memory_usage()
    memory_increase = memory_usage_after - memory_usage_before

    # Run benchmark for each query
    query_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        benchmark_task = progress.add_task(
            f"Benchmarking {system_name}...", total=len(queries) * iterations
        )

        for query_data in queries:
            query_text = query_data["text"]
            relevant_indices = set(query_data["relevant_indices"])
            query_type = query_data.get("type", "unknown")

            # Run multiple iterations for reliable timing
            iteration_times = []
            iteration_results = []

            for _ in range(iterations):
                # Measure retrieval time
                results = system.retrieve(query_text, top_k=top_k)
                progress.advance(benchmark_task)

            # Use the last iteration's results for metrics
            results = iteration_results[-1]

            # Debug results - add this to see what's being returned
            if system_name == "MemoryWeave Hybrid":
                console.print(f"[dim]Query: '{query_text}'[/dim]")
                console.print(f"[dim]Expected relevant indices: {relevant_indices}[/dim]")
                console.print(f"[dim]Results count: {len(results)}[/dim]")
                if results:
                    console.print(f"[dim]First result sample: {results[0]}[/dim]")

            # Extract document IDs - handle different ID formats
            retrieved_doc_ids = []
            for r in results:
                if "memory_id" in r:
                    id_val = r["memory_id"]
                elif "id" in r:
                    id_val = r["id"]
                else:
                    continue

                # Convert to int if possible
                if isinstance(id_val, str) and id_val.isdigit():
                    retrieved_doc_ids.append(int(id_val))
                elif isinstance(id_val, int):
                    retrieved_doc_ids.append(id_val)

            # Calculate metrics
            retrieved_set = set(retrieved_doc_ids[:top_k])
            relevant_retrieved = retrieved_set.intersection(relevant_indices)

            # Recall and precision
            recall = len(relevant_retrieved) / len(relevant_indices) if relevant_indices else 0
            precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0

            # Mean Reciprocal Rank (MRR)
            mrr = 0
            for i, doc_id in enumerate(retrieved_doc_ids):
                if doc_id in relevant_indices:
                    mrr = 1.0 / (i + 1)
                    break

            # Store query results
            query_result = {
                "query": query_text,
                "query_type": query_type,
                "retrieval_time": np.mean(iteration_times),
                "retrieved_ids": retrieved_doc_ids[:top_k],
                "relevant_ids": list(relevant_indices),
                "recall": recall,
                "precision": precision,
                "mrr": mrr,
                "retrieved_count": len(results),
            }

            query_results.append(query_result)
            recall_scores.append(recall)
            precision_scores.append(precision)
            query_times.append(np.mean(iteration_times))

    # Calculate overall metrics
    avg_recall = np.mean(recall_scores)
    avg_precision = np.mean(precision_scores)
    avg_f1 = (
        2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0
    )
    avg_query_time = np.mean(query_times)

    # Return comprehensive results
    return {
        "system_name": system_name,
        "avg_recall": avg_recall,
        "avg_precision": avg_precision,
        "avg_f1": avg_f1,
        "avg_query_time": avg_query_time,
        "load_time": load_time,
        "memory_usage_mb": memory_increase,
        "query_results": query_results,
        "documents_count": len(documents),
        "iterations": iterations,
    }


def display_results_table(results):
    """Display benchmark results in a formatted table."""
    table = Table(title="Retrieval Systems Benchmark Results")

    # Add columns
    table.add_column("System", style="cyan")
    table.add_column("Recall", style="green")
    table.add_column("Precision", style="green")
    table.add_column("F1 Score", style="green")
    table.add_column("Query Time (s)", style="yellow")
    table.add_column("Load Time (s)", style="yellow")
    table.add_column("Memory (MB)", style="magenta")

    # Add rows for each system
    for result in results:
        table.add_row(
            result["system_name"],
            f"{result['avg_recall']:.3f}",
            f"{result['avg_precision']:.3f}",
            f"{result['avg_f1']:.3f}",
            f"{result['avg_query_time']:.5f}",
            f"{result['load_time']:.3f}",
            f"{result['memory_usage_mb']:.1f}" if HAS_PSUTIL else "N/A",
        )

    console.print(table)

    # Add a detailed breakdown by query type
    console.print("\n[bold]Performance by Query Type[/bold]")

    for result in results:
        console.print(f"\n[bold cyan]{result['system_name']}[/bold cyan]")

        # Group queries by type
        query_types = {}
        for query_result in result["query_results"]:
            query_type = query_result["query_type"]
            if query_type not in query_types:
                query_types[query_type] = []
            query_types[query_type].append(query_result)

        # Create type breakdown table
        type_table = Table(show_header=True)
        type_table.add_column("Query Type")
        type_table.add_column("Avg Recall")
        type_table.add_column("Avg Precision")
        type_table.add_column("Avg Time (s)")

        for query_type, queries in query_types.items():
            type_recall = np.mean([q["recall"] for q in queries])
            type_precision = np.mean([q["precision"] for q in queries])
            type_time = np.mean([q["retrieval_time"] for q in queries])

            type_table.add_row(
                query_type,
                f"{type_recall:.3f}",
                f"{type_precision:.3f}",
                f"{type_time:.5f}",
            )

        console.print(type_table)


def save_results(results, output_dir=OUTPUT_DIR):
    """Save benchmark results to a JSON file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"retrieval_benchmark_{timestamp}.json")

    # Add metadata
    output_data = {
        "timestamp": timestamp,
        "systems_tested": [r["system_name"] for r in results],
        "results": results,
    }

    # Save to file
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)

    console.print(f"[bold green]Results saved to:[/bold green] {filename}")
    return filename


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark different context retrieval strategies")
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of iterations for timing (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Directory to save results (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--test-type", default="general", help="Type of test data to use (default: general)"
    )
    parser.add_argument("--no-charts", action="store_true", help="Disable chart generation")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["all"],
        help="Systems to test (memoryweave, hybrid, standard_rag, bm25, vector_sim, all)",
    )
    return parser.parse_args()


def create_charts(results_file, output_dir=OUTPUT_DIR):
    """Create visualization charts from benchmark results."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        console.print("[yellow]Matplotlib or pandas not available. Charts not generated.[/yellow]")
        console.print("[yellow]Install with: pip install matplotlib pandas[/yellow]")
        return

    # Create charts directory
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Load results
    with open(results_file) as f:
        data = json.load(f)

    # Extract main results
    systems = []
    recalls = []
    precisions = []
    f1_scores = []
    query_times = []

    for result in data["results"]:
        systems.append(result["system_name"])
        recalls.append(result["avg_recall"])
        precisions.append(result["avg_precision"])
        f1_scores.append(result["avg_f1"])
        query_times.append(result["avg_query_time"])

    # Create dataframe
    df = pd.DataFrame(
        {
            "System": systems,
            "Recall": recalls,
            "Precision": precisions,
            "F1 Score": f1_scores,
            "Query Time": query_times,
        }
    )

    # 1. Performance metrics chart
    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    index = np.arange(len(systems))

    plt.bar(index, df["Recall"], bar_width, label="Recall", color="#5DA5DA")
    plt.bar(index + bar_width, df["Precision"], bar_width, label="Precision", color="#F15854")
    plt.bar(index + 2 * bar_width, df["F1 Score"], bar_width, label="F1 Score", color="#60BD68")

    plt.xlabel("Retrieval System")
    plt.ylabel("Score")
    plt.title("Retrieval Quality Metrics")
    plt.xticks(index + bar_width, systems, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(charts_dir, "quality_metrics.png"), dpi=300)

    # 2. Query time comparison
    plt.figure(figsize=(10, 6))
    plt.bar(systems, df["Query Time"], color="#5DA5DA")
    plt.xlabel("Retrieval System")
    plt.ylabel("Average Query Time (seconds)")
    plt.title("Retrieval Speed Comparison")
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(df["Query Time"]):
        plt.text(i, v + 0.0001, f"{v:.5f}s", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()

    plt.savefig(os.path.join(charts_dir, "speed_comparison.png"), dpi=300)

    # 3. Query type performance
    # Prepare data by query type
    query_types = set()
    type_data = {}

    for result in data["results"]:
        system = result["system_name"]
        type_data[system] = {}

        # Group queries by type
        for query in result["query_results"]:
            query_type = query["query_type"]
            query_types.add(query_type)

            if query_type not in type_data[system]:
                type_data[system][query_type] = []

            type_data[system][query_type].append(
                {
                    "recall": query["recall"],
                    "precision": query["precision"],
                    "time": query["retrieval_time"],
                }
            )

    # Calculate averages by type
    type_averages = {}
    for system, types in type_data.items():
        type_averages[system] = {}
        for query_type, queries in types.items():
            type_averages[system][query_type] = {
                "recall": np.mean([q["recall"] for q in queries]),
                "precision": np.mean([q["precision"] for q in queries]),
                "time": np.mean([q["time"] for q in queries]),
            }

    # Create query type chart
    query_types = sorted(list(query_types))
    fig, axes = plt.subplots(1, len(query_types), figsize=(15, 6), sharey=True)

    for i, query_type in enumerate(query_types):
        ax = axes[i] if len(query_types) > 1 else axes

        type_recalls = []
        for system in systems:
            if query_type in type_averages[system]:
                type_recalls.append(type_averages[system][query_type]["recall"])
            else:
                type_recalls.append(0)

        ax.bar(systems, type_recalls, color="#5DA5DA")
        ax.set_title(f"{query_type.capitalize()} Queries")
        ax.set_xticklabels(systems, rotation=45, ha="right")

        if i == 0:
            ax.set_ylabel("Recall Score")

    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "query_type_performance.png"), dpi=300)

    console.print(f"[bold green]Charts saved to:[/bold green] {charts_dir}")


def main():
    """Run the retrieval benchmark."""
    args = parse_arguments()

    console.print(
        Panel.fit(
            "[bold cyan]Context Retrieval Strategies Benchmark[/bold cyan]\n\n"
            f"Iterations: [yellow]{args.iterations}[/yellow]\n"
            f"Top-K: [yellow]{args.top_k}[/yellow]\n"
            f"Test Type: [yellow]{args.test_type}[/yellow]\n"
            f"Systems: [yellow]{', '.join(args.systems)}[/yellow]",
            border_style="cyan",
        )
    )

    # Load test data
    console.print("[bold]Loading test data...[/bold]")
    documents, queries = load_test_data(args.test_type)
    console.print(f"Loaded {len(documents)} documents and {len(queries)} queries")

    # Initialize embedding model (shared across systems)
    console.print("[bold]Initializing embedding model...[/bold]")
    embedding_model = _get_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Systems to test
    systems_to_test = []
    test_all = "all" in args.systems

    if test_all or "memoryweave" in args.systems:
        systems_to_test.append(
            (
                "MemoryWeave Standard",
                MemoryWeaveRAM(embedding_model, use_hybrid=False),
            )
        )

    if test_all or "hybrid" in args.systems:
        systems_to_test.append(
            (
                "MemoryWeave Hybrid",
                MemoryWeaveRAM(embedding_model, use_hybrid=True),
            )
        )

    if test_all or "standard_rag" in args.systems:
        systems_to_test.append(("Standard RAG", StandardRAG(embedding_model)))

    if test_all or "vector_sim" in args.systems:
        systems_to_test.append(("Vector Similarity", VectorSimilarityRetriever(embedding_model)))

    if (test_all or "bm25" in args.systems) and HAS_BM25:
        systems_to_test.append(("BM25", BM25Retriever()))

    console.print(f"[bold]Testing {len(systems_to_test)} retrieval systems[/bold]")

    # Run benchmarks
    results = []
    for system_name, system in systems_to_test:
        result = benchmark_retrieval_system(
            system_name, system, documents, queries, top_k=args.top_k, iterations=args.iterations
        )
        results.append(result)

    # Display results
    display_results_table(results)

    # Save results
    results_file = save_results(results, args.output_dir)

    # Create charts
    if not args.no_charts:
        try:
            create_charts(results_file, args.output_dir)
        except Exception as e:
            console.print(f"[yellow]Error creating charts: {e}[/yellow]")


if __name__ == "__main__":
    main()
