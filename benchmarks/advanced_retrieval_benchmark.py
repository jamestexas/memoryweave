# memoryweave/benchmarks/advanced_retrieval_benchmark.py
# !/usr/bin/env python3
"""
Advanced Retrieval Benchmark for MemoryWeave

This script tests the MemoryWeave hybrid approach against more challenging scenarios:
- Different document sizes
- Temporal queries
- Complex queries requiring inference
- Different collection sizes
"""

import argparse
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta

import numpy as np
from retrieval_benchmark import (
    BM25Retriever,
    MemoryWeaveRAM,
    StandardRAG,
    VectorSimilarityRetriever,
    get_memory_usage,
)
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from utils.perf_timer import PerformanceTimer, timer

# Import MemoryWeave components
from memoryweave.api.llm_provider import LLMProvider
from memoryweave.components.retriever import _get_embedder

# Setup logging and console
console = Console()
logger = logging.getLogger("advanced_benchmark")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "./benchmark_results/advanced"


def generate_test_data():
    """Generate more diverse test data sets."""
    test_sets = {
        "short_docs": {"documents": [], "queries": []},
        "long_docs": {"documents": [], "queries": []},
        "temporal": {"documents": [], "queries": []},
        "inference": {"documents": [], "queries": []},
    }

    # Set 1: Short documents (50-150 chars)
    short_docs = [
        "Python is a high-level programming language with simple syntax.",
        "The Earth orbits the Sun at a distance of about 150 million kilometers.",
        "Coffee contains caffeine which can help people stay alert.",
        "Leonardo da Vinci painted the Mona Lisa in the early 16th century.",
        "Tokyo is the capital city of Japan and has over 13 million people.",
        "Oxygen is essential for human respiration and survival.",
        "The Great Wall of China is over 13,000 miles long.",
        "Dolphins are highly intelligent marine mammals.",
        "Mozart composed over 600 works during his lifetime.",
        "Mount Everest is the highest mountain on Earth.",
    ]

    short_queries = [
        {
            "text": "What programming language has simple syntax?",
            "relevant_indices": [0],
            "type": "factual",
        },
        {"text": "How far is Earth from the Sun?", "relevant_indices": [1], "type": "factual"},
        {"text": "What effect does caffeine have?", "relevant_indices": [2], "type": "factual"},
        {"text": "Who painted the Mona Lisa?", "relevant_indices": [3], "type": "factual"},
        {"text": "What is the tallest mountain?", "relevant_indices": [9], "type": "factual"},
    ]

    # Set 2: Long documents (500+ chars)
    long_docs = [
        "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals and humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term 'artificial intelligence' is often used to describe machines (or computers) that mimic cognitive functions that humans associate with the human mind, such as learning and problem solving. As machines become increasingly capable, tasks considered to require 'intelligence' are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.",
        "The Python programming language was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands as a successor to the ABC programming language, which was inspired by SETL, capable of exception handling and interfacing with the Amoeba operating system. Its implementation began in December 1989. Van Rossum shouldered sole responsibility for the project, as the lead developer, until July 12, 2018, when he announced his 'permanent vacation' from his responsibilities as Python's 'benevolent dictator for life', a title the Python community bestowed upon him to reflect his long-term commitment as the project's chief decision-maker. In January 2019, active Python core developers elected a five-member Steering Council to lead the project.",
        "Climate change includes both global warming driven by human-induced emissions of greenhouse gases and the resulting large-scale shifts in weather patterns. Though there have been previous periods of climatic change, since the mid-20th century humans have had an unprecedented impact on Earth's climate system and caused change on a global scale. The largest driver of warming is the emission of greenhouse gases, of which more than 90% are carbon dioxide and methane. Fossil fuel burning (coal, oil, and natural gas) for energy consumption is the main source of these emissions, with additional contributions from agriculture, deforestation, and manufacturing. Temperature rise is also affected by climate feedback mechanisms, such as the loss of sunlight-reflecting snow and ice, increased water vapor, and changes to carbon sinks.",
        "Quantum computing is the exploitation of collective properties of quantum states, such as superposition and entanglement, to perform computation. The devices that perform quantum computations are known as quantum computers. They are believed to be able to solve certain computational problems, such as integer factorization (which underlies RSA encryption), substantially faster than classical computers. The study of quantum computing is a subfield of quantum information science. It is likely that quantum computers will have the ability to solve certain classes of problems that are not efficiently solvable on classical computers, an achievement known as 'quantum supremacy.' These include simulating quantum many-body systems, and solving certain instances of constraint satisfaction, satisfiability, and optimization problems related to searching unsorted databases.",
        "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries. It occurred after the Crisis of the Late Middle Ages and was associated with great social change. In addition to the standard periodization, proponents of a 'long Renaissance' may put its beginning in the 14th century and its end in the 17th century. The traditional view focuses more on the early modern aspects of the Renaissance and argues that it was a break from the past, but many historians today focus more on its medieval aspects and argue that it was an extension of the Middle Ages. However, the beginnings of the period – the early Renaissance of the 15th century and the Italian Proto-Renaissance from around 1250 or 1300 – overlap considerably with the Late Middle Ages, conventionally dated to c.1250–1500, and the Middle Ages themselves were a long period filled with gradual changes, like the Modern Period.",
    ]

    long_queries = [
        {
            "text": "What is artificial intelligence and how is it defined?",
            "relevant_indices": [0],
            "type": "conceptual",
        },
        {
            "text": "Who created the Python programming language?",
            "relevant_indices": [1],
            "type": "factual",
        },
        {
            "text": "What are the main causes of climate change?",
            "relevant_indices": [2],
            "type": "inference",
        },
        {
            "text": "How does quantum computing relate to encryption?",
            "relevant_indices": [3],
            "type": "inference",
        },
        {
            "text": "When did the Renaissance period occur in history?",
            "relevant_indices": [4],
            "type": "factual",
        },
    ]

    # Set 3: Temporal documents (with dates)
    now = datetime.now()
    dates = [
        now - timedelta(days=5),
        now - timedelta(days=10),
        now - timedelta(days=30),
        now - timedelta(days=90),
        now - timedelta(days=365),
    ]

    temporal_docs = [
        f"Meeting notes ({dates[0].strftime('%Y-%m-%d')}): Discussed the new product launch scheduled for next month. Team agreed on final design specifications and marketing strategy. John will handle coordination with manufacturing.",
        f"Project update ({dates[1].strftime('%Y-%m-%d')}): Phase 1 of the database migration completed successfully. Some performance issues were identified and will be addressed in Phase 2. User feedback has been mostly positive.",
        f"Quarterly review ({dates[2].strftime('%Y-%m-%d')}): Q1 sales exceeded expectations by 15%. New market expansion is proceeding according to plan. Customer retention has improved by 7% since implementing the loyalty program.",
        f"Strategy meeting ({dates[3].strftime('%Y-%m-%d')}): Board approved the five-year expansion plan. Key markets identified include Southeast Asia and Eastern Europe. R&D budget increased by 20% to support new product development.",
        f"Annual report ({dates[4].strftime('%Y-%m-%d')}): Company achieved 12% YOY growth. Challenges included supply chain disruptions and increasing raw material costs. Outlook for next year remains positive with projected 15% growth.",
    ]

    temporal_queries = [
        {
            "text": "What was discussed in the recent meeting?",
            "relevant_indices": [0],
            "type": "temporal",
        },
        {
            "text": "What happened in the project update from last week?",
            "relevant_indices": [1],
            "type": "temporal",
        },
        {
            "text": "What were the quarterly results from last month?",
            "relevant_indices": [2],
            "type": "temporal",
        },
        {
            "text": "What did the board approve a few months ago?",
            "relevant_indices": [3],
            "type": "temporal",
        },
        {
            "text": "What challenges were mentioned in last year's report?",
            "relevant_indices": [4],
            "type": "temporal",
        },
    ]

    # Set 4: Inference documents (requiring reading between the lines)
    inference_docs = [
        "Company XYZ reported a 30% drop in quarterly revenue. Their stock price fell 15% in early trading. The CEO announced a restructuring plan that includes closing three manufacturing plants.",
        "The town's water reservoir is at 20% capacity after three years of below-average rainfall. Local officials have implemented stage 2 water restrictions, banning lawn watering and car washing.",
        "A new study of 5,000 patients showed those taking Drug A had 45% fewer heart attacks than those on the placebo. However, 15% of participants reported experiencing headaches and dizziness.",
        "The university's physics department received a $50 million grant from the National Science Foundation. They plan to build a new quantum computing lab and hire five additional faculty members.",
        "Global smartphone sales declined for the third consecutive quarter. Analysts cite market saturation and consumers keeping devices longer as main factors. Meanwhile, wearable technology sales increased by 25%.",
    ]

    inference_queries = [
        {
            "text": "Is Company XYZ doing well financially?",
            "relevant_indices": [0],
            "type": "inference",
        },
        {"text": "Is there a drought in the town?", "relevant_indices": [1], "type": "inference"},
        {"text": "Is Drug A safe to use?", "relevant_indices": [2], "type": "inference"},
        {
            "text": "What fields will benefit from the university's new funding?",
            "relevant_indices": [3],
            "type": "inference",
        },
        {
            "text": "Why are people buying fewer smartphones?",
            "relevant_indices": [4],
            "type": "inference",
        },
    ]

    # Assign to test sets
    test_sets["short_docs"]["documents"] = short_docs
    test_sets["short_docs"]["queries"] = short_queries
    test_sets["long_docs"]["documents"] = long_docs
    test_sets["long_docs"]["queries"] = long_queries
    test_sets["temporal"]["documents"] = temporal_docs
    test_sets["temporal"]["queries"] = temporal_queries
    test_sets["inference"]["documents"] = inference_docs
    test_sets["inference"]["queries"] = inference_queries

    return test_sets


def generate_large_collection(size=100):
    """Generate a larger collection of documents for scaling tests."""
    documents = []
    relevant_docs = []

    # Add some core documents that will be targets for queries
    core_topics = [
        "artificial intelligence",
        "climate change",
        "quantum physics",
        "renewable energy",
        "space exploration",
    ]

    for i, topic in enumerate(core_topics):
        doc = f"This document contains important information about {topic}. It is a key document that should be retrieved when asking about {topic}. The document ID is {i} and it contains specialized terminology related to {topic}."
        documents.append(doc)
        relevant_docs.append(i)

    # Add filler documents up to the desired size
    filler_topics = [
        "agriculture",
        "biology",
        "chemistry",
        "economics",
        "education",
        "engineering",
        "finance",
        "geology",
        "healthcare",
        "history",
        "literature",
        "mathematics",
        "music",
        "philosophy",
        "politics",
        "psychology",
        "sociology",
        "sports",
        "technology",
        "tourism",
    ]

    for i in range(5, size):
        topic = secrets.choice(filler_topics)
        doc = f"Document {i} about {topic}. This is a filler document with information about {topic}. It contains common terms and phrases related to {topic}."
        documents.append(doc)

    # Create queries targeting the core documents
    queries = []
    for i, topic in enumerate(core_topics):
        queries.append(
            {
                "text": f"Tell me about {topic}",
                "relevant_indices": [i],
                "type": "factual",
            }
        )

    return {"documents": documents, "queries": queries}


@timer
def benchmark_retrieval_system(system_name, system, documents, queries, top_k=10, iterations=3):
    """Benchmark a retrieval system's performance."""
    timer = PerformanceTimer()
    # Results storage
    recall_scores = []
    precision_scores = []
    f1_scores = []
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
                timer.start("retrieval")
                results = system.retrieve(query_text, top_k=top_k)
                retrieval_time = timer.stop("retrieval")

                iteration_times.append(retrieval_time)
                iteration_results.append(results)
                progress.advance(benchmark_task)

            # Use the last iteration's results for metrics
            results = iteration_results[-1]

            # Add detailed debug log here
            logger.debug(f"Raw retrieved results for query '{query_text}': {results}")

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
            logger.debug(f"Query: '{query_text}'")
            logger.debug(f"Retrieved IDs: {retrieved_doc_ids}")
            logger.debug(f"Expected relevant IDs: {relevant_indices}")
            logger.debug(f"Intersection: {retrieved_set.intersection(relevant_indices)}")
            relevant_retrieved = retrieved_set.intersection(relevant_indices)

            # Recall and precision
            recall = len(relevant_retrieved) / len(relevant_indices) if relevant_indices else 0
            precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0

            # F1 score
            f1 = 0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)

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
                "f1": f1,
                "mrr": mrr,
                "retrieved_count": len(results),
            }

            query_results.append(query_result)
            recall_scores.append(recall)
            precision_scores.append(precision)
            f1_scores.append(f1)
            query_times.append(np.mean(iteration_times))

    # Calculate overall metrics
    avg_recall = np.mean(recall_scores)
    avg_precision = np.mean(precision_scores)
    avg_f1 = np.mean(f1_scores)
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


def run_benchmark_by_type(test_type, test_data, embedding_model, hybrid_instance=None):
    """Run benchmark for a specific test type."""
    console.print(f"\n[bold magenta]Running {test_type} benchmark[/bold magenta]")

    documents = test_data["documents"]
    queries = test_data["queries"]
    # If a hybrid instance is provided, reuse it. Otherwise, create a new one.
    if hybrid_instance is None:
        hybrid_instance = MemoryWeaveRAM(
            embedding_model, use_hybrid=True, two_stage_retrieval=False
        )
    # Create systems to test
    systems = [
        ("Hybrid", hybrid_instance),
        ("BM25", BM25Retriever()),
        ("Vector", VectorSimilarityRetriever(embedding_model)),
        ("RAG", StandardRAG(embedding_model)),
    ]

    # Run benchmarks
    results = []
    for system_name, system in systems:
        full_name = f"{system_name} ({test_type})"
        result = benchmark_retrieval_system(
            full_name, system, documents, queries, top_k=5, iterations=3
        )
        results.append(result)

    return results


def run_scaling_test(embedding_model, sizes: list[int] | None = None):
    """Run benchmark with different corpus sizes."""
    sizes = sizes or [10, 50, 100]
    console.print("\n[bold yellow]Running scaling benchmark[/bold yellow]")

    all_results = []

    for size in sizes:
        console.print(f"\n[bold]Testing with {size} documents[/bold]")
        test_data = generate_large_collection(size)

        # Use only hybrid for scaling tests
        system = MemoryWeaveRAM(embedding_model, use_hybrid=True, two_stage_retrieval=False)

        result = benchmark_retrieval_system(
            f"Hybrid ({size} docs)",
            system,
            test_data["documents"],
            test_data["queries"],
            top_k=5,
            iterations=2,  # Fewer iterations for scaling tests
        )

        all_results.append(result)

    return all_results


def run_configuration_test(embedding_model, test_data):
    """Run benchmark with different hybrid configurations."""
    console.print("\n[bold cyan]Running configuration benchmark[/bold cyan]")

    documents = test_data["documents"]
    queries = test_data["queries"]

    # Test different configurations
    configs = [
        {"name": "Hybrid (Default)", "settings": {}},
        {"name": "Hybrid (Low Threshold)", "settings": {"confidence_threshold": 0.01}},
        {
            "name": "Hybrid (No Chunking)",
            "settings": {"chunking": {"adaptive_chunk_threshold": 100000}},
        },
        {"name": "Hybrid (Keyword Focus)", "settings": {"keyword_boost_factor": 0.7}},
    ]

    results = []

    for config in configs:
        # Create system with this configuration
        system = MemoryWeaveRAM(embedding_model, use_hybrid=True, two_stage_retrieval=False)
        doc_id = system.add_memory("Python is a simple programming language.")
        results = system.retrieve("What is Python?")
        print(f"RESULTS {results}\nDOC ID: {doc_id}\n")  # Should not be empty
        # Apply configuration
        if "confidence_threshold" in config["settings"]:
            system.system.strategy.confidence_threshold = config["settings"]["confidence_threshold"]

        if "keyword_boost_factor" in config["settings"]:
            system.system.strategy.keyword_boost_factor = config["settings"]["keyword_boost_factor"]

        if "chunking" in config["settings"]:
            chunking_settings = config["settings"]["chunking"]
            system.system.configure_chunking(**chunking_settings)

        # Run benchmark
        result = benchmark_retrieval_system(
            config["name"], system, documents, queries, top_k=5, iterations=3
        )

        results.append(result)

    return results


def display_results_table(results, title="Benchmark Results"):
    """Display benchmark results in a formatted table."""
    table = Table(title=title)

    # Add columns
    table.add_column("System", style="cyan")
    table.add_column("Recall", style="green")
    table.add_column("Precision", style="green")
    table.add_column("F1 Score", style="green")
    table.add_column("Query Time (s)", style="yellow")
    table.add_column("Load Time (s)", style="yellow")
    if any("memory_usage_mb" in r for r in results):
        table.add_column("Memory (MB)", style="magenta")

    # Add rows for each system
    for result in results:
        row = [
            result["system_name"],
            f"{result['avg_recall']:.3f}",
            f"{result['avg_precision']:.3f}",
            f"{result['avg_f1']:.3f}",
            f"{result['avg_query_time']:.5f}",
            f"{result['load_time']:.3f}",
        ]

        if "memory_usage_mb" in result:
            row.append(f"{result.get('memory_usage_mb', 0):.1f}")

        table.add_row(*row)

    console.print(table)


def save_results(results, output_dir=OUTPUT_DIR, test_type="advanced"):
    """Save benchmark results to a JSON file."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{test_type}_benchmark_{timestamp}.json")

    # Add metadata
    output_data = {
        "timestamp": timestamp,
        "test_type": test_type,
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
    parser = argparse.ArgumentParser(description="Advanced Benchmark for MemoryWeave")

    parser.add_argument(
        "--test-types",
        nargs="+",
        default=["all"],
        choices=["all", "short_docs", "long_docs", "temporal", "inference", "scaling", "configs"],
        help="Types of tests to run",
    )

    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Directory to save results (default: {OUTPUT_DIR})",
    )

    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of iterations for timing (default: 3)"
    )

    parser.add_argument("--no-charts", action="store_true", help="Disable chart generation")

    return parser.parse_args()


def main():
    """Run the advanced benchmarks."""
    args = parse_arguments()

    console.print(
        Panel.fit(
            "[bold cyan]Advanced MemoryWeave Benchmark[/bold cyan]\n\n"
            f"Test Types: [yellow]{', '.join(args.test_types)}[/yellow]\n"
            f"Iterations: [yellow]{args.iterations}[/yellow]",
            border_style="cyan",
        )
    )

    # Initialize embedding model
    console.print("[bold]Initializing embedding model...[/bold]")
    embedding_model = _get_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Generate test data
    console.print("[bold]Generating test data...[/bold]")
    test_sets = generate_test_data()

    # All results for final reporting
    all_results = []

    # Run selected tests
    test_types = args.test_types
    if "all" in test_types:
        test_types = ["short_docs", "long_docs", "temporal", "inference", "scaling", "configs"]
    llm_provider = LLMProvider(
        model_name=DEFAULT_MODEL,
    )
    hybrid_system = MemoryWeaveRAM(
        embedding_model, llm_provider=llm_provider, use_hybrid=True, two_stage_retrieval=False
    )

    # Document type tests
    if "short_docs" in test_types:
        results = run_benchmark_by_type(
            "short_docs", test_sets["short_docs"], embedding_model, hybrid_instance=hybrid_system
        )
        display_results_table(results, "Short Documents Benchmark")
        all_results.extend(results)
        save_results(results, args.output_dir, "short_docs")

    if "long_docs" in test_types:
        results = run_benchmark_by_type(
            "long_docs", test_sets["long_docs"], embedding_model, hybrid_instance=hybrid_system
        )
        display_results_table(results, "Long Documents Benchmark")
        all_results.extend(results)
        save_results(results, args.output_dir, "long_docs")

    if "temporal" in test_types:
        results = run_benchmark_by_type(
            "temporal", test_sets["temporal"], embedding_model, hybrid_instance=hybrid_system
        )
        display_results_table(results, "Temporal Queries Benchmark")
        all_results.extend(results)
        save_results(results, args.output_dir, "temporal")

    if "inference" in test_types:
        results = run_benchmark_by_type(
            "inference", test_sets["inference"], embedding_model, hybrid_instance=hybrid_system
        )
        display_results_table(results, "Inference Queries Benchmark")
        all_results.extend(results)
        save_results(results, args.output_dir, "inference")

    # Scaling test
    if "scaling" in test_types:
        results = run_scaling_test(embedding_model, sizes=[10, 50, 100])
        display_results_table(results, "Scaling Benchmark")
        all_results.extend(results)
        save_results(results, args.output_dir, "scaling")

    # Configuration test
    if "configs" in test_types:
        results = run_configuration_test(embedding_model, test_sets["short_docs"])
        display_results_table(results, "Configuration Benchmark")
        all_results.extend(results)
        save_results(results, args.output_dir, "configs")

    # Save combined results
    if len(test_types) > 1:
        save_results(all_results, args.output_dir, "combined")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled - starting benchmark")
    main()
