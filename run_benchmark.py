#!/usr/bin/env python3
"""
MemoryWeave Unified Benchmark Runner

This script provides a single entry point for running various MemoryWeave benchmarks
using configuration files. It supports running contextual fabric benchmarks,
memory retrieval benchmarks, baseline comparisons, and synthetic benchmarks.

Usage:
  python run_benchmark.py --config <config_file> [--output <output_file>]
  [--memories <num>] [--no-viz]

Examples:
  python run_benchmark.py --config configs/contextual_fabric_benchmark.yaml
  python run_benchmark.py --config configs/memory_retrieval_benchmark.yaml --memories 200
  python run_benchmark.py --config configs/baseline_comparison.yaml --output results.json
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rich_click as click
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

# Import necessary modules
from memoryweave.baselines import BM25Retriever, VectorBaselineRetriever
from memoryweave.benchmarks.contextual_fabric_benchmark import ContextualFabricBenchmark
from memoryweave.benchmarks.memory_retrieval_benchmark import (
    BenchmarkConfig as MRBConfig,
    MemoryRetrievalBenchmark,
)
from memoryweave.benchmarks.visualize_contextual_fabric import (
    create_f1_comparison_chart,
    create_summary_chart,
)
from memoryweave.components.memory_manager import MemoryManager
from memoryweave.evaluation.baseline_comparison import BaselineComparison, BaselineConfig

# Import benchmark implementation
from memoryweave.evaluation.synthetic.benchmark import (
    BenchmarkConfig as SynBConfig,
    SyntheticBenchmark,
)
from memoryweave.interfaces.retrieval import Query, QueryType
from memoryweave.storage.memory_store import Memory, MemoryStore

# set up rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("memoryweave")
console = Console()


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    type: str  # "contextual_fabric", "memory_retrieval", "baseline", or "synthetic"
    description: str = ""

    # Common parameters
    memories: int = 100
    queries: int = 20
    embedding_dim: int = 384
    output_file: str = "benchmark_results.json"
    visualize: bool = True

    # Specific parameters for each benchmark type
    parameters: dict[str, Any] = field(default_factory=dict)

    # Configurations to test (list of dicts)
    configurations: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_file(cls, config_file: str) -> "BenchmarkConfig":
        """Load configuration from a YAML or JSON file."""
        path = Path(config_file)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        if path.suffix.lower() in [".yaml", ".yml"]:
            with open(path) as f:
                config_data = yaml.safe_load(f)
        elif path.suffix.lower() == ".json":
            with open(path) as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")

        return cls(**config_data)


class UnifiedBenchmark:
    """Unified benchmark system for MemoryWeave."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize with benchmark configuration."""
        self.config = config
        self.results = {}

    def run(self) -> dict[str, Any]:
        """Run the benchmark according to configuration."""
        benchmark_type = self.config.type.lower()

        console.print(
            f"[bold green]Running {benchmark_type} benchmark: {self.config.name}[/bold green]"
        )

        # Select and run appropriate benchmark
        if benchmark_type == "contextual_fabric":
            return self._run_contextual_fabric()
        elif benchmark_type == "memory_retrieval":
            return self._run_memory_retrieval()
        elif benchmark_type == "baseline":
            return self._run_baseline_comparison()
        elif benchmark_type == "synthetic":
            return self._run_synthetic()
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")

    def _run_contextual_fabric(self) -> dict[str, Any]:
        """Run the contextual fabric benchmark."""
        # Extract parameters
        memories = self.config.memories
        embedding_dim = self.config.embedding_dim
        output_file = self.config.output_file

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            # Import benchmark implementation

            # Initialize benchmark
            benchmark = ContextualFabricBenchmark(embedding_dim=embedding_dim)

            # Update parameters if provided
            if self.config.parameters:
                for component, params in self.config.parameters.items():
                    if component == "contextual_fabric_strategy" and hasattr(
                        benchmark, "contextual_fabric_strategy"
                    ):
                        benchmark.contextual_fabric_strategy.initialize(params)
                    elif component == "baseline_strategy" and hasattr(
                        benchmark, "baseline_strategy"
                    ):
                        benchmark.baseline_strategy.initialize(params)
                    # Add more component configurations as needed

            # Run benchmark
            console.print(f"Running benchmark with [cyan]{memories}[/cyan] memories...")
            results = benchmark.run_benchmark(num_memories=memories, output_file=output_file)

            # Display summary
            self._display_summary(results)

            # Visualize results if requested
            if self.config.visualize:
                output_dir = Path(output_file).parent / "charts"
                os.makedirs(output_dir, exist_ok=True)
                try:
                    create_f1_comparison_chart(
                        results, str(output_dir / "contextual_fabric_comparison.png")
                    )
                    create_summary_chart(results, str(output_dir / "contextual_fabric_summary.png"))
                    console.print(f"[green]Visualizations saved to {output_dir}[/green]")
                except ImportError:
                    console.print(
                        "[yellow]Visualization module not found. Skipping visualization.[/yellow]"
                    )

            return results

        except ImportError:
            console.print(
                "[bold red]Error: contextual_fabric_benchmark module not found.[/bold red]"
            )
            console.print("Make sure the memoryweave package is installed correctly.")
            raise

        except Exception as e:
            console.print(f"[bold red]Error running contextual fabric benchmark: {e}[/bold red]")
            raise

    def _run_memory_retrieval(self) -> dict[str, Any]:
        """Run the memory retrieval benchmark."""
        # Extract parameters
        memories = self.config.memories
        queries = self.config.queries
        output_file = self.config.output_file

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            # Convert configurations to expected format
            configs = []

            for cfg in self.config.configurations:
                configs.append(MRBConfig(**cfg))

            # Initialize and run benchmark
            benchmark = MemoryRetrievalBenchmark(configs)
            benchmark.generate_test_data(num_memories=memories, num_queries=queries)
            results = benchmark.run_benchmark(save_path=output_file)

            console.print(f"[green]Results saved to {output_file}[/green]")
            console.print("[green]Visualization saved to benchmark_results.png[/green]")

            return results

        except ImportError:
            console.print(
                "[bold red]Error: memory_retrieval_benchmark module not found.[/bold red]"
            )
            console.print("Make sure the memoryweave package is installed correctly.")
            raise

        except Exception as e:
            console.print(f"[bold red]Error running memory retrieval benchmark: {e}[/bold red]")
            raise

    def _run_baseline_comparison(self) -> dict[str, Any]:
        """Run the baseline comparison benchmark."""
        # Extract dataset path from parameters
        dataset_path = self.config.parameters.get("dataset", "sample_baseline_dataset.json")

        # Other parameters
        max_results = self.config.parameters.get("max_results", 10)
        threshold = self.config.parameters.get("threshold", 0.0)
        output_file = self.config.output_file

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            # Load dataset
            console.print(f"Loading dataset: [cyan]{dataset_path}[/cyan]")

            with open(dataset_path) as f:
                data = json.load(f)

            # Create memory objects
            memories = []
            for mem_data in data.get("memories", []):
                embedding = (
                    np.array(mem_data.get("embedding", [])) if "embedding" in mem_data else None
                )

                memory = Memory(
                    id=mem_data.get("id", str(len(memories))),
                    embedding=embedding,
                    content=mem_data.get("content", {"text": "", "metadata": {}}),
                    metadata=mem_data.get("metadata", {}),
                )
                memories.append(memory)

            # Create query objects
            queries = []
            for q_data in data.get("queries", []):
                embedding = np.array(q_data.get("embedding", [])) if "embedding" in q_data else None

                query = Query(
                    text=q_data.get("text", ""),
                    embedding=embedding,
                    query_type=QueryType.UNKNOWN,
                    extracted_keywords=q_data.get("keywords", []),
                    extracted_entities=q_data.get("entities", []),
                )
                queries.append(query)

            # Get relevant memory IDs
            relevant_ids = data.get("relevant_ids", [])

            console.print(
                f"Loaded dataset with [cyan]{len(memories)}[/cyan] memories and [cyan]{len(queries)}[/cyan] queries"
            )

            # Initialize memory manager
            memory_store = MemoryStore()
            memory_store.add_multiple(memories)
            memory_manager = MemoryManager(memory_store=memory_store)

            # Create retriever adaptable for baseline comparison
            class MemoryWeaveRetriever:
                """Simplified retriever that works with MemoryManager for benchmark purposes."""

                def __init__(self, memory_manager):
                    self.memory_manager = memory_manager

                def retrieve(self, query, top_k=10, threshold=0.0, **kwargs):
                    """Basic vector similarity retrieval implementation."""
                    # For benchmark purposes, implement a simple vector similarity search
                    # that's compatible with our Memory objects

                    if query.embedding is None or not hasattr(query, "embedding"):
                        return {
                            "memories": [],
                            "scores": [],
                            "strategy": "memoryweave",
                            "parameters": {"max_results": top_k, "threshold": threshold},
                            "metadata": {"query_time": 0.0},
                        }

                    # Get all memories
                    all_memories = self.memory_manager.get_all_memories()
                    if not all_memories:
                        return {
                            "memories": [],
                            "scores": [],
                            "strategy": "memoryweave",
                            "parameters": {"max_results": top_k, "threshold": threshold},
                            "metadata": {"query_time": 0.0},
                        }

                    # Calculate similarities
                    query_embedding = np.array(query.embedding).reshape(1, -1)
                    results = []

                    for memory in all_memories:
                        if memory.embedding is not None:
                            # Use cosine similarity
                            memory_embedding = np.array(memory.embedding).reshape(1, -1)

                            # Ensure same dimensions by padding if necessary
                            if memory_embedding.shape[1] < query_embedding.shape[1]:
                                pad_width = query_embedding.shape[1] - memory_embedding.shape[1]
                                memory_embedding = np.pad(
                                    memory_embedding, ((0, 0), (0, pad_width))
                                )
                            elif memory_embedding.shape[1] > query_embedding.shape[1]:
                                # Use only the first dimensions of memory embedding
                                memory_embedding = memory_embedding[:, : query_embedding.shape[1]]

                            # Calculate dot product
                            similarity = np.dot(query_embedding, memory_embedding.T)[0][0]

                            # Normalize
                            query_norm = np.linalg.norm(query_embedding)
                            memory_norm = np.linalg.norm(memory_embedding)
                            if query_norm > 0 and memory_norm > 0:
                                similarity = similarity / (query_norm * memory_norm)

                            if similarity >= threshold:
                                results.append((memory, float(similarity)))

                    # Sort by similarity (descending)
                    results.sort(key=lambda x: x[1], reverse=True)

                    # Take top_k
                    results = results[:top_k]

                    return {
                        "memories": [memory for memory, _ in results],
                        "scores": [score for _, score in results],
                        "strategy": "memoryweave",
                        "parameters": {"max_results": top_k, "threshold": threshold},
                        "metadata": {"query_time": 0.0},
                    }

            # Create memoryweave retriever
            memoryweave_retriever = MemoryWeaveRetriever(memory_manager)

            # Load baseline configurations from file
            config_path = self.config.parameters.get("config", "baselines_config.yaml")
            console.print(f"Loading baseline configurations from: [cyan]{config_path}[/cyan]")

            retriever_classes = {
                "bm25": BM25Retriever,
                "vector": VectorBaselineRetriever,
            }

            # Load configurations
            with open(config_path) as f:
                configs = yaml.safe_load(f)

            baseline_configs = []
            for config in configs:
                retriever_type = config.get("type")
                if retriever_type not in retriever_classes:
                    console.print(
                        f"[yellow]Warning: Unknown baseline type '{retriever_type}', skipping[/yellow]"
                    )
                    continue

                baseline_configs.append(
                    BaselineConfig(
                        name=config.get("name", retriever_type),
                        retriever_class=retriever_classes[retriever_type],
                        parameters=config.get("parameters", {}),
                    )
                )

            console.print(f"Loaded [cyan]{len(baseline_configs)}[/cyan] baseline configurations")

            # Create comparison framework
            comparison = BaselineComparison(
                memory_manager=memory_manager,
                memoryweave_retriever=memoryweave_retriever,
                baseline_configs=baseline_configs,
                metrics=["precision", "recall", "f1", "mrr"],
            )

            # Run comparison
            with Progress(
                TextColumn("[bold green]{task.description}"),
                BarColumn(),
                TextColumn("[cyan]{task.completed}/{task.total}"),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task("Running baseline comparison...", total=len(queries))

                # Run comparison
                result = comparison.run_comparison(
                    queries=queries,
                    relevant_memory_ids=relevant_ids,
                    max_results=max_results,
                    threshold=threshold,
                )

                progress.update(task, completed=len(queries))

            # Save results
            comparison.save_results(result, output_file)
            console.print(f"[green]Saved comparison results to {output_file}[/green]")

            # Generate visualization if requested
            if self.config.visualize:
                viz_path = output_file.replace(".json", "_viz.png")
                comparison.visualize_results(result, viz_path)
                console.print(f"[green]Saved visualization to {viz_path}[/green]")

                # Generate HTML report if requested
                html_path = output_file.replace(".json", "_report.html")
                comparison.generate_html_report(result, html_path)
                console.print(f"[green]Saved HTML report to {html_path}[/green]")

            # Display summary
            self._display_baseline_summary(result)

            return result

        except ImportError as e:
            console.print(f"[bold red]Error importing required modules: {e}[/bold red]")
            console.print("Make sure the memoryweave package is installed correctly.")
            raise

        except FileNotFoundError as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            raise

        except Exception as e:
            console.print(f"[bold red]Error running baseline comparison: {e}[/bold red]")
            raise

    def _run_synthetic(self) -> dict[str, Any]:
        """Run the synthetic benchmark."""
        # Extract parameters
        dataset_path = self.config.parameters.get("dataset")
        save_path = self.config.output_file
        random_seed = self.config.parameters.get("random_seed", 42)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            # Convert configurations to expected format
            configs = []

            for cfg in self.config.configurations:
                configs.append(SynBConfig(**cfg))

            # Initialize benchmark
            benchmark = SyntheticBenchmark(
                configs=configs,
                dataset_path=dataset_path,
                random_seed=random_seed,
            )

            # Run benchmark
            console.print("Running synthetic benchmark...")
            results = benchmark.run_benchmark(save_path=save_path)

            console.print(f"[green]Results saved to {save_path}[/green]")
            console.print("[green]Visualization saved to synthetic_benchmark_results.png[/green]")

            return results

        except ImportError:
            console.print("[bold red]Error: synthetic benchmark module not found.[/bold red]")
            console.print("Make sure the memoryweave package is installed correctly.")
            raise

        except Exception as e:
            console.print(f"[bold red]Error running synthetic benchmark: {e}[/bold red]")
            raise

    def _display_summary(self, results: dict[str, Any]) -> None:
        """Display a summary of benchmark results."""
        if not results:
            console.print("[yellow]No results to display[/yellow]")
            return

        # Check if we have summary metrics
        if "summary" in results:
            summary = results["summary"]

            table = Table(title="Benchmark Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Baseline", justify="right")
            table.add_column("Contextual Fabric", justify="right")
            table.add_column("Improvement", justify="right", style="green")

            baseline_f1 = summary.get("average_baseline_f1", 0)
            fabric_f1 = summary.get("average_fabric_f1", 0)
            improvement = summary.get("average_improvement", 0)

            table.add_row(
                "F1 Score",
                f"{baseline_f1:.4f}",
                f"{fabric_f1:.4f}",
                f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}",
            )

            console.print(table)

            # Print memory count
            console.print(f"Number of memories: [cyan]{summary.get('num_memories', 'N/A')}[/cyan]")
            console.print(
                f"Number of test cases: [cyan]{summary.get('num_test_cases', 'N/A')}[/cyan]"
            )

    def _display_baseline_summary(self, result) -> None:
        """Display a summary of baseline comparison results."""
        # Print MemoryWeave metrics
        mw_metrics = result.memoryweave_metrics["average"]

        table = Table(title="Baseline Comparison Results")
        table.add_column("System", style="cyan")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("MRR", justify="right")

        # Add MemoryWeave row
        table.add_row(
            "MemoryWeave",
            f"{mw_metrics.get('precision', 0):.4f}",
            f"{mw_metrics.get('recall', 0):.4f}",
            f"{mw_metrics.get('f1', 0):.4f}",
            f"{mw_metrics.get('mrr', 0):.4f}",
        )

        # Add baseline rows
        for name, metrics in result.baseline_metrics.items():
            avg_metrics = metrics["average"]
            table.add_row(
                name,
                f"{avg_metrics.get('precision', 0):.4f}",
                f"{avg_metrics.get('recall', 0):.4f}",
                f"{avg_metrics.get('f1', 0):.4f}",
                f"{avg_metrics.get('mrr', 0):.4f}",
            )

        console.print(table)


# setup rich_click
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "bold italic"
click.rich_click.ERRORS_SUGGESTION = "Try running the '--help' flag for more information."
click.rich_click.MAX_WIDTH = 100


@click.command(
    help="""
# MemoryWeave Unified Benchmark Runner

This tool provides a consistent interface for running different types of MemoryWeave benchmarks using configuration files.

## Benchmark Types:
- **contextual_fabric**: Evaluates the contextual fabric architecture
- **memory_retrieval**: Compares different memory retrieval configurations
- **baseline**: Compares MemoryWeave against standard baseline methods
- **synthetic**: Tests with synthetic data having controlled properties
"""
)
@click.option(
    "--config",
    "-c",
    required=True,
    help="Path to benchmark configuration file (YAML/JSON)",
    type=click.Path(exists=True),
)
@click.option(
    "--output", "-o", default=None, help="Path to save benchmark results (overrides config file)"
)
@click.option(
    "--memories",
    "-m",
    type=int,
    default=None,
    help="Number of memories to use (overrides config file)",
)
@click.option(
    "--queries",
    "-q",
    type=int,
    default=None,
    help="Number of queries to run (overrides config file)",
)
@click.option("--no-viz", is_flag=True, help="Disable visualization generation")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(config, output, memories, queries, no_viz, debug):
    """Run the unified benchmark system."""
    # set up logging level
    if debug:
        logger.setLevel(logging.DEBUG)

    try:
        # Load configuration
        benchmark_config = BenchmarkConfig.from_file(config)

        # Override with command line options if provided
        if output:
            benchmark_config.output_file = output
        if memories:
            benchmark_config.memories = memories
        if queries:
            benchmark_config.queries = queries
        if no_viz:
            benchmark_config.visualize = False

        # Print configuration summary
        console.print(
            Panel.fit(
                f"[bold]Benchmark:[/bold] {benchmark_config.name}\n"
                f"[bold]Type:[/bold] {benchmark_config.type}\n"
                f"[bold]Memories:[/bold] {benchmark_config.memories}\n"
                f"[bold]Output:[/bold] {benchmark_config.output_file}",
                title="MemoryWeave Benchmark",
                border_style="green",
            )
        )

        # Run benchmark
        start_time = time.time()
        benchmark = UnifiedBenchmark(benchmark_config)
        results = benchmark.run()
        elapsed_time = time.time() - start_time

        console.print(
            f"\n[bold green]Benchmark completed successfully in {elapsed_time:.2f} seconds![/bold green]"
        )
        console.print(f"Results saved to: [cyan]{benchmark_config.output_file}[/cyan]")

        return 0

    except Exception as e:
        console.print(f"[bold red]Error running benchmark: {e}[/bold red]")
        if debug:
            import traceback

            console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
