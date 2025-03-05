#!/usr/bin/env python
"""
MemoryWeave Unified Benchmark System

A consolidated benchmarking tool for evaluating different MemoryWeave configurations
against various datasets and test scenarios.

Features:
- Configuration-driven benchmarking
- Rich CLI output
- Comprehensive visualizations
- Support for multiple benchmark types
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import rich_click as click
import yaml

# Import benchmark implementations
from contextual_fabric_benchmark import ContextualFabricBenchmark
from memory_retrieval_benchmark import BenchmarkConfig as MRBConfig, MemoryRetrievalBenchmark
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich_click import RichCommand, RichGroup

from memoryweave.evaluation.baseline_comparison import BaselineComparison
from memoryweave.evaluation.synthetic.benchmark import SyntheticBenchmark

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

        # Initialize benchmark
        benchmark = ContextualFabricBenchmark(embedding_dim=embedding_dim)

        # Update parameters if provided
        if self.config.parameters:
            for component, params in self.config.parameters.items():
                if component == "contextual_fabric_strategy" and hasattr(
                    benchmark, "contextual_fabric_strategy"
                ):
                    benchmark.contextual_fabric_strategy.initialize(params)
                elif component == "baseline_strategy" and hasattr(benchmark, "baseline_strategy"):
                    benchmark.baseline_strategy.initialize(params)
                # Add more component configurations as needed

        # Run benchmark
        console.print(f"Running benchmark with [cyan]{memories}[/cyan] memories...")
        results = benchmark.run_benchmark(num_memories=memories, output_file=output_file)

        # Display summary
        self._display_summary(results)

        return results

    def _run_memory_retrieval(self) -> dict[str, Any]:
        """Run the memory retrieval benchmark."""
        # Extract parameters
        memories = self.config.memories
        queries = self.config.queries
        output_file = self.config.output_file

        # Convert configurations to expected format

        configs = []

        for cfg in self.config.configurations:
            configs.append(MRBConfig(**cfg))

        # Initialize and run benchmark
        benchmark = MemoryRetrievalBenchmark(configs)
        benchmark.generate_test_data(num_memories=memories, num_queries=queries)
        results = benchmark.run_benchmark(save_path=output_file)

        return results

    def _run_baseline_comparison(self) -> dict[str, Any]:
        """Run the baseline comparison benchmark."""
        # Extract dataset path from parameters
        dataset_path = self.config.parameters.get("dataset", "sample_baseline_dataset.json")

        # Other parameters
        max_results = self.config.parameters.get("max_results", 10)
        threshold = self.config.parameters.get("threshold", 0.0)

        # Run the baseline comparison script logic
        try:
            # Define output_file to be used for saving results and visualization
            output_file = self.config.output_file

            from run_baseline_comparison import get_retriever, load_baseline_configs, load_dataset

            # Load dataset
            dataset = load_dataset(dataset_path)
            console.print(
                f"Loaded dataset with [cyan]{len(dataset['memories'])}[/cyan] memories and [cyan]{len(dataset['queries'])}[/cyan] queries"
            )

            # Load baselines config
            config_path = self.config.parameters.get("config", "baselines_config.yaml")
            baseline_configs = load_baseline_configs(config_path)
            console.print(f"Loaded [cyan]{len(baseline_configs)}[/cyan] baseline configurations")

            # Initialize memory manager
            from memoryweave.components.memory_manager import MemoryManager
            from memoryweave.interfaces.memory import MemoryStore

            memory_store = StandardMemoryStore()
memory_adapter = MemoryAdapter(memory_store)
            memory_store.add_multiple(dataset["memories"])
            memory_manager = MemoryManager(memory_store=memory_store)

            # Initialize retriever
            retriever_type = self.config.parameters.get("retriever", "similarity")
            memoryweave_retriever = get_retriever(retriever_type, memory_manager=memory_manager)
            console.print(f"Using MemoryWeave retriever: [cyan]{retriever_type}[/cyan]")

            # Create comparison framework
            comparison = BaselineComparison(
                memory_manager=memory_manager,
                memoryweave_retriever=memoryweave_retriever,
                baseline_configs=baseline_configs,
                metrics=["precision", "recall", "f1", "mrr"],
            )

            # Run comparison
            console.print("Running baseline comparison...")
            result = comparison.run_comparison(
                queries=dataset["queries"],
                relevant_memory_ids=dataset["relevant_ids"],
                max_results=max_results,
                threshold=threshold,
            )

            # Save results
            comparison.save_results(result, output_file)
            console.print(f"Saved comparison results to [green]{output_file}[/green]")

            # Generate visualization if requested
            if self.config.visualize:
                viz_path = output_file.replace(".json", "_viz.png")
                comparison.visualize_results(result, viz_path)
                console.print(f"Saved visualization to [green]{viz_path}[/green]")

            # Print summary
            self._display_baseline_summary(result)

            return result

        except Exception as e:
            console.print(f"[bold red]Error running baseline comparison: {e}[/bold red]")
            raise

    def _run_synthetic(self) -> dict[str, Any]:
        """Run the synthetic benchmark."""
        # Extract parameters
        dataset_path = self.config.parameters.get("dataset")
        save_path = self.config.output_file
        random_seed = self.config.parameters.get("random_seed", 42)

        # Convert configurations to expected format
        from memoryweave.evaluation.synthetic.benchmark import BenchmarkConfig as SynBConfig

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

        return results

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

    def _display_baseline_summary(self, result) -> None:
        """Display a summary of baseline comparison results."""
        # Print MemoryWeave metrics
        mw_metrics = result.memoryweave_metrics["average"]
        console.print("\n[bold]MemoryWeave Performance:[/bold]")
        console.print(f"  Precision: [cyan]{mw_metrics.get('precision', 0):.4f}[/cyan]")
        console.print(f"  Recall: [cyan]{mw_metrics.get('recall', 0):.4f}[/cyan]")
        console.print(f"  F1: [cyan]{mw_metrics.get('f1', 0):.4f}[/cyan]")
        console.print(f"  MRR: [cyan]{mw_metrics.get('mrr', 0):.4f}[/cyan]")

        # Print baseline metrics
        console.print("\n[bold]Baseline Performance:[/bold]")
        for name, metrics in result.baseline_metrics.items():
            avg_metrics = metrics["average"]
            console.print(f"[bold]{name}:[/bold]")
            console.print(f"  Precision: [cyan]{avg_metrics.get('precision', 0):.4f}[/cyan]")
            console.print(f"  Recall: [cyan]{avg_metrics.get('recall', 0):.4f}[/cyan]")
            console.print(f"  F1: [cyan]{avg_metrics.get('f1', 0):.4f}[/cyan]")
            console.print(f"  MRR: [cyan]{avg_metrics.get('mrr', 0):.4f}[/cyan]")


# CLI setup
click.RichCommand = RichCommand
click.RichGroup = RichGroup


@click.command()
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
    """
    Run the MemoryWeave unified benchmark system.

    This tool provides a consistent interface for running different types of
    MemoryWeave benchmarks using configuration files.
    """
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
        benchmark = UnifiedBenchmark(benchmark_config)
        results = benchmark.run()

        console.print("\n[bold green]Benchmark completed successfully![/bold green]")
        console.print(f"Results saved to: [cyan]{benchmark_config.output_file}[/cyan]")
        console.print(f"Results: {results}")

    except Exception as e:
        console.print(f"[bold red]Error running benchmark: {e}[/bold red]")
        if debug:
            import traceback

            console.print(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
