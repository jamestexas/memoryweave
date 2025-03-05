"""
Optimized RAM Benchmark Framework for MemoryWeave

This module provides an optimized version of the RAM benchmark framework
that uses shared model instances and implements proper memory management
to significantly reduce memory usage during benchmarking.
"""

import gc
import json
import logging
import os
import time
import traceback
from datetime import datetime
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import psutil
import rich_click as click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Configure logging
console = Console(highlight=True)
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_path=False, rich_tracebacks=True)],
)

# Silence other loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)

logger = logging.getLogger("ram_benchmark")

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"


# Define system types for benchmarking
class SystemType(str, Enum):
    """Types of systems to benchmark."""

    MEMORYWEAVE_HYBRID = "memoryweave_hybrid"  # New hybrid approach
    MEMORYWEAVE_CHUNKED = "memoryweave_chunked"
    MEMORYWEAVE_STANDARD = "memoryweave_standard"
    STANDARD_RAG = "standard_rag"
    RECENCY_BIASED = "recency_biased"
    CONTEXT_WINDOW = "context_window"


# Shared resources container
class SharedResources:
    """Container for shared resources to minimize memory usage."""

    def __init__(self):
        """Initialize the shared resources container."""
        self.llm_provider = None
        self.embedding_model = None
        self.token_usage = {}

    def clear_resources(self):
        """Clear and release resources."""
        self.llm_provider = None
        self.embedding_model = None
        gc.collect()


class OptimizedBenchmarkRunner:
    """Memory-efficient benchmark runner for MemoryWeave systems."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embedding_model: str = None,
        systems_to_test: list[SystemType] = None,
        scenarios_to_run: list[str] = None,
        output_dir: str = "./benchmark_results",
        max_memories_per_scenario: int = 500,
        debug: bool = False,
        sequential_testing: bool = True,
    ):
        """Initialize the benchmark runner."""
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.output_dir = output_dir
        self.max_memories_per_scenario = max_memories_per_scenario
        self.debug = debug
        self.sequential_testing = sequential_testing

        # Set default systems to test if not provided
        if systems_to_test is None:
            self.systems_to_test = [
                SystemType.MEMORYWEAVE_HYBRID,  # Test new hybrid approach
                SystemType.MEMORYWEAVE_STANDARD,
                SystemType.STANDARD_RAG,
                # Don't test chunked by default to save memory
                # SystemType.MEMORYWEAVE_CHUNKED,
                # SystemType.RECENCY_BIASED,
                # SystemType.CONTEXT_WINDOW,
            ]
        else:
            self.systems_to_test = systems_to_test

        # Set up scenarios
        from evaluations.ram_benchmark import (
            ConversationalMemory,
            DocumentRetrieval,
            LargeContextHandling,
            MixedKnowledge,
            TemporalReferences,
        )

        self.all_scenarios = {
            "document": DocumentRetrieval(),
            "conversation": ConversationalMemory(),
            "temporal": TemporalReferences(),
            "mixed": MixedKnowledge(),
            "large_context": LargeContextHandling(),
        }

        # Filter scenarios based on input
        if scenarios_to_run:
            self.scenarios = {k: v for k, v in self.all_scenarios.items() if k in scenarios_to_run}
        else:
            self.scenarios = self.all_scenarios

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize shared resources
        self.shared_resources = SharedResources()

        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "systems_tested": [s.value for s in self.systems_to_test],
            "scenarios_run": list(self.scenarios.keys()),
            "scenario_results": {},
            "system_metrics": {},
        }

        # Configure logging based on debug setting
        if debug:
            logger.setLevel(logging.DEBUG)

        # Track memory usage
        self.memory_tracker = {"peak_usage": {}, "detailed": {}}

    def initialize_shared_resources(self):
        """Initialize shared resources used by all systems."""
        console.print("[bold cyan]Initializing Shared Resources[/bold cyan]")

        try:
            # Import here to avoid loading all modules at startup
            from memoryweave.api.llm_provider import LLMProvider
            from memoryweave.components.retriever import _get_embedder

            # Make sure embedding_model has a default value
            if self.embedding_model is None:
                self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
                console.print(
                    f"No embedding model specified, using default: {self.embedding_model}",
                    style="bold yellow",
                )

            # Initialize embedding model (shared across all systems)
            console.print(f"Loading embedding model: {self.embedding_model}")
            self.shared_resources.embedding_model = _get_embedder(
                model_name=self.embedding_model,
                device="mps",
            )

            # Initialize LLM provider (shared across all systems)
            console.print(f"Loading LLM: {self.model_name}")
            self.shared_resources.llm_provider = LLMProvider(
                model_name=self.model_name,
                device="mps",
            )

            # Record initial memory usage
            initial_memory = self._get_memory_usage()
            console.print(f"Initial memory usage: {initial_memory:.2f} MB")

            console.print("[green]✓[/green] Shared resources initialized")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to initialize shared resources: {str(e)}")
            if self.debug:
                console.print(traceback.format_exc())
            return False

        return True

    def create_system(self, system_type: SystemType):
        """Create a specific system type using shared resources."""
        if system_type == SystemType.MEMORYWEAVE_HYBRID:
            # Import the hybrid implementation
            from memoryweave.api.hybrid_memory_weave import HybridMemoryWeaveAPI

            # Create the system with shared resources
            system = HybridMemoryWeaveAPI(
                model_name=self.model_name,
                embedding_model_name=self.embedding_model,
                debug=self.debug,
            )

            # Replace the models with shared instances
            system.llm_provider = self.shared_resources.llm_provider
            system.embedding_model = self.shared_resources.embedding_model

            console.print("  [green]✓[/green] HybridMemoryWeave initialized")
            return system

        elif system_type == SystemType.MEMORYWEAVE_CHUNKED:
            # Import the chunked implementation
            from memoryweave.api.chunked_memory_weave import ChunkedMemoryWeaveAPI

            # Create the system with shared resources
            system = ChunkedMemoryWeaveAPI(
                model_name=self.model_name,
                embedding_model_name=self.embedding_model,
                debug=self.debug,
            )

            # Replace the models with shared instances
            system.llm_provider = self.shared_resources.llm_provider
            system.embedding_model = self.shared_resources.embedding_model

            console.print("  [green]✓[/green] ChunkedMemoryWeave initialized")
            return system

        elif system_type == SystemType.MEMORYWEAVE_STANDARD:
            # Import the standard implementation
            from memoryweave.api.memory_weave import MemoryWeaveAPI

            # Create the system with shared resources
            system = MemoryWeaveAPI(
                model_name=self.model_name,
                embedding_model_name=self.embedding_model,
                debug=self.debug,
            )

            # Replace the models with shared instances
            system.llm_provider = self.shared_resources.llm_provider
            system.embedding_model = self.shared_resources.embedding_model

            console.print("  [green]✓[/green] MemoryWeave (Standard) initialized")
            return system

        elif system_type == SystemType.STANDARD_RAG:
            # Import the standard implementation
            from memoryweave.api.memory_weave import MemoryWeaveAPI

            # Create a standard RAG system for comparison
            system = MemoryWeaveAPI(
                model_name=self.model_name,
                embedding_model_name=self.embedding_model,
                enable_category_management=False,
                enable_personal_attributes=False,
                enable_semantic_coherence=False,
                enable_dynamic_thresholds=False,
                debug=self.debug,
            )

            # Replace the models with shared instances
            system.llm_provider = self.shared_resources.llm_provider
            system.embedding_model = self.shared_resources.embedding_model

            # Configure the system to use only similarity retrieval
            system.strategy.initialize({
                "confidence_threshold": 0.1,
                "similarity_weight": 1.0,  # Only use similarity
                "associative_weight": 0.0,  # Disable associative retrieval
                "temporal_weight": 0.0,  # Disable temporal relevance
                "activation_weight": 0.0,  # Disable activation boosting
            })

            console.print("  [green]✓[/green] Standard RAG initialized")
            return system

        elif system_type == SystemType.RECENCY_BIASED:
            # Import the standard implementation
            from memoryweave.api.memory_weave import MemoryWeaveAPI

            # Create a recency-biased system
            system = MemoryWeaveAPI(
                model_name=self.model_name,
                embedding_model_name=self.embedding_model,
                enable_category_management=False,
                enable_personal_attributes=False,
                enable_semantic_coherence=False,
                debug=self.debug,
            )

            # Replace the models with shared instances
            system.llm_provider = self.shared_resources.llm_provider
            system.embedding_model = self.shared_resources.embedding_model

            # Configure the system to prioritize recency
            system.strategy.initialize({
                "confidence_threshold": 0.1,
                "similarity_weight": 0.5,  # Some weight on similarity
                "associative_weight": 0.0,  # Disable associative retrieval
                "temporal_weight": 0.5,  # Heavy weight on temporal relevance
                "activation_weight": 0.0,  # Disable activation patterns
            })

            console.print("  [green]✓[/green] Recency-biased system initialized")
            return system

        elif system_type == SystemType.CONTEXT_WINDOW:
            # Import the standard implementation
            from memoryweave.api.memory_weave import MemoryWeaveAPI

            # Create a simple context window system
            system = MemoryWeaveAPI(
                model_name=self.model_name,
                embedding_model_name=self.embedding_model,
                debug=self.debug,
            )

            # Replace the models with shared instances
            system.llm_provider = self.shared_resources.llm_provider
            system.embedding_model = self.shared_resources.embedding_model

            console.print("  [green]✓[/green] Context window system initialized")
            return system

        else:
            console.print(f"[red]✗[/red] Unknown system type: {system_type}")
            return None

    def run_benchmarks(self):
        """Run all benchmarks with optimized memory usage."""
        # Initialize shared resources
        if not self.initialize_shared_resources():
            console.print("[red]Failed to initialize shared resources, aborting benchmark[/red]")
            return

        # Track overall metrics
        system_metrics = {
            system_type.value: {"avg_time": 0, "avg_accuracy": 0, "memory_mb": 0}
            for system_type in self.systems_to_test
        }

        # Run benchmarks for each scenario
        for scenario_name, scenario in self.scenarios.items():
            console.print(f"\n[bold cyan]Running benchmark: {scenario.name}[/bold cyan]")
            console.print(f"[dim]{scenario.description}[/dim]\n")

            # Get memories for this scenario (with limit)
            memories = scenario.get_memories()[: self.max_memories_per_scenario]
            queries = scenario.get_queries()

            # Display scenario stats
            console.print(f"[bold]Memory count:[/bold] {len(memories)}")
            console.print(f"[bold]Query count:[/bold] {len(queries)}")

            # Run benchmark for each system
            scenario_results = {}

            for system_type in self.systems_to_test:
                console.print(f"\n[bold]Testing: {system_type.value}[/bold]")

                # Clear memory before creating each system
                self._clear_memory()

                try:
                    # Create system with shared resources
                    system = self.create_system(system_type)
                    if system is None:
                        console.print(
                            f"[yellow]Skipping {system_type.value} (creation failed)[/yellow]"
                        )
                        continue

                    # Record memory usage after initialization
                    init_memory = self._get_memory_usage()
                    peak_memory = init_memory
                    self.memory_tracker["peak_usage"][system_type] = init_memory

                    # Add memories in batches with progress tracking
                    batch_size = 50
                    memory_count = len(memories)
                    batch_count = (memory_count + batch_size - 1) // batch_size

                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[bold blue]{task.description}"),
                        BarColumn(),
                        TimeElapsedColumn(),
                    ) as progress:
                        memory_task = progress.add_task(
                            f"Adding {memory_count} memories to {system_type.value}...",
                            total=memory_count,
                        )

                        for i in range(0, memory_count, batch_size):
                            batch = memories[i : i + batch_size]
                            for memory in batch:
                                text = memory.get("text", "")
                                metadata = {k: v for k, v in memory.items() if k != "text"}
                                system.add_memory(text, metadata)
                                progress.advance(memory_task)

                            # Track peak memory usage
                            current_memory = self._get_memory_usage()
                            peak_memory = max(peak_memory, current_memory)

                            # Explicitly run garbage collection between batches
                            gc.collect()

                    # Run queries with memory monitoring
                    console.print(f"\nRunning {len(queries)} queries on {system_type.value}...")

                    query_results = []
                    query_times = []
                    accuracy_scores = []

                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[bold blue]{task.description}"),
                        BarColumn(),
                        TimeElapsedColumn(),
                    ) as progress:
                        query_task = progress.add_task(
                            f"Processing queries on {system_type.value}...", total=len(queries)
                        )

                        for query in queries:
                            # Measure query time
                            start_time = time.time()

                            # Get response
                            response = system.chat(query, max_new_tokens=150)

                            # Calculate elapsed time
                            elapsed = time.time() - start_time
                            query_times.append(elapsed)

                            # Calculate accuracy based on expected answers
                            expected_answers = scenario.get_expected_answers(query)
                            accuracy = self._calculate_accuracy(response, expected_answers)
                            accuracy_scores.append(accuracy)

                            # Store result
                            query_results.append({
                                "query": query,
                                "response": response,
                                "time_seconds": elapsed,
                                "accuracy": accuracy,
                                "expected": expected_answers,
                            })

                            progress.advance(query_task)

                            # Track peak memory usage
                            current_memory = self._get_memory_usage()
                            peak_memory = max(peak_memory, current_memory)

                    # Calculate memory increase from baseline
                    memory_increase = peak_memory - init_memory

                    # Calculate summary metrics
                    avg_time = sum(query_times) / len(query_times) if query_times else 0
                    avg_accuracy = (
                        sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
                    )

                    # Update system metrics
                    system_metrics[system_type.value]["avg_time"] += avg_time / len(self.scenarios)
                    system_metrics[system_type.value]["avg_accuracy"] += avg_accuracy / len(
                        self.scenarios
                    )
                    system_metrics[system_type.value]["memory_mb"] = max(
                        system_metrics[system_type.value]["memory_mb"], memory_increase
                    )

                    # Display summary for this system
                    console.print(f"\n[bold]Results for {system_type.value}:[/bold]")
                    console.print(f"  Average query time: [cyan]{avg_time:.3f}s[/cyan]")
                    console.print(f"  Average accuracy: [cyan]{avg_accuracy:.2f}[/cyan]")
                    console.print(f"  Memory usage: [cyan]{memory_increase:.2f} MB[/cyan]")

                    # Store result
                    scenario_results[system_type.value] = {
                        "queries": query_results,
                        "avg_time": avg_time,
                        "avg_accuracy": avg_accuracy,
                        "memory_usage_mb": memory_increase,
                    }

                    # Track detailed memory stats
                    if system_type not in self.memory_tracker["detailed"]:
                        self.memory_tracker["detailed"][system_type] = []

                    self.memory_tracker["detailed"][system_type].append({
                        "scenario": scenario_name,
                        "initial_mb": init_memory,
                        "peak_mb": peak_memory,
                        "increase_mb": memory_increase,
                    })

                    # Clean up system resources
                    del system
                    self._clear_memory()

                except Exception as e:
                    console.print(f"[red]Error testing {system_type.value}: {str(e)}[/red]")
                    if self.debug:
                        console.print(traceback.format_exc())

            # Save scenario results
            self.results["scenario_results"][scenario_name] = scenario_results

        # Save overall system metrics
        self.results["system_metrics"] = system_metrics

        # Display overall results
        self.display_results()

        # Save results to file
        self.save_results()

    def _calculate_accuracy(self, response: str, expected_answers: list[str]) -> float:
        """Calculate accuracy score based on presence of expected answers in response."""
        if not expected_answers:
            return 0.0

        response_lower = response.lower()
        found_count = sum(1 for ans in expected_answers if ans.lower() in response_lower)

        # Partial credit: divide found by total expected
        return found_count / len(expected_answers)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB

    def _clear_memory(self):
        """Force memory cleanup."""
        gc.collect()

        # Try a more aggressive collection on supported platforms
        if hasattr(gc, "collect") and callable(gc.collect):
            try:
                gc.collect(2)  # Collect using the most thorough collection method
            except TypeError:
                gc.collect()

    def display_results(self):
        """Display benchmark results in a nice table format."""
        console.print("\n[bold cyan]Overall Benchmark Results[/bold cyan]")

        # Create a table for overall system comparison
        table = Table(title="System Performance Comparison")

        # Add columns
        table.add_column("System", style="cyan")
        table.add_column("Avg Time (s)", style="yellow")
        table.add_column("Avg Accuracy", style="green")
        table.add_column("Memory Usage (MB)", style="magenta")

        # Add rows for each system
        for system_type in self.systems_to_test:
            system_name = system_type.value
            metrics = self.results["system_metrics"].get(system_name, {})

            if metrics:
                table.add_row(
                    system_name,
                    f"{metrics['avg_time']:.3f}",
                    f"{metrics['avg_accuracy']:.2f}",
                    f"{metrics['memory_mb']:.2f}",
                )

        console.print(table)

        # Create comparison charts
        self._create_comparative_charts()

        # Display memory efficiency comparison
        self._display_memory_efficiency()

    def _display_memory_efficiency(self):
        """Display memory efficiency comparison between systems."""
        if not self.memory_tracker["detailed"]:
            return

        console.print("\n[bold cyan]Memory Efficiency Details[/bold cyan]")

        # Create a table for memory efficiency
        table = Table(title="Memory Efficiency by Scenario")

        # Add columns
        table.add_column("System", style="cyan")
        table.add_column("Scenario", style="yellow")
        table.add_column("Initial (MB)", style="blue")
        table.add_column("Peak (MB)", style="magenta")
        table.add_column("Increase (MB)", style="green")

        # Add rows for each system and scenario
        for system_type, scenarios in self.memory_tracker["detailed"].items():
            for i, scenario in enumerate(scenarios):
                if isinstance(system_type, SystemType):
                    system_name = system_type.value
                else:
                    system_name = str(system_type)

                table.add_row(
                    system_name if i == 0 else "",
                    scenario["scenario"],
                    f"{scenario['initial_mb']:.2f}",
                    f"{scenario['peak_mb']:.2f}",
                    f"{scenario['increase_mb']:.2f}",
                )

        console.print(table)

    def _create_comparative_charts(self):
        """Create comparative visualizations of the results."""
        try:
            # Check if we have valid results to plot
            if not self.results["system_metrics"]:
                console.print("[yellow]No metrics available for visualization[/yellow]")
                return

            # Extract data for plotting
            systems = []
            times = []
            accuracies = []
            memories = []

            for system, metrics in self.results["system_metrics"].items():
                systems.append(system)
                times.append(metrics["avg_time"])
                accuracies.append(metrics["avg_accuracy"])
                memories.append(metrics["memory_mb"])

            # Set up figure with 3 subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            # Plot response times
            ax1.bar(systems, times, color="skyblue")
            ax1.set_title("Average Response Time")
            ax1.set_ylabel("Time (seconds)")
            ax1.tick_params(axis="x", rotation=45)

            # Plot accuracy scores
            ax2.bar(systems, accuracies, color="lightgreen")
            ax2.set_title("Average Accuracy")
            ax2.set_ylabel("Accuracy Score")
            ax2.tick_params(axis="x", rotation=45)

            # Plot memory usage
            ax3.bar(systems, memories, color="salmon")
            ax3.set_title("Memory Usage")
            ax3.set_ylabel("Memory (MB)")
            ax3.tick_params(axis="x", rotation=45)

            # Adjust layout and save figure
            plt.tight_layout()

            # Save the chart
            chart_path = os.path.join(self.output_dir, "performance_comparison.png")
            plt.savefig(chart_path)

            console.print(
                f"\n[bold green]Performance comparison chart saved to:[/bold green] {chart_path}"
            )

            # Create memory efficiency chart
            if self.memory_tracker["detailed"]:
                self._create_memory_efficiency_chart()

        except Exception as e:
            console.print(f"[yellow]Failed to create charts: {str(e)}[/yellow]")
            if self.debug:
                console.print(traceback.format_exc())

    def _create_memory_efficiency_chart(self):
        """Create a chart showing memory efficiency metrics."""
        try:
            # Extract data for the chart
            systems = []
            scenarios = []
            memory_increases = []

            for system_type, scenario_data in self.memory_tracker["detailed"].items():
                if isinstance(system_type, SystemType):
                    system_name = system_type.value
                else:
                    system_name = str(system_type)

                for scenario in scenario_data:
                    systems.append(system_name)
                    scenarios.append(scenario["scenario"])
                    memory_increases.append(scenario["increase_mb"])

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))

            # Determine unique systems and scenarios
            unique_systems = list(dict.fromkeys(systems))
            unique_scenarios = list(dict.fromkeys(scenarios))

            # Prepare data for grouped bar chart
            bar_width = 0.2
            index = np.arange(len(unique_scenarios))

            # Plot bars for each system
            for i, system in enumerate(unique_systems):
                system_increases = [
                    memory_increases[j]
                    for j in range(len(systems))
                    if systems[j] == system and scenarios[j] in unique_scenarios
                ]

                # Pad with zeros if missing scenarios
                if len(system_increases) < len(unique_scenarios):
                    system_increases.extend([0] * (len(unique_scenarios) - len(system_increases)))

                offset = bar_width * (i - len(unique_systems) / 2 + 0.5)
                ax.bar(index + offset, system_increases, bar_width, label=system)

            # Set chart labels and title
            ax.set_xlabel("Scenario")
            ax.set_ylabel("Memory Increase (MB)")
            ax.set_title("Memory Usage by System and Scenario")
            ax.set_xticks(index)
            ax.set_xticklabels(unique_scenarios, rotation=45, ha="right")
            ax.legend()

            plt.tight_layout()

            # Save the chart
            chart_path = os.path.join(self.output_dir, "memory_efficiency_comparison.png")
            plt.savefig(chart_path)

            console.print(
                f"[bold green]Memory efficiency chart saved to:[/bold green] {chart_path}"
            )

        except Exception as e:
            console.print(f"[yellow]Failed to create memory efficiency chart: {str(e)}[/yellow]")
            if self.debug:
                console.print(traceback.format_exc())

    def save_results(self):
        """Save benchmark results to a JSON file."""
        # Create a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")

        try:
            # Add memory tracking info to results
            self.results["memory_tracking"] = self.memory_tracker

            # Write results to file
            with open(filename, "w") as f:
                json.dump(self.results, f, indent=2)

            console.print(f"\n[bold green]Results saved to:[/bold green] {filename}")

        except Exception as e:
            console.print(f"[yellow]Failed to save results: {str(e)}[/yellow]")
            if self.debug:
                console.print(traceback.format_exc())


@click.command()
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help=f"Name of the Hugging Face model to load (default: {DEFAULT_MODEL})",
)
@click.option(
    "--embedding-model",
    default=None,
    help="Name of the embedding model to use (default: use MemoryWeave's default)",
)
@click.option(
    "--systems",
    multiple=True,
    type=click.Choice([s.value for s in SystemType]),
    default=["memoryweave_hybrid", "memoryweave_standard", "standard_rag"],
    show_default=True,
    help="Specific systems to test (can specify multiple)",
)
@click.option(
    "--scenarios",
    multiple=True,
    type=click.Choice(["document", "conversation", "temporal", "mixed", "large_context"]),
    default=["mixed"],
    show_default=True,
    help="Specific scenarios to run (can specify multiple)",
)
@click.option(
    "--output-dir",
    default="./benchmark_results",
    help="Directory to save benchmark results (default: ./benchmark_results)",
)
@click.option(
    "--max-memories",
    default=500,
    help="Maximum number of memories to use per scenario (default: 500)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging for more detailed output",
)
@click.option(
    "--sequential",
    is_flag=True,
    default=True,
    help="Test systems sequentially to minimize memory usage (default: True)",
)
def main(model, embedding_model, systems, scenarios, output_dir, max_memories, debug, sequential):
    """
    Optimized RAM Benchmark: Retrieval Augmented Memory Benchmarking Tool

    This tool benchmarks MemoryWeave against other memory/retrieval systems across
    multiple scenarios to evaluate performance, accuracy, and resource usage.
    It uses shared model instances and efficient memory management.
    """
    # Configure logging based on debug flag
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        console.print("[yellow]Debug logging enabled[/yellow]")

    # Print banner
    console.print(
        Panel.fit(
            "[bold cyan]Optimized RAM Benchmark: Retrieval Augmented Memory[/bold cyan]\n\n"
            f"Model: [yellow]{model}[/yellow]\n"
            f"Output directory: [yellow]{output_dir}[/yellow]\n"
            "This benchmark compares different memory and retrieval approaches.",
            border_style="cyan",
        )
    )

    # Convert systems to enum types
    systems_to_test = [SystemType(s) for s in systems] if systems else None

    # Initialize and run benchmark
    try:
        benchmark = OptimizedBenchmarkRunner(
            model_name=model,
            embedding_model=embedding_model,
            systems_to_test=systems_to_test,
            scenarios_to_run=scenarios,
            output_dir=output_dir,
            max_memories_per_scenario=max_memories,
            debug=debug,
            sequential_testing=sequential,
        )

        benchmark.run_benchmarks()

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Benchmark interrupted by user[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error running benchmark: {str(e)}[/bold red]")
        if debug:
            console.print(traceback.format_exc())


if __name__ == "__main__":
    main()
