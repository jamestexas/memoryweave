#!/usr/bin/env python3
"""
memoryweave_performance_showcase.py

A comprehensive showcase of MemoryWeave's performance advantages in different memory scenarios.
This script runs benchmarks comparing MemoryWeave against a baseline (no memory) approach and
visualizes the results with detailed analytics.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rich_click as click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Import the MemoryWeave API
from memoryweave.api import MemoryWeaveAPI

# setup logging
console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(markup=True)])
logger = logging.getLogger("memoryweave_showcase")

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"

# Define different benchmark scenarios
SCENARIOS = dict(
    factual=dict(
        description="Basic factual recall",
        facts=[
            "My name is Alex Thompson and I'm 32 years old.",
            "I work as a data scientist at TechCorp.",
            "I live in Seattle with my dog named Rex.",
            "I graduated from Stanford University in 2015.",
            "My favorite color is teal and I prefer coffee over tea.",
        ],
    ),
    temporal=dict(
        description="Temporal and sequence-sensitive recall",
        facts=[
            "Yesterday I went hiking at Mount Rainier.",
            "This morning I had oatmeal for breakfast.",
            "Last week I visited my parents in Portland.",
            "Three days ago I watched a documentary about space.",
            "Two months ago I started learning Spanish.",
        ],
        queries=[
            ("What did I do yesterday?", ["hiking", "Mount Rainier"]),
            ("What did I eat this morning?", ["oatmeal", "breakfast"]),
            ("Where were my parents when I visited them?", ["Portland"]),
            ("What did I watch recently?", ["documentary", "space"]),
            ("What language am I learning?", ["Spanish"]),
        ],
    ),
    contextual=dict(
        description="Contextual understanding with inference",
        facts=[
            "I bought a new Tesla last month, it's electric blue.",
            "My sister Emma just had twins named Lily and Lucas.",
            "I'm allergic to peanuts so I have to be careful with Thai food.",
            "I've been playing piano since I was 7 years old.",
            "I'm planning a trip to Japan during cherry blossom season.",
        ],
        queries=[
            ("What color is my car?", ["blue", "electric blue"]),
            ("What are my niece and nephew's names?", ["Lily", "Lucas", "twins"]),
            ("Why do I need to be careful with Thai food?", ["allergic", "peanuts"]),
            ("What instrument do I play?", ["piano"]),
            ("When am I planning to visit Japan?", ["cherry blossom"]),
        ],
    ),
    conversational_drift=dict(
        description="Memory recall after conversational drift",
        facts=[
            "I'm writing a novel about time travel.",
            "My grandfather was a famous chess player in the 1960s.",
            "I collect vintage vinyl records from the 1970s.",
            "I once met Neil Armstrong at a conference in Houston.",
            "I'm training for a marathon that's happening in October.",
        ],
        distractors=[
            "Do you think AI will surpass human intelligence?",
            "What's your take on climate change?",
            "Can you explain how blockchain works?",
            "What are your thoughts on remote work?",
            "How would you solve the traffic congestion in cities?",
        ],
        queries=[
            ("What am I writing?", ["novel", "time travel"]),
            ("What did my grandfather do?", ["chess", "player", "1960s"]),
            ("What do I collect?", ["vinyl", "records", "1970s"]),
            ("Who did I meet in Houston?", ["Neil Armstrong"]),
            ("What am I training for?", ["marathon", "October"]),
        ],
    ),
)


class MemoryWeaveShowcase:
    """Showcase MemoryWeave's performance across different scenarios."""

    def __init__(self, model_name: str, debug: bool = False):
        """Initialize the showcase with specified model."""
        self.model_name = model_name
        self.debug = debug

        # Track statistics
        self.stats = {
            "with_memory": {
                "successes": 0,
                "partials": 0,
                "failures": 0,
                "total": 0,
                "avg_time": 0,
                "by_scenario": {},
            },
            "no_memory": {
                "successes": 0,
                "partials": 0,
                "failures": 0,
                "total": 0,
                "avg_time": 0,
                "by_scenario": {},
            },
        }

        # Initialize scenario-specific stats
        for scenario in SCENARIOS:
            self.stats["with_memory"]["by_scenario"][scenario] = {
                "successes": 0,
                "partials": 0,
                "failures": 0,
                "total": 0,
            }
            self.stats["no_memory"]["by_scenario"][scenario] = {
                "successes": 0,
                "partials": 0,
                "failures": 0,
                "total": 0,
            }

        if debug:
            logger.setLevel(logging.DEBUG)

        # set timestamp for results
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_scenario(self, scenario_key: str) -> dict[str, Any]:
        """Run a specific benchmark scenario and return results."""
        scenario = SCENARIOS[scenario_key]

        console.print(
            f"\n[bold cyan]Running scenario:[/bold cyan] [yellow]{scenario_key}[/yellow] - {scenario['description']}"
        )

        # Track scenario results
        results = {
            "scenario": scenario_key,
            "description": scenario["description"],
            "with_memory": [],
            "no_memory": [],
            "memory_advantages": [],
        }

        # Run with memory
        console.print("[bold green]Testing with MemoryWeave enabled...[/bold green]")
        with_memory_results = self._run_test(scenario, scenario_key, True)
        results["with_memory"] = with_memory_results

        # Run without memory
        console.print("[bold yellow]Testing with no memory (baseline)...[/bold yellow]")
        no_memory_results = self._run_test(scenario, scenario_key, False)
        results["no_memory"] = no_memory_results

        # Calculate advantages
        advantages = []
        query_indices = range(min(len(with_memory_results), len(no_memory_results)))

        for i in query_indices:
            mem_result = with_memory_results[i]
            no_mem_result = no_memory_results[i]

            # Calculate advantage for this query
            advantage = {
                "query": mem_result["query"],
                "expected": mem_result["expected"],
                "result_difference": self._get_result_difference(mem_result, no_mem_result),
                "time_difference": mem_result["time"] - no_mem_result["time"],
                "with_memory_found": mem_result["found_keywords"],
                "no_memory_found": no_mem_result["found_keywords"],
            }
            advantages.append(advantage)

        results["memory_advantages"] = advantages

        return results

    def _run_test(self, scenario: dict, scenario_key: str, with_memory: bool) -> list[dict]:
        """Run a test with or without memory enabled."""
        # Initialize MemoryWeave API for this test
        api = MemoryWeaveAPI(model_name=self.model_name, debug=self.debug)

        # Track results for this test
        results = []
        total_time = 0

        mode = "WITH" if with_memory else "NO"
        stats_key = "with_memory" if with_memory else "no_memory"

        # Add all facts to memory if using memory
        if with_memory:
            for fact in scenario["facts"]:
                api.add_memory(fact)
                if self.debug:
                    console.print(f"[dim]Added memory: {fact}[/dim]")

        # If this is the conversational drift scenario, add distractors
        if "distractors" in scenario and with_memory:
            for distractor in scenario["distractors"]:
                # Just process the distractor messages to shift conversation
                api.chat(distractor)
                if self.debug:
                    console.print(f"[dim]Processed distractor: {distractor}[/dim]")

        # Process each query
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[{mode} MEMORY] Processing queries...", total=len(scenario["queries"])
            )

            for query, expected in scenario["queries"]:
                # Measure query time
                start_time = time.time()

                # Generate response
                if with_memory:
                    response = api.chat(query)
                else:
                    response = api.chat_without_memory(query)

                # Calculate elapsed time
                elapsed = time.time() - start_time
                total_time += elapsed

                # Check for expected keywords
                found = [kw.lower() in response.lower() for kw in expected]
                found_count = sum(found)
                found_keywords = [kw for i, kw in enumerate(expected) if found[i]]

                # Determine success level
                if found_count == len(expected):
                    result = "success"
                    self.stats[stats_key]["successes"] += 1
                    self.stats[stats_key]["by_scenario"][scenario_key]["successes"] += 1
                elif found_count > 0:
                    result = "partial"
                    self.stats[stats_key]["partials"] += 1
                    self.stats[stats_key]["by_scenario"][scenario_key]["partials"] += 1
                else:
                    result = "failure"
                    self.stats[stats_key]["failures"] += 1
                    self.stats[stats_key]["by_scenario"][scenario_key]["failures"] += 1

                # Update total counts
                self.stats[stats_key]["total"] += 1
                self.stats[stats_key]["by_scenario"][scenario_key]["total"] += 1

                # Record result
                results.append(
                    {
                        "query": query,
                        "expected": expected,
                        "response": response,
                        "result": result,
                        "found_keywords": found_keywords,
                        "time": elapsed,
                    }
                )

                # Update progress
                progress.update(task, advance=1)

        # Calculate average time
        avg_time = total_time / len(scenario["queries"]) if scenario["queries"] else 0
        self.stats[stats_key]["avg_time"] = avg_time

        return results

    def _get_result_difference(self, mem_result: dict, no_mem_result: dict) -> str:
        """Determine the difference in results between memory and no-memory approaches."""
        mem_status = mem_result["result"]
        no_mem_status = no_mem_result["result"]

        if mem_status == no_mem_status:
            return "equal"
        elif mem_status == "success" and no_mem_status == "partial":
            return "better"
        elif mem_status == "partial" and no_mem_status == "failure":
            return "better"
        elif mem_status == "success" and no_mem_status == "failure":
            return "much_better"
        elif mem_status == "partial" and no_mem_status == "success":
            return "worse"
        elif mem_status == "failure" and no_mem_status == "partial":
            return "worse"
        elif mem_status == "failure" and no_mem_status == "success":
            return "much_worse"
        else:
            return "unknown"

    def run_all_scenarios(self) -> dict:
        """Run all benchmark scenarios and return combined results."""
        all_results = {"timestamp": self.timestamp, "model": self.model_name, "scenarios": {}}

        console.print(
            Panel.fit(
                "[bold cyan]MemoryWeave Performance Showcase[/bold cyan]\n\n"
                f"Model: [yellow]{self.model_name}[/yellow]\n"
                f"Testing across [green]{len(SCENARIOS)}[/green] scenarios with varied memory requirements",
                title="Starting Benchmark",
                border_style="green",
            )
        )

        for scenario_key in SCENARIOS:
            scenario_results = self.run_scenario(scenario_key)
            all_results["scenarios"][scenario_key] = scenario_results

        # Add summary statistics
        all_results["summary"] = self._calculate_summary()

        # Save results to file
        self._save_results(all_results)

        # Display final results
        self.display_results(all_results)

        return all_results

    def _calculate_summary(self) -> dict:
        """Calculate summary statistics from all test runs."""
        with_mem = self.stats["with_memory"]
        no_mem = self.stats["no_memory"]

        # Calculate overall scores
        with_mem_score = (
            (with_mem["successes"] + 0.5 * with_mem["partials"]) / with_mem["total"]
            if with_mem["total"] > 0
            else 0
        )
        no_mem_score = (
            (no_mem["successes"] + 0.5 * no_mem["partials"]) / no_mem["total"]
            if no_mem["total"] > 0
            else 0
        )

        # Calculate per-scenario scores
        scenario_scores = {}
        for scenario in SCENARIOS:
            wm_scenario = with_mem["by_scenario"][scenario]
            nm_scenario = no_mem["by_scenario"][scenario]

            wm_score = (
                (wm_scenario["successes"] + 0.5 * wm_scenario["partials"]) / wm_scenario["total"]
                if wm_scenario["total"] > 0
                else 0
            )
            nm_score = (
                (nm_scenario["successes"] + 0.5 * nm_scenario["partials"]) / nm_scenario["total"]
                if nm_scenario["total"] > 0
                else 0
            )

            scenario_scores[scenario] = {
                "with_memory_score": wm_score,
                "no_memory_score": nm_score,
                "improvement": wm_score - nm_score,
            }

        return {
            "with_memory_score": with_mem_score,
            "no_memory_score": no_mem_score,
            "overall_improvement": with_mem_score - no_mem_score,
            "with_memory_stats": with_mem,
            "no_memory_stats": no_mem,
            "scenario_scores": scenario_scores,
        }

    def _save_results(self, results: dict) -> None:
        """Save benchmark results to a JSON file."""
        filename = f"memoryweave_showcase_results_{self.timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        console.print(f"\n[green]Results saved to {filename}[/green]")

    def display_results(self, results: dict) -> None:
        """Display formatted results from the benchmark."""
        summary = results["summary"]

        # Overall comparison table
        table = Table(title="MemoryWeave vs. No Memory: Overall Performance")

        table.add_column("Metric", style="cyan")
        table.add_column("With Memory", style="green")
        table.add_column("No Memory", style="yellow")
        table.add_column("Difference", style="magenta")

        # Format as percentages
        wm_score = summary["with_memory_score"] * 100
        nm_score = summary["no_memory_score"] * 100
        diff = summary["overall_improvement"] * 100

        table.add_row(
            "Overall Score",
            f"{wm_score:.1f}%",
            f"{nm_score:.1f}%",
            f"[bold green]+{diff:.1f}%[/bold green]"
            if diff > 0
            else f"[bold red]{diff:.1f}%[/bold red]",
        )

        # Success rates
        wm = summary["with_memory_stats"]
        nm = summary["no_memory_stats"]

        table.add_row(
            "Success Rate",
            f"{wm['successes']}/{wm['total']} ({wm['successes'] / wm['total'] * 100:.1f}%)"
            if wm["total"] > 0
            else "N/A",
            f"{nm['successes']}/{nm['total']} ({nm['successes'] / nm['total'] * 100:.1f}%)"
            if nm["total"] > 0
            else "N/A",
            f"+{wm['successes'] - nm['successes']}",
        )

        table.add_row(
            "Partial Success",
            f"{wm['partials']}/{wm['total']} ({wm['partials'] / wm['total'] * 100:.1f}%)"
            if wm["total"] > 0
            else "N/A",
            f"{nm['partials']}/{nm['total']} ({nm['partials'] / nm['total'] * 100:.1f}%)"
            if nm["total"] > 0
            else "N/A",
            f"{wm['partials'] - nm['partials']:+d}",
        )

        table.add_row(
            "Failure Rate",
            f"{wm['failures']}/{wm['total']} ({wm['failures'] / wm['total'] * 100:.1f}%)"
            if wm["total"] > 0
            else "N/A",
            f"{nm['failures']}/{nm['total']} ({nm['failures'] / nm['total'] * 100:.1f}%)"
            if nm["total"] > 0
            else "N/A",
            f"{wm['failures'] - nm['failures']:+d}",
        )

        # Average response time
        time_diff = wm["avg_time"] - nm["avg_time"]
        time_pct = (time_diff / nm["avg_time"] * 100) if nm["avg_time"] > 0 else 0

        table.add_row(
            "Avg Response Time",
            f"{wm['avg_time']:.2f}s",
            f"{nm['avg_time']:.2f}s",
            f"{time_diff:+.2f}s ({time_pct:+.1f}%)",
        )

        console.print("\n")
        console.print(table)

        # Per-scenario table
        scenario_table = Table(title="Performance by Scenario Type")

        scenario_table.add_column("Scenario", style="cyan")
        scenario_table.add_column("Description", style="dim")
        scenario_table.add_column("With Memory", style="green")
        scenario_table.add_column("No Memory", style="yellow")
        scenario_table.add_column("Improvement", style="magenta")

        for scenario_key, scenario_data in summary["scenario_scores"].items():
            wm_score = scenario_data["with_memory_score"] * 100
            nm_score = scenario_data["no_memory_score"] * 100
            improvement = scenario_data["improvement"] * 100

            scenario_table.add_row(
                scenario_key.capitalize(),
                SCENARIOS[scenario_key]["description"],
                f"{wm_score:.1f}%",
                f"{nm_score:.1f}%",
                f"[bold green]+{improvement:.1f}%[/bold green]"
                if improvement > 0
                else f"[bold red]{improvement:.1f}%[/bold red]",
            )

        console.print("\n")
        console.print(scenario_table)

        # Generate and display charts
        self._generate_charts(results)

    def _generate_charts(self, results: dict) -> None:
        """Generate visualization charts for the results."""
        try:
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(15, 12))

            # 1. Overall comparison bar chart
            ax1 = fig.add_subplot(2, 2, 1)

            summary = results["summary"]
            scenario_scores = summary["scenario_scores"]

            # Prepare data for overall comparison
            scenarios = list(scenario_scores.keys())
            wm_scores = [scenario_scores[s]["with_memory_score"] * 100 for s in scenarios]
            nm_scores = [scenario_scores[s]["no_memory_score"] * 100 for s in scenarios]

            x = np.arange(len(scenarios))
            width = 0.35

            ax1.bar(x - width / 2, wm_scores, width, label="With Memory", color="#5cb85c")
            ax1.bar(x + width / 2, nm_scores, width, label="No Memory", color="#f0ad4e")

            ax1.set_ylabel("Score (%)")
            ax1.set_title("MemoryWeave Performance by Scenario")
            ax1.set_xticks(x)
            ax1.set_xticklabels([s.capitalize() for s in scenarios])
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.7)

            for i, v in enumerate(wm_scores):
                ax1.text(i - width / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=9)

            for i, v in enumerate(nm_scores):
                ax1.text(i + width / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=9)

            # 2. Success/Partial/Failure breakdown
            ax2 = fig.add_subplot(2, 2, 2)

            wm_stats = summary["with_memory_stats"]
            nm_stats = summary["no_memory_stats"]

            # Calculate percentages
            wm_success_pct = (
                wm_stats["successes"] / wm_stats["total"] * 100 if wm_stats["total"] > 0 else 0
            )
            wm_partial_pct = (
                wm_stats["partials"] / wm_stats["total"] * 100 if wm_stats["total"] > 0 else 0
            )
            wm_failure_pct = (
                wm_stats["failures"] / wm_stats["total"] * 100 if wm_stats["total"] > 0 else 0
            )

            nm_success_pct = (
                nm_stats["successes"] / nm_stats["total"] * 100 if nm_stats["total"] > 0 else 0
            )
            nm_partial_pct = (
                nm_stats["partials"] / nm_stats["total"] * 100 if nm_stats["total"] > 0 else 0
            )
            nm_failure_pct = (
                nm_stats["failures"] / nm_stats["total"] * 100 if nm_stats["total"] > 0 else 0
            )

            categories = ["Success", "Partial", "Failure"]
            wm_values = [wm_success_pct, wm_partial_pct, wm_failure_pct]
            nm_values = [nm_success_pct, nm_partial_pct, nm_failure_pct]

            x = np.arange(len(categories))

            ax2.bar(
                x - width / 2,
                wm_values,
                width,
                label="With Memory",
                color=["#5cb85c", "#5bc0de", "#d9534f"],
            )
            ax2.bar(
                x + width / 2,
                nm_values,
                width,
                label="No Memory",
                color=["#5cb85c", "#5bc0de", "#d9534f"],
                alpha=0.6,
            )

            ax2.set_ylabel("Percentage (%)")
            ax2.set_title("Response Quality Breakdown")
            ax2.set_xticks(x)
            ax2.set_xticklabels(categories)
            ax2.legend()
            ax2.grid(True, linestyle="--", alpha=0.7)

            for i, v in enumerate(wm_values):
                ax2.text(i - width / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=9)

            for i, v in enumerate(nm_values):
                ax2.text(i + width / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=9)

            # 3. Improvement percentage by scenario
            ax3 = fig.add_subplot(2, 2, 3)

            improvements = [scenario_scores[s]["improvement"] * 100 for s in scenarios]
            colors = ["#5cb85c" if imp > 0 else "#d9534f" for imp in improvements]

            ax3.bar(scenarios, improvements, color=colors)
            ax3.set_ylabel("Improvement (%)")
            ax3.set_title("Memory Advantage by Scenario")
            ax3.set_xticklabels([s.capitalize() for s in scenarios])
            ax3.grid(True, linestyle="--", alpha=0.7)

            for i, v in enumerate(improvements):
                ax3.text(i, v + 2 if v > 0 else v - 6, f"{v:+.1f}%", ha="center", fontsize=9)

            # 4. Overall score comparison
            ax4 = fig.add_subplot(2, 2, 4)

            overall_wm_score = summary["with_memory_score"] * 100
            overall_nm_score = summary["no_memory_score"] * 100
            overall_improvement = summary["overall_improvement"] * 100

            ax4.bar(
                ["With Memory", "No Memory"],
                [overall_wm_score, overall_nm_score],
                color=["#5cb85c", "#f0ad4e"],
            )

            ax4.set_ylabel("Overall Score (%)")
            ax4.set_title(f"Overall Performance (Improvement: {overall_improvement:+.1f}%)")
            ax4.grid(True, linestyle="--", alpha=0.7)

            for i, v in enumerate([overall_wm_score, overall_nm_score]):
                ax4.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=9)

            plt.tight_layout()

            # Save figure
            chart_filename = f"memoryweave_showcase_charts_{self.timestamp}.png"
            plt.savefig(chart_filename, dpi=300, bbox_inches="tight")
            console.print(f"\n[green]Performance charts saved to {chart_filename}[/green]")

            # Show interactive plot
            plt.show()

        except Exception as e:
            console.print(f"[yellow]Warning: Could not generate charts: {e}[/yellow]")
            import traceback

            traceback.print_exc()


@click.command(
    help="MemoryWeave Performance Showcase - Compare memory-enhanced vs. baseline performance",
)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    help=f"Name of the Hugging Face model to load (default: {DEFAULT_MODEL})",
)
@click.option(
    "--scenario",
    type=click.Choice(
        choices=["all", *SCENARIOS],
    ),
    default="all",
    help="Specific scenario to run (default: all)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging for more detailed output.",
)
def main(model, scenario, debug):
    """Run the MemoryWeave performance showcase."""
    # Run the showcase
    showcase = MemoryWeaveShowcase(model, debug)

    if scenario == "all":
        showcase.run_all_scenarios()
    else:
        # Run a single scenario
        results = {"timestamp": showcase.timestamp, "model": model, "scenarios": {}}

        scenario_results = showcase.run_scenario(scenario)
        results["scenarios"][scenario] = scenario_results

        # Add summary statistics
        results["summary"] = showcase._calculate_summary()

        # Save and display results
        showcase._save_results(results)
        showcase.display_results(results)


if __name__ == "__main__":
    main()
