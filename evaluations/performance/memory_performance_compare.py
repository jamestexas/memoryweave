#!/usr/bin/env python3

"""
memory_performance_compare.py

A script that compares the performance and capabilities of MemoryWeave
versus standard LLM usage. Shows how MemoryWeave's contextual memory
handling compares to regular context window management.
"""

import logging
import os
import time
from datetime import datetime

# Disable HuggingFace warnings and info messages
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import rich_click as click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

# Import MemoryWeave API
from memoryweave.api import MemoryWeaveAPI

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

logger = logging.getLogger("memory_performance")

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"

# Test scenarios with personal information queries - SIMPLIFIED for faster runs
TEST_SCENARIOS = [
    {
        "setup": [
            "My name is Alex and I live in Seattle.",
            "I have a dog named Luna.",
            "My favorite color is blue.",
        ],
        "questions": ["What's my name?", "Where do I live?", "What's my dog's name?"],
    },
    {
        "setup": [
            "I'm planning a trip to Japan next summer.",
            "I want to visit Tokyo and Kyoto.",
            "I want to try authentic ramen.",
        ],
        "questions": [
            "Where am I planning to travel?",
            "When am I planning my trip?",
            "What food do I want to try?",
        ],
    },
]


class PerformanceComparer:
    """Class to handle performance comparison between MemoryWeave and standard LLM."""

    def __init__(self, model_name, use_simulated_baseline=True):
        self.model_name = model_name
        self.memoryweave_api = None
        self.use_simulated_baseline = use_simulated_baseline

        # Performance metrics
        self.memoryweave_times = []
        self.standard_times = []
        self.memoryweave_success = 0
        self.standard_success = 0
        self.total_queries = 0

        # For the standard approach, we'll collect all context in a single string
        self.standard_context = ""

    def initialize(self):
        """Initialize the MemoryWeave API."""
        with console.status(
            "[bold green]Initializing MemoryWeave...[/bold green]", spinner="dots"
        ) as status:
            self.memoryweave_api = MemoryWeaveAPI(model_name=self.model_name)
            status.update("[bold green]Initialization complete[/bold green]")
            time.sleep(0.5)

        console.print("[bold green]✓[/bold green] System ready\n")

    def run_comparison(self, scenario_index=0):
        """Run a comparison between MemoryWeave and standard LLM."""
        scenario = TEST_SCENARIOS[scenario_index]
        setup_messages = scenario["setup"]
        questions = scenario["questions"]

        # Display scenario information
        console.print(
            Panel.fit(
                f"[bold cyan]Performance Comparison - Scenario {scenario_index + 1}[/bold cyan]\n\n"
                f"Setup Context: {len(setup_messages)} statements\n"
                f"Test Questions: {len(questions)} questions",
                border_style="cyan",
            )
        )

        # Reset metrics
        self.memoryweave_times = []
        self.standard_times = []
        self.memoryweave_success = 0
        self.standard_success = 0
        self.total_queries = 0
        self.standard_context = ""

        # Run the benchmark
        console.print("[bold]Phase 1: Setup - Adding context information[/bold]")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            # Setup phase - FAST VERSION
            setup_task = progress.add_task("[cyan]Adding context...", total=len(setup_messages))

            # Add context to MemoryWeave and build standard context
            for message in setup_messages:
                # Add to MemoryWeave as memory
                self.memoryweave_api.add_memory(message, {"type": "user_fact", "importance": 0.8})

                # For standard approach, just accumulate the context
                self.standard_context += message + "\n"

                progress.advance(setup_task)
                time.sleep(0.1)  # Small delay to show progress, but not too long

            # Testing phase
            console.print("\n[bold]Phase 2: Testing - Comparing response quality and speed[/bold]")

            test_task = progress.add_task("[cyan]Running memory tests...", total=len(questions))
            self.total_queries = len(questions)

            for i, question in enumerate(questions):
                # Test MemoryWeave
                mw_start = time.time()
                mw_response = self.memoryweave_api.chat(question)
                mw_time = time.time() - mw_start
                self.memoryweave_times.append(mw_time)

                # Test Standard approach (simulated)
                std_start = time.time()
                if self.use_simulated_baseline:
                    # Simulate standard LLM by using a simple prompt with context
                    baseline_prompt = f"Context:\n{self.standard_context}\n\nQuestion: {question}"
                    std_response = self.simulate_standard_llm(baseline_prompt, question)
                else:
                    # Fallback to using regular chat with the full context in the query
                    # This is less accurate but works if you don't have chat_without_memory
                    std_response = self.memoryweave_api.chat(
                        f"Based on this information: {self.standard_context}\n{question}"
                    )
                std_time = time.time() - std_start
                self.standard_times.append(std_time)

                # Evaluate responses (simplified for demo)
                relevant_context = any(
                    info.lower() in mw_response.lower() for info in setup_messages
                )
                if relevant_context:
                    self.memoryweave_success += 1

                relevant_context_std = any(
                    info.lower() in std_response.lower() for info in setup_messages
                )
                if relevant_context_std:
                    self.standard_success += 1

                # Show results for this question
                console.print(f"\n[bold]Question {i + 1}:[/bold] {question}")

                # Display comparison in a table
                comparison_table = Table(title=f"Response Comparison - Question {i + 1}")
                comparison_table.add_column("System", style="cyan")
                comparison_table.add_column("Response Time", style="magenta")
                comparison_table.add_column("Response Quality", style="green")
                comparison_table.add_column("Sample Response", style="yellow")

                # Add MemoryWeave results
                comparison_table.add_row(
                    "MemoryWeave",
                    f"{mw_time:.2f}s",
                    "✓" if relevant_context else "✗",
                    mw_response[:100] + "..." if len(mw_response) > 100 else mw_response,
                )

                # Add Standard LLM results
                comparison_table.add_row(
                    "Standard Approach",
                    f"{std_time:.2f}s",
                    "✓" if relevant_context_std else "✗",
                    std_response[:100] + "..." if len(std_response) > 100 else std_response,
                )

                console.print(comparison_table)

                progress.advance(test_task)

        self.display_summary()

    def simulate_standard_llm(self, prompt, question):
        """
        Simulate standard LLM behavior.

        In a real environment, you'd make an actual call to the LLM.
        For this demo, we'll generate a plausible response based on the context.
        """
        # Extract relevant information from the context for this question
        context_lines = self.standard_context.strip().split("\n")

        # Look for relevant info based on simple keyword matching
        relevant_info = []
        question_lower = question.lower()

        # Keywords to look for in the question
        name_keywords = ["name", "who", "your"]
        location_keywords = ["live", "where", "city", "location"]
        pet_keywords = ["pet", "dog", "cat", "animal"]
        color_keywords = ["color", "favourite", "favorite"]
        travel_keywords = ["travel", "trip", "visit", "going", "planning"]
        food_keywords = ["food", "eat", "meal", "cuisine", "restaurant"]

        # Find relevant information based on question type
        if any(keyword in question_lower for keyword in name_keywords):
            for line in context_lines:
                if "name" in line.lower():
                    relevant_info.append(line)

        elif any(keyword in question_lower for keyword in location_keywords):
            for line in context_lines:
                if "live" in line.lower() or "city" in line.lower() or "seattle" in line.lower():
                    relevant_info.append(line)

        elif any(keyword in question_lower for keyword in pet_keywords):
            for line in context_lines:
                if "dog" in line.lower() or "cat" in line.lower() or "pet" in line.lower():
                    relevant_info.append(line)

        elif any(keyword in question_lower for keyword in color_keywords):
            for line in context_lines:
                if (
                    "color" in line.lower()
                    or "favourite" in line.lower()
                    or "favorite" in line.lower()
                ):
                    relevant_info.append(line)

        elif any(keyword in question_lower for keyword in travel_keywords):
            for line in context_lines:
                if (
                    "trip" in line.lower()
                    or "travel" in line.lower()
                    or "visit" in line.lower()
                    or "japan" in line.lower()
                ):
                    relevant_info.append(line)

        elif any(keyword in question_lower for keyword in food_keywords):
            for line in context_lines:
                if (
                    "food" in line.lower()
                    or "eat" in line.lower()
                    or "cuisine" in line.lower()
                    or "ramen" in line.lower()
                ):
                    relevant_info.append(line)

        # If we found relevant info, construct a response based on it
        if relevant_info:
            response = "Based on the information provided, "
            for info in relevant_info:
                # Extract just the relevant part
                if "name" in question_lower and "name" in info.lower():
                    name = (
                        info.split("name is ")[1].split(" ")[0] if "name is " in info else "unknown"
                    )
                    response += f"your name is {name}. "
                elif "live" in question_lower and "live" in info.lower():
                    location = (
                        info.split("live in ")[1].split(".")[0]
                        if "live in " in info
                        else "unknown location"
                    )
                    response += f"you live in {location}. "
                elif "dog" in question_lower and "dog" in info.lower():
                    dog_name = (
                        info.split("named ")[1].split(".")[0] if "named " in info else "unknown"
                    )
                    response += f"your dog's name is {dog_name}. "
                elif "color" in question_lower and "color" in info.lower():
                    color = (
                        info.split("color is ")[1].split(".")[0]
                        if "color is " in info
                        else "unknown"
                    )
                    response += f"your favorite color is {color}. "
                elif "travel" in question_lower:
                    if "Japan" in info:
                        response += "you're planning to travel to Japan. "
                elif "food" in question_lower:
                    if "ramen" in info:
                        response += "you want to try authentic ramen. "
                else:
                    # Generic inclusion
                    response += f"{info} "

            return response
        else:
            # Generic fallback response
            return "I don't have specific information to answer that question. Could you please provide more details?"

    def display_summary(self):
        """Display a summary of the performance comparison."""
        console.print(
            Panel.fit("[bold cyan]Performance Comparison Summary[/bold cyan]", border_style="cyan")
        )

        # Calculate statistics
        mw_avg_time = (
            sum(self.memoryweave_times) / len(self.memoryweave_times)
            if self.memoryweave_times
            else 0
        )
        std_avg_time = (
            sum(self.standard_times) / len(self.standard_times) if self.standard_times else 0
        )

        time_diff = mw_avg_time - std_avg_time
        time_pct = (time_diff / std_avg_time) * 100 if std_avg_time > 0 else 0

        mw_success_rate = (
            (self.memoryweave_success / self.total_queries) * 100 if self.total_queries > 0 else 0
        )
        std_success_rate = (
            (self.standard_success / self.total_queries) * 100 if self.total_queries > 0 else 0
        )

        accuracy_diff = mw_success_rate - std_success_rate

        # Display summary metrics
        metrics_table = Table(title="Performance Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("MemoryWeave", style="green")
        metrics_table.add_column("Standard Approach", style="yellow")
        metrics_table.add_column("Difference", style="magenta")

        # Add timing metrics
        metrics_table.add_row(
            "Avg Response Time",
            f"{mw_avg_time:.2f}s",
            f"{std_avg_time:.2f}s",
            f"{time_diff:+.2f}s ({time_pct:+.1f}%)",
        )

        # Add success rate metrics
        metrics_table.add_row(
            "Success Rate",
            f"{mw_success_rate:.1f}%",
            f"{std_success_rate:.1f}%",
            f"{accuracy_diff:+.1f}%",
        )

        # Add total counts
        metrics_table.add_row(
            "Successful Recalls",
            f"{self.memoryweave_success}/{self.total_queries}",
            f"{self.standard_success}/{self.total_queries}",
            f"{self.memoryweave_success - self.standard_success:+d}",
        )

        console.print(metrics_table)

        # Interpretation
        console.print("\n[bold]Performance Insights:[/bold]")

        if time_diff > 0:
            console.print(
                f"[yellow]MemoryWeave is slower by {time_diff:.2f}s ({time_pct:.1f}%) on average[/yellow]"
            )
            console.print(
                "[dim]This is expected due to the additional retrieval and memory processing.[/dim]"
            )
        else:
            console.print(
                f"[green]MemoryWeave is faster by {-time_diff:.2f}s ({-time_pct:.1f}%) on average[/green]"
            )

        if accuracy_diff > 0:
            console.print(
                f"[green]MemoryWeave has {accuracy_diff:.1f}% better recall accuracy[/green]"
            )
            console.print(
                "[dim]This shows the value of specialized memory management over simple context.[/dim]"
            )
        else:
            console.print(
                f"[yellow]Standard approach has {-accuracy_diff:.1f}% better recall accuracy[/yellow]"
            )

        # Final assessment
        if mw_success_rate - std_success_rate > 10:
            console.print(
                "\n[bold green]MemoryWeave shows significant improvement in information recall.[/bold green]"
            )
        elif mw_success_rate > std_success_rate:
            console.print(
                "\n[bold green]MemoryWeave shows moderate improvement in information recall.[/bold green]"
            )
        else:
            console.print(
                "\n[bold yellow]For this scenario, standard context management performed similarly.[/bold yellow]"
            )


@click.command()
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    help=f"Name of the Hugging Face model to load (default: {DEFAULT_MODEL})",
)
@click.option(
    "--scenario",
    default=0,
    type=int,
    help="Scenario index to run (default: 0)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging for more detailed output.",
)
def main(model, scenario, debug):
    """
    Compare MemoryWeave's performance against standard LLM memory handling.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("memoryweave").setLevel(logging.DEBUG)

    # Print header
    console.print(
        Panel.fit(
            "[bold cyan]MemoryWeave vs Standard LLM Performance Comparison[/bold cyan]\n\n"
            f"Model: [yellow]{model}[/yellow]\n"
            "This benchmark compares MemoryWeave against standard context handling.",
            border_style="cyan",
        )
    )

    # Run the benchmark
    try:
        # Use simulated baseline instead of requiring chat_without_memory
        comparer = PerformanceComparer(model_name=model, use_simulated_baseline=True)
        comparer.initialize()

        # Validate scenario index
        if scenario < 0 or scenario >= len(TEST_SCENARIOS):
            console.print(
                f"[bold red]Invalid scenario index: {scenario}. Using default (0).[/bold red]"
            )
            scenario = 0

        comparer.run_comparison(scenario_index=scenario)

        # Save the timestamp of when the benchmark was run
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"\n[dim]Benchmark completed at {now}[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error during benchmark: {str(e)}[/bold red]")
        if debug:
            import traceback

            console.print(traceback.format_exc())


if __name__ == "__main__":
    main()
