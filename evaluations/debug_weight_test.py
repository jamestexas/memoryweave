#!/usr/bin/env python3
"""
debug_weight_test.py

This script tests conflict resolution weighting by:
  - Inserting an initial fact ("My favorite color is blue.")
  - Inserting a conflicting fact ("Actually, my favorite color is red.")
  - Running a retrieval query ("What is my favorite color?") and printing out
    the raw retrieval results and their breakdowns.

This helps diagnose which weighting factors (similarity, associative, temporal, activation)
are dominating the retrieval scores.
"""

import logging
import time

import rich_click as click
from rich.console import Console

from memoryweave.api import MemoryWeaveAPI

# Disable verbose logging from SemanticCoherenceProcessor
logging.getLogger("memoryweave.components.post_processors").setLevel(logging.WARNING)

console = Console()


@click.command(help="Debug weight test: print raw retrieval breakdowns for a conflict scenario.")
@click.option("--model", default="unsloth/Llama-3.2-3B-Instruct", help="Model name to load.")
@click.option("--debug", is_flag=True, help="Enable debug logging for more details.")
def main(model: str, debug: bool):
    console.print("Initializing MemoryWeave API...", style="bold cyan")
    api = MemoryWeaveAPI(model_name=model)

    # Optionally clear existing memories if needed
    # api.clear_memories()  # Uncomment if your API supports clearing

    # Insert initial fact
    initial_fact = "My favorite color is blue."
    console.print("\nInserting initial fact:", style="bold cyan")
    console.print(f"  {initial_fact}", style="dim cyan")
    api.add_memory(initial_fact)
    time.sleep(1)

    # Insert conflicting fact
    conflicting_fact = "Actually, my favorite color is red."
    console.print("\nInserting conflicting fact:", style="bold red")
    console.print(f"  {conflicting_fact}", style="dim red")
    api.add_memory(conflicting_fact)
    time.sleep(1)

    # Run retrieval query directly (bypassing full chat)
    query = "What is my favorite color?"
    console.print("\nRunning retrieval for query:", style="bold cyan")
    console.print(f"  {query}", style="dim cyan")

    # Call the retrieve method to get raw retrieval details
    retrieval_results = api.retrieve(query, top_k=10)

    if retrieval_results:
        console.print("\nRetrieval breakdown:", style="bold yellow")
        for i, res in enumerate(retrieval_results, start=1):
            similarity = res.get("similarity_contribution", 0.0)
            associative = res.get("associative_contribution", 0.0)
            temporal = res.get("temporal_contribution", 0.0)
            activation = res.get("activation_contribution", 0.0)
            total = res.get("relevance_score", 0.0)
            content = res.get("content", "")
            console.print(
                f"Result {i}: Score={total:.3f} (sim={similarity:.3f}, assoc={associative:.3f}, "
                f"temp={temporal:.3f}, act={activation:.3f})\nContent: {content}",
                style="dim magenta",
            )
    else:
        console.print("No retrieval results returned.", style="bold red")


if __name__ == "__main__":
    main()
