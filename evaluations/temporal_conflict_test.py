#!/usr/bin/env python3
"""
temporal_conflict_test.py

This test simulates both temporal evolution and conflict resolution:
    - Inserts an initial fact.
    - Sequentially updates that fact with conflicting information over several turns.
    - Periodically queries to observe when the system switches from one fact to the next.
    - Measures and reports the responses.

This helps assess how quickly the system updates its memory activations and manages conflicting information.
"""

import logging
import time

import rich_click as click
from rich.console import Console

from memoryweave.api import MemoryWeaveAPI

# Disable verbose logging from SemanticCoherenceProcessor
logging.getLogger("memoryweave.components.post_processors").setLevel(logging.WARNING)

console = Console()


@click.command(
    help="Temporal Conflict Dynamics Test: Insert an initial fact, update it over time, and query periodically to observe changes."
)
@click.option("--model", default="unsloth/Llama-3.2-3B-Instruct", help="Model name to load.")
@click.option("--debug", is_flag=True, help="Enable debug logging for more details.")
def main(model: str, debug: bool):
    console.print("Initializing MemoryWeave API...", style="bold cyan")
    api = MemoryWeaveAPI(model_name=model)

    # Insert the initial fact.
    initial_fact = "My favorite color is blue."
    console.print("\nInserting initial fact:", style="bold cyan")
    console.print(f"  {initial_fact}", style="dim cyan")
    api.add_memory(initial_fact)
    time.sleep(1)  # Give the system a moment to process the insertion.

    # Query initially.
    query = "What is my favorite color?"
    console.print("\nInitial query:", style="bold cyan")
    response = api.chat(query, max_new_tokens=50)
    console.print(f"Response: {response}", style="dim cyan")

    # Sequentially update with conflicting facts.
    updates = [
        "Actually, my favorite color is green.",
        "No, scratch that; it's yellow.",
        "Now, I prefer red over any other color.",
    ]

    for idx, update in enumerate(updates, start=1):
        console.print(f"\nInserting update {idx}:", style="bold red")
        console.print(f"  {update}", style="dim red")
        api.add_memory(update)
        time.sleep(1)  # Pause to allow internal updates.

        console.print(f"\nQuery after update {idx}:", style="bold cyan")
        response = api.chat(query, max_new_tokens=50)
        console.print(f"Response: {response}", style="dim cyan")
        time.sleep(1)

    # Final query after all updates.
    console.print("\nFinal query after all updates:", style="bold cyan")
    final_response = api.chat(query, max_new_tokens=50)
    console.print(f"Final Response: {final_response}", style="dim cyan")


if __name__ == "__main__":
    main()
