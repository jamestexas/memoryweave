#!/usr/bin/env python3
"""
conflict_test.py

This test simulates a conflict resolution scenario by:
  - Inserting an initial fact.
  - Querying for that fact.
  - Inserting a conflicting fact.
  - Querying again to see which fact is retrieved.
  - Optionally inserting a distractor memory to further stress retrieval.

This test helps you verify whether MemoryWeave correctly updates or prioritizes new information when conflicts arise.
"""

import logging
import time

import rich_click as click
from rich.console import Console

from memoryweave.api import MemoryWeaveAPI

logging.getLogger("memoryweave.components.post_processors").setLevel(logging.WARNING)

console = Console()


@click.command(
    help="Conflict resolution test: insert conflicting facts and check retrieval behavior."
)
@click.option("--model", default="unsloth/Llama-3.2-3B-Instruct", help="Model name to load.")
@click.option("--debug", is_flag=True, help="Enable debug logging for more details.")
def main(model: str, debug: bool):
    console.print("Initializing MemoryWeave API...")
    api = MemoryWeaveAPI(model_name=model)

    # Insert the initial fact.
    initial_fact = "My favorite color is blue."
    console.print("\nInserting initial fact:")
    console.print(f"  {initial_fact}")
    api.add_memory(initial_fact)
    time.sleep(1)  # Give the system a moment to process the insertion.

    # Query for favorite color.
    query = "What is my favorite color?"
    console.print("\nQuerying for favorite color (initial):", style="bold cyan")
    response1 = api.chat(query, max_new_tokens=50)
    console.print(f"Response: {response1}", style="dim cyan")

    # Insert a conflicting fact.
    conflicting_fact = "Actually, my favorite color is red."
    console.print("\nInserting conflicting fact:", style="bold red")
    console.print(f"  {conflicting_fact}", style="dim red")
    api.add_memory(conflicting_fact)
    time.sleep(1)

    # Query again after conflict.
    console.print("\nQuerying for favorite color (after conflict):", style="bold cyan")
    response2 = api.chat(query, max_new_tokens=50)
    console.print(f"Response: {response2}", style="dim cyan")

    # Optionally insert a distractor memory.
    distractor = "I love hiking on weekends."
    console.print("\nInserting a distractor memory:", style="bold cyan")
    console.print(f"  {distractor}", style="dim cyan")
    api.add_memory(distractor)
    time.sleep(1)

    # Final query to see if the system consistently reflects the update.
    console.print("\nFinal query for favorite color:", style="bold cyan")
    response3 = api.chat(query, max_new_tokens=50)
    console.print(f"Response: {response3}", style="dim cyan")


if __name__ == "__main__":
    main()
