#!/usr/bin/env python3
"""
stress_test.py

A stress test for MemoryWeave that:
  - Preloads a set number of synthetic memories.
  - Runs a series of retrieval queries using chat_without_memory (with minimal token generation).
  - Measures and prints per-query, total, and average retrieval times.

This script is designed to stress test the memory store and retrieval logic without full inference.
"""

import logging
import secrets
import time

import rich_click as click
from rich import print
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from memoryweave.api import MemoryWeaveAPI

logging.getLogger("memoryweave.components.post_processors").setLevel(logging.WARNING)


@click.command(
    help="Stress test MemoryWeave by preloading synthetic memories and running retrieval queries."
)
@click.option(
    "--model",
    default="unsloth/Llama-3.2-3B-Instruct",
    help="Name of the Hugging Face model to load (default: unsloth/Llama-3.2-3B-Instruct)",
)
@click.option(
    "--num-memories",
    default=500,
    help="Number of synthetic memories to preload (default: 500)",
)
@click.option(
    "--num-queries",
    default=100,
    help="Number of retrieval queries to run (default: 100)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging for more detailed output.",
)
def main(model: str, num_memories: int, num_queries: int, debug: bool):
    # Initialize MemoryWeave API
    print("Initializing MemoryWeave LLM...")
    llm = MemoryWeaveAPI(model_name=model)

    # Bulk insert synthetic memories with a progress bar
    topics = ["sports", "tech", "music", "travel", "food"]
    print(f"Adding {num_memories} synthetic memories...")
    mem_start = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description} ({task.completed}/{task.total})"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Adding memories...", total=num_memories)
        for i in range(num_memories):
            fact = f"Synthetic fact {i}: This is a detail about {secrets.choice(topics)}."
            llm.add_memory(fact)
            progress.advance(task)
    mem_elapsed = time.time() - mem_start
    print(f"Added {num_memories} memories in {mem_elapsed:.2f} seconds.\n")

    # Run retrieval queries with a progress bar and measure total query time
    query_times = []
    print(f"Running {num_queries} retrieval queries...")
    query_start = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description} ({task.completed}/{task.total})"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Running queries...", total=num_queries)
        for i in range(num_queries):
            fact_id = secrets.randbelow(num_memories)
            query = f"What did I say about synthetic fact {fact_id}?"
            start = time.time()
            response = llm.chat(query, max_new_tokens=10)
            elapsed_query = time.time() - start
            query_times.append(elapsed_query)
            progress.advance(task)

            # Optionally print detailed output for the first few queries if debug is enabled.
            if debug and i < 5:
                print(f"\nQuery {i + 1}: {query}")
                print(f"Response: {response}")
                print(f"Time: {elapsed_query:.2f} seconds")

    total_query_time = time.time() - query_start
    avg_time = sum(query_times) / len(query_times)
    print(f"\nTotal query phase time: {total_query_time:.2f} seconds")
    print(f"Average query time over {num_queries} queries: {avg_time:.2f} seconds")

    # If the responses don't contain the expected synthetic fact details,
    # you might need to add further debugging within the MemoryWeave retrieval logic.
    # For instance, check retrieval scores or the number of retrieved memories.


if __name__ == "__main__":
    main()
