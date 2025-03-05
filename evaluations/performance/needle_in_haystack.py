#!/usr/bin/env python3
"""
needle_in_haystack_test.py

This test simulates a needle-in-a-haystack scenario by:
  - Preloading a large number of synthetic distractor memories.
  - Inserting a few key facts that contain the "needle" details.
  - Running retrieval queries for each key fact.
  - Reporting how many key facts are correctly recalled, along with timing metrics.

This is especially useful when the total amount of information exceeds the LLM’s context window,
forcing the retrieval system to select only the most relevant memories.
"""

import logging
import random
import time

import rich_click as click
from rich import print
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from memoryweave.api import MemoryWeaveAPI

logging.getLogger("memoryweave.components.post_processors").setLevel(logging.WARNING)


@click.command(
    help="Needle-in-a-haystack test: Preload many synthetic distractors and key facts, then run targeted retrieval queries."
)
@click.option(
    "--model",
    default="unsloth/Llama-3.2-3B-Instruct",
    help="Name of the Hugging Face model to load.",
)
@click.option(
    "--num-distractors",
    default=1000,
    help="Number of distractor memories to insert (default: 1000).",
)
@click.option("--num-key-facts", default=5, help="Number of key facts to insert (default: 5).")
@click.option("--debug", is_flag=True, help="Enable debug logging for more detailed output.")
def main(model: str, num_distractors: int, num_key_facts: int, debug: bool):
    # Optionally suppress verbose logging from SemanticCoherenceProcessor
    print("Initializing MemoryWeave API...")
    api = MemoryWeaveAPI(model_name=model)

    # Insert distractor memories
    distractor_topics = ["weather", "sports", "music", "food", "technology"]
    print(f"Inserting {num_distractors} distractor memories...")
    start_time = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description} ({task.completed}/{task.total})"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Adding distractors...", total=num_distractors)
        for i in range(num_distractors):
            fact = f"Distractor {i}: Random fact about {random.choice(distractor_topics)}."
            api.add_memory(fact)
            progress.advance(task)
    distractor_time = time.time() - start_time
    print(f"Inserted {num_distractors} distractors in {distractor_time:.2f} seconds.\n")

    # Insert key facts (the needles) – you can adjust these as needed
    print(f"Inserting {num_key_facts} key facts with specific details...")
    sample_key_facts = [
        "My secret code is 12345.",
        "I have a pet dog named Rex.",
        "I prefer tea over coffee.",
        "My favorite book is '1984'.",
        "I once visited Paris in the spring.",
    ]
    key_facts = sample_key_facts[:num_key_facts]
    for fact in key_facts:
        api.add_memory(fact)
    print("Key facts inserted.\n")

    # Define retrieval queries for the key facts along with expected substrings.
    queries = [
        ("What is my secret code?", "12345"),
        ("What is the name of my pet?", "Rex"),
        ("Do I prefer coffee or tea?", "tea"),
        ("What is my favorite book?", "1984"),
        ("Where did I visit in the spring?", "Paris"),
    ]
    queries = queries[:num_key_facts]

    # Run retrieval queries and check if expected substrings are present.
    correct_count = 0
    query_times = []
    print("Running key fact retrieval queries...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description} ({task.completed}/{task.total})"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Running queries...", total=len(queries))
        for i, (query, expected_substr) in enumerate(queries):
            start = time.time()
            # Use chat to trigger memory retrieval
            response = api.chat(query, max_new_tokens=50)
            elapsed_query = time.time() - start
            query_times.append(elapsed_query)
            progress.advance(task)
            # Check if expected substring is present (case-insensitive)
            if expected_substr.lower() in response.lower():
                correct_count += 1
            print(f"\nQuery {i + 1}: {query}")
            print(f"Expected: {expected_substr}")
            print(f"Response: {response}")
            print(f"Time: {elapsed_query:.2f} seconds")

    total_query_time = sum(query_times)
    avg_query_time = total_query_time / len(query_times)
    print("\n--- Retrieval Performance ---")
    print(f"Correct retrievals: {correct_count}/{len(queries)}")
    print(f"Total query phase time: {total_query_time:.2f} seconds")
    print(f"Average query time: {avg_query_time:.2f} seconds")

    # Optionally, you could further analyze these responses (e.g., computing F1 scores)
    # by comparing expected substrings to the retrieved content.


if __name__ == "__main__":
    main()
