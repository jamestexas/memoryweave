#!/usr/bin/env python3
"""
benchmark_scenario.py

A script that demonstrates how MemoryWeave can recall multiple user facts
compared to a baseline (no-memory) approach. We define a set of facts,
insert them into a conversation, then ask about them. We check the LLM's
answers for correctness.
"""

import logging
import time

import rich_click as click

# Import our wrapper class
from memoryweave_llm_wrapper import MemoryWeaveLLM
from rich import print
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"

# We'll define some user facts that we want to share and then quiz the LLM about.
USER_FACTS = [
    # (fact statement, question, expected substring)
    ("I am Alicia", "What's my name?", ["Alicia"]),
    ("I work as a nurse", "What is my occupation?", ["nurse"]),
    ("My favorite color is green", "What's my favorite color?", ["green"]),
    ("I have a cat named Lucy", "What's my pet's name?", ["Lucy"]),
    ("I live in Chicago", "Which city do I live in?", ["Chicago"]),
    ("I am allergic to peanuts", "What is my allergy?", ["peanuts"]),
    ("I like painting on weekends", "What activities do I enjoy?", ["paint", "painting"]),
    ("I love sushi", "Which food do I like best?", ["sushi"]),
    ("I have a spouse named Jordan", "What's my spouse's name?", ["Jordan"]),
    ("I dream of visiting Japan", "Which country do I dream of?", ["Japan"]),
]

# We can optionally lead into these facts with a preamble:
PREAMBLE = "Hello, I'd like to share some details about myself:\n"

console = Console()
FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(markup=True),  # allow colors in terminal
    ],
)
logger = logging.getLogger(__name__)


def run_benchmark(model_name: str, with_memory: bool):
    """
    Runs a multi-turn conversation where:
      1. We share user facts (hard-coded).
      2. We ask about them.
      3. We measure how accurately the LLM recalled them.
    If `with_memory=False`, we skip MemoryWeave's retrieval (call chat_without_memory).
    Otherwise, we use MemoryWeave's normal chat.
    Returns a list of (question, answer, expected, success_code) plus query times.
    """

    # 1) Initialize the MemoryWeave LLM
    llm = MemoryWeaveLLM(model_name=model_name)

    mode_label = "WITH Memory" if with_memory else "NO Memory"
    logger.info(
        f"\n[bold cyan]--- Running Benchmark in mode: {mode_label} ---[/bold cyan]",
        extra=dict(markup=True),
    )

    # Keep track of how long each query takes
    query_times = []

    # Keep track of results for each question
    results = []

    # 2) Start by "telling" the system the user facts
    # We'll do this in a single user message. Alternatively, we can do them turn by turn.
    # We'll put them all in one user message so it doesn't take too many queries.
    joined_facts = "\n".join(fact for (fact, _, _) in USER_FACTS)
    user_message_facts = PREAMBLE + joined_facts

    start_time = time.time()
    if with_memory:
        _ = llm.chat(user_message_facts, max_new_tokens=100)
    else:
        _ = llm.chat_without_memory(user_message_facts, max_new_tokens=100)
    elapsed = time.time() - start_time
    query_times.append(elapsed)
    logger.info(f"Shared user facts. (Took {elapsed:.2f}s)")

    # 3) Now we ask about each fact in turn
    for fact, question, expected_substrs in USER_FACTS:
        # Ask the question
        logger.info(f"[bold yellow]User:[/bold yellow] {question}", extra=dict(markup=True))
        start_time = time.time()
        if with_memory:
            assistant_reply = llm.chat(question, max_new_tokens=100)
        else:
            assistant_reply = llm.chat_without_memory(question, max_new_tokens=100)
        elapsed = time.time() - start_time
        query_times.append(elapsed)

        # Log the assistant's response
        logger.info(
            f"[bold green]Assistant:[/bold green] {assistant_reply} (took {elapsed:.2f}s)\n",
            extra=dict(markup=True),
        )

        # 4) Check if the answer includes the expected substring(s)
        # We'll say it's a "success" if ALL expected substrings appear,
        # "partial" if at least one substring appears, "fail" if none appear.
        # (You can tweak this logic as you see fit.)
        assistant_lower = assistant_reply.lower()
        found_count = sum(1 for es in expected_substrs if es.lower() in assistant_lower)

        if found_count == len(expected_substrs):
            success_code = "success"
        elif found_count > 0:
            success_code = "partial"
        else:
            success_code = "fail"

        results.append((question, assistant_reply, expected_substrs, success_code))

    # 5) Summarize
    return results, query_times, llm.get_conversation_history()


def summarize_results(results):
    """
    Summarize success vs fail counts, return a dict of stats.
    """
    stats = {"successes": 0, "failures": 0, "partial": 0, "total": len(results)}
    for _, _, _, code in results:
        if code == "success":
            stats["successes"] += 1
        elif code == "partial":
            stats["partial"] += 1
        elif code == "fail":
            stats["failures"] += 1

    # We can define a simple recall "score"
    # success = 1 point, partial = 0.5 points, fail = 0
    points = stats["successes"] + 0.5 * stats["partial"]
    possible = stats["total"]
    if possible == 0:
        stats["score"] = 0.0
    else:
        stats["score"] = (points / possible) * 100.0
    return stats


def display_table(name: str, stats: dict, avg_time: float):
    """
    Show a small table summarizing the results for "with memory" or "no memory".
    """
    table = Table(title=f"{name} Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Successes", str(stats["successes"]))
    table.add_row("Partial", str(stats["partial"]))
    table.add_row("Failures", str(stats["failures"]))
    table.add_row("Score (Weighted)", f"{stats['score']:.1f}%")
    table.add_row("Total Queries", str(stats["total"]))
    table.add_row("Avg Query Time", f"{avg_time:.2f}s")

    console.print(table)


@click.command()
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    help=f"Name of the Hugging Face model to load (default: {DEFAULT_MODEL})",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging for more detailed output.",
)
def main(model, debug):
    # Possibly enable debug
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print("[yellow]Debug logging enabled[/yellow]")

    # 1) Run with memory
    with_mem_results, with_mem_times, _with_mem_history = run_benchmark(model, True)
    # Summarize
    with_mem_stats = summarize_results(with_mem_results)
    with_mem_avg = sum(with_mem_times) / len(with_mem_times)

    # 2) Run no memory
    no_mem_results, no_mem_times, _no_mem_history = run_benchmark(model, False)
    # Summarize
    no_mem_stats = summarize_results(no_mem_results)
    no_mem_avg = sum(no_mem_times) / len(no_mem_times)

    # 3) Print comparison
    display_table("WITH Memory", with_mem_stats, with_mem_avg)
    display_table("NO Memory", no_mem_stats, no_mem_avg)

    # Show improvement
    improvement = with_mem_stats["score"] - no_mem_stats["score"]
    logger.info(
        f"\n[bold cyan]Score improvement with memory: {improvement:.1f} percentage points[/bold cyan]"
    )


if __name__ == "__main__":
    main()
