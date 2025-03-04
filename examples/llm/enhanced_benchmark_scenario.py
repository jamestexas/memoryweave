#!/usr/bin/env python3
"""
enhanced_benchmark_scenario.py

A script that demonstrates MemoryWeave's recall capabilities in various scenarios,
including factual recall, conversational context, and nuanced queries.
"""

import logging
import os
import sys
import time

import rich_click as click
from rich import print
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Adjust path if memoryweave_llm_wrapper.py is one folder up or in the same dir:
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memoryweave_llm_wrapper import MemoryWeaveLLM

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"

# Enhanced User Facts and Scenarios
TEST_SCENARIOS = [
    # 1. Factual Recall (Basic)
    ("My name is Alex.", "What is my name?", ["Alex"], "factual"),
    ("I work as a software engineer.", "What is my occupation?", ["software engineer"], "factual"),
    # 2. Conversational Context (Multi-turn)
    (
        "I like to hike.",
        "That's nice. What activities do I enjoy?",
        ["hike"],
        "contextual",
    ),
    (
        "My favorite food is pizza.",
        "What did I say my favorite food was?",
        ["pizza"],
        "contextual",
    ),
    # 3. Nuanced Queries (Inference)
    (
        "I have a cat named Whiskers. Whiskers loves to chase mice.",
        "What does my cat like to chase?",
        ["mice"],
        "inference",
    ),
    (
        "I live in a city with many tall buildings.",
        "Do I live in a rural area?",
        ["no"],
        "inference",
    ),
    # 4. Long Term Memory (After context window shifting)
    (
        "I visited Paris last summer.",
        "Where did I travel last summer?",
        ["Paris"],
        "long_term",
    ),
    (
        "My favorite book is 'The Hitchhiker's Guide to the Galaxy'.",
        "What book did I mention I liked?",
        ["Hitchhiker"],
        "long_term",
    ),
    # 5. Ambiguous Questions
    (
        "I enjoy playing music.",
        "What do I like to do?",
        ["music", "play"],
        "ambiguous"
    ),
    (
        "Today is a beautiful day.",
        "How is the weather?",
        ["beautiful", "day"],
        "ambiguous"
    )
]

PREAMBLE = "Let's talk about me."

console = Console()
FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(markup=True)],
)
logger = logging.getLogger(__name__)


def run_benchmark(model_name: str, with_memory: bool):
    llm = MemoryWeaveLLM(model_name=model_name)
    mode_label = "WITH Memory" if with_memory else "NO Memory"
    logger.info(f"\n[bold cyan]--- Running Benchmark in mode: {mode_label} ---[/bold cyan]", extra=dict(markup=True))

    query_times = []
    results = []

    # Inject all the facts first.
    joined_facts = "\n".join(fact for (fact, _, _, _) in TEST_SCENARIOS)
    user_message_facts = PREAMBLE + "\n" + joined_facts

    start_time = time.time()
    if with_memory:
        _ = llm.chat(user_message_facts, max_new_tokens=200)
    else:
        _ = llm.chat_without_memory(user_message_facts, max_new_tokens=200)
    elapsed = time.time() - start_time
    query_times.append(elapsed)
    logger.info(f"Shared user facts. (Took {elapsed:.2f}s)")

    # Test each scenario.
    for fact, question, expected_substrs, scenario_type in TEST_SCENARIOS:
        logger.info(f"[bold yellow]User:[/bold yellow] {question}", extra=dict(markup=True))
        start_time = time.time()
        if with_memory:
            assistant_reply = llm.chat(question, max_new_tokens=200)
        else:
            assistant_reply = llm.chat_without_memory(question, max_new_tokens=200)
        elapsed = time.time() - start_time
        query_times.append(elapsed)

        logger.info(f"[bold green]Assistant:[/bold green] {assistant_reply} (took {elapsed:.2f}s)\n", extra=dict(markup=True))

        assistant_lower = assistant_reply.lower()
        found_count = sum(1 for es in expected_substrs if es.lower() in assistant_lower)

        if found_count == len(expected_substrs):
            success_code = "success"
        elif found_count > 0:
            success_code = "partial"
        else:
            success_code = "fail"

        results.append((question, assistant_reply, expected_substrs, success_code, scenario_type))

    return results, query_times, llm.get_conversation_history()


def summarize_results(results):
    stats = {"successes": 0, "failures": 0, "partial": 0, "total": len(results)}
    scenario_stats = {}
    for _, _, _, code, scenario_type in results:
        if code == "success":
            stats["successes"] += 1
        elif code == "partial":
            stats["partial"] += 1
        elif code == "fail":
            stats["failures"] += 1

        if scenario_type not in scenario_stats:
            scenario_stats[scenario_type] = {"successes": 0, "failures": 0, "partial": 0, "total": 0}
        scenario_stats[scenario_type]["total"] += 1
        if code == "success":
            scenario_stats[scenario_type]["successes"] += 1
        elif code == "partial":
            scenario_stats[scenario_type]["partial"] += 1
        elif code == "fail":
            scenario_stats[scenario_type]["failures"] += 1

    points = stats["successes"] + 0.5 * stats["partial"]
    possible = stats["total"]
    if possible == 0:
        stats["score"] = 0.0
    else:
        stats["score"] = (points / possible) * 100.0
    return stats, scenario_stats

def display_table(name: str, stats: dict, avg_time: float, scenario_stats: dict):
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

    scenario_table = Table(title=f"{name} Scenario Breakdown")
    scenario_table.add_column("Scenario", style="cyan")
    scenario_table.add_column("Successes", style="green")
    scenario_table.add_column("Failures", style="red")
    scenario_table.add_column("Partial", style="yellow")
    for scenario, s_stats in scenario_stats.items():
        scenario_table.add_row(scenario, str(s_stats["successes"]), str(s_stats["failures"]), str(s_stats["partial"]))
    console.print(scenario_table)

@click.command()
@click.option("--model", default=DEFAULT_MODEL, help=f"Name of the Hugging Face model to load (default: {DEFAULT_MODEL})")
@click.option("--debug", is_flag=True, help="Enable debug logging for more detailed output.")
def main(model, debug):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print("[yellow]Debug logging is enabled[/yellow]")

    with_mem_results, with_mem_times, with_mem_history = run_benchmark(model, True)
    with_mem_stats, with_mem_scenario_stats = summarize_results(with_mem_results)
    with_mem_avg = sum(with_mem_times) / len(with_mem_times)

    no_mem_results, no_mem_times, no_mem_history = run_benchmark(model, False)
    no_mem_stats, no_mem_scenario_stats = summarize_results(no_mem_results)
    no_mem_avg = sum(no_mem_times) / len(no_mem_times)

    display_table("WITH Memory", with_mem_stats, with_mem_avg, with_mem_scenario_stats)
    display_table("NO Memory", no_mem_stats, no_mem_avg, no_mem_scenario_stats)

    improvement = with_mem_stats["score"] - no_mem_stats["score"]
    logger.info(f"\n[bold cyan]Score improvement with memory: {improvement:.1f} percentage points[/bold cyan]")


if __name__ == "__main__":
    main()