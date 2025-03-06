#!/usr/bin/env python3
"""
mw_temporal_benchmark_detailed.py

A detailed benchmark that tests only MemoryWeave's handling of temporal references.
It adds several user facts (with associated timestamps) and then runs queries to test
whether MemoryWeave correctly incorporates temporal context.
For each query, a per-query breakdown table is shown with weighted contributions,
and aggregate statistics are logged at the end.
The conversation is recorded and saved to a log file.
"""

import logging
import secrets
import time
from datetime import datetime

import rich_click as click
from memoryweave_llm_wrapper import MemoryWeaveLLM
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

console = Console(record=True)
logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler(markup=True)])
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
# To record the conversation (queries and responses)
CONVERSATION_LOG = []  # list of (query, response, breakdown dict)
# Define temporal reference types and test scenarios
TEMPORAL_REFERENCES = [
    "yesterday",
    "last week",
    "earlier today",
    "this morning",
    "previously",
    "before",
    "a while ago",
]

TEST_SCENARIOS = [
    ("I visited the Golden Gate Bridge {when}.", "Where did I go {temporal_ref}?"),
    ("I had pasta with mushrooms for dinner {when}.", "What did I eat {temporal_ref}?"),
    ("I watched a documentary about dolphins {when}.", "What show did I watch {temporal_ref}?"),
    (
        "I learned that Jupiter has 79 known moons {when}.",
        "Tell me what I learned about Jupiter {temporal_ref}.",
    ),
    ("My friend Emma called me about her new job {when}.", "Who called me {temporal_ref}?"),
    ("I fixed my bicycle's flat tire {when}.", "What did I repair {temporal_ref}?"),
]


def format_timestamp(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def configure_time_periods():
    """Configure different time periods for temporal reference testing."""
    now = time.time()
    yesterday = now - 86400
    last_week = now - 604800
    earlier_today = now - 21600
    this_morning = now - 43200

    time_periods = {
        "yesterday": yesterday,
        "last week": last_week,
        "earlier today": earlier_today,
        "this morning": this_morning,
        "previously": last_week,
        "before": earlier_today,
        "a while ago": last_week,
    }

    return time_periods


def add_benchmark_facts(mw, time_periods, scenarios):
    """Add facts to MemoryWeave for benchmarking with appropriate temporal context."""
    console.print(
        "Starting Benchmark: Adding Facts",
        style="bold cyan",
        new_line_start=True,
    )
    fact_metadata = []

    for i, (fact_template, _) in enumerate(TEST_SCENARIOS[:scenarios]):
        temporal_ref = secrets.choice(TEMPORAL_REFERENCES)
        timestamp = time_periods[temporal_ref]

        # Simple conversion for display
        if timestamp == time_periods["yesterday"]:
            when_str = "yesterday"
        elif timestamp == time_periods["last week"]:
            when_str = "7 days ago"
        elif timestamp == time_periods["earlier today"]:
            when_str = "earlier today"
        elif timestamp == time_periods["this morning"]:
            when_str = "this morning"
        else:
            when_str = "recently"

        fact = fact_template.format(when=when_str)
        console.print(f"\n[bold]Adding fact {i + 1}:[/bold] {fact}")
        console.print(f"Associated time: {format_timestamp(timestamp)}", style="dim")
        mw.add_memory(fact, {"type": "user_fact", "created_at": timestamp, "importance": 0.8})
        fact_metadata.append((fact, temporal_ref))

        # Add a distractor
        distractor = "I'm thinking about buying a new laptop."
        mw.add_memory(
            distractor, {"type": "user_message", "created_at": time.time(), "importance": 0.4}
        )

    return fact_metadata


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
@click.option("--scenarios", type=int, default=4, help="Number of scenarios to test (default: 4)")
@click.option(
    "--logfile", default="mw_temporal_benchmark_log.txt", help="File to write conversation log"
)
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(model, scenarios, logfile, debug):
    if debug:
        logger.setLevel(logging.DEBUG)
        console.log("[yellow]Debug logging enabled[/yellow]")

    console.rule("[bold cyan]MemoryWeave-Only Temporal Benchmark (Detailed)[/bold cyan]")
    mw = MemoryWeaveLLM(model_name=model)

    # Get time periods from the configuration function
    time_periods = configure_time_periods()

    # Add facts for benchmarking
    fact_metadata = add_benchmark_facts(mw, time_periods, scenarios)

    console.rule("[bold green]Running Queries[/bold green]")

    # For aggregate metrics
    agg_breakdowns = dict(
        similarity=[],
        associative=[],
        temporal=[],
        activation=[],
        total=[],
    )

    # Run queries and print per-query breakdown tables
    for i, (_fact, temporal_ref) in enumerate(fact_metadata):
        _, query_template = TEST_SCENARIOS[i]
        query = query_template.format(temporal_ref=temporal_ref)
        console.print(f"\n[bold]Query {i + 1}:[/bold] {query}")
        response = mw.chat(query)
        console.print(f"[bold]Response:[/bold] {response}")

        # Get breakdown info by performing a retrieval debug call
        query_embedding = mw.embedding_model.encode(query, show_progress_bar=False)
        context = {
            "query": query,
            "current_time": time.time(),
            "memory_store": mw.memory_store_adapter,
        }
        results = mw.strategy.retrieve(query_embedding, top_k=10, context=context)
        if results:
            top = results[0]
            breakdown = {
                "similarity": top.get("similarity_contribution", 0.0),
                "associative": top.get("associative_contribution", 0.0),
                "temporal": top.get("temporal_contribution", 0.0),
                "activation": top.get("activation_contribution", 0.0),
                "total": top.get("relevance_score", 0.0),
            }
        else:
            breakdown = {
                "similarity": 0,
                "associative": 0,
                "temporal": 0,
                "activation": 0,
                "total": 0,
            }

        for key in agg_breakdowns:
            agg_breakdowns[key].append(breakdown[key])

        # Build a per-query breakdown table using Rich
        table = Table(title=f"Query {i + 1} Breakdown", show_edge=True)
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Score", style="magenta", justify="right")
        table.add_row("Similarity", f"{breakdown['similarity']:.3f}")
        table.add_row("Associative", f"{breakdown['associative']:.3f}")
        table.add_row("Temporal", f"{breakdown['temporal']:.3f}")
        table.add_row("Activation", f"{breakdown['activation']:.3f}")
        table.add_row("Total", f"{breakdown['total']:.3f}")
        console.print(table)

        CONVERSATION_LOG.append((query, response, breakdown))

    # Compute aggregate statistics
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    agg_table = Table(title="Aggregate Metrics", show_edge=True)
    agg_table.add_column("Component", style="cyan", no_wrap=True)
    agg_table.add_column("Average Score", style="magenta", justify="right")
    agg_table.add_row("Similarity", f"{avg(agg_breakdowns['similarity']):.3f}")
    agg_table.add_row("Associative", f"{avg(agg_breakdowns['associative']):.3f}")
    agg_table.add_row("Temporal", f"{avg(agg_breakdowns['temporal']):.3f}")
    agg_table.add_row("Activation", f"{avg(agg_breakdowns['activation']):.3f}")
    agg_table.add_row("Total", f"{avg(agg_breakdowns['total']):.3f}")
    console.rule("[bold green]Aggregate Breakdown[/bold green]")
    console.print(agg_table)

    # Optionally write the recorded console output to a file
    try:
        output_text = console.export_text()
        with open(logfile, "w", encoding="utf-8") as f:
            f.write(output_text)
        console.print(f"\n[green]Full benchmark output saved to {logfile}[/green]")
    except Exception as e:
        console.print(f"[red]Error writing log file: {e}[/red]")

    console.rule("[bold green]Benchmark Complete[/bold green]")


if __name__ == "__main__":
    main()
