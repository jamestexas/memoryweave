#!/usr/bin/env python3
"""
memoryweave_basic_demo.py

A simple demonstration of the core MemoryWeaveAPI functionality.
This example shows the basic initialization, adding memories, and
generating responses with memory-enhanced context.
"""

import logging

import rich_click as click
from rich import print
from rich.console import Console
from rich.logging import RichHandler

# Import the MemoryWeave API
from memoryweave.api import MemoryWeaveAPI

# Set up logging
console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("memoryweave")

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"


@click.command()
@click.option(
    "--model", default=DEFAULT_MODEL, help=f"Hugging Face model to use (default: {DEFAULT_MODEL})"
)
@click.option("--debug", is_flag=True, help="Enable debug logging for detailed information")
def main(model, debug):
    """Demonstrate core MemoryWeaveAPI functionality."""
    if debug:
        logger.setLevel(logging.DEBUG)
        print("[yellow]Debug logging enabled[/yellow]")

    print("[bold cyan]Starting MemoryWeave Basic Demo[/bold cyan]")

    # Initialize the API
    memory_api = MemoryWeaveAPI(model_name=model, debug=debug)

    # Add some memories
    print("\n[bold magenta]Adding memories about a fictional user:[/bold magenta]")

    memories = [
        "My name is Jordan and I work as a software engineer.",
        "I live in Toronto with my cat Luna.",
        "I enjoy rock climbing and photography on weekends.",
        "I'm planning a trip to Japan next spring.",
    ]

    for memory in memories:
        memory_api.add_memory(memory)
        print(f"[green]Added memory:[/green] {memory}")

    # Run a conversation to demonstrate memory retrieval
    print("\n[bold magenta]Starting a conversation with memory retrieval:[/bold magenta]")

    questions = [
        "What do I do for work?",
        "What's my pet's name?",
        "What hobbies did I mention earlier?",
        "Where am I planning to travel?",
        "Tell me everything you know about me.",
    ]

    for question in questions:
        print(f"\n[bold blue]User:[/bold blue] {question}")

        response = memory_api.chat(question)
        print(f"[bold green]Assistant:[/bold green] {response}")

    # Demonstrate search functionality
    print("\n[bold magenta]Searching memories by keyword:[/bold magenta]")

    search_results = memory_api.search_by_keyword("weekend")
    print("[cyan]Memories about weekends:[/cyan]")
    for result in search_results:
        print(f"- {result['content']} (Score: {result['relevance_score']:.2f})")

    # Display conversation history
    print("\n[bold magenta]Conversation history:[/bold magenta]")

    history = memory_api.get_conversation_history()
    for turn in history:
        role = turn["role"]
        content = turn["content"]
        color = "blue" if role == "user" else "green"
        print(f"[bold {color}]{role.capitalize()}:[/bold {color}] {content}")

    print("\n[bold cyan]Demo complete![/bold cyan]")


if __name__ == "__main__":
    main()
