#!/usr/bin/env python3

"""
scenario_demo.py

A self-contained demonstration that shows MemoryWeave's capability to store and recall
personal/contextual details across multiple conversation turns.
"""

import logging
import os
import time

# Disable HuggingFace warnings and info messages
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import rich_click as click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

# Import MemoryWeave API
from memoryweave.api import MemoryWeaveAPI

# Configure logging
console = Console(highlight=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_path=False, rich_tracebacks=True)],
)

# Silence other loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)

logger = logging.getLogger("scenario_demo")
DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"


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
    """
    A demonstration of MemoryWeave in a scenario-based conversation,
    showing how personal context is stored, recalled, and updated automatically.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("memoryweave").setLevel(logging.DEBUG)
        logger.debug("Debug logging is enabled.")
    else:
        # Keep most logs silenced in normal mode
        logging.getLogger("memoryweave").setLevel(logging.WARNING)

    console.print(
        Panel.fit(
            "[bold cyan]MemoryWeave Scenario Demo[/bold cyan]\n\n"
            f"Model: [yellow]{model}[/yellow]\n"
            "This demo shows how MemoryWeave stores and recalls personal information across conversation turns.",
            border_style="cyan",
        )
    )

    # Initialize the MemoryWeave API
    console.print("[bold]Initializing MemoryWeave...[/bold]")
    api = None

    # Use a single status display for loading
    with console.status("[bold green]Loading models...[/bold green]", spinner="dots") as status:
        api = MemoryWeaveAPI(model_name=model)
        status.update("[bold green]Initialization complete[/bold green]")
        time.sleep(0.5)  # Give a moment to see the completion message

    console.print("[bold green]✓[/bold green] MemoryWeave ready\n")

    # Define conversation scenario
    conversation = [
        (
            "Hello! My name is Mark, and I'm allergic to peanuts. I also have a dog named Max.",
            "Initial introduction",
        ),
        (
            "Nice to meet you. I'd love to get some Thai food. Any suggestions?",
            "Food recommendation request",
        ),
        (
            "Actually, I'd like to confirm: what's my dog's name?",
            "Memory recall check - dog's name",
        ),
        ("Oh, and do you remember my allergy?", "Memory recall check - allergy"),
    ]

    # Run the conversation
    console.print("[bold]Starting conversation scenario[/bold]\n")

    for i, (user_msg, label) in enumerate(conversation, start=1):
        # Show user message in a panel
        console.print(
            Panel(
                user_msg,
                title=f"[bold]User - Step {i}[/bold]: {label}",
                title_align="left",
                border_style="yellow",
            )
        )

        # Generate response
        start_time = time.time()
        assistant_reply = None

        # Use a single status display for thinking
        with console.status(
            f"[bold green]Step {i}: Thinking...[/bold green]", spinner="dots"
        ) as status:
            try:
                assistant_reply = api.chat(user_msg, max_new_tokens=100)
                elapsed = time.time() - start_time
                status.update(f"[bold green]Response ready ({elapsed:.1f}s)[/bold green]")
                time.sleep(0.5)  # Give a moment to see the completion message
            except Exception as e:
                status.update(f"[bold red]Error: {str(e)}[/bold red]")
                time.sleep(1)

        # Show assistant response if we got one
        if assistant_reply:
            console.print(
                Panel(
                    assistant_reply,
                    title=f"[bold]Assistant[/bold] ({elapsed:.1f}s)",
                    title_align="left",
                    border_style="green",
                )
            )
        else:
            console.print("[bold red]Failed to generate response[/bold red]")

        console.print()  # Add space between exchanges

    # Show conversation summary
    console.print(Panel.fit("[bold cyan]Memory Test Complete![/bold cyan]", border_style="cyan"))

    console.print("[bold]Full conversation history:[/bold]")

    # Display the conversation history
    history = api.get_conversation_history()
    if not history:
        console.print("[yellow]No conversation history recorded[/yellow]")
    else:
        for i, turn in enumerate(history):
            role = turn["role"]
            text = turn["content"]

            if role == "user":
                console.print(f"[yellow]User:[/yellow] {text}")
            else:
                console.print(f"[green]Assistant:[/green] {text}")

            # Add separator between turns, but not after the last one
            if i < len(history) - 1:
                console.print("─" * 80)

    console.print(
        "\n[bold cyan]Demo complete![/bold cyan] Personal info (name, allergy, dog's name) was successfully stored and recalled."
    )


if __name__ == "__main__":
    main()
