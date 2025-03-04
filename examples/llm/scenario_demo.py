#!/usr/bin/env python3

"""
scenario_demo.py

A self-contained demonstration that shows MemoryWeave's capability to store and recall
personal/contextual details across multiple conversation turns. It uses Rich logging
and a CLI interface via rich_click for a user-friendly experience.

Usage:
  pip install rich_click

  python scenario_demo.py
    or
  python scenario_demo.py --model "some-other-hf-model"
    or
  python scenario_demo.py --debug
"""

import time
import logging

# rich_click as a drop-in for Click to get pretty terminal formatting
import rich_click as click
from rich import print
from rich.console import Console
from rich.logging import RichHandler

# Our MemoryWeave LLM wrapper (assumes memoryweave_llm_wrapper.py is in the same directory)
from memoryweave_llm_wrapper import MemoryWeaveLLM

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,  # default to INFO level
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()]  # route logging through Rich for pretty logs
)
logger = logging.getLogger(__name__)
console = Console(highlight=True)

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
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging is enabled.")

    console.log("Starting the MemoryWeave scenario demo...", style="bold cyan")
    console.log(f"[bold]Using model[/bold]: {model}")

    # 1) Initialize the MemoryWeave LLM
    llm = MemoryWeaveLLM(model_name=model)

    # 2) Scenario demonstration
    # We go through a sequence of user messages and show how MemoryWeave
    # accumulates personal details. Then the user references them.

    conversation = [
        # (User message, short label/description for logging)
        ("Hello! My name is Mark, and I'm allergic to peanuts. I also have a dog named Max.", "Initial info"),
        ("Nice to meet you. I'd love to get some Thai food. Any suggestions?", "Allergy-based request"),
        ("Actually, I'd like to confirm: what's my dog's name?", "Memory recall check"),
        ("Oh, and do you remember my allergy?", "Another memory recall check"),
    ]

    # We'll step through each conversation message:
    for i, (user_msg, label) in enumerate(conversation, start=1):
        # Show user message
        console.log(f"\n[bold yellow]User (Step {i} - {label}):[/bold yellow] {user_msg}")

        # Let MemoryWeave respond
        start_time = time.time()
        try:
            assistant_reply = llm.chat(user_msg, max_new_tokens=100)
        except Exception as e:
            logger.error(f"[red]Error generating assistant response:[/red] {e}")
            continue
        elapsed = time.time() - start_time

        # Print the assistant's reply
        console.log(f"[bold green]Assistant (Step {i}):[/bold green] {assistant_reply}")
        logger.debug(f"Response took {elapsed:.2f}s")

    # 3) Wrap up
    console.log("\n[bold cyan]Scenario complete![/bold cyan]")
    console.log("Below is the final conversation history as stored in MemoryWeave:\n")

    # 4) Show entire conversation from the LLM's perspective
    history = llm.get_conversation_history()
    for turn in history:
        role = turn["role"]
        text = turn["content"]
        role_str = "User" if role == "user" else "Assistant"
        # We color user messages in yellow, assistant in green
        color = "yellow" if role == "user" else "green"
        console.log(f"[bold {color}]{role_str}:[/bold {color}] {text}")

    console.log(
        "[bold magenta]Done![/bold magenta] You can see how personal info (name, allergy, dog's name)"
        " was stored and reused across multiple turns.\n", 
        newline_start=True,
    )


if __name__ == "__main__":
    main()
