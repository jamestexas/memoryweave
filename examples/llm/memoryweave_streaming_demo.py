#!/usr/bin/env python3
"""
memoryweave_streaming_demo.py

Demonstrates MemoryWeave's API with streaming responses and memory-enhanced conversations.
Features colorful terminal output and a clean interactive experience.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator

import rich_click as click
from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Prompt

# Import the API
from memoryweave.api import MemoryWeaveAPI

# Setup logging with rich
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("memoryweave")

# Create console for pretty output
console = Console()

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"


class MemoryWeaveDemo:
    """Demo class for MemoryWeave with streaming responses."""

    def __init__(self, model_name: str, debug: bool = False):
        """Initialize the MemoryWeave demo."""
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)

        # Initialize the API
        self.api = MemoryWeaveAPI(
            model_name=model_name,
            debug=debug,
        )

        # Set up streaming
        # Note: The API now handles streaming internally

    def add_memories(self, memories: list[str]) -> None:
        """Add initial memories to the system."""
        console.print("[bold cyan]Adding initial memories...[/bold cyan]")
        for memory in memories:
            try:
                self.api.add_memory(memory)
                console.print(f"[green]Added:[/green] {memory}")
            except Exception as e:
                console.print(f"[red]Error adding memory:[/red] {str(e)}")

    def chat_sync(self, user_input: str) -> str:
        """Process a user message and generate a response (non-streaming)."""
        try:
            response = self.api.chat(user_input)
            return response
        except Exception as e:
            console.print(f"[red]Error generating response:[/red] {str(e)}")
            return "I encountered an error processing your request."

    async def chat_async(self, user_input: str) -> AsyncGenerator[str, None]:
        """Process a user message and generate a streaming response."""
        try:
            async for token in self.api.chat_stream(user_input):
                yield token
        except Exception as e:
            console.print(f"[red]Error in streaming response:[/red] {str(e)}")
            yield "I encountered an error processing your request."

    def run_interactive_demo(self) -> None:
        """Run an interactive demo allowing the user to chat with MemoryWeave."""
        console.print("[bold magenta]=== MemoryWeave Interactive Demo ===[/bold magenta]")
        console.print(
            "Type your messages below. Type 'exit' or 'quit' to end the demo.\n"
            "Type 'add memory: <text>' to add a new memory.\n"
        )

        # Add some initial memories for testing
        initial_memories = [
            "My name is Alex.",
            "I live in San Francisco.",
            "I have a golden retriever named Max.",
            "I enjoy hiking on weekends.",
        ]
        self.add_memories(initial_memories)

        # Main interaction loop
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

                # Check for exit command
                if user_input.lower() in ["exit", "quit"]:
                    console.print("[bold cyan]Exiting demo. Goodbye![/bold cyan]")
                    break

                # Check for memory addition command
                if user_input.lower().startswith("add memory:"):
                    memory_text = user_input[len("add memory:") :].strip()
                    self.api.add_memory(memory_text)
                    console.print(f"[green]Added memory:[/green] {memory_text}")
                    continue

                # Process user message
                console.print("")  # Add space before assistant response

                # Stream the response
                asyncio.run(self._stream_response(user_input))

            except KeyboardInterrupt:
                console.print("\n[bold cyan]Interrupted. Exiting demo.[/bold cyan]")
                break
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")

    async def _stream_response(self, user_input: str) -> None:
        """Stream the assistant's response token by token."""
        console.print("[bold green]Assistant:[/bold green] ", end="")

        full_response = ""
        start_time = time.time()

        async for token in self.chat_async(user_input):
            full_response += token
            console.print(token, end="")

        elapsed = time.time() - start_time

        # Print timing info if debug is enabled
        if self.debug:
            console.print(f"\n[dim](Response generated in {elapsed:.2f}s)[/dim]")

        # Show evidence if debug is enabled
        if self.debug:
            self._display_reasoning()

    def _display_reasoning(self) -> None:
        """Display the reasoning and evidence used for the response (for debugging)."""
        try:
            # In the new API, we would need to implement a way to access internal reasoning
            # This is a placeholder for future implementation
            pass

        except Exception as e:
            logger.error(f"Error displaying reasoning: {e}")


@click.command()
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    help=f"Name of the model to use (default: {DEFAULT_MODEL})",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode with extra logging",
)
def main(model: str, debug: bool) -> None:
    """
    Run the MemoryWeave streaming demo.

    This demo showcases MemoryWeave's ability to maintain context through a conversation,
    providing memory-enhanced responses in a streaming fashion.
    """
    try:
        demo = MemoryWeaveDemo(model_name=model, debug=debug)
        demo.run_interactive_demo()
    except KeyboardInterrupt:
        console.print("\n[bold cyan]Demo interrupted. Exiting.[/bold cyan]")
    except Exception as e:
        console.print(f"[bold red]Error in demo:[/bold red] {str(e)}")
        if debug:
            console.print_exception()


if __name__ == "__main__":
    main()
