#!/usr/bin/env python3
"""
chunking_demo.py

A demonstration of how MemoryWeave's chunking capability improves handling of large contexts.
This script shows the benefits of chunking for memory representation and retrieval accuracy.
"""

import logging
import time
from typing import Any

import rich_click as click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Import the chunked MemoryWeave API
from memoryweave.api.chunked_memory_weave import ChunkedMemoryWeaveAPI

# Setup console and logging
console = Console(highlight=True)
FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"

# Example large text for demonstration
LARGE_TEXT = """
Artificial neural networks (ANNs) are computing systems inspired by the biological neural networks that constitute animal brains. The neural network itself is not an algorithm, but rather a framework for many different machine learning algorithms to work together and process complex data inputs.

Neural networks are a series of algorithms that mimic the operations of a human brain to recognize relationships between vast amounts of data. They are used in a variety of applications in computer vision, speech recognition, natural language processing, and many other fields.

The concept of artificial neural networks began in the late 1940s with the work of Warren McCulloch and Walter Pitts, who created a computational model for neural networks based on mathematics and algorithms called threshold logic. In the 1950s, Frank Rosenblatt created the perceptron, an algorithm for pattern recognition based on a two-layer learning computer network using simple addition and subtraction.

The basic unit of computation in a neural network is the neuron, also called a node or unit. It receives input from some other nodes, or from an external source and computes an output. Each input has an associated weight (w), which is assigned on the basis of its relative importance to other inputs. The node applies a function f (called the Activation Function) to the weighted sum of its inputs.

The activation function is usually a step function, sigmoid function, or ReLU (Rectified Linear Unit) function. The purpose of the activation function is to introduce non-linearity into the output of a neuron. This is important because most real world data is non-linear and we want neurons to learn these non-linear representations.

Deep learning is part of a broader family of machine learning methods based on artificial neural networks. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, audio recognition, social network filtering, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases superior to human experts.

The term "deep" usually refers to the number of hidden layers in the neural network. Traditional neural networks only contain 2-3 hidden layers, while deep networks can have as many as 150. Deep learning models are trained by using large sets of labeled data and neural network architectures that learn features directly from the data without the need for manual feature extraction.

In recent years, deep learning has achieved remarkable success in various fields and has become a hot topic in artificial intelligence research. This success has been driven by the increasing availability of large amounts of data, the improvements in computational power, and the development of new algorithms and network architectures.
"""

# Example conversation for demonstration
EXAMPLE_CONVERSATION = [
    {
        "role": "user",
        "content": "I'm planning a trip to Europe next month. I'm thinking of visiting Paris, Rome, and Barcelona. What do you recommend I see in each city?",
    },
    {
        "role": "assistant",
        "content": "That sounds like a wonderful European adventure! Here are some recommendations for each city:\n\nParis:\n1. Eiffel Tower - Iconic symbol of Paris with stunning views\n2. Louvre Museum - Home to thousands of artworks including the Mona Lisa\n3. Notre-Dame Cathedral - Historic Gothic masterpiece (note: still under restoration)\n4. Montmartre and Sacré-Cœur - Artistic neighborhood with beautiful basilica\n5. Seine River cruise - A relaxing way to see many landmarks\n\nRome:\n1. Colosseum - Ancient amphitheater showcasing Roman engineering\n2. Vatican City - St. Peter's Basilica and the Vatican Museums with the Sistine Chapel\n3. Trevi Fountain - Baroque masterpiece (don't forget to throw a coin!)\n4. Roman Forum - Ancient government center with ruins to explore\n5. Pantheon - Remarkably preserved ancient temple\n\nBarcelona:\n1. Sagrada Família - Gaudí's unfinished masterpiece basilica\n2. Park Güell - Whimsical park with Gaudí's unique architecture\n3. La Rambla - Famous pedestrian street with shops and cafes\n4. Gothic Quarter - Medieval area with narrow streets and historic buildings\n5. Barceloneta Beach - If you want to relax by the Mediterranean\n\nWould you like more specific recommendations for any of these cities?",
    },
    {
        "role": "user",
        "content": "That's great information! I'm particularly interested in the food. What dishes should I try in each city?",
    },
    {
        "role": "assistant",
        "content": "You're in for a culinary treat across these cities! Here are some must-try dishes in each location:\n\nParis Food Specialties:\n1. Croissants and Pain au Chocolat - For breakfast or a snack from a local boulangerie\n2. Steak Frites - Classic bistro dish of steak with french fries\n3. Coq au Vin - Chicken braised with wine, mushrooms, and garlic\n4. Beef Bourguignon - Beef stewed in red wine with vegetables\n5. Escargots - Snails prepared with garlic and butter\n6. Crème Brûlée or Macarons - For dessert\n7. Cheese plate with varieties like Camembert, Brie, and Roquefort\n\nRome Food Specialties:\n1. Carbonara - Pasta with egg, cheese, pancetta, and black pepper\n2. Cacio e Pepe - Pasta with Pecorino Romano cheese and black pepper\n3. Supplì - Fried rice balls with mozzarella center\n4. Roman-style pizza - Thin, crispy crust unlike any other\n5. Saltimbocca alla Romana - Veal with prosciutto and sage\n6. Artichokes (Carciofi) - When in season, prepared Roman-style\n7. Gelato - Italian ice cream from authentic gelaterias\n\nBarcelona Food Specialties:\n1. Paella - Saffron rice dish, often with seafood in Barcelona\n2. Tapas - Small plates like patatas bravas (spicy potatoes) and gambas al ajillo (garlic shrimp)\n3. Jamón ibérico - Cured ham, often served with pan con tomate (bread with tomato)\n4. Bombas - Potato and meat croquettes with spicy sauce\n5. Crema Catalana - Similar to crème brûlée but with citrus flavors\n6. Churros con chocolate - Fried dough pastries with thick hot chocolate\n7. Seafood dishes - Barcelona's coastal location means excellent fresh seafood\n\nDon't miss visiting local markets in each city: Marché d'Aligre in Paris, Mercato di Testaccio in Rome, and La Boqueria in Barcelona. They're perfect for sampling local specialties!\n\nAny particular dish from this list that sounds most appealing to you?",
    },
]


class ChunkingDemo:
    """Class for demonstrating MemoryWeave's chunking capabilities."""

    def __init__(self, model_name=DEFAULT_MODEL, debug=False):
        """Initialize the chunking demo."""
        self.model_name = model_name
        self.debug = debug

        # Enable debug logging if requested
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("Debug logging enabled")

        # Initialize the chunked API
        self.api = ChunkedMemoryWeaveAPI(
            model_name=model_name,
            debug=debug,
            # Configure chunking parameters
            auto_chunk_threshold=300,  # Smaller threshold for demo purposes
            enable_auto_chunking=True,
        )

        # Configure chunker for demo settings
        self.api.configure_chunking(
            chunk_size=150,  # Smaller chunks for demo
            chunk_overlap=30,
            min_chunk_size=30,
            respect_paragraphs=True,
        )

        # Track added memories
        self.added_memories = []

    def run_demo(self):
        """Run the chunking demonstration."""
        console.print("\n[bold cyan]MemoryWeave Chunking Demonstration[/bold cyan]")
        console.print("This demo shows how chunking improves handling of large contexts")

        # Part 1: Add a large text memory
        console.print("\n[bold green]Part 1: Adding Large Text Memory[/bold green]")
        self._demo_large_text_memory()

        # Part 2: Add a conversation memory
        console.print("\n[bold green]Part 2: Adding Conversation Memory[/bold green]")
        self._demo_conversation_memory()

        # Part 3: Test retrieval from chunks
        console.print("\n[bold green]Part 3: Testing Chunk Retrieval[/bold green]")
        self._demo_chunk_retrieval()

        # Part 4: Show chunking statistics
        console.print("\n[bold green]Part 4: Chunking Statistics[/bold green]")
        self._show_chunking_statistics()

    def _demo_large_text_memory(self):
        """Demonstrate adding and chunking a large text memory."""
        console.print("Adding a large text about neural networks...")
        console.print(f"Text length: {len(LARGE_TEXT)} characters")

        # Add the large text
        start_time = time.time()
        memory_id = self.api.add_memory(
            LARGE_TEXT,
            {
                "type": "article",
                "topic": "neural networks",
                "importance": 0.8,
            },
        )
        elapsed = time.time() - start_time

        self.added_memories.append({"id": memory_id, "type": "article"})

        # Get chunks for this memory
        chunks = self.api.get_memory_chunks(memory_id)

        console.print(f"Memory added with ID: {memory_id} in {elapsed:.2f} seconds")
        console.print(f"Memory was split into {len(chunks)} chunks")

        # Show the first few chunks
        self._display_chunks(chunks[:3], "Sample Chunks from Neural Networks Text")

    def _demo_conversation_memory(self):
        """Demonstrate adding and chunking a conversation memory."""
        console.print("Adding a multi-turn conversation about European travel...")

        # Add the conversation
        start_time = time.time()
        memory_id = self.api.add_conversation_memory(
            EXAMPLE_CONVERSATION,
            {
                "type": "conversation",
                "topic": "travel",
                "importance": 0.7,
            },
        )
        elapsed = time.time() - start_time

        self.added_memories.append({"id": memory_id, "type": "conversation"})

        # Get chunks for this memory
        chunks = self.api.get_memory_chunks(memory_id)

        console.print(f"Conversation memory added with ID: {memory_id} in {elapsed:.2f} seconds")
        console.print(f"Conversation was split into {len(chunks)} chunks")

        # Show the first few chunks
        self._display_chunks(chunks[:3], "Sample Chunks from Travel Conversation")

    def _demo_chunk_retrieval(self):
        """Demonstrate retrieving information from chunks."""
        # Test queries for retrieval
        queries = [
            "Tell me about activation functions in neural networks",
            "What food should I try in Rome?",
            "How does deep learning differ from traditional neural networks?",
            "What are some tourist attractions in Barcelona?",
        ]

        # Run queries and show results
        for query in queries:
            console.print(f"\n[bold]Query:[/bold] {query}")

            # Retrieve memories
            start_time = time.time()
            memories = self.api.retrieve(query, top_k=3)
            elapsed = time.time() - start_time

            console.print(f"Retrieved {len(memories)} relevant memories in {elapsed:.2f} seconds")

            # Show result summary
            if memories:
                self._display_retrieval_results(memories, query)
            else:
                console.print("[yellow]No relevant memories found[/yellow]")

            # Generate a response
            console.print("\n[bold]Generating response with memory enhanced context...[/bold]")
            start_time = time.time()
            response = self.api.chat(query)
            elapsed = time.time() - start_time

            console.print(f"[green]Response (in {elapsed:.2f}s):[/green] {response}")

    def _show_chunking_statistics(self):
        """Show statistics about chunking."""
        stats = self.api.get_chunking_statistics()

        table = Table(title="Chunking Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        for key, value in stats.items():
            # Format values for better display
            if isinstance(value, (int, float)):
                if key == "avg_chunks_per_memory":
                    formatted = f"{value:.2f}"
                else:
                    formatted = str(value)
            else:
                formatted = str(value)

            # Format key for better display
            display_key = key.replace("_", " ").title()

            table.add_row(display_key, formatted)

        console.print(table)

    def _display_chunks(self, chunks: list[dict[str, Any]], title: str):
        """Display chunks in a formatted table."""
        if not chunks:
            console.print("[yellow]No chunks to display[/yellow]")
            return

        table = Table(title=title)
        table.add_column("Index", style="cyan", width=5)
        table.add_column("Text", style="green")
        table.add_column("Metadata", style="magenta", width=20)

        for chunk in chunks:
            # Format metadata for display
            metadata = chunk.get("metadata", {})
            meta_str = "\n".join(
                f"{k}: {v}" for k, v in metadata.items() if k not in ("text", "chunk_text")
            )

            table.add_row(
                str(chunk.get("chunk_index", "?")),
                chunk.get("text", "")[:100] + "..."
                if len(chunk.get("text", "")) > 100
                else chunk.get("text", ""),
                meta_str[:100] + "..." if len(meta_str) > 100 else meta_str,
            )

        console.print(table)

    def _display_retrieval_results(self, memories: list[dict[str, Any]], query: str):
        """Display retrieval results from chunks."""
        table = Table(title=f"Top Retrieval Results for: {query[:40]}...")
        table.add_column("Rank", style="cyan", width=5)
        table.add_column("Content", style="green")
        table.add_column("Score", style="magenta", width=10)
        table.add_column("Memory Info", style="blue", width=20)

        for i, memory in enumerate(memories):
            # Extract memory info
            memory_id = memory.get("memory_id", "?")
            if memory_id == "?":
                memory_type = "unknown"
            else:
                # Find the memory type from our added memories
                memory_type = next(
                    (m["type"] for m in self.added_memories if m["id"] == memory_id), "unknown"
                )

            # Format chunking info
            chunk_info = ""
            if "chunk_index" in memory:
                chunk_info = f"Chunk {memory.get('chunk_index')}"
                if "has_sequential_chunks" in memory:
                    chunk_info += (
                        " (sequential)"
                        if memory.get("has_sequential_chunks")
                        else " (non-sequential)"
                    )

            # Create memory info string
            memory_info = f"ID: {memory_id}\nType: {memory_type}"
            if chunk_info:
                memory_info += f"\n{chunk_info}"

            table.add_row(
                str(i + 1),
                memory.get("content", "")[:100] + "..."
                if len(memory.get("content", "")) > 100
                else memory.get("content", ""),
                f"{memory.get('relevance_score', 0.0):.3f}",
                memory_info,
            )

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
    """Run a demonstration of MemoryWeave's chunking capabilities."""
    # Run the chunking demo
    demo = ChunkingDemo(model_name=model, debug=debug)
    demo.run_demo()


if __name__ == "__main__":
    main()
