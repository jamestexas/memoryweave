#!/usr/bin/env python3
"""
visualize_memory_fabric.py

Visualizes how MemoryWeave's contextual fabric differs from traditional approaches
by showing memory activation, associative links, and temporal episodes.

This creates interactive visualizations that demonstrate the unique aspects of
MemoryWeave's memory retrieval capabilities.

Usage:
  uv run python examples/llm/visualize_memory_fabric.py --mode static
  uv run python examples/llm/visualize_memory_fabric.py --mode comparison --query "What are my hobbies?"
  uv run python examples/llm/visualize_memory_fabric.py --help
"""

import logging
import os
import random
import time
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rich_click as click
from matplotlib.figure import Figure
from rich.console import Console
from rich.logging import RichHandler

# Import the MemoryWeave API
from memoryweave.api import MemoryWeaveAPI

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True)],
)
logger = logging.getLogger("memoryweave")

if findspec("matplotlib") is not None:
    import matplotlib

    matplotlib.use("TkAgg")  # Use TkAgg backend for interactive plotting
else:
    logger.warning("[bold yellow]Warning: matplotlib not found, plotting disabled[/bold yellow]")


console = Console()

# Test memories with different categories and relationships
SAMPLE_MEMORIES = [
    # Personal facts
    "My name is Alex Thompson.",
    "I was born on March 15, 1988.",
    "I grew up in Portland, Oregon.",
    "I have a sister named Emma.",
    "My favorite color is blue.",
    # Work related
    "I work as a software engineer.",
    "I've been at my company for 5 years.",
    "I specialize in Python and React development.",
    "I'm working on a project called DataFlow.",
    "Our team has 7 members.",
    # Hobbies
    "I enjoy hiking on weekends.",
    "I've been learning to play guitar for 6 months.",
    "I like to cook Italian food.",
    "I read science fiction books regularly.",
    "I've recently started practicing meditation.",
    # Recent events
    "I went to a concert last weekend.",
    "I had lunch at Chez Michel yesterday.",
    "I finished reading 'Dune' last night.",
    "I have a dentist appointment next Thursday.",
    "I'm planning a trip to Japan in April.",
]


class MemoryFabricVisualizer:
    """Visualize MemoryWeave's contextual fabric approach."""

    def __init__(
        self,
        model_name: str = "unsloth/Llama-3.2-3B-Instruct",
        output_dir: str = "./memory_viz",
        debug: bool = False,
    ):
        """
        Initialize the visualizer with a MemoryWeave instance.

        Args:
            model_name: Name of the model to use
            output_dir: Directory to save visualizations
            debug: Enable debug output
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.debug = debug

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize the API
        self.api = MemoryWeaveAPI(model_name=model_name, debug=debug)

        # Common attributes for visualization
        self.fig: Optional[Figure] = None
        self.graph = nx.Graph()
        self.memory_map = {}  # Maps memory IDs to text
        self.memory_embeddings = {}  # Maps memory IDs to embeddings
        self.memory_categories = {}  # Maps memory IDs to categories
        self.memory_timestamps = {}  # Maps memory IDs to creation timestamps
        self.memory_activations = {}  # Simulated activations for visualization

        # For animation
        self.anim = None
        self.activated_nodes = set()

        # Initialize with sample memories
        self._add_initial_memories()

        console.print(f"Visualizer initialized with model: {model_name}", style="bold green")
        console.print(f"Visualizations will be saved to: {output_dir}", style="bold green")

    def _add_initial_memories(self):
        """Add sample memories with timestamps spread across time."""
        now = time.time()
        day_seconds = 86400  # Seconds in a day

        # Create timestamps spread over the last 30 days
        timestamps = [
            now - random.randint(0, 30) * day_seconds for _ in range(len(SAMPLE_MEMORIES))
        ]

        console.print("Adding sample memories...", style="bold cyan")

        # Add memories with their timestamps
        for i, (memory_text, timestamp) in enumerate(zip(SAMPLE_MEMORIES, timestamps)):
            # Group into categories (artificial for visualization)
            if i < 5:
                category = "personal"
            elif i < 10:
                category = "work"
            elif i < 15:
                category = "hobbies"
            else:
                category = "events"

            metadata = {
                "type": "fact",
                "category": category,
                "created_at": timestamp,
                "importance": random.uniform(0.5, 1.0),
            }

            # Add memory to API
            memory_id = self.api.add_memory(memory_text, metadata)
            if self.debug:
                console.print(f"Added memory {memory_id}: {memory_text[:30]}...", style="dim")

            # Store for visualization
            self.memory_map[memory_id] = memory_text

            # Store metadata for visualization
            self.memory_categories[memory_id] = category
            self.memory_timestamps[memory_id] = timestamp

            # For visualization, store embedding
            embedding = self.api.embedding_model.encode(memory_text, show_progress_bar=False)
            self.memory_embeddings[memory_id] = embedding

            # Assign a random initial activation value
            self.memory_activations[memory_id] = random.uniform(0.1, 0.3)

        # Create artificial associative links based on text similarity
        self._build_artificial_links()

        console.print(f"Added {len(SAMPLE_MEMORIES)} initial memories", style="bold green")

    def _build_artificial_links(self):
        """
        Build artificial associative links between memories based on embedding similarity.
        This simulates the associative linking that would happen in the full MemoryWeave system.
        """
        self.associative_links = {}

        # For each memory, find its closest neighbors
        for id1, emb1 in self.memory_embeddings.items():
            similarities = []
            for id2, emb2 in self.memory_embeddings.items():
                if id1 != id2:
                    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarities.append((id2, sim))

            # Sort by similarity and keep top 3
            similarities.sort(key=lambda x: x[1], reverse=True)
            self.associative_links[id1] = similarities[:3]

        # Create artificial temporal episodes by grouping memories by time period
        self.temporal_episodes = {}
        episode_id = 0

        # Group by rough time periods (each 3-day period is one episode)
        grouped_by_time = {}
        for memory_id, timestamp in self.memory_timestamps.items():
            time_period = int(timestamp / (3 * 86400))  # 3-day periods
            if time_period not in grouped_by_time:
                grouped_by_time[time_period] = []
            grouped_by_time[time_period].append(memory_id)

        # Create episodes
        for time_period, memory_ids in grouped_by_time.items():
            if len(memory_ids) >= 2:  # Only create episodes with at least 2 memories
                self.temporal_episodes[episode_id] = memory_ids
                episode_id += 1

        if self.debug:
            console.print(f"Created {len(self.associative_links)} associative links", style="dim")
            console.print(f"Created {len(self.temporal_episodes)} temporal episodes", style="dim")

    def _build_memory_graph(self):
        """Build a graph representation of the memory structure."""
        self.graph.clear()

        # Add memory nodes
        for memory_id, text in self.memory_map.items():
            # Truncate text for display
            short_text = text[:30] + "..." if len(text) > 30 else text
            category = self.memory_categories.get(memory_id, "unknown")
            timestamp = self.memory_timestamps.get(memory_id, 0)

            # Format timestamp for display
            date_str = datetime.fromtimestamp(timestamp).strftime("%m/%d/%Y")

            # Get activation level
            activation = self.memory_activations.get(memory_id, 0.1)

            # Add node with attributes
            self.graph.add_node(
                memory_id,
                text=short_text,
                category=category,
                timestamp=timestamp,
                date_str=date_str,
                activation=activation,
            )

        # Add associative links
        for memory_id, links in self.associative_links.items():
            for linked_id, strength in links:
                if linked_id in self.memory_map:  # Ensure the target exists
                    self.graph.add_edge(memory_id, linked_id, weight=strength, type="associative")

        # Add temporal episode links (connect memories in the same episode)
        for _episode_id, memory_ids in self.temporal_episodes.items():
            for i in range(len(memory_ids)):
                for j in range(i + 1, len(memory_ids)):
                    self.graph.add_edge(memory_ids[i], memory_ids[j], weight=0.3, type="temporal")

    def visualize_static(self):
        """Create a static visualization of the memory fabric."""
        console.print("Building memory graph...", style="bold cyan")
        self._build_memory_graph()

        console.print("Generating static visualization...", style="bold cyan")
        plt.figure(figsize=(14, 10))

        # Define node colors by category
        category_colors = {
            "personal": "royalblue",
            "work": "forestgreen",
            "hobbies": "darkorange",
            "events": "firebrick",
            "unknown": "gray",
        }

        # Get node colors, sizes, and positions
        node_colors = [category_colors[self.graph.nodes[n]["category"]] for n in self.graph]

        # Node size based on activation (scaled)
        node_sizes = [300 + 500 * self.graph.nodes[n].get("activation", 0.1) for n in self.graph]

        # Create layout based on temporal relationship
        # Newer items on right, older on left
        pos = {}
        for node in self.graph.nodes:
            timestamp = self.graph.nodes[node].get("timestamp", 0)
            category = self.graph.nodes[node].get("category", "unknown")

            # Normalize timestamp to 0-1 range
            min_time = min(n.get("timestamp", 0) for _, n in self.graph.nodes(data=True))
            max_time = max(n.get("timestamp", 0) for _, n in self.graph.nodes(data=True))
            time_range = max_time - min_time if max_time > min_time else 1
            x_pos = (timestamp - min_time) / time_range

            # Y position based on category
            category_map = {"personal": 0.2, "work": 0.4, "hobbies": 0.6, "events": 0.8}
            y_pos = category_map.get(category, 0.5) + random.uniform(-0.05, 0.05)

            pos[node] = np.array([x_pos, y_pos])

        # Draw the graph
        nx.draw_networkx_nodes(
            self.graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8
        )

        # Draw associative edges in blue
        associative_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True) if d.get("type") == "associative"
        ]
        nx.draw_networkx_edges(
            self.graph, pos, edgelist=associative_edges, edge_color="blue", alpha=0.6, width=1.5
        )

        # Draw temporal edges in red
        temporal_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True) if d.get("type") == "temporal"
        ]
        nx.draw_networkx_edges(
            self.graph, pos, edgelist=temporal_edges, edge_color="red", alpha=0.4, width=1.0
        )

        # Add labels
        labels = {n: self.graph.nodes[n]["text"] for n in self.graph}
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8, font_color="black")

        # Add legend
        plt.legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="royalblue",
                    markersize=10,
                    label="Personal",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="forestgreen",
                    markersize=10,
                    label="Work",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="darkorange",
                    markersize=10,
                    label="Hobbies",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="firebrick",
                    markersize=10,
                    label="Events",
                ),
                plt.Line2D([0], [0], color="blue", lw=2, label="Associative Link"),
                plt.Line2D([0], [0], color="red", lw=2, label="Temporal Link"),
            ]
        )

        plt.title("MemoryWeave Contextual Fabric Visualization")
        plt.tight_layout()

        # Save the visualization
        output_path = os.path.join(self.output_dir, "memory_fabric_static.png")
        plt.savefig(output_path, dpi=300)
        console.print(f"Saved static visualization to {output_path}", style="bold green")

        # Show the plot
        console.print(
            "Displaying visualization (close the plot window to continue)...", style="bold cyan"
        )
        plt.show()

    def visualize_comparison(self, query: str):
        """
        Create a visualization comparing MemoryWeave's contextual fabric approach
        to traditional vector-only retrieval.
        """
        console.print(f"Processing query: '{query}'", style="bold cyan")

        # Build the graph
        self._build_memory_graph()

        # Get query embedding
        query_embedding = self.api.embedding_model.encode(query, show_progress_bar=False)

        # Retrieve using MemoryWeave API
        memory_results = self.api.retrieve(query, top_k=5)

        # Simulate activations - in a real system, we'd get these from the API
        # Here we're just increasing activation for retrieved memories
        initial_activations = self.memory_activations.copy()

        # Extract memory IDs that would be retrieved by MemoryWeave's contextual approach
        # In a real system, this would come from the API
        # Here we're simulating by using the results plus some associative/temporal neighbors
        memoryweave_ids = []

        # First add direct matches
        for result in memory_results:
            result_id = result.get("id", "")
            if result_id in self.memory_map:
                memoryweave_ids.append(result_id)
                # Update activation
                self.memory_activations[result_id] = min(
                    1.0, self.memory_activations[result_id] + 0.3
                )

        # Then add some associative neighbors to simulate the contextual fabric
        for mem_id in memoryweave_ids.copy():
            if mem_id in self.associative_links:
                for neighbor_id, _ in self.associative_links[mem_id][:1]:  # Add one neighbor
                    if neighbor_id not in memoryweave_ids:
                        memoryweave_ids.append(neighbor_id)
                        # Update activation (less than direct matches)
                        self.memory_activations[neighbor_id] = min(
                            1.0, self.memory_activations[neighbor_id] + 0.15
                        )

        # Compute simple vector similarity for comparison (traditional approach)
        similarities = []
        for node in self.graph.nodes():
            if node in self.memory_embeddings:
                embedding = self.memory_embeddings[node]
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((node, similarity))

        # Get top 5 similarities for vector approach
        similarities.sort(key=lambda x: x[1], reverse=True)
        vector_ids = [node for node, _ in similarities[:5]]

        console.print("Creating comparison visualization...", style="bold cyan")

        # Create two side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Shared layout
        pos = {}
        for node in self.graph.nodes:
            timestamp = self.graph.nodes[node].get("timestamp", 0)
            category = self.graph.nodes[node].get("category", "unknown")

            # Normalize timestamp to 0-1 range
            min_time = min(n.get("timestamp", 0) for _, n in self.graph.nodes(data=True))
            max_time = max(n.get("timestamp", 0) for _, n in self.graph.nodes(data=True))
            time_range = max_time - min_time if max_time > min_time else 1
            x_pos = (timestamp - min_time) / time_range

            # Y position based on category
            category_map = {"personal": 0.2, "work": 0.4, "hobbies": 0.6, "events": 0.8}
            y_pos = category_map.get(category, 0.5) + random.uniform(-0.05, 0.05)

            pos[node] = np.array([x_pos, y_pos])

        # Define node colors by category
        category_colors = {
            "personal": "royalblue",
            "work": "forestgreen",
            "hobbies": "darkorange",
            "events": "firebrick",
            "unknown": "gray",
        }

        # Plot MemoryWeave results
        node_colors_mw = []
        node_sizes_mw = []

        for n in self.graph:
            if n in memoryweave_ids:
                node_colors_mw.append("yellow")
                node_sizes_mw.append(600)
            else:
                node_colors_mw.append(category_colors[self.graph.nodes[n]["category"]])
                node_sizes_mw.append(300)

        nx.draw_networkx_nodes(
            self.graph, pos, node_color=node_colors_mw, node_size=node_sizes_mw, alpha=0.8, ax=ax1
        )

        # Draw associative edges in blue
        associative_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True) if d.get("type") == "associative"
        ]
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=associative_edges,
            edge_color="blue",
            alpha=0.6,
            width=1.5,
            ax=ax1,
        )

        # Draw temporal edges in red
        temporal_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True) if d.get("type") == "temporal"
        ]
        nx.draw_networkx_edges(
            self.graph, pos, edgelist=temporal_edges, edge_color="red", alpha=0.4, width=1.0, ax=ax1
        )

        # Add labels
        labels = {n: self.graph.nodes[n]["text"] for n in self.graph}
        nx.draw_networkx_labels(
            self.graph, pos, labels=labels, font_size=8, font_color="black", ax=ax1
        )

        ax1.set_title(f"MemoryWeave Contextual Fabric Results\nQuery: '{query}'")
        ax1.axis("off")

        # Plot Vector Similarity results
        node_colors_vs = []
        node_sizes_vs = []

        for n in self.graph:
            if n in vector_ids:
                node_colors_vs.append("yellow")
                node_sizes_vs.append(600)
            else:
                node_colors_vs.append(category_colors[self.graph.nodes[n]["category"]])
                node_sizes_vs.append(300)

        nx.draw_networkx_nodes(
            self.graph, pos, node_color=node_colors_vs, node_size=node_sizes_vs, alpha=0.8, ax=ax2
        )

        # Only draw structural edges for clarity
        nx.draw_networkx_labels(
            self.graph, pos, labels=labels, font_size=8, font_color="black", ax=ax2
        )

        ax2.set_title(f"Traditional Vector Similarity Results\nQuery: '{query}'")
        ax2.axis("off")

        # Add legend to figure
        fig.legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="royalblue",
                    markersize=10,
                    label="Personal",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="forestgreen",
                    markersize=10,
                    label="Work",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="darkorange",
                    markersize=10,
                    label="Hobbies",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="firebrick",
                    markersize=10,
                    label="Events",
                ),
                plt.Line2D([0], [0], color="blue", lw=2, label="Associative Link"),
                plt.Line2D([0], [0], color="red", lw=2, label="Temporal Link"),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="yellow",
                    markersize=10,
                    label="Retrieved",
                ),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=7,
        )

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "memory_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        console.print(f"Saved comparison to {output_path}", style="bold green")

        # Reset activations
        self.memory_activations = initial_activations

        # Show the visualization
        console.print(
            "Displaying visualization (close the plot window to continue)...", style="bold cyan"
        )
        plt.show()


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["static", "comparison"]),
    default="comparison",
    help="Visualization mode to use",
)
@click.option(
    "--query",
    type=str,
    default="What are my hobbies?",
    help="Query to use for comparison mode",
)
@click.option(
    "--model",
    type=str,
    default="unsloth/Llama-3.2-3B-Instruct",
    help="Hugging Face model name to use",
)
@click.option(
    "--output",
    type=str,
    default="./memory_viz",
    help="Directory to save visualizations",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug output",
)
def main(mode, query, model, output, debug):
    """Visualize MemoryWeave's contextual fabric approach to memory management."""
    console.print("MemoryWeave Memory Fabric Visualizer", style="bold cyan")
    console.print(f"Mode: {mode}", style="bold cyan")

    if mode == "comparison":
        console.print(f"Query: {query}", style="bold cyan")

    console.print(f"Model: {model}", style="bold cyan")
    console.print(f"Output directory: {output}", style="bold cyan")

    # Create visualizer
    visualizer = MemoryFabricVisualizer(model_name=model, output_dir=output, debug=debug)

    # Run visualization based on mode
    if mode == "static":
        visualizer.visualize_static()
    elif mode == "comparison":
        visualizer.visualize_comparison(query)


if __name__ == "__main__":
    main()
