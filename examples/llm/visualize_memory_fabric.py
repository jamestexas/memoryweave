#!/usr/bin/env python3
"""
visualize_memory_fabric.py

Visualizes how MemoryWeave's contextual fabric differs from traditional approaches
by showing memory activation, associative links, and temporal episodes.

This creates an interactive visualization that helps demonstrate the unique
aspects of MemoryWeave's memory retrieval capabilities.
"""

import argparse
import random
import time
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

# Import our wrapper class
from memoryweave_llm_wrapper import MemoryWeaveLLM
from rich.console import Console

try:
    import matplotlib

    matplotlib.use("TkAgg")  # Use TkAgg backend for interactive plotting
except ImportError:
    pass  # Fall back to default backend if TkAgg is not available

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
        llm: Optional[MemoryWeaveLLM] = None,
        model_name: str = "unsloth/Llama-3.2-3B-Instruct",
    ):
        """Initialize the visualizer with a MemoryWeave LLM instance."""
        self.llm = llm or MemoryWeaveLLM(model_name=model_name)
        self.fig: Optional[Figure] = None
        self.graph = nx.Graph()
        self.memory_map = {}  # Maps memory IDs to text
        self.memory_embeddings = {}  # Maps memory IDs to embeddings
        self.memory_categories = {}  # Maps memory IDs to categories
        self.memory_timestamps = {}  # Maps memory IDs to creation timestamps

        # For animation
        self.anim = None
        self.activated_nodes = set()

        # Initialize with some memories
        self._add_initial_memories()

    def _add_initial_memories(self):
        """Add sample memories with timestamps spread across time."""
        now = time.time()
        day_seconds = 86400  # Seconds in a day

        # Create timestamps spread over the last 30 days
        timestamps = [
            now - random.randint(0, 30) * day_seconds for _ in range(len(SAMPLE_MEMORIES))
        ]

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

            memory_id = self.llm.add_memory(
                memory_text,
                {
                    "type": "fact",
                    "category": category,
                    "created_at": timestamp,
                    "importance": random.uniform(0.5, 1.0),
                },
            )

            # Store for visualization
            self.memory_map[memory_id] = memory_text
            # Create embedding if not already created by MemoryWeave
            if not hasattr(self.llm.memory_store_adapter, "memory_embeddings"):
                embedding = self.llm.embedding_model.encode(memory_text, show_progress_bar=False)
                self.memory_embeddings[memory_id] = embedding
            self.memory_categories[memory_id] = category
            self.memory_timestamps[memory_id] = timestamp

        console.print(f"[green]Added {len(SAMPLE_MEMORIES)} initial memories[/green]")

        # Initialize associative links
        if hasattr(self.llm, "associative_linker") and self.llm.associative_linker:
            self.llm.associative_linker._rebuild_all_links()
            console.print("[green]Built associative links[/green]")

        # Initialize temporal episodes
        if hasattr(self.llm, "temporal_context") and self.llm.temporal_context:
            self.llm.temporal_context._build_episodes()
            console.print("[green]Built temporal episodes[/green]")

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

            # Add node with attributes
            self.graph.add_node(
                memory_id,
                text=short_text,
                category=category,
                timestamp=timestamp,
                date_str=date_str,
                activation=self.llm.activation_manager.get_activation(memory_id)
                if hasattr(self.llm, "activation_manager")
                else 0.1,
            )

        # Add associative links
        if hasattr(self.llm, "associative_linker") and self.llm.associative_linker:
            for memory_id in self.memory_map:
                links = self.llm.associative_linker.get_associative_links(memory_id)
                for linked_id, strength in links:
                    if linked_id in self.memory_map:  # Ensure the target exists
                        self.graph.add_edge(
                            memory_id, linked_id, weight=strength, type="associative"
                        )

        # Add temporal episode links (connect memories in the same episode)
        if (
            hasattr(self.llm, "temporal_context")
            and self.llm.temporal_context
            and hasattr(self.llm.temporal_context, "memory_to_episode")
        ):
            # Group memories by episode
            episode_memories = {}
            for memory_id, episode_id in self.llm.temporal_context.memory_to_episode.items():
                if memory_id in self.memory_map:  # Ensure the memory exists
                    if episode_id not in episode_memories:
                        episode_memories[episode_id] = []
                    episode_memories[episode_id].append(memory_id)

            # Connect memories in the same episode
            for episode_id, memories in episode_memories.items():
                for i in range(len(memories)):
                    for j in range(i + 1, len(memories)):
                        self.graph.add_edge(memories[i], memories[j], weight=0.3, type="temporal")

    def visualize_static(self):
        """Create a static visualization of the memory fabric."""
        self._build_memory_graph()

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
        plt.savefig("memory_fabric_static.png", dpi=300)
        plt.show()

    def visualize_interactive(self, query: str):
        """
        Create an interactive visualization showing how MemoryWeave activates
        and traverses the memory fabric when processing a query.
        """
        self._build_memory_graph()

        # Execute the query to capture activation patterns
        console.print(f"[cyan]Processing query: {query}[/cyan]")

        # Get query embedding
        query_embedding = self.llm.embedding_model.encode(query, show_progress_bar=False)

        # Get initial activations
        initial_activations = {
            n: self.llm.activation_manager.get_activation(n) for n in self.graph.nodes()
        }

        # Retrieve using MemoryWeave to trigger activation updates
        memory_results = self.llm.strategy.retrieve(
            query_embedding=query_embedding,
            top_k=5,
            context={"query": query, "current_time": time.time()},
        )

        retrieved_ids = [
            str(r.get("memory_id"))
            for r in memory_results
            if "memory_id" in r and str(r["memory_id"]) in self.graph
        ]
        console.print(f"[green]Retrieved {len(retrieved_ids)} memories[/green]")

        # Get updated activations
        updated_activations = {
            n: self.llm.activation_manager.get_activation(n) for n in self.graph.nodes()
        }

        # Compute activation changes
        activation_changes = {
            n: updated_activations[n] - initial_activations[n] for n in self.graph.nodes()
        }

        # Create a graph layout
        category_order = {"personal": 0, "work": 1, "hobbies": 2, "events": 3, "unknown": 4}

        pos = {}
        for node in self.graph.nodes:
            timestamp = self.graph.nodes[node].get("timestamp", 0)
            category = self.graph.nodes[node].get("category", "unknown")

            # Normalize timestamp to 0-1 range for x position
            min_time = min(n.get("timestamp", 0) for _, n in self.graph.nodes(data=True))
            max_time = max(n.get("timestamp", 0) for _, n in self.graph.nodes(data=True))
            time_range = max_time - min_time if max_time > min_time else 1
            x_pos = (timestamp - min_time) / time_range

            # Y position based on category
            y_pos = 0.8 - (category_order.get(category, 4) * 0.2)
            y_pos += random.uniform(-0.05, 0.05)  # Add jitter

            pos[node] = np.array([x_pos, y_pos])

        # Create figure and first frame
        self.fig, ax = plt.subplots(figsize=(14, 10))

        # Define a function for animation updates
        frame_count = 10  # Number of animation frames
        all_frames_data = []

        # Compute frames data
        for frame in range(frame_count + 1):
            t = frame / frame_count  # Time parameter (0 to 1)

            # Interpolate activation levels
            current_activations = {
                n: initial_activations[n] + (t * activation_changes[n]) for n in self.graph.nodes()
            }

            # Track which nodes to highlight (progressively add retrieved nodes)
            highlight_nodes = set()
            for i, node_id in enumerate(retrieved_ids):
                if i <= int(t * len(retrieved_ids)):
                    highlight_nodes.add(node_id)

            all_frames_data.append((current_activations, highlight_nodes))

        def update(frame):
            ax.clear()

            current_activations, highlight_nodes = all_frames_data[frame]

            # Define node colors by category
            category_colors = {
                "personal": "royalblue",
                "work": "forestgreen",
                "hobbies": "darkorange",
                "events": "firebrick",
                "unknown": "gray",
            }

            # Node size based on activation (scaled)
            node_sizes = [300 + 1000 * current_activations[n] for n in self.graph]

            # Node colors with highlighted nodes in brighter colors
            node_colors = []
            for n in self.graph:
                base_color = category_colors[self.graph.nodes[n]["category"]]
                if n in highlight_nodes:
                    # Make the color brighter for highlighted nodes
                    node_colors.append("yellow")
                else:
                    node_colors.append(base_color)

            # Draw the nodes
            nx.draw_networkx_nodes(
                self.graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax
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
                ax=ax,
            )

            # Draw temporal edges in red
            temporal_edges = [
                (u, v) for u, v, d in self.graph.edges(data=True) if d.get("type") == "temporal"
            ]
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=temporal_edges,
                edge_color="red",
                alpha=0.4,
                width=1.0,
                ax=ax,
            )

            # Add labels
            labels = {n: self.graph.nodes[n]["text"] for n in self.graph}
            nx.draw_networkx_labels(
                self.graph, pos, labels=labels, font_size=8, font_color="black", ax=ax
            )

            # Add legend
            ax.legend(
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
                ]
            )

            ax.set_title(f"MemoryWeave Activation for Query: '{query}'")
            ax.axis("off")

            return ax

        # Create animation
        self.anim = FuncAnimation(
            self.fig, update, frames=len(all_frames_data), interval=300, repeat=True
        )

        plt.tight_layout()
        self.anim.save("memory_activation.gif", writer="pillow", fps=4)
        plt.show()

    def visualize_comparison(self, query: str):
        """
        Create a visualization comparing MemoryWeave's contextual fabric approach
        to traditional vector-only retrieval.
        """
        self._build_memory_graph()

        # Get query embedding
        query_embedding = self.llm.embedding_model.encode(query, show_progress_bar=False)

        # Retrieve using MemoryWeave
        memory_results = self.llm.strategy.retrieve(
            query_embedding=query_embedding,
            top_k=5,
            context={"query": query, "current_time": time.time()},
        )

        memoryweave_ids = [
            str(r.get("memory_id"))
            for r in memory_results
            if "memory_id" in r and str(r["memory_id"]) in self.graph
        ]

        # Compute simple vector similarity for comparison
        similarities = []
        for node in self.graph.nodes():
            if node in self.memory_embeddings:
                embedding = self.memory_embeddings[node]
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((node, similarity))

        # Get top 5 similarities
        similarities.sort(key=lambda x: x[1], reverse=True)
        vector_ids = [node for node, _ in similarities[:5]]

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
        plt.savefig("memory_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize MemoryWeave's contextual fabric")
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Llama-3.2-3B-Instruct",
        help="Hugging Face model name to use",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["static", "interactive", "comparison"],
        default="comparison",
        help="Visualization mode to use",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are my hobbies?",
        help="Query to use for interactive or comparison modes",
    )

    args = parser.parse_args()

    console.print("[bold cyan]MemoryWeave Fabric Visualizer[/bold cyan]")
    console.print(f"Loading model: {args.model}")

    visualizer = MemoryFabricVisualizer(model_name=args.model)

    if args.mode == "static":
        console.print("Creating static visualization...")
        visualizer.visualize_static()
    elif args.mode == "interactive":
        console.print(f"Creating interactive visualization for query: '{args.query}'")
        visualizer.visualize_interactive(args.query)
    elif args.mode == "comparison":
        console.print(f"Creating comparison visualization for query: '{args.query}'")
        visualizer.visualize_comparison(args.query)


if __name__ == "__main__":
    main()
