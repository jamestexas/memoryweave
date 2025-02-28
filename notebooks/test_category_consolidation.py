"""
Tests the category consolidation functionality of MemoryWeave.
"""

import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from transformers import AutoModel, AutoTokenizer

from memoryweave.core import ContextualMemory, MemoryEncoder

# Create a custom theme
custom_theme = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red bold",
        "category": "magenta",
        "title": "bold cyan",
        "section": "bold yellow",
        "vigilance": "bold magenta",
    }
)

# Initialize rich console with our theme
console = Console(theme=custom_theme)


# Helper class for sentence embedding
class EmbeddingModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling
        attention_mask = inputs["attention_mask"]
        embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts

        return mean_pooled.numpy()[0]


def generate_test_data():
    """Generate test data organized into clear topic clusters with some overlap."""

    # Programming language memories (12)
    programming_memories = [
        # Python (4)
        {
            "text": "Python is a high-level programming language known for readability and simplicity.",
            "subtopic": "python",
        },
        {
            "text": "Python's syntax allows programmers to express concepts in fewer lines of code.",
            "subtopic": "python",
        },
        {
            "text": "Python supports multiple programming paradigms, including procedural and object-oriented.",
            "subtopic": "python",
        },
        {
            "text": "Python has a large standard library and active community support.",
            "subtopic": "python",
        },
        # JavaScript (4)
        {
            "text": "JavaScript is the primary language for web development and browser interactions.",
            "subtopic": "javascript",
        },
        {
            "text": "JavaScript enables interactive web pages and is essential to web applications.",
            "subtopic": "javascript",
        },
        {
            "text": "JavaScript is a prototype-based language with first-class functions.",
            "subtopic": "javascript",
        },
        {
            "text": "JavaScript frameworks like React and Angular help build complex web applications.",
            "subtopic": "javascript",
        },
        # Rust (4)
        {"text": "Rust provides memory safety without garbage collection.", "subtopic": "rust"},
        {
            "text": "Rust's ownership system prevents memory bugs at compile time.",
            "subtopic": "rust",
        },
        {"text": "Rust combines high-level ergonomics with low-level control.", "subtopic": "rust"},
        {
            "text": "Rust is increasingly used for systems programming and performance-critical applications.",
            "subtopic": "rust",
        },
    ]

    # Machine Learning memories (12)
    ml_memories = [
        # Neural Networks (4)
        {
            "text": "Neural networks are composed of layers of interconnected nodes.",
            "subtopic": "neural_networks",
        },
        {
            "text": "Deep neural networks have transformed computer vision and natural language processing.",
            "subtopic": "neural_networks",
        },
        {
            "text": "Convolutional neural networks are specialized for processing grid-like data such as images.",
            "subtopic": "neural_networks",
        },
        {
            "text": "Recurrent neural networks are designed to recognize patterns in sequences of data.",
            "subtopic": "neural_networks",
        },
        # Transformers (4)
        {
            "text": "Transformer models use self-attention mechanisms to process sequential data.",
            "subtopic": "transformers",
        },
        {
            "text": "GPT models are based on the transformer architecture for natural language processing.",
            "subtopic": "transformers",
        },
        {
            "text": "BERT is a transformer-based model designed for understanding context in text.",
            "subtopic": "transformers",
        },
        {
            "text": "Transformer models have achieved state-of-the-art results in many NLP tasks.",
            "subtopic": "transformers",
        },
        # Reinforcement Learning (4)
        {
            "text": "Reinforcement learning involves agents learning through trial and error.",
            "subtopic": "reinforcement_learning",
        },
        {
            "text": "In reinforcement learning, agents aim to maximize cumulative rewards.",
            "subtopic": "reinforcement_learning",
        },
        {
            "text": "Q-learning is a value-based reinforcement learning algorithm.",
            "subtopic": "reinforcement_learning",
        },
        {
            "text": "Policy gradient methods directly optimize the policy in reinforcement learning.",
            "subtopic": "reinforcement_learning",
        },
    ]

    # Cuisine memories (12)
    cuisine_memories = [
        # Italian (4)
        {
            "text": "Italian cuisine is known for pasta, pizza, and a variety of regional dishes.",
            "subtopic": "italian",
        },
        {
            "text": "Italian cooking emphasizes high-quality, fresh ingredients.",
            "subtopic": "italian",
        },
        {
            "text": "Italian cuisine varies greatly by region, from seafood in coastal areas to meat inland.",
            "subtopic": "italian",
        },
        {
            "text": "Traditional Italian meals often include multiple courses, from antipasto to dolce.",
            "subtopic": "italian",
        },
        # Japanese (4)
        {
            "text": "Japanese cuisine includes sushi, sashimi, and a variety of noodle dishes.",
            "subtopic": "japanese",
        },
        {
            "text": "Japanese cooking emphasizes seasonal ingredients and beautiful presentation.",
            "subtopic": "japanese",
        },
        {
            "text": "Umami, the fifth taste, is an important concept in Japanese cuisine.",
            "subtopic": "japanese",
        },
        {
            "text": "Traditional Japanese meals follow the ichiju-sansai structure: one soup, three sides.",
            "subtopic": "japanese",
        },
        # Mexican (4)
        {
            "text": "Mexican cuisine features corn, beans, and chili peppers as staple ingredients.",
            "subtopic": "mexican",
        },
        {
            "text": "Mexican food is known for its vibrant flavors and diverse regional styles.",
            "subtopic": "mexican",
        },
        {
            "text": "Traditional Mexican cooking techniques include nixtamalization of corn.",
            "subtopic": "mexican",
        },
        {
            "text": "Mexican cuisine has influenced cooking styles worldwide, especially in the Americas.",
            "subtopic": "mexican",
        },
    ]

    # Cross-domain connections (6)
    cross_domain_memories = [
        # Python + ML
        {
            "text": "Python is the most popular language for machine learning and data science.",
            "subtopic": "python_ml",
        },
        {
            "text": "Libraries like TensorFlow and PyTorch implement neural networks in Python.",
            "subtopic": "python_ml",
        },
        # JavaScript + ML
        {
            "text": "TensorFlow.js allows running machine learning models in JavaScript.",
            "subtopic": "js_ml",
        },
        {
            "text": "JavaScript frameworks are increasingly incorporating ML capabilities for web applications.",
            "subtopic": "js_ml",
        },
        # Cooking + Technology
        {
            "text": "Machine learning is being used to analyze flavors and create new recipes.",
            "subtopic": "tech_cooking",
        },
        {
            "text": "Computational gastronomy uses data science to understand food pairings and cooking techniques.",
            "subtopic": "tech_cooking",
        },
    ]

    # Combine all memories with their categories
    all_memories = (
        [(mem["text"], "programming", mem["subtopic"]) for mem in programming_memories]
        + [(mem["text"], "machine_learning", mem["subtopic"]) for mem in ml_memories]
        + [(mem["text"], "cuisine", mem["subtopic"]) for mem in cuisine_memories]
        + [(mem["text"], "cross_domain", mem["subtopic"]) for mem in cross_domain_memories]
    )

    # Shuffle the memories
    random.shuffle(all_memories)

    return all_memories


def visualize_category_graph(memory, title="Category Similarity Graph"):
    """
    Visualize the category structure as a graph where nodes are categories
    and edges represent similarity above a threshold.
    """
    try:
        import networkx as nx

        # Get similarity matrix
        sim_matrix = memory.category_similarity_matrix()
        if len(sim_matrix) == 0:
            console.print("[warning]No categories to visualize[/warning]")
            return

        # Get category statistics
        stats = memory.get_category_statistics()
        category_counts = stats["memories_per_category"]

        # Create graph
        G = nx.Graph()

        # Add nodes (categories)
        for cat_idx in range(len(sim_matrix)):
            # Node size based on number of memories
            size = category_counts.get(cat_idx, 0)
            G.add_node(cat_idx, size=size)

        # Add edges (similarities above threshold)
        threshold = 0.5  # Similarity threshold for drawing edges
        for i in range(len(sim_matrix)):
            for j in range(i + 1, len(sim_matrix)):
                similarity = sim_matrix[i, j]
                if similarity > threshold:
                    G.add_edge(i, j, weight=similarity)

        # Create visualization
        plt.figure(figsize=(12, 10))

        # Node positions - using spring layout
        pos = nx.spring_layout(G, seed=42)

        # Node sizes based on category sizes
        node_sizes = [G.nodes[node]["size"] * 100 + 100 for node in G.nodes()]

        # Draw the graph
        nx.draw_networkx_nodes(
            G, pos, node_size=node_sizes, alpha=0.8, node_color="lightblue", edgecolors="black"
        )

        # Draw edges with width proportional to similarity
        for u, v, data in G.edges(data=True):
            width = data["weight"] * 3  # Scale width based on similarity
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.5)

        # Draw labels
        nx.draw_networkx_labels(G, pos)

        plt.title(title)
        plt.axis("off")

        # Save the visualization
        plt.tight_layout()
        plt.savefig(f"{title.lower().replace(' ', '_')}.png")
        console.print(
            f"[success]Category graph visualization saved as '{title.lower().replace(' ', '_')}.png'[/success]"
        )

    except ImportError:
        console.print("[warning]NetworkX not installed. Cannot visualize category graph.[/warning]")


def test_category_consolidation():
    console.print(Panel.fit("Testing Category Consolidation", style="title"))

    # Load embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    console.print(f"Loading embedding model: [bold]{model_name}[/bold]", style="info")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embedding_model = EmbeddingModelWrapper(model, tokenizer)

    # Generate test data
    all_memories = generate_test_data()
    console.print(f"Generated [bold]{len(all_memories)}[/bold] test memories", style="info")

    # Group memories by category
    category_counts = Counter([category for _, category, _ in all_memories])
    subtopic_counts = Counter([subtopic for _, _, subtopic in all_memories])

    # Print distribution
    console.print("\n[bold]Memory Distribution:[/bold]")
    for category, count in category_counts.items():
        console.print(f"  {category}: {count} memories")

    console.print("\n[bold]Subtopic Distribution:[/bold]")
    for subtopic, count in subtopic_counts.items():
        console.print(f"  {subtopic}: {count} memories")

    # Initialize memory with fragmentation-prone settings
    console.print(
        "\n[bold]Initializing memory with high vigilance but with consolidation enabled[/bold]",
        style="section",
    )
    memory_with_consolidation = ContextualMemory(
        embedding_dim=model.config.hidden_size,
        use_art_clustering=True,
        vigilance_threshold=0.8,  # High vigilance = many initial categories
        learning_rate=0.2,
        enable_category_consolidation=True,
        consolidation_threshold=0.6,  # Moderate threshold for merging
        min_category_size=2,
        consolidation_frequency=15,  # Consolidate every 15 memories
    )

    # Also initialize a baseline memory without consolidation for comparison
    memory_baseline = ContextualMemory(
        embedding_dim=model.config.hidden_size,
        use_art_clustering=True,
        vigilance_threshold=0.8,  # Same high vigilance
        learning_rate=0.2,
        enable_category_consolidation=False,  # No consolidation
    )

    # Create encoder
    encoder = MemoryEncoder(embedding_model)

    # Add all memories to both memory systems
    # Track category growth
    consolidation_categories = []
    baseline_categories = []

    # Also track which memories got consolidated to enable analysis
    memory_mappings_consolidation = []
    memory_mappings_baseline = []

    console.print(
        "\n[bold]Adding memories to both systems and tracking category formation...[/bold]",
        style="section",
    )

    for i, (text, category, subtopic) in enumerate(all_memories):
        # Create metadata
        metadata = {"category": category, "subtopic": subtopic}

        # Get embedding
        embedding, _ = encoder.encode_concept(
            concept=category, description=text, related_concepts=[subtopic]
        )

        # Add to memory with consolidation
        idx_consolidation = memory_with_consolidation.add_memory(embedding, text, metadata)
        cat_idx_consolidation = memory_with_consolidation.memory_categories[idx_consolidation]
        memory_mappings_consolidation.append(
            (idx_consolidation, cat_idx_consolidation, category, subtopic)
        )

        # Add to baseline memory
        idx_baseline = memory_baseline.add_memory(embedding, text, metadata)
        cat_idx_baseline = memory_baseline.memory_categories[idx_baseline]
        memory_mappings_baseline.append((idx_baseline, cat_idx_baseline, category, subtopic))

        # Track category counts
        consolidation_categories.append(len(memory_with_consolidation.category_prototypes))
        baseline_categories.append(len(memory_baseline.category_prototypes))

        # Show progress every 10 memories
        if (i + 1) % 10 == 0 or i == len(all_memories) - 1:
            console.print(
                f"Added {i + 1} memories - Consolidation: {consolidation_categories[-1]} categories, "
                + f"Baseline: {baseline_categories[-1]} categories"
            )

            # Show if consolidation happened
            if i > 0 and consolidation_categories[-1] < consolidation_categories[-2]:
                console.print(
                    f"  [success]Consolidation occurred![/success] Reduced from {consolidation_categories[-2]} to {consolidation_categories[-1]} categories"
                )

    # Get final statistics
    consolidation_stats = memory_with_consolidation.get_category_statistics()
    baseline_stats = memory_baseline.get_category_statistics()

    # Print comparison
    console.print("\n[bold]Final Results Comparison:[/bold]", style="section")
    comparison_table = Table(title="Category Consolidation Comparison")
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("With Consolidation", style="green")
    comparison_table.add_column("Baseline (No Consolidation)", style="yellow")

    comparison_table.add_row(
        "Final Categories",
        str(consolidation_stats["num_categories"]),
        str(baseline_stats["num_categories"]),
    )

    # Calculate average category size
    avg_size_consolidation = np.mean(list(consolidation_stats["memories_per_category"].values()))
    avg_size_baseline = np.mean(list(baseline_stats["memories_per_category"].values()))

    comparison_table.add_row(
        "Average Category Size", f"{avg_size_consolidation:.2f}", f"{avg_size_baseline:.2f}"
    )

    # Calculate standard deviation of category sizes
    std_size_consolidation = np.std(list(consolidation_stats["memories_per_category"].values()))
    std_size_baseline = np.std(list(baseline_stats["memories_per_category"].values()))

    comparison_table.add_row(
        "Category Size StdDev", f"{std_size_consolidation:.2f}", f"{std_size_baseline:.2f}"
    )

    # Find max category size
    max_size_consolidation = max(consolidation_stats["memories_per_category"].values())
    max_size_baseline = max(baseline_stats["memories_per_category"].values())

    comparison_table.add_row(
        "Largest Category Size", str(max_size_consolidation), str(max_size_baseline)
    )

    console.print(comparison_table)

    # Create a graph visualization of the category structure
    visualize_category_graph(memory_with_consolidation, "Category Structure With Consolidation")
    visualize_category_graph(memory_baseline, "Category Structure Without Consolidation")

    # Analyze category purity
    console.print("\n[bold]Analyzing Category Purity:[/bold]", style="section")
    console.print("(How well categories align with the original topics and subtopics)")

    # Create table for consolidated memory analysis
    table_consolidation = Table(title="Categories in Consolidated Memory")
    table_consolidation.add_column("Category ID", style="cyan")
    table_consolidation.add_column("Size", style="magenta")
    table_consolidation.add_column("Topic Distribution", style="green")
    table_consolidation.add_column("Subtopic Distribution", style="yellow")

    # Analyze each category in the consolidated memory
    for cat_idx, count in sorted(
        consolidation_stats["memories_per_category"].items(), key=lambda x: x[1], reverse=True
    ):
        # Find memories in this category
        cat_memories = [m for m in memory_mappings_consolidation if m[1] == cat_idx]

        # Count topics and subtopics
        topics = Counter([m[2] for m in cat_memories])
        subtopics = Counter([m[3] for m in cat_memories])

        # Format for display
        topics_str = ", ".join([f"{t}:{c}" for t, c in topics.most_common(3)])
        subtopics_str = ", ".join([f"{s}:{c}" for s, c in subtopics.most_common(3)])

        table_consolidation.add_row(str(cat_idx), str(count), topics_str, subtopics_str)

    console.print(table_consolidation)

    # Test manual consolidation with different thresholds
    console.print(
        "\n[bold]Testing Manual Consolidation with Different Thresholds:[/bold]", style="section"
    )

    # Create a new memory instance with many initial categories (high vigilance)
    memory_manual = ContextualMemory(
        embedding_dim=model.config.hidden_size,
        use_art_clustering=True,
        vigilance_threshold=0.85,  # Very high vigilance = many categories
        learning_rate=0.2,
        enable_category_consolidation=False,  # Will trigger manually
    )

    # Add all memories first
    for text, category, subtopic in all_memories:
        metadata = {"category": category, "subtopic": subtopic}
        embedding, _ = encoder.encode_concept(concept=category, description=text)
        memory_manual.add_memory(embedding, text, metadata)

    # Get initial category count
    initial_stats = memory_manual.get_category_statistics()
    initial_count = initial_stats["num_categories"]
    console.print(f"Initial categories before manual consolidation: {initial_count}")

    # Try different thresholds
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    results = []

    for threshold in thresholds:
        # Clone the memory to test each threshold independently
        memory_clone = memory_manual  # In a real implementation, you'd deep copy

        # Manually consolidate
        final_count = memory_clone.consolidate_categories_manually(threshold)

        # Get current stats
        stats = memory_clone.get_category_statistics()
        avg_size = np.mean(list(stats["memories_per_category"].values()))

        results.append((threshold, final_count, avg_size))

        console.print(
            f"Threshold {threshold:.1f}: {initial_count} → {final_count} categories (avg size: {avg_size:.2f})"
        )

    # Create a simple bar chart
    plt.figure(figsize=(10, 6))
    thresholds = [r[0] for r in results]
    category_counts = [r[1] for r in results]
    avg_sizes = [r[2] for r in results]

    x = np.arange(len(thresholds))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x - width / 2, category_counts, width, label="Category Count", color="blue", alpha=0.7)
    ax1.set_xlabel("Consolidation Threshold")
    ax1.set_ylabel("Number of Categories", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, avg_sizes, width, label="Avg Size", color="red", alpha=0.7)
    ax2.set_ylabel("Average Category Size", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Effect of Consolidation Threshold")
    plt.xticks(x, [f"{t:.1f}" for t in thresholds])

    # Add a line showing initial categories
    ax1.axhline(y=initial_count, color="gray", linestyle="--", alpha=0.7)
    ax1.text(len(thresholds) - 1, initial_count + 0.5, f"Initial ({initial_count})", color="gray")

    fig.tight_layout()
    plt.savefig("consolidation_thresholds.png")
    console.print(
        "[success]Threshold comparison chart saved as 'consolidation_thresholds.png'[/success]"
    )

    # Test retrieval quality
    console.print("\n[bold]Testing Retrieval Quality:[/bold]", style="section")

    test_queries = [
        "Tell me about Python programming",
        "How do neural networks work?",
        "I'm interested in Japanese cuisine",
        "What is the connection between Python and machine learning?",
        "Tell me about different programming languages",
    ]

    for query in test_queries:
        console.print(f"\n[bold]Query:[/bold] {query}")

        # Get query embedding
        query_embedding = embedding_model.encode(query)

        # Retrieve from consolidated memory
        console.print("[bold]Results from consolidated memory:[/bold]")
        results_consolidation = memory_with_consolidation.retrieve_memories(
            query_embedding, top_k=3, use_categories=True
        )

        if results_consolidation:
            for i, (idx, score, metadata) in enumerate(results_consolidation):
                category = metadata.get("category", "unknown")
                subtopic = metadata.get("subtopic", "unknown")
                console.print(
                    f"  {i + 1}. [{category}:{subtopic}] {metadata.get('text', '')[:50]}... (Score: {score:.3f})"
                )
        else:
            console.print("  No results found")

        # Retrieve from baseline memory
        console.print("[bold]Results from baseline memory:[/bold]")
        results_baseline = memory_baseline.retrieve_memories(
            query_embedding, top_k=3, use_categories=True
        )

        if results_baseline:
            for i, (idx, score, metadata) in enumerate(results_baseline):
                category = metadata.get("category", "unknown")
                subtopic = metadata.get("subtopic", "unknown")
                console.print(
                    f"  {i + 1}. [{category}:{subtopic}] {metadata.get('text', '')[:50]}... (Score: {score:.3f})"
                )
        else:
            console.print("  No results found")

    console.print("\n[bold]Test completed successfully![/bold]", style="success")
    return memory_with_consolidation, memory_baseline


if __name__ == "__main__":
    test_category_consolidation()
