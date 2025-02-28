import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

from memoryweave.core import ContextualMemory, MemoryEncoder

# Create a custom theme for consistent styling
custom_theme = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red bold",
        "category": "magenta",
        "score": "cyan",
        "tech": "blue",
        "travel": "green",
        "food": "yellow",
        "technology": "blue",
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


def test_memory_with_vigilance_strategy(
    strategy, tech_memories, travel_memories, food_memories, embedding_model
):
    """Test memory with a specific vigilance strategy."""

    console.print(Panel.fit(f"Testing {strategy.title()} Vigilance Strategy", style="title"))

    # Initialize memory with dynamic vigilance
    embedding_dim = 384  # For MiniLM-L6
    memory = ContextualMemory(
        embedding_dim=embedding_dim,
        use_art_clustering=True,
        vigilance_threshold=0.7,  # Starting vigilance
        learning_rate=0.25,  # Increased learning rate
        dynamic_vigilance=True,
        vigilance_strategy=strategy,
        min_vigilance=0.5,
        max_vigilance=0.9,
        target_categories=3,  # Target for category_based strategy
    )

    # Create encoder
    encoder = MemoryEncoder(embedding_model)

    # Add all memories
    all_memories = []
    all_embeddings = []

    # Track vigilance changes
    vigilance_values = []
    category_counts = []

    # Add memories and track statistics
    memories_list = [
        ("technology", tech_memories),
        ("travel", travel_memories),
        ("food", food_memories),
    ]

    memory_index = 0
    for concept, concept_memories in memories_list:
        console.print(f"\n[bold]Adding {concept} memories...[/bold]", style=concept)
        for text in concept_memories:
            embedding, metadata = encoder.encode_concept(concept=concept, description=text)
            idx = memory.add_memory(embedding, text, metadata)
            all_memories.append((idx, text, concept))
            all_embeddings.append(embedding)

            # Track vigilance and category count
            stats = memory.get_category_statistics()
            vigilance_values.append(memory.vigilance_threshold)
            category_counts.append(stats["num_categories"])

            memory_index += 1

            # Show progress every 5 memories
            if memory_index % 5 == 0 or memory_index == len(tech_memories) + len(
                travel_memories
            ) + len(food_memories):
                console.print(
                    f"  Memory #{memory_index}: Vigilance = {memory.vigilance_threshold:.3f}, Categories = {stats['num_categories']}",
                    style="vigilance",
                )

    # Print final category statistics
    console.print("\n[bold]Final Category Statistics:[/bold]", style="category")
    stats = memory.get_category_statistics()
    console.print(
        f"Number of categories formed: [bold]{stats['num_categories']}[/bold] (with vigilance = {memory.vigilance_threshold:.3f})",
        style="category",
    )

    # Create a table for category statistics
    table = Table(title="Memories per Category")
    table.add_column("Category", style="magenta")
    table.add_column("Count", style="cyan")
    table.add_column("Activation", style="green")

    # Sort by category size (descending)
    sorted_categories = sorted(
        stats["memories_per_category"].items(), key=lambda x: x[1], reverse=True
    )

    for cat_idx, count in sorted_categories:
        activation = stats["category_activations"].get(cat_idx, 0)
        table.add_row(str(cat_idx), str(count), f"{activation:.2f}")

    console.print(table)

    # Analyze category coherence if we have multiple categories
    if stats["num_categories"] > 1:
        console.print("\n[bold]Analyzing category coherence...[/bold]")

        # Calculate intra-category similarity
        intra_category_similarities = []
        for cat_idx in stats["memories_per_category"].keys():
            # Get all memories in this category
            cat_memory_indices = np.where(memory.memory_categories == cat_idx)[0]
            if len(cat_memory_indices) > 1:  # Need at least 2 memories for similarity
                cat_embeddings = memory.memory_embeddings[cat_memory_indices]
                # Calculate pairwise similarities
                similarities = np.dot(cat_embeddings, cat_embeddings.T)
                # Get upper triangle (excluding diagonal)
                upper_tri = similarities[np.triu_indices(len(cat_embeddings), k=1)]
                if len(upper_tri) > 0:
                    avg_similarity = np.mean(upper_tri)
                    intra_category_similarities.append((cat_idx, avg_similarity))

        # Calculate average inter-category similarity
        inter_category_similarities = []
        categories = list(stats["memories_per_category"].keys())
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i + 1 :]:
                # Get memories from both categories
                cat1_indices = np.where(memory.memory_categories == cat1)[0]
                cat2_indices = np.where(memory.memory_categories == cat2)[0]

                if len(cat1_indices) > 0 and len(cat2_indices) > 0:
                    cat1_embeddings = memory.memory_embeddings[cat1_indices]
                    cat2_embeddings = memory.memory_embeddings[cat2_indices]

                    # Calculate cross-category similarities
                    similarities = np.dot(cat1_embeddings, cat2_embeddings.T)
                    avg_similarity = np.mean(similarities)
                    inter_category_similarities.append(((cat1, cat2), avg_similarity))

        # Print results
        if intra_category_similarities:
            console.print("\nIntra-category similarities (higher is better):")
            for cat_idx, similarity in intra_category_similarities:
                console.print(f"  Category {cat_idx}: {similarity:.3f}")

            avg_intra = np.mean([s for _, s in intra_category_similarities])
            console.print(f"Average intra-category similarity: {avg_intra:.3f}")

        if inter_category_similarities:
            # Calculate average across all pairs
            avg_inter = np.mean([s for _, s in inter_category_similarities])
            console.print(f"Average inter-category similarity: {avg_inter:.3f}")

            if intra_category_similarities:
                separation = avg_intra - avg_inter
                console.print(f"\nCategory separation (higher is better): {separation:.3f}")

    # Visualize vigilance changes and category growth
    plt.figure(figsize=(12, 5))
    memory_indices = list(range(1, len(vigilance_values) + 1))

    plt.subplot(1, 2, 1)
    plt.plot(memory_indices, vigilance_values, "b-", linewidth=2)
    plt.xlabel("Memory Count")
    plt.ylabel("Vigilance Threshold")
    plt.title(f"Vigilance Changes ({strategy.title()})")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(memory_indices, category_counts, "r-", linewidth=2)
    plt.xlabel("Memory Count")
    plt.ylabel("Number of Categories")
    plt.title(f"Category Growth ({strategy.title()})")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    filename = f"vigilance_{strategy}.png"
    plt.savefig(filename)
    console.print(f"Visualization saved as '{filename}'")

    # Test retrieval with a query
    console.print("\n[bold]Testing retrieval with a query:[/bold]")
    query = "Tell me about programming languages"
    console.print(f"Query: {query}")

    # Get query embedding
    query_embedding = embedding_model.encode(query)

    # Retrieve with category-based approach
    category_results = memory.retrieve_memories(query_embedding, top_k=3, use_categories=True)

    if category_results:
        for idx, score, metadata in category_results:
            original_idx, text, category = [(i, t, c) for i, t, c in all_memories if i == idx][0]
            style = "tech" if category == "technology" else category
            console.print(f"  [{category}] {text[:50]}... (Score: {score:.3f})", style=style)
    else:
        console.print("  No results found", style="warning")

    # Visualize memory clusters
    try:
        console.print("\n[bold]Visualizing memory clusters...[/bold]")

        # Convert embeddings to 2D using PCA
        all_embeddings_array = np.array(all_embeddings)
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(all_embeddings_array)

        # Create a scatter plot
        plt.figure(figsize=(10, 8))

        # Define colors for each category
        category_colors = {"technology": "blue", "travel": "green", "food": "orange"}

        # Plot each memory point
        for i, (idx, text, category) in enumerate(all_memories):
            cat_idx = memory.memory_categories[idx]
            plt.scatter(
                embeddings_2d[i, 0],
                embeddings_2d[i, 1],
                color=category_colors.get(category, "gray"),
                s=100,
                alpha=0.7,
            )
            plt.text(
                embeddings_2d[i, 0] + 0.02, embeddings_2d[i, 1] + 0.02, f"C{cat_idx}", fontsize=9
            )

        # Add category centers
        if memory.use_art_clustering and len(memory.category_prototypes) > 0:
            # Project category prototypes to 2D
            category_centers_2d = pca.transform(memory.category_prototypes)

            # Plot category centers
            for cat_idx, center in enumerate(category_centers_2d):
                plt.scatter(
                    center[0], center[1], color="red", marker="*", s=200, edgecolors="black"
                )
                plt.text(
                    center[0] + 0.02, center[1] + 0.02, f"Cat {cat_idx}", fontsize=12, weight="bold"
                )

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=10,
                label="Technology",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markersize=10,
                label="Travel",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="orange",
                markersize=10,
                label="Food",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="red",
                markersize=15,
                label="Category Center",
            ),
        ]
        plt.legend(handles=legend_elements, loc="upper right")

        # Add title and labels
        plt.title(f"Memory Clusters with {strategy.title()} Vigilance")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, linestyle="--", alpha=0.7)

        # Save the plot
        plt.tight_layout()
        filename = f"clusters_{strategy}.png"
        plt.savefig(filename)
        console.print(f"Visualization saved as '{filename}'")
    except Exception as e:
        console.print(f"[bold red]Error creating visualization: {e}[/bold red]")

    return memory, stats


def main():
    console.print(
        Panel.fit(
            "Testing Dynamic Vigilance Strategies for ART-Inspired Memory Clustering", style="title"
        )
    )

    # Load embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    console.print(f"Loading embedding model: [bold]{model_name}[/bold]", style="info")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embedding_model = EmbeddingModelWrapper(model, tokenizer)

    # Test data
    tech_memories = [
        "Python is a popular programming language for AI development.",
        "TensorFlow and PyTorch are leading deep learning frameworks.",
        "GPT models use transformer architectures for natural language processing.",
        "Reinforcement learning involves agents learning through trial and error.",
        "Neural networks are composed of layers of interconnected nodes.",
    ]

    travel_memories = [
        "Paris is known as the City of Light and is famous for the Eiffel Tower.",
        "Japan's cherry blossom season typically occurs in spring.",
        "The Great Barrier Reef in Australia is the world's largest coral reef system.",
        "Venice is famous for its canals and gondola rides.",
        "Machu Picchu is an ancient Incan citadel located in Peru.",
    ]

    food_memories = [
        "Sushi is a Japanese dish featuring vinegared rice and various ingredients.",
        "Italian cuisine is known for pasta, pizza, and gelato.",
        "Thai food often combines sweet, sour, salty, and spicy flavors.",
        "French pastries include croissants, macarons, and éclairs.",
        "Indian curry dishes vary widely by region and spice combinations.",
    ]

    # Test each vigilance strategy
    strategies = ["decreasing", "increasing", "category_based", "density_based"]

    results = {}
    for strategy in strategies:
        console.print("\n" + "=" * 80, style="section")
        memory, stats = test_memory_with_vigilance_strategy(
            strategy, tech_memories, travel_memories, food_memories, embedding_model
        )
        results[strategy] = {
            "categories": stats["num_categories"],
            "final_vigilance": memory.vigilance_threshold,
        }

    # Compare results
    console.print("\n" + "=" * 80, style="section")
    console.print(Panel.fit("Comparison of Dynamic Vigilance Strategies", style="title"))

    table = Table(title="Strategy Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Categories", style="magenta")
    table.add_column("Final Vigilance", style="green")

    for strategy, stats in results.items():
        table.add_row(strategy.title(), str(stats["categories"]), f"{stats['final_vigilance']:.3f}")

    console.print(table)
    console.print("\n[bold]Test completed successfully![/bold]", style="success")


if __name__ == "__main__":
    main()
