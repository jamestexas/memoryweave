"""
Tests the confidence thresholding functionality of MemoryWeave.
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
        "threshold": "bold magenta",
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
    """Generate test data with varying degrees of relevance to test queries."""

    # Python programming memories
    python_memories = [
        {
            "text": "Python is a high-level programming language known for readability and simplicity.",
            "relevance": "high",
        },
        {
            "text": "Python supports multiple programming paradigms including procedural and object-oriented.",
            "relevance": "high",
        },
        {
            "text": "Python has a large standard library and active community support.",
            "relevance": "high",
        },
        {
            "text": "Python's syntax allows programmers to express concepts in fewer lines of code.",
            "relevance": "high",
        },
        {
            "text": "Python is commonly used for web development, data analysis, and artificial intelligence.",
            "relevance": "medium",
        },
        {
            "text": "Python was created by Guido van Rossum in the late 1980s.",
            "relevance": "medium",
        },
        {
            "text": "Python is named after the British comedy group Monty Python.",
            "relevance": "low",
        },
        {
            "text": "Python is one of many programming languages used in software development.",
            "relevance": "low",
        },
    ]

    # Machine learning memories
    ml_memories = [
        {
            "text": "Machine learning is a subset of artificial intelligence focused on data-based learning.",
            "relevance": "high",
        },
        {
            "text": "Neural networks are composed of layers of interconnected nodes.",
            "relevance": "high",
        },
        {
            "text": "Deep learning uses multiple layers of neural networks for complex pattern recognition.",
            "relevance": "high",
        },
        {
            "text": "Supervised learning requires labeled training data to make predictions.",
            "relevance": "high",
        },
        {
            "text": "Machine learning algorithms improve through experience without explicit programming.",
            "relevance": "medium",
        },
        {
            "text": "Reinforcement learning involves agents learning through trial and error.",
            "relevance": "medium",
        },
        {
            "text": "The term 'machine learning' was coined by Arthur Samuel in 1959.",
            "relevance": "low",
        },
        {
            "text": "Machine learning competitions are often hosted on platforms like Kaggle.",
            "relevance": "low",
        },
    ]

    # Travel memories (unrelated to programming/ML)
    travel_memories = [
        {
            "text": "Paris is known as the City of Light and is famous for the Eiffel Tower.",
            "relevance": "unrelated",
        },
        {
            "text": "Japan's cherry blossom season typically occurs in spring.",
            "relevance": "unrelated",
        },
        {
            "text": "The Great Barrier Reef in Australia is the world's largest coral reef system.",
            "relevance": "unrelated",
        },
        {"text": "Venice is famous for its canals and gondola rides.", "relevance": "unrelated"},
    ]

    # Combine all memories with their categories
    all_memories = (
        [(mem["text"], "python", mem["relevance"]) for mem in python_memories]
        + [(mem["text"], "machine_learning", mem["relevance"]) for mem in ml_memories]
        + [(mem["text"], "travel", mem["relevance"]) for mem in travel_memories]
    )

    # Shuffle the memories
    random.shuffle(all_memories)

    return all_memories


def test_confidence_thresholding():
    console.print(Panel.fit("Testing Confidence Thresholding", style="title"))

    # Load embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    console.print(f"Loading embedding model: [bold]{model_name}[/bold]", style="info")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embedding_model = EmbeddingModelWrapper(model, tokenizer)

    # Generate test data
    all_memories = generate_test_data()
    console.print(f"Generated [bold]{len(all_memories)}[/bold] test memories", style="info")

    # Group memories by category and relevance
    category_counts = Counter([category for _, category, _ in all_memories])
    relevance_counts = Counter([relevance for _, _, relevance in all_memories])

    # Print distribution
    console.print("\n[bold]Memory Distribution:[/bold]")
    for category, count in category_counts.items():
        console.print(f"  {category}: {count} memories")

    console.print("\n[bold]Relevance Distribution:[/bold]")
    for relevance, count in relevance_counts.items():
        console.print(f"  {relevance}: {count} memories")

    # Initialize memory
    console.print("\n[bold]Initializing memory[/bold]", style="section")
    memory = ContextualMemory(
        embedding_dim=model.config.hidden_size,
        default_confidence_threshold=0.0,  # Start with no threshold
        semantic_coherence_check=False,  # Will test separately
        adaptive_retrieval=False,  # Will test separately
    )

    # Create encoder
    encoder = MemoryEncoder(embedding_model)

    # Add all memories
    memory_mappings = []

    console.print("\n[bold]Adding memories to the system...[/bold]", style="section")

    for i, (text, category, relevance) in enumerate(all_memories):
        # Create metadata
        metadata = {"category": category, "relevance": relevance}

        # Get embedding
        embedding, _ = encoder.encode_concept(
            concept=category, description=text, related_concepts=[relevance]
        )

        # Add to memory
        idx = memory.add_memory(embedding, text, metadata)
        memory_mappings.append((idx, text, category, relevance))

        # Show progress every 10 memories
        if (i + 1) % 10 == 0 or i == len(all_memories) - 1:
            console.print(f"Added {i + 1} memories")

    # Test queries
    test_queries = [
        "Tell me about Python programming",
        "How do neural networks work?",
        "What is machine learning?",
        "What are the best travel destinations?",
        "How does supervised learning work?",
    ]

    # Test different confidence thresholds
    thresholds = [0.0, 0.3, 0.5, 0.7]

    # Create a table to track results
    results_table = Table(title="Retrieval Results by Threshold")
    results_table.add_column("Query", style="cyan")
    results_table.add_column("Threshold", style="magenta")
    results_table.add_column("Results", style="green")
    results_table.add_column("Relevance Distribution", style="yellow")

    # Track precision metrics
    precision_by_threshold = {threshold: [] for threshold in thresholds}

    console.print("\n[bold]Testing different confidence thresholds...[/bold]", style="section")

    for query in test_queries:
        console.print(f"\n[bold]Query:[/bold] {query}")

        # Get query embedding
        query_embedding = embedding_model.encode(query)

        for threshold in thresholds:
            console.print(f"[threshold]Threshold: {threshold}[/threshold]")

            # Retrieve with current threshold
            results = memory.retrieve_memories(
                query_embedding, top_k=5, confidence_threshold=threshold
            )

            # Count relevance distribution
            relevance_dist = Counter()

            if results:
                for idx, score, metadata in results:
                    relevance = metadata.get("relevance", "unknown")
                    category = metadata.get("category", "unknown")
                    console.print(
                        f"  [{category}:{relevance}] {metadata.get('text', '')[:50]}... (Score: {score:.3f})"
                    )
                    relevance_dist[relevance] += 1

                # Calculate precision (high + medium relevance / total)
                precision = (relevance_dist["high"] + relevance_dist["medium"]) / len(results)
                precision_by_threshold[threshold].append(precision)
            else:
                console.print("  No results found")

            # Add to results table
            results_table.add_row(
                query if threshold == 0.0 else "",
                f"{threshold:.1f}",
                f"{len(results)}",
                ", ".join([f"{k}:{v}" for k, v in relevance_dist.items()]),
            )

    console.print("\n")
    console.print(results_table)

    # Calculate average precision by threshold
    avg_precision = {t: np.mean(p) if p else 0 for t, p in precision_by_threshold.items()}

    # Create precision chart
    plt.figure(figsize=(10, 6))
    thresholds_list = list(avg_precision.keys())
    precision_list = list(avg_precision.values())

    plt.bar(thresholds_list, precision_list, color="blue", alpha=0.7)
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Average Precision")
    plt.title("Effect of Confidence Threshold on Retrieval Precision")
    plt.xticks(thresholds_list)
    plt.ylim(0, 1.1)

    for i, v in enumerate(precision_list):
        plt.text(thresholds_list[i], v + 0.05, f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig("confidence_threshold_precision.png")
    console.print(
        "[success]Precision chart saved as 'confidence_threshold_precision.png'[/success]"
    )

    # Test semantic coherence check
    console.print("\n[bold]Testing semantic coherence check...[/bold]", style="section")

    # Create a memory with semantic coherence check enabled
    memory_coherence = ContextualMemory(
        embedding_dim=model.config.hidden_size,
        default_confidence_threshold=0.3,  # Moderate threshold
        semantic_coherence_check=True,
        coherence_threshold=0.2,
    )

    # Add the same memories
    for text, category, relevance in all_memories:
        metadata = {"category": category, "relevance": relevance}
        embedding, _ = encoder.encode_concept(concept=category, description=text)
        memory_coherence.add_memory(embedding, text, metadata)

    # Compare with and without coherence check
    for query in test_queries[:2]:  # Just test a couple of queries
        console.print(f"\n[bold]Query:[/bold] {query}")
        query_embedding = embedding_model.encode(query)

        # Without coherence check
        console.print("[bold]Without coherence check:[/bold]")
        results_no_coherence = memory.retrieve_memories(
            query_embedding, top_k=5, confidence_threshold=0.3
        )

        if results_no_coherence:
            for idx, score, metadata in results_no_coherence:
                relevance = metadata.get("relevance", "unknown")
                category = metadata.get("category", "unknown")
                console.print(
                    f"  [{category}:{relevance}] {metadata.get('text', '')[:50]}... (Score: {score:.3f})"
                )
        else:
            console.print("  No results found")

        # With coherence check
        console.print("[bold]With coherence check:[/bold]")
        results_with_coherence = memory_coherence.retrieve_memories(
            query_embedding, top_k=5, confidence_threshold=0.3
        )

        if results_with_coherence:
            for idx, score, metadata in results_with_coherence:
                relevance = metadata.get("relevance", "unknown")
                category = metadata.get("category", "unknown")
                console.print(
                    f"  [{category}:{relevance}] {metadata.get('text', '')[:50]}... (Score: {score:.3f})"
                )
        else:
            console.print("  No results found")

    # Test adaptive k selection
    console.print("\n[bold]Testing adaptive k selection...[/bold]", style="section")

    # Create a memory with adaptive k selection enabled
    memory_adaptive = ContextualMemory(
        embedding_dim=model.config.hidden_size,
        default_confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=True,
    )

    # Add the same memories
    for text, category, relevance in all_memories:
        metadata = {"category": category, "relevance": relevance}
        embedding, _ = encoder.encode_concept(concept=category, description=text)
        memory_adaptive.add_memory(embedding, text, metadata)

    # Compare with and without adaptive k
    for query in test_queries[:2]:  # Just test a couple of queries
        console.print(f"\n[bold]Query:[/bold] {query}")
        query_embedding = embedding_model.encode(query)

        # Without adaptive k
        console.print("[bold]With fixed k=5:[/bold]")
        results_fixed_k = memory.retrieve_memories(
            query_embedding, top_k=5, confidence_threshold=0.3
        )

        if results_fixed_k:
            for idx, score, metadata in results_fixed_k:
                relevance = metadata.get("relevance", "unknown")
                category = metadata.get("category", "unknown")
                console.print(
                    f"  [{category}:{relevance}] {metadata.get('text', '')[:50]}... (Score: {score:.3f})"
                )
        else:
            console.print("  No results found")

        # With adaptive k
        console.print("[bold]With adaptive k selection:[/bold]")
        results_adaptive_k = memory_adaptive.retrieve_memories(
            query_embedding, top_k=5, confidence_threshold=0.3
        )

        if results_adaptive_k:
            for idx, score, metadata in results_adaptive_k:
                relevance = metadata.get("relevance", "unknown")
                category = metadata.get("category", "unknown")
                console.print(
                    f"  [{category}:{relevance}] {metadata.get('text', '')[:50]}... (Score: {score:.3f})"
                )
        else:
            console.print("  No results found")

    console.print("\n[bold]Test completed successfully![/bold]", style="success")
    return memory


if __name__ == "__main__":
    test_confidence_thresholding()
