"""
Utilities for analyzing memory retrieval performance and distributions.
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def analyze_query_similarities(
    memory_system: Dict[str, Any],
    query: str,
    expected_relevant_indices: Optional[List[int]] = None,
    plot: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze similarity distributions for a specific query.

    Args:
        memory_system: Dictionary containing memory components
        query: The query to analyze
        expected_relevant_indices: Indices of memories that should be relevant
        plot: Whether to generate plots
        save_path: Path to save the plot

    Returns:
        Dictionary with analysis results
    """
    memory = memory_system.get("memory")
    retriever = memory_system.get("retriever")
    embedding_model = memory_system.get("embedding_model")

    if not all([memory, retriever, embedding_model]):
        raise ValueError("Memory system must contain memory, retriever, and embedding model")

    # Get query embedding
    query_embedding = embedding_model.encode(query)

    # Calculate raw similarities with all memories
    similarities = np.dot(memory.memory_embeddings, query_embedding)

    # Get activation-boosted similarities
    boosted_similarities = similarities * memory.activation_levels

    # Get memories retrieved with current settings
    retrieved = retriever.retrieve_for_context(query, top_k=10)
    retrieved_indices = [
        item.get("memory_id") for item in retrieved if isinstance(item.get("memory_id"), int)
    ]

    # Prepare results
    results = {
        "query": query,
        "raw_similarities": similarities,
        "boosted_similarities": boosted_similarities,
        "activation_levels": memory.activation_levels,
        "retrieved_indices": retrieved_indices,
        "retrieved_scores": [
            item.get("relevance_score")
            for item in retrieved
            if isinstance(item.get("memory_id"), int)
        ],
        "memory_count": len(memory.memory_embeddings),
        "threshold_used": retriever.confidence_threshold,
    }

    # Add statistics
    results["raw_similarity_stats"] = {
        "mean": float(np.mean(similarities)),
        "median": float(np.median(similarities)),
        "min": float(np.min(similarities)),
        "max": float(np.max(similarities)),
        "std": float(np.std(similarities)),
    }

    # Add expected vs actual analysis if expected indices provided
    if expected_relevant_indices:
        results["expected_indices"] = expected_relevant_indices

        # Calculate precision and recall
        if retrieved_indices:
            precision = sum(
                1 for idx in retrieved_indices if idx in expected_relevant_indices
            ) / len(retrieved_indices)
        else:
            precision = 0.0

        if expected_relevant_indices:
            recall = sum(1 for idx in retrieved_indices if idx in expected_relevant_indices) / len(
                expected_relevant_indices
            )
        else:
            recall = 1.0

        # Calculate F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        results["metrics"] = {"precision": precision, "recall": recall, "f1": f1}

        # Get similarities of expected memories
        expected_similarities = similarities[expected_relevant_indices]
        expected_boosted = boosted_similarities[expected_relevant_indices]

        results["expected_similarity_stats"] = {
            "mean": float(np.mean(expected_similarities)),
            "median": float(np.median(expected_similarities)),
            "min": float(np.min(expected_similarities)),
            "max": float(np.max(expected_similarities)),
            "std": float(np.std(expected_similarities)),
        }

        # Check if any expected memories are below threshold
        below_threshold = [
            (idx, float(similarities[idx]), float(boosted_similarities[idx]))
            for idx in expected_relevant_indices
            if boosted_similarities[idx] < retriever.confidence_threshold
        ]

        results["below_threshold"] = below_threshold

    # Generate plots if requested
    if plot:
        plt.figure(figsize=(12, 8))

        # Plot histogram of all similarities
        plt.subplot(2, 1, 1)
        plt.hist(similarities, bins=30, alpha=0.5, label="Raw Similarities")
        plt.hist(boosted_similarities, bins=30, alpha=0.5, label="Activation-Boosted")

        # Add vertical line for threshold
        plt.axvline(
            x=retriever.confidence_threshold,
            color="r",
            linestyle="--",
            label=f"Threshold ({retriever.confidence_threshold:.2f})",
        )

        # Mark retrieved memories
        if retrieved_indices:
            retrieved_sims = boosted_similarities[retrieved_indices]
            plt.scatter(
                retrieved_sims,
                np.zeros_like(retrieved_sims) + 1,  # y position
                color="green",
                marker="o",
                s=100,
                label="Retrieved",
            )

        # Mark expected memories if provided
        if expected_relevant_indices:
            expected_sims = boosted_similarities[expected_relevant_indices]
            plt.scatter(
                expected_sims,
                np.zeros_like(expected_sims) + 2,  # y position
                color="blue",
                marker="x",
                s=100,
                label="Expected",
            )

        plt.title(f'Similarity Distribution for Query: "{query}"')
        plt.xlabel("Similarity Score")
        plt.ylabel("Frequency")
        plt.legend()

        # Plot activation levels
        plt.subplot(2, 1, 2)
        plt.bar(range(len(memory.activation_levels)), memory.activation_levels, alpha=0.7)

        # Mark retrieved memories
        if retrieved_indices:
            plt.bar(
                retrieved_indices,
                memory.activation_levels[retrieved_indices],
                color="green",
                alpha=0.7,
                label="Retrieved",
            )

        # Mark expected memories if provided
        if expected_relevant_indices:
            plt.bar(
                expected_relevant_indices,
                memory.activation_levels[expected_relevant_indices],
                color="blue",
                alpha=0.7,
                label="Expected",
            )

        plt.title("Memory Activation Levels")
        plt.xlabel("Memory Index")
        plt.ylabel("Activation Level")
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            results["plot_saved_to"] = save_path
        else:
            plt.show()

    return results


def visualize_memory_categories(
    memory_system: Dict[str, Any], save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Visualize memory categories and their relationships.

    Args:
        memory_system: Dictionary containing memory components
        save_path: Path to save the visualization

    Returns:
        Dictionary with category statistics
    """
    memory = memory_system.get("memory")

    if not memory or not memory.use_art_clustering:
        return {"error": "Memory system does not use ART clustering"}

    # Get category statistics
    stats = memory.get_category_statistics()

    # Get similarity matrix between categories
    sim_matrix = memory.category_similarity_matrix()

    # Plot category relationships
    plt.figure(figsize=(12, 10))

    # Plot similarity matrix as heatmap
    plt.subplot(2, 1, 1)
    plt.imshow(sim_matrix, cmap="viridis")
    plt.colorbar(label="Similarity")
    plt.title("Category Similarity Matrix")
    plt.xlabel("Category Index")
    plt.ylabel("Category Index")

    # Add category sizes
    category_counts = stats.get("memories_per_category", {})
    categories = sorted(category_counts.keys())
    counts = [category_counts.get(cat, 0) for cat in categories]

    plt.subplot(2, 1, 2)
    plt.bar(categories, counts)
    plt.title("Memories per Category")
    plt.xlabel("Category Index")
    plt.ylabel("Number of Memories")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        stats["plot_saved_to"] = save_path
    else:
        plt.show()

    return stats


def analyze_retrieval_performance(
    memory_system: Dict[str, Any],
    test_queries: List[Tuple[str, List[int]]],
    parameter_variations: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze retrieval performance across different parameter settings.

    Args:
        memory_system: Dictionary containing memory components
        test_queries: List of (query, expected_indices) tuples
        parameter_variations: List of parameter dictionaries to test
        save_path: Path to save the results

    Returns:
        Dictionary with performance metrics for each parameter setting
    """
    memory = memory_system.get("memory")
    retriever = memory_system.get("retriever")

    if not all([memory, retriever]):
        raise ValueError("Memory system must contain memory and retriever")

    # Store original parameters to restore later
    original_params = {
        "confidence_threshold": retriever.confidence_threshold,
        "adaptive_k_factor": retriever.adaptive_k_factor,
        "semantic_coherence_check": retriever.semantic_coherence_check,
        "use_two_stage_retrieval": retriever.use_two_stage_retrieval,
        "query_type_adaptation": retriever.query_type_adaptation,
    }

    results = []

    # Test each parameter variation
    for params in parameter_variations:
        # Update retriever parameters
        for param, value in params.items():
            if hasattr(retriever, param):
                setattr(retriever, param, value)

        # Run queries and collect metrics
        metrics = []
        for query, expected_indices in test_queries:
            retrieved = retriever.retrieve_for_context(query, top_k=10)
            retrieved_indices = [
                item.get("memory_id")
                for item in retrieved
                if isinstance(item.get("memory_id"), int)
            ]

            # Calculate precision and recall
            if retrieved_indices:
                precision = sum(1 for idx in retrieved_indices if idx in expected_indices) / len(
                    retrieved_indices
                )
            else:
                precision = 0.0

            if expected_indices:
                recall = sum(1 for idx in retrieved_indices if idx in expected_indices) / len(
                    expected_indices
                )
            else:
                recall = 1.0

            # Calculate F1
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            metrics.append(
                {
                    "query": query,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "retrieved_count": len(retrieved_indices),
                    "expected_count": len(expected_indices),
                    "correct_retrieved": [
                        idx for idx in retrieved_indices if idx in expected_indices
                    ],
                    "missed": [idx for idx in expected_indices if idx not in retrieved_indices],
                }
            )

        # Calculate average metrics
        avg_precision = np.mean([m["precision"] for m in metrics])
        avg_recall = np.mean([m["recall"] for m in metrics])
        avg_f1 = np.mean([m["f1"] for m in metrics])

        results.append(
            {
                "parameters": params,
                "avg_precision": float(avg_precision),
                "avg_recall": float(avg_recall),
                "avg_f1": float(avg_f1),
                "detailed_metrics": metrics,
            }
        )

    # Restore original parameters
    for param, value in original_params.items():
        if hasattr(retriever, param):
            setattr(retriever, param, value)

    # Plot results
    if save_path:
        plt.figure(figsize=(12, 8))

        # Extract parameter names and values for x-axis labels
        param_labels = []
        for params in parameter_variations:
            label = ", ".join([f"{k}={v}" for k, v in params.items()])
            param_labels.append(label)

        # Plot precision, recall, and F1
        x = np.arange(len(results))
        width = 0.25

        plt.bar(x - width, [r["avg_precision"] for r in results], width, label="Precision")
        plt.bar(x, [r["avg_recall"] for r in results], width, label="Recall")
        plt.bar(x + width, [r["avg_f1"] for r in results], width, label="F1")

        plt.xlabel("Parameter Settings")
        plt.ylabel("Score")
        plt.title("Retrieval Performance Across Parameter Settings")
        plt.xticks(x, param_labels, rotation=45, ha="right")
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)

    return {
        "results": results,
        "best_f1": max(results, key=lambda r: r["avg_f1"]),
        "best_precision": max(results, key=lambda r: r["avg_precision"]),
        "best_recall": max(results, key=lambda r: r["avg_recall"]),
    }
