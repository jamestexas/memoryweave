"""Example showing performance improvements with ANN vector store.

This script demonstrates the performance difference between the standard vector store
and the new ANN-based vector store implementation for different memory store sizes.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from memoryweave.storage.vector_store import (
    ActivationVectorStore,
    ANNActivationVectorStore,
    ANNVectorStore,
    SimpleVectorStore,
    get_optimal_faiss_config,
)


def generate_random_vectors(count: int, dimension: int = 768) -> list[np.ndarray]:
    """Generate random embedding vectors."""
    return [np.random.randn(dimension).astype(np.float32) for _ in range(count)]


def benchmark_vector_stores(
    memory_counts: list[int],
    dimension: int = 768,
    num_queries: int = 10,
    k: int = 10,
) -> dict[str, dict[str, list[float]]]:
    """Benchmark different vector store implementations across memory scales."""
    results = {
        "SimpleVectorStore": {"add_time": [], "search_time": [], "memory_size": []},
        "ActivationVectorStore": {"add_time": [], "search_time": [], "memory_size": []},
        "ANNVectorStore": {"add_time": [], "search_time": [], "memory_size": []},
        "ANNActivationVectorStore": {"add_time": [], "search_time": [], "memory_size": []},
    }

    # Generate query vectors once
    query_vectors = generate_random_vectors(num_queries, dimension)

    for memory_count in memory_counts:
        # Generate vectors for this test
        vectors = generate_random_vectors(memory_count, dimension)

        # Determine optimal FAISS config for this memory scale
        scale = "small" if memory_count < 100 else "medium" if memory_count < 500 else "large"
        faiss_config = get_optimal_faiss_config(scale, dimension)

        # Test SimpleVectorStore
        store = SimpleVectorStore()
        add_time = measure_add_time(store, vectors)
        search_time = measure_search_time(store, query_vectors, k)
        results["SimpleVectorStore"]["add_time"].append(add_time)
        results["SimpleVectorStore"]["search_time"].append(search_time)
        results["SimpleVectorStore"]["memory_size"].append(memory_count)

        # Test ActivationVectorStore
        store = ActivationVectorStore(activation_weight=0.2)
        add_time = measure_add_time(store, vectors)
        search_time = measure_search_time(store, query_vectors, k)
        results["ActivationVectorStore"]["add_time"].append(add_time)
        results["ActivationVectorStore"]["search_time"].append(search_time)
        results["ActivationVectorStore"]["memory_size"].append(memory_count)

        # For ANN tests, we need to make sure we have enough vectors for the cluster count
        # Skip ANN tests if we have fewer vectors than the minimum required
        if memory_count < 100:  # Skip for small test sets
            # Just add empty data to keep the results structure consistent
            results["ANNVectorStore"]["add_time"].append(0)
            results["ANNVectorStore"]["search_time"].append(0)
            results["ANNVectorStore"]["memory_size"].append(memory_count)
            results["ANNActivationVectorStore"]["add_time"].append(0)
            results["ANNActivationVectorStore"]["search_time"].append(0)
            results["ANNActivationVectorStore"]["memory_size"].append(memory_count)
            continue

        # Only test ANNVectorStore for large enough memory sets
        # Test ANNVectorStore
        # Adjust index settings to avoid training errors
        if memory_count < 200:
            # Use Flat index for small datasets (100-200 memories)
            ann_index_type = "Flat"
        else:
            # Use IVF with a reasonable number of clusters based on data size
            # FAISS requires at least 39*k data points for k clusters (we'll use a safer 40*k)
            # so we ensure the number of clusters is always appropriate for the dataset size
            max_possible_clusters = memory_count // 40
            # Use at least 5 clusters but don't exceed max_possible_clusters
            num_clusters = min(50, max(max_possible_clusters, 5))
            ann_index_type = f"IVF{num_clusters},Flat"

        store = ANNVectorStore(
            dimension=dimension,
            index_type=ann_index_type,
            nprobe=faiss_config["nprobe"],
            build_threshold=faiss_config["build_threshold"],
            quantize=faiss_config["quantize"],
        )
        add_time = measure_add_time(store, vectors)
        search_time = measure_search_time(store, query_vectors, k)
        results["ANNVectorStore"]["add_time"].append(add_time)
        results["ANNVectorStore"]["search_time"].append(search_time)
        results["ANNVectorStore"]["memory_size"].append(memory_count)

        # Test ANNActivationVectorStore
        store = ANNActivationVectorStore(
            activation_weight=0.2,
            dimension=dimension,
            index_type=ann_index_type,  # Use the same adjusted index type from above
            nprobe=faiss_config["nprobe"],
            build_threshold=faiss_config["build_threshold"],
            quantize=faiss_config["quantize"],
        )
        add_time = measure_add_time(store, vectors)
        search_time = measure_search_time(store, query_vectors, k)
        results["ANNActivationVectorStore"]["add_time"].append(add_time)
        results["ANNActivationVectorStore"]["search_time"].append(search_time)
        results["ANNActivationVectorStore"]["memory_size"].append(memory_count)

        # Print progress
        print(f"Completed benchmark for {memory_count} memories")

    return results


def measure_add_time(store, vectors: list[np.ndarray]) -> float:
    """Measure time to add vectors to store."""
    start_time = time.time()
    for i, vector in enumerate(vectors):
        store.add(f"id_{i}", vector)
    return time.time() - start_time


def measure_search_time(store, query_vectors: list[np.ndarray], k: int) -> float:
    """Measure time to search vectors in store."""
    search_times = []
    for query in query_vectors:
        start_time = time.time()
        _ = store.search(query, k)
        search_times.append(time.time() - start_time)
    return sum(search_times) / len(search_times)


def plot_results(
    results: dict[str, dict[str, list[float]]], output_path: str = "vector_store_benchmark.png"
):
    """Plot benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Prepare data for plotting
    memory_sizes = results["SimpleVectorStore"]["memory_size"]

    # Search time plot (log scale)
    ax1.set_title("Search Time Comparison (Lower is Better)")
    ax1.set_xlabel("Memory Store Size")
    ax1.set_ylabel("Average Search Time (s) - Log Scale")
    ax1.set_yscale("log")

    # Add time plot
    ax2.set_title("Add Time Comparison (Lower is Better)")
    ax2.set_xlabel("Memory Store Size")
    ax2.set_ylabel("Total Add Time (s)")

    # Plot each store type
    colors = ["blue", "green", "red", "purple"]
    for i, (store_name, store_data) in enumerate(results.items()):
        ax1.plot(
            memory_sizes, store_data["search_time"], marker="o", label=store_name, color=colors[i]
        )
        ax2.plot(
            memory_sizes, store_data["add_time"], marker="o", label=store_name, color=colors[i]
        )

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Results plotted and saved to {output_path}")


def main():
    """Run the benchmark."""
    # Memory counts to test (small, medium, large)
    memory_counts = [50, 100, 500, 1000, 5000]

    # Run benchmark
    print("Starting vector store benchmark...")
    results = benchmark_vector_stores(memory_counts)

    # Plot results
    plot_results(results)

    # Print search time improvement factor for largest memory size
    largest_count = max(memory_counts)
    simple_time = results["SimpleVectorStore"]["search_time"][-1]
    ann_time = results["ANNVectorStore"]["search_time"][-1]
    improvement = simple_time / ann_time if ann_time > 0 else float("inf")

    print(f"\nPerformance Summary for {largest_count} memories:")
    print(f"SimpleVectorStore search time: {simple_time:.6f}s")
    print(f"ANNVectorStore search time: {ann_time:.6f}s")
    print(f"Improvement factor: {improvement:.2f}x faster")

    # Print performance table
    print("\nDetailed Performance Comparison:")
    print(f"{'Memory Count':<15} {'Simple (s)':<15} {'ANN (s)':<15} {'Improvement':<15}")
    print("-" * 60)

    for i, count in enumerate(memory_counts):
        simple = results["SimpleVectorStore"]["search_time"][i]
        ann = results["ANNVectorStore"]["search_time"][i]
        imp = simple / ann if ann > 0 else float("inf")
        print(f"{count:<15} {simple:<15.6f} {ann:<15.6f} {imp:<15.2f}x")


if __name__ == "__main__":
    main()
