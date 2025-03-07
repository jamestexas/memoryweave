#!/usr/bin/env python3
"""
Example showing how to use the baseline comparison framework.
"""

import json
from pathlib import Path

import numpy as np

from memoryweave.baselines import BM25Retriever, VectorBaselineRetriever
from memoryweave.components.memory_manager import MemoryManager
from memoryweave.evaluation.baseline_comparison import BaselineComparison, BaselineConfig
from memoryweave.interfaces.memory import Memory
from memoryweave.interfaces.retrieval import Query, QueryType
from memoryweave.storage.adapter import MemoryAdapter
from memoryweave.storage.memory_store import StandardMemoryStore


class RetrievalStrategyAdapter:
    """Adapter to make SimilarityRetrievalStrategy compatible with the baseline comparison framework."""

    def __init__(self, retrieval_strategy):
        """Initialize with a retrieval strategy."""
        self.retrieval_strategy = retrieval_strategy

    def retrieve(self, query, top_k=10, threshold=0.0, **kwargs):
        """Adapter method to work with the baseline comparison interface."""
        # Extract the query embedding
        query_embedding = query.embedding

        # Create a context dictionary with necessary parameters
        context = {
            "memory": self.retrieval_strategy.memory,
            "query": query.text,
            "top_k": top_k,
            "adapted_retrieval_params": {"confidence_threshold": threshold},
        }

        # Call the strategy's retrieve method with the correct arguments
        results = self.retrieval_strategy.retrieve(query_embedding, top_k, context)

        # Convert the results to the expected format
        memories = []
        scores = []

        # Get the memory store
        memory_store = self.retrieval_strategy.memory

        for result in results:
            memory_id = result.get("memory_id")
            if hasattr(memory_store, "get") and callable(memory_store.get):
                memory = memory_store.get(memory_id)
                memories.append(memory)
                scores.append(result.get("relevance_score", 0.0))

        # Return in the expected format
        return {
            "memories": memories,
            "scores": scores,
            "strategy": "memoryweave",
            "parameters": {"max_results": top_k, "threshold": threshold},
            "metadata": {
                "query_time": 0.0  # We don't track query time here
            },
        }


def main():
    """Run a simple baseline comparison example."""
    # Load the sample dataset
    dataset_path = Path(__file__).parent.parent / "sample_baseline_dataset.json"

    with open(dataset_path) as f:
        data = json.load(f)

    # Create memory objects
    memories = []
    for mem_data in data.get("memories", []):
        embedding = np.array(mem_data.get("embedding", [])) if "embedding" in mem_data else None

        memory = Memory(
            id=mem_data.get("id", str(len(memories))),
            embedding=embedding,
            content=mem_data.get("content", {"text": "", "metadata": {}}),
            metadata=mem_data.get("metadata", {}),
        )
        memories.append(memory)

    # Create query objects
    queries = []
    for q_data in data.get("queries", []):
        embedding = np.array(q_data.get("embedding", [])) if "embedding" in q_data else None

        query = Query(
            text=q_data.get("text", ""),
            embedding=embedding,
            query_type=QueryType.UNKNOWN,
            extracted_keywords=q_data.get("keywords", []),
            extracted_entities=q_data.get("entities", []),
        )
        queries.append(query)

    # Get relevant memory IDs
    relevant_ids = data.get("relevant_ids", [])

    # Initialize memory manager
    memory_store = StandardMemoryStore()
    MemoryAdapter(memory_store)
    memory_store.add_multiple(memories)
    memory_manager = MemoryManager(memory_store=memory_store)

    # Create a mock solution since SimilarityRetrievalStrategy expects ContextualMemory
    # For our benchmark purposes, we'll create a simplified version that works with MemoryManager

    class MemoryWeaveRetriever:
        """Simplified retriever that works with MemoryManager for benchmark purposes."""

        def __init__(self, memory_manager):
            self.memory_manager = memory_manager

        def retrieve(self, query, top_k=10, threshold=0.0, **kwargs):
            """Basic vector similarity retrieval implementation."""
            # For benchmark purposes, implement a simple vector similarity search
            # that's compatible with our Memory objects

            if query.embedding is None or not hasattr(query, "embedding"):
                return {
                    "memories": [],
                    "scores": [],
                    "strategy": "memoryweave",
                    "parameters": {"max_results": top_k, "threshold": threshold},
                    "metadata": {"query_time": 0.0},
                }

            # Get all memories
            all_memories = self.memory_manager.get_all_memories()
            if not all_memories:
                return {
                    "memories": [],
                    "scores": [],
                    "strategy": "memoryweave",
                    "parameters": {"max_results": top_k, "threshold": threshold},
                    "metadata": {"query_time": 0.0},
                }

            # Calculate similarities
            query_embedding = np.array(query.embedding).reshape(1, -1)
            results = []

            for memory in all_memories:
                if memory.embedding is not None:
                    # Use cosine similarity
                    memory_embedding = np.array(memory.embedding).reshape(1, -1)

                    # Ensure same dimensions by padding if necessary
                    if memory_embedding.shape[1] < query_embedding.shape[1]:
                        pad_width = query_embedding.shape[1] - memory_embedding.shape[1]
                        memory_embedding = np.pad(memory_embedding, ((0, 0), (0, pad_width)))
                    elif memory_embedding.shape[1] > query_embedding.shape[1]:
                        # Use only the first dimensions of memory embedding
                        memory_embedding = memory_embedding[:, : query_embedding.shape[1]]

                    # Calculate dot product
                    similarity = np.dot(query_embedding, memory_embedding.T)[0][0]

                    # Normalize
                    query_norm = np.linalg.norm(query_embedding)
                    memory_norm = np.linalg.norm(memory_embedding)
                    if query_norm > 0 and memory_norm > 0:
                        similarity = similarity / (query_norm * memory_norm)

                    if similarity >= threshold:
                        results.append((memory, float(similarity)))

            # Sort by similarity (descending)
            results.sort(key=lambda x: x[1], reverse=True)

            # Take top_k
            results = results[:top_k]

            return {
                "memories": [memory for memory, _ in results],
                "scores": [score for _, score in results],
                "strategy": "memoryweave",
                "parameters": {"max_results": top_k, "threshold": threshold},
                "metadata": {"query_time": 0.0},
            }

    # Use our simplified retriever
    memoryweave_retriever = MemoryWeaveRetriever(memory_manager)

    # Define baseline configurations
    baseline_configs = [
        BaselineConfig(
            name="bm25", retriever_class=BM25Retriever, parameters={"b": 0.75, "k1": 1.2}
        ),
        BaselineConfig(
            name="vector_search",
            retriever_class=VectorBaselineRetriever,
            parameters={"use_exact_search": True},
        ),
    ]

    # Create comparison framework
    comparison = BaselineComparison(
        memory_manager=memory_manager,
        memoryweave_retriever=memoryweave_retriever,
        baseline_configs=baseline_configs,
        metrics=["precision", "recall", "f1", "mrr"],
    )

    # Run comparison
    print("Running baseline comparison...")
    result = comparison.run_comparison(
        queries=queries, relevant_memory_ids=relevant_ids, max_results=5, threshold=0.0
    )

    # Save results
    output_path = "baseline_comparison_example.json"
    comparison.save_results(result, output_path)
    print(f"Saved comparison results to {output_path}")

    # Generate visualization
    visualization_path = "baseline_comparison_chart.png"
    comparison.visualize_results(result, visualization_path)
    print(f"Saved visualization to {visualization_path}")

    # Generate HTML report
    html_report_path = "baseline_comparison_report.html"
    comparison.generate_html_report(
        result, html_report_path, title="MemoryWeave Baseline Comparison Example"
    )
    print(f"Saved HTML report to {html_report_path}")

    # Print summary
    print("\nComparison Summary:")
    print("-------------------")

    # Print MemoryWeave metrics
    mw_metrics = result.memoryweave_metrics["average"]
    print("MemoryWeave (SimilarityRetriever):")
    print(f"  Precision: {mw_metrics.get('precision', 0):.4f}")
    print(f"  Recall: {mw_metrics.get('recall', 0):.4f}")
    print(f"  F1: {mw_metrics.get('f1', 0):.4f}")
    print(f"  MRR: {mw_metrics.get('mrr', 0):.4f}")
    print(f"  Avg Query Time: {result.runtime_stats['memoryweave'].get('avg_query_time', 0):.5f}s")

    # Print baseline metrics
    for name, metrics in result.baseline_metrics.items():
        print(f"\n{name}:")
        avg_metrics = metrics["average"]
        print(f"  Precision: {avg_metrics.get('precision', 0):.4f}")
        print(f"  Recall: {avg_metrics.get('recall', 0):.4f}")
        print(f"  F1: {avg_metrics.get('f1', 0):.4f}")
        print(f"  MRR: {avg_metrics.get('mrr', 0):.4f}")
        print(f"  Avg Query Time: {result.runtime_stats[name].get('avg_query_time', 0):.5f}s")


if __name__ == "__main__":
    main()
