"""
Benchmark script for MemoryWeave retrievers.

This script benchmarks different configurations of the MemoryWeave system,
focusing on retrieval quality and performance metrics.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import os

import torch
from transformers import AutoModel, AutoTokenizer

from memoryweave.core import ContextualMemory, ContextualRetriever, MemoryEncoder
from memoryweave.evaluation import coherence_score, context_relevance, response_consistency

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Helper class for sentence embedding
class EmbeddingModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        attention_mask = inputs['attention_mask']
        embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts
        
        return mean_pooled.numpy()[0]


class MemoryBenchmark:
    """Benchmark different configurations of the MemoryWeave system."""
    
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the benchmark.
        
        Args:
            embedding_model_name: Name of the embedding model to use
        """
        print(f"Loading embedding model: {embedding_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        model = AutoModel.from_pretrained(embedding_model_name)
        self.embedding_model = EmbeddingModelWrapper(model, tokenizer)
        self.embedding_dim = model.config.hidden_size
        
        # Datasets
        self.memory_data = None
        self.test_queries = None
        self.relevance_judgments = None
        
        # Results
        self.results = defaultdict(list)
        
    def load_test_data(self, memory_data=None, test_queries=None, relevance_judgments=None):
        """
        Load test data for benchmarking.
        
        If not provided, generate synthetic test data.
        
        Args:
            memory_data: List of (text, category, subcategory) tuples for memory items
            test_queries: List of (query_text, category) tuples for test queries
            relevance_judgments: Dict mapping (query_idx, memory_idx) to relevance score
        """
        if memory_data is None:
            # Generate synthetic memory data
            print("Generating synthetic memory data...")
            self.memory_data = self._generate_synthetic_memories()
        else:
            self.memory_data = memory_data
            
        if test_queries is None:
            # Generate synthetic test queries
            print("Generating synthetic test queries...")
            self.test_queries = self._generate_synthetic_queries()
        else:
            self.test_queries = test_queries
            
        if relevance_judgments is None:
            # Generate synthetic relevance judgments
            print("Generating synthetic relevance judgments...")
            self.relevance_judgments = self._generate_synthetic_relevance()
        else:
            self.relevance_judgments = relevance_judgments
            
        print(f"Loaded {len(self.memory_data)} memories and {len(self.test_queries)} test queries")
    
    def _generate_synthetic_memories(self) -> List[Tuple[str, str, str]]:
        """Generate synthetic memory data for testing."""
        # Categories and subcategories for synthetic data
        categories = {
            "technology": ["programming", "ai", "hardware", "software", "gadgets"],
            "science": ["physics", "biology", "chemistry", "astronomy", "medicine"],
            "arts": ["music", "painting", "literature", "film", "photography"],
            "history": ["ancient", "medieval", "modern", "war", "politics"],
            "personal": ["preferences", "experiences", "opinions", "relationships", "habits"]
        }
        
        memories = []
        
        # Generate memories for each category and subcategory
        for category, subcategories in categories.items():
            for subcategory in subcategories:
                # Generate multiple memories per subcategory
                for i in range(5):  # 5 memories per subcategory
                    text = f"This is a {subcategory} memory about {category} (#{i+1})"
                    memories.append((text, category, subcategory))
        
        return memories
    
    def _generate_synthetic_queries(self) -> List[Tuple[str, str]]:
        """Generate synthetic test queries."""
        # Extract unique categories from memory data
        categories = set(category for _, category, _ in self.memory_data)
        
        queries = []
        
        # Generate queries for each category
        for category in categories:
            # Direct query
            queries.append((f"Tell me about {category}", category))
            # Question query
            queries.append((f"What do you know about {category}?", category))
            # Specific query
            queries.append((f"I'm interested in learning more about {category}", category))
            
        # Add some cross-category queries
        if len(categories) >= 2:
            categories_list = list(categories)
            queries.append((f"Compare {categories_list[0]} and {categories_list[1]}", "cross-category"))
            
        return queries
    
    def _generate_synthetic_relevance(self) -> Dict[Tuple[int, int], float]:
        """Generate synthetic relevance judgments."""
        relevance = {}
        
        # For each query
        for query_idx, (query_text, query_category) in enumerate(self.test_queries):
            # For each memory
            for memory_idx, (memory_text, memory_category, memory_subcategory) in enumerate(self.memory_data):
                # Calculate relevance based on category match
                if query_category == memory_category:
                    relevance[(query_idx, memory_idx)] = 1.0  # Highly relevant
                elif query_category == "cross-category" and memory_category in query_text:
                    relevance[(query_idx, memory_idx)] = 0.8  # Relevant for cross-category queries
                else:
                    relevance[(query_idx, memory_idx)] = 0.0  # Not relevant
                    
        return relevance
    
    def benchmark_configuration(
        self,
        config_name: str,
        use_art_clustering: bool = False,
        confidence_threshold: float = 0.0,
        semantic_coherence_check: bool = False,
        adaptive_retrieval: bool = False,
        enable_category_consolidation: bool = False,
        retrieval_strategy: str = "hybrid",
    ):
        """
        Benchmark a specific configuration of the memory system.
        
        Args:
            config_name: Name to identify this configuration in results
            use_art_clustering: Whether to use ART-inspired clustering
            confidence_threshold: Minimum similarity score for retrieval
            semantic_coherence_check: Whether to check semantic coherence
            adaptive_retrieval: Whether to adaptively select k
            enable_category_consolidation: Whether to enable category consolidation
            retrieval_strategy: Retrieval strategy to use
        """
        print(f"\nBenchmarking configuration: {config_name}")
        
        # Initialize memory system
        memory = ContextualMemory(
            embedding_dim=self.embedding_dim,
            use_art_clustering=use_art_clustering,
            default_confidence_threshold=confidence_threshold,
            semantic_coherence_check=semantic_coherence_check,
            adaptive_retrieval=adaptive_retrieval,
            enable_category_consolidation=enable_category_consolidation,
        )
        
        encoder = MemoryEncoder(self.embedding_model)
        
        retriever = ContextualRetriever(
            memory=memory,
            embedding_model=self.embedding_model,
            retrieval_strategy=retrieval_strategy,
            confidence_threshold=confidence_threshold,
            semantic_coherence_check=semantic_coherence_check,
            adaptive_retrieval=adaptive_retrieval,
        )
        
        # Populate memory
        memory_start_time = time.time()
        for mem_idx, (text, category, subcategory) in enumerate(self.memory_data):
            # Encode memory
            embedding, metadata = encoder.encode_concept(
                concept=category,
                description=text,
                related_concepts=[subcategory]
            )
            
            # Add to memory
            memory.add_memory(embedding, text, metadata)
            
        memory_time = time.time() - memory_start_time
        print(f"Memory population time: {memory_time:.2f} seconds")
        
        # Benchmark retrieval
        retrieval_start_time = time.time()
        precision_at_k = []
        recall_at_k = []
        retrieval_times = []
        
        for query_idx, (query_text, query_category) in enumerate(self.test_queries):
            # Time retrieval
            query_start_time = time.time()
            retrieved = retriever.retrieve_for_context(query_text, top_k=5)
            query_time = time.time() - query_start_time
            retrieval_times.append(query_time)
            
            # Calculate precision and recall
            retrieved_indices = [item.get("memory_id") for item in retrieved]
            retrieved_indices = [idx for idx in retrieved_indices if isinstance(idx, int)]
            
            # Get relevant memories for this query
            relevant_indices = [
                memory_idx for memory_idx in range(len(self.memory_data))
                if self.relevance_judgments.get((query_idx, memory_idx), 0.0) > 0.5
            ]
            
            # Precision: how many retrieved items are relevant
            if len(retrieved_indices) > 0:
                precision = sum(1 for idx in retrieved_indices if idx in relevant_indices) / len(retrieved_indices)
            else:
                precision = 0.0
                
            # Recall: how many relevant items were retrieved
            if len(relevant_indices) > 0:
                recall = sum(1 for idx in retrieved_indices if idx in relevant_indices) / len(relevant_indices)
            else:
                recall = 1.0  # No relevant items, so perfect recall
                
            precision_at_k.append(precision)
            recall_at_k.append(recall)
            
            # Print some results
            if query_idx < 3:  # Only print the first few queries to avoid clutter
                print(f"\nQuery: {query_text}")
                print(f"Retrieved {len(retrieved)} memories in {query_time:.3f} seconds")
                print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")
                
                # Print top 3 retrieved memories
                for i, mem in enumerate(retrieved[:3]):
                    relevance = "✓" if mem.get("memory_id") in relevant_indices else "✗"
                    print(f"  {i+1}. [{relevance}] {mem.get('text', '')[:50]}... (Score: {mem.get('relevance_score', 0):.3f})")
                
        retrieval_time = time.time() - retrieval_start_time
        print(f"Total retrieval time: {retrieval_time:.2f} seconds")
        
        # Calculate average metrics
        avg_precision = np.mean(precision_at_k)
        avg_recall = np.mean(recall_at_k)
        avg_retrieval_time = np.mean(retrieval_times)
        
        print(f"Average precision: {avg_precision:.3f}")
        print(f"Average recall: {avg_recall:.3f}")
        print(f"Average retrieval time: {avg_retrieval_time:.3f} seconds per query")
        
        # Store results
        self.results["configuration"].append(config_name)
        self.results["avg_precision"].append(avg_precision)
        self.results["avg_recall"].append(avg_recall)
        self.results["avg_retrieval_time"].append(avg_retrieval_time)
        self.results["memory_time"].append(memory_time)
        self.results["use_art_clustering"].append(use_art_clustering)
        self.results["confidence_threshold"].append(confidence_threshold)
        self.results["semantic_coherence_check"].append(semantic_coherence_check)
        self.results["adaptive_retrieval"].append(adaptive_retrieval)
        self.results["enable_category_consolidation"].append(enable_category_consolidation)
        self.results["retrieval_strategy"].append(retrieval_strategy)
        
    def generate_report(self, output_file=None):
        """
        Generate a report of the benchmark results.
        
        Args:
            output_file: Optional file to save the report to
        """
        if not self.results:
            print("No results to report. Run benchmark_configuration first.")
            return
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Print summary table
        print("\n=== Benchmark Results ===")
        
        # Sort by precision
        sorted_df = results_df.sort_values("avg_precision", ascending=False)
        print(sorted_df[["configuration", "avg_precision", "avg_recall", "avg_retrieval_time"]].to_string(index=False))
        
        # Create precision vs. recall plot
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df["avg_recall"], results_df["avg_precision"], s=100, alpha=0.7)
        
        # Add labels to each point
        for i, config in enumerate(results_df["configuration"]):
            plt.annotate(
                config, 
                (results_df["avg_recall"][i], results_df["avg_precision"][i]),
                xytext=(7, 0), 
                textcoords='offset points'
            )
            
        plt.xlabel("Average Recall")
        plt.ylabel("Average Precision")
        plt.title("Precision vs. Recall for Different Configurations")
        plt.grid(True, alpha=0.3)
        
        # Set axis limits with some padding
        plt.xlim(max(0, min(results_df["avg_recall"]) - 0.05), min(1, max(results_df["avg_recall"]) + 0.05))
        plt.ylim(max(0, min(results_df["avg_precision"]) - 0.05), min(1, max(results_df["avg_precision"]) + 0.05))
        
        # Save or show the plot
        if output_file:
            output_path = f"output/{output_file}_precision_recall.png"
            plt.savefig(output_path)
            print(f"Saved precision-recall plot to {output_path}")
            
            # Also save the results data
            csv_path = f"output/{output_file}_results.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")
        else:
            output_path = "output/benchmark_precision_recall.png"
            plt.savefig(output_path)
            print(f"Saved precision-recall plot to {output_path}")
            
            # Also save the results data
            csv_path = "output/benchmark_results.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")
            
        # Create a bar chart comparing retrieval times
        plt.figure(figsize=(12, 6))
        bar_positions = np.arange(len(results_df["configuration"]))
        
        plt.bar(bar_positions, results_df["avg_retrieval_time"], alpha=0.7)
        plt.xticks(bar_positions, results_df["configuration"], rotation=45, ha="right")
        plt.ylabel("Average Retrieval Time (seconds)")
        plt.title("Retrieval Performance for Different Configurations")
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save or show the plot
        if output_file:
            output_path = f"output/{output_file}_retrieval_time.png"
            plt.savefig(output_path)
            print(f"Saved retrieval time plot to {output_path}")
        else:
            output_path = "output/benchmark_retrieval_time.png"
            plt.savefig(output_path)
            print(f"Saved retrieval time plot to {output_path}")
        
        return results_df


def run_benchmark():
    """
    Run the benchmark with various configurations.
    """
    benchmark = MemoryBenchmark()
    benchmark.load_test_data()
    
    # Test baseline configuration
    benchmark.benchmark_configuration(
        config_name="Baseline",
        use_art_clustering=False,
        confidence_threshold=0.0,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
    )
    
    # Test confidence thresholding
    benchmark.benchmark_configuration(
        config_name="Confidence Threshold (0.3)",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
    )
    
    # Test confidence thresholding + semantic coherence
    benchmark.benchmark_configuration(
        config_name="Conf + Semantic Coherence",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=True,
        adaptive_retrieval=False,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
    )
    
    # Test adaptive retrieval
    benchmark.benchmark_configuration(
        config_name="Adaptive Retrieval",
        use_art_clustering=False,
        confidence_threshold=0.3,
        semantic_coherence_check=False,
        adaptive_retrieval=True,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
    )
    
    # Test ART clustering
    benchmark.benchmark_configuration(
        config_name="ART Clustering",
        use_art_clustering=True,
        confidence_threshold=0.0,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        enable_category_consolidation=False,
        retrieval_strategy="hybrid",
    )
    
    # Test ART clustering + consolidation
    benchmark.benchmark_configuration(
        config_name="ART + Consolidation",
        use_art_clustering=True,
        confidence_threshold=0.0,
        semantic_coherence_check=False,
        adaptive_retrieval=False,
        enable_category_consolidation=True,
        retrieval_strategy="hybrid",
    )
    
    # Test full featured configuration
    benchmark.benchmark_configuration(
        config_name="Full Features",
        use_art_clustering=True,
        confidence_threshold=0.3,
        semantic_coherence_check=True,
        adaptive_retrieval=True,
        enable_category_consolidation=True,
        retrieval_strategy="hybrid",
    )
    
    # Generate report
    benchmark.generate_report(output_file="memory_benchmark")


if __name__ == "__main__":
    """
    Run the benchmark with:
    python notebooks/benchmark_memory.py
    """
    run_benchmark()
