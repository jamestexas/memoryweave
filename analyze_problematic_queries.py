"""
Analyze problematic queries and test enhanced retrieval mechanisms.

This script helps identify why certain queries might be failing to retrieve
relevant memories and tests the effectiveness of the enhanced retrieval mechanisms.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

from memoryweave.core import ContextualMemory, ContextualRetriever, MemoryEncoder
from memoryweave.utils.analysis import analyze_query_similarities, visualize_memory_categories, analyze_retrieval_performance

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

def load_test_data(file_path="datasets/evaluation_queries.json"):
    """Load test data from the evaluation queries file."""
    with open(file_path) as f:
        data = json.load(f)
    
    # Separate queries by category
    personal_queries = []
    general_queries = []
    domain_specific_queries = []
    
    for item in data:
        query = item["query"]
        expected = item["expected_answer"]
        category = item["category"]
        
        if category == "personal":
            personal_queries.append((query, expected, category))
        elif category == "general":
            general_queries.append((query, expected, category))
        elif category == "domain-specific":
            domain_specific_queries.append((query, expected, category))
    
    return {
        "personal": personal_queries,
        "general": general_queries,
        "domain_specific": domain_specific_queries,
        "all": personal_queries + general_queries + domain_specific_queries
    }

def populate_memory(memory_system, queries):
    """Populate memory with test data."""
    memory = memory_system["memory"]
    encoder = memory_system["encoder"]
    
    # Add all expected answers to memory
    for query, expected, category in queries:
        # Encode memory
        embedding, metadata = encoder.encode_concept(
            concept=category,
            description=expected,
            related_concepts=[query.split()[0]]  # Use first word of query as related concept
        )
        
        # Add to memory
        memory.add_memory(embedding, expected, metadata)
    
    print(f"Populated memory with {len(queries)} items")

def analyze_query_types(memory_system, queries_by_category):
    """Analyze performance for different query types."""
    retriever = memory_system["retriever"]
    
    results = {}
    
    # Test each category
    for category, queries in queries_by_category.items():
        if category == "all":
            continue
            
        print(f"\nAnalyzing {category} queries...")
        category_results = []
        
        for query, expected, _ in queries[:5]:  # Test first 5 queries of each category
            print(f"\nQuery: {query}")
            print(f"Expected: {expected}")
            
            # Retrieve with default settings
            retrieved = retriever.retrieve_for_context(query, top_k=5)
            
            # Check if expected answer is in retrieved memories
            found = False
            for item in retrieved:
                if expected.lower() in str(item.get("text", "")).lower() or expected.lower() in str(item.get("description", "")).lower():
                    found = True
                    break
            
            print(f"Found expected answer: {'✓' if found else '✗'}")
            print(f"Retrieved {len(retrieved)} memories")
            
            # Analyze similarities
            analysis = analyze_query_similarities(
                memory_system=memory_system,
                query=query,
                plot=True,
                save_path=f"output/query_analysis_{category}_{len(category_results)}.png"
            )
            
            category_results.append({
                "query": query,
                "expected": expected,
                "found": found,
                "retrieved_count": len(retrieved),
                "analysis": {
                    "raw_similarity_stats": analysis["raw_similarity_stats"],
                    "threshold_used": analysis["threshold_used"],
                }
            })
        
        results[category] = category_results
    
    return results

def test_enhanced_retrieval(memory_system, queries_by_category):
    """Test the enhanced retrieval mechanisms."""
    retriever = memory_system["retriever"]
    
    # Define parameter variations to test
    parameter_variations = [
        # Baseline
        {"confidence_threshold": 0.3, "adaptive_k_factor": 0.3, "use_two_stage_retrieval": False, "query_type_adaptation": False},
        # Two-stage retrieval
        {"confidence_threshold": 0.3, "adaptive_k_factor": 0.3, "use_two_stage_retrieval": True, "query_type_adaptation": False},
        # Query type adaptation
        {"confidence_threshold": 0.3, "adaptive_k_factor": 0.3, "use_two_stage_retrieval": False, "query_type_adaptation": True},
        # Combined approach
        {"confidence_threshold": 0.3, "adaptive_k_factor": 0.3, "use_two_stage_retrieval": True, "query_type_adaptation": True},
        # Lower threshold
        {"confidence_threshold": 0.15, "adaptive_k_factor": 0.3, "use_two_stage_retrieval": True, "query_type_adaptation": True},
        # Less conservative adaptive K
        {"confidence_threshold": 0.3, "adaptive_k_factor": 0.15, "use_two_stage_retrieval": True, "query_type_adaptation": True},
        # Optimized combined approach
        {"confidence_threshold": 0.2, "adaptive_k_factor": 0.15, "use_two_stage_retrieval": True, "query_type_adaptation": True},
    ]
    
    results = {}
    
    # Test each category
    for category, queries in queries_by_category.items():
        if category == "all":
            continue
            
        print(f"\nTesting enhanced retrieval for {category} queries...")
        
        # Prepare test queries with expected indices
        test_queries = []
        for i, (query, expected, _) in enumerate(queries[:5]):  # Test first 5 queries of each category
            # Find memory indices that contain the expected answer
            expected_indices = []
            for j, metadata in enumerate(memory_system["memory"].memory_metadata):
                if expected.lower() in str(metadata.get("text", "")).lower() or expected.lower() in str(metadata.get("description", "")).lower():
                    expected_indices.append(j)
            
            test_queries.append((query, expected_indices))
        
        # Analyze retrieval performance across parameter variations
        performance = analyze_retrieval_performance(
            memory_system=memory_system,
            test_queries=test_queries,
            parameter_variations=parameter_variations,
            save_path=f"output/retrieval_performance_{category}.png"
        )
        
        # Convert the results to be JSON serializable
        serializable_performance = {
            "results": [
                {
                    "parameters": result["parameters"],
                    "avg_precision": float(result["avg_precision"]),
                    "avg_recall": float(result["avg_recall"]),
                    "avg_f1": float(result["avg_f1"]),
                    "detailed_metrics": [
                        {
                            "query": metric["query"],
                            "precision": float(metric["precision"]),
                            "recall": float(metric["recall"]),
                            "f1": float(metric["f1"]),
                            "retrieved_count": int(metric["retrieved_count"]),
                            "expected_count": int(metric["expected_count"]),
                            "correct_retrieved": [int(idx) for idx in metric["correct_retrieved"]],
                            "missed": [int(idx) for idx in metric["missed"]],
                        }
                        for metric in result["detailed_metrics"]
                    ]
                }
                for result in performance["results"]
            ],
            "best_f1": {
                "parameters": performance["best_f1"]["parameters"],
                "avg_precision": float(performance["best_f1"]["avg_precision"]),
                "avg_recall": float(performance["best_f1"]["avg_recall"]),
                "avg_f1": float(performance["best_f1"]["avg_f1"]),
            },
            "best_precision": {
                "parameters": performance["best_precision"]["parameters"],
                "avg_precision": float(performance["best_precision"]["avg_precision"]),
                "avg_recall": float(performance["best_precision"]["avg_recall"]),
                "avg_f1": float(performance["best_precision"]["avg_f1"]),
            },
            "best_recall": {
                "parameters": performance["best_recall"]["parameters"],
                "avg_precision": float(performance["best_recall"]["avg_precision"]),
                "avg_recall": float(performance["best_recall"]["avg_recall"]),
                "avg_f1": float(performance["best_recall"]["avg_f1"]),
            }
        }
        
        results[category] = serializable_performance
    
    return results

def main():
    """Run the analysis."""
    print("Loading embedding model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embedding_model = EmbeddingModelWrapper(model, tokenizer)
    
    # Initialize memory components
    embedding_dim = model.config.hidden_size
    memory = ContextualMemory(
        embedding_dim=embedding_dim,
        use_art_clustering=True,
        vigilance_threshold=0.75,
        default_confidence_threshold=0.3,
        adaptive_retrieval=True,
        semantic_coherence_check=True,
    )
    encoder = MemoryEncoder(embedding_model)
    retriever = ContextualRetriever(
        memory=memory,
        embedding_model=embedding_model,
        confidence_threshold=0.3,
        semantic_coherence_check=True,
        adaptive_retrieval=True,
        adaptive_k_factor=0.3,
        use_two_stage_retrieval=False,
        query_type_adaptation=False,
    )
    
    memory_system = {
        "memory": memory,
        "encoder": encoder,
        "retriever": retriever,
        "embedding_model": embedding_model
    }
    
    # Load test data
    print("Loading test data...")
    queries_by_category = load_test_data()
    
    # Populate memory
    print("Populating memory...")
    populate_memory(memory_system, queries_by_category["all"])
    
    # Analyze query types
    print("Analyzing query types...")
    query_analysis = analyze_query_types(memory_system, queries_by_category)
    
    # Visualize memory categories
    print("Visualizing memory categories...")
    category_stats = visualize_memory_categories(
        memory_system=memory_system,
        save_path="output/memory_categories.png"
    )
    
    # Test enhanced retrieval
    print("Testing enhanced retrieval mechanisms...")
    retrieval_results = test_enhanced_retrieval(memory_system, queries_by_category)
    
    # Save results
    with open("output/query_analysis_results.json", "w") as f:
        json.dump(query_analysis, f, indent=2)
    
    # Convert category_stats to be JSON serializable
    serializable_stats = {}
    for key, value in category_stats.items():
        if isinstance(value, dict):
            serializable_stats[key] = {}
            for k, v in value.items():
                if isinstance(k, tuple):
                    # Convert tuple keys to strings
                    serializable_stats[key][str(k)] = v
                else:
                    serializable_stats[key][k] = v
        else:
            serializable_stats[key] = value
    
    with open("output/category_stats.json", "w") as f:
        json.dump(serializable_stats, f, indent=2)
    
    with open("output/retrieval_results.json", "w") as f:
        json.dump(retrieval_results, f, indent=2)
    
    print("\nAnalysis complete. Results saved to output directory.")
    
    # Print summary of findings
    print("\nSummary of Findings:")
    
    # Summarize query analysis
    for category, results in query_analysis.items():
        found_count = sum(1 for r in results if r["found"])
        print(f"{category.capitalize()} queries: {found_count}/{len(results)} successful retrievals")
    
    # Summarize best retrieval configuration
    for category, performance in retrieval_results.items():
        best_config = performance["best_f1"]
        print(f"\nBest configuration for {category} queries (F1={best_config['avg_f1']:.2f}):")
        for param, value in best_config["parameters"].items():
            print(f"  {param}: {value}")

if __name__ == "__main__":
    main()
