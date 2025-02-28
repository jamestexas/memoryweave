"""
Diagnostic Analysis for MemoryWeave

This script performs targeted analysis to identify core issues with memory retrieval,
focusing on the differences between personal and factual queries, embedding space
analysis, and threshold optimization.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve
from transformers import AutoTokenizer, AutoModel

from memoryweave.core import ContextualMemory, ContextualRetriever, MemoryEncoder
from memoryweave.utils.nlp_extraction import NLPExtractor
from memoryweave.utils.analysis import analyze_query_similarities

# Create output directory if it doesn't exist
os.makedirs("diagnostic_output", exist_ok=True)

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

class DiagnosticAnalysis:
    """
    Performs diagnostic analysis on the memory retrieval system to identify core issues.
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the diagnostic analysis.
        
        Args:
            model_name: Name of the embedding model to use
        """
        print(f"Loading embedding model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        self.embedding_model = EmbeddingModelWrapper(model, tokenizer)
        self.embedding_dim = model.config.hidden_size
        
        # Initialize NLP extractor for query classification
        self.nlp_extractor = NLPExtractor()
        
        # Initialize memory components
        self.memory = ContextualMemory(embedding_dim=self.embedding_dim)
        self.encoder = MemoryEncoder(self.embedding_model)
        self.retriever = ContextualRetriever(
            memory=self.memory,
            embedding_model=self.embedding_model
        )
        
        self.memory_system = {
            "memory": self.memory,
            "encoder": self.encoder,
            "retriever": self.retriever,
            "embedding_model": self.embedding_model
        }
        
        # Test data
        self.personal_queries = []
        self.factual_queries = []
        self.all_memories = []
        self.relevance_judgments = {}
        
    def load_test_data(self, queries_file="datasets/evaluation_queries.json"):
        """
        Load test data and separate into personal and factual queries.
        
        Args:
            queries_file: Path to the evaluation queries file
        """
        print(f"Loading test data from {queries_file}")
        
        try:
            with open(queries_file) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File {queries_file} not found")
            return
            
        # Separate queries by type
        for item in data:
            query = item["query"]
            expected = item["expected_answer"]
            category = item["category"]
            
            # Use NLP extractor to classify query type
            query_types = self.nlp_extractor.identify_query_type(query)
            primary_type = max(query_types.items(), key=lambda x: x[1])[0]
            
            # Store query with metadata
            query_item = {
                "query": query,
                "expected": expected,
                "category": category,
                "detected_type": primary_type,
                "type_scores": query_types
            }
            
            # Add to appropriate list based on detected type
            if primary_type == "personal":
                self.personal_queries.append(query_item)
            else:
                self.factual_queries.append(query_item)
                
            # Add expected answer to memories
            embedding, metadata = self.encoder.encode_concept(
                concept=category,
                description=expected,
                related_concepts=[primary_type]
            )
            
            memory_id = len(self.all_memories)
            self.memory.add_memory(embedding, expected, metadata)
            self.all_memories.append({
                "id": memory_id,
                "text": expected,
                "category": category,
                "embedding": embedding
            })
            
            # Create relevance judgment
            query_id = len(self.personal_queries) + len(self.factual_queries) - 1
            self.relevance_judgments[(query_id, memory_id)] = 1.0
            
        print(f"Loaded {len(self.personal_queries)} personal queries and {len(self.factual_queries)} factual queries")
        print(f"Added {len(self.all_memories)} memories to the system")
        
    def analyze_query_classification(self):
        """
        Analyze the accuracy of query classification.
        """
        print("\nAnalyzing query classification...")
        
        # Combine all queries
        all_queries = self.personal_queries + self.factual_queries
        
        # Check classification against category
        correct_count = 0
        misclassified = []
        
        for query in all_queries:
            # Simple heuristic: personal queries should be classified as personal,
            # and general/domain-specific queries should be classified as factual
            expected_type = "personal" if query["category"] == "personal" else "factual"
            detected_type = query["detected_type"]
            
            if expected_type == detected_type:
                correct_count += 1
            else:
                misclassified.append({
                    "query": query["query"],
                    "expected_type": expected_type,
                    "detected_type": detected_type,
                    "type_scores": query["type_scores"]
                })
        
        accuracy = correct_count / len(all_queries) if all_queries else 0
        print(f"Query classification accuracy: {accuracy:.2%}")
        
        # Print misclassified queries
        if misclassified:
            print("\nMisclassified queries:")
            for i, item in enumerate(misclassified[:5]):  # Show first 5
                print(f"{i+1}. Query: \"{item['query']}\"")
                print(f"   Expected: {item['expected_type']}, Detected: {item['detected_type']}")
                print(f"   Scores: {item['type_scores']}")
                
        # Save results
        with open("diagnostic_output/query_classification.json", "w") as f:
            json.dump({
                "accuracy": accuracy,
                "misclassified": misclassified
            }, f, indent=2)
            
        return accuracy, misclassified
        
    def analyze_embedding_space(self):
        """
        Analyze the embedding space to understand memory clustering.
        """
        print("\nAnalyzing embedding space...")
        
        # Extract embeddings
        memory_embeddings = np.array([m["embedding"] for m in self.all_memories])
        
        # Create query embeddings
        all_queries = self.personal_queries + self.factual_queries
        query_embeddings = np.array([
            self.embedding_model.encode(q["query"]) for q in all_queries
        ])
        
        # Combine embeddings for visualization
        combined_embeddings = np.vstack([memory_embeddings, query_embeddings])
        
        # Create labels
        memory_labels = [m["category"] for m in self.all_memories]
        query_labels = [f"Query ({q['category']})" for q in all_queries]
        combined_labels = memory_labels + query_labels
        
        # Use t-SNE to reduce dimensionality for visualization
        print("Performing t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_embeddings)-1))
        reduced_embeddings = tsne.fit_transform(combined_embeddings)
        
        # Split back into memories and queries
        memory_points = reduced_embeddings[:len(memory_embeddings)]
        query_points = reduced_embeddings[len(memory_embeddings):]
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Plot memories
        categories = set(memory_labels)
        category_colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        category_to_color = dict(zip(categories, category_colors))
        
        for category in categories:
            indices = [i for i, label in enumerate(memory_labels) if label == category]
            plt.scatter(
                memory_points[indices, 0],
                memory_points[indices, 1],
                label=f"Memory ({category})",
                alpha=0.7
            )
        
        # Plot queries
        query_categories = set(q["category"] for q in all_queries)
        for category in query_categories:
            indices = [i for i, q in enumerate(all_queries) if q["category"] == category]
            plt.scatter(
                query_points[indices, 0],
                query_points[indices, 1],
                marker='X',
                s=100,
                label=f"Query ({category})",
                alpha=0.9
            )
        
        plt.title("t-SNE Visualization of Embedding Space")
        plt.legend()
        plt.tight_layout()
        plt.savefig("diagnostic_output/embedding_space.png")
        
        # Calculate and analyze distances
        print("Analyzing distances in embedding space...")
        
        # Calculate distances between queries and their expected memories
        query_to_memory_distances = []
        
        for query_idx, query in enumerate(all_queries):
            query_embedding = query_embeddings[query_idx]
            
            # Find the memory containing the expected answer
            expected_memory_indices = []
            for memory_idx, memory in enumerate(self.all_memories):
                if query["expected"] in memory["text"]:
                    expected_memory_indices.append(memory_idx)
            
            if expected_memory_indices:
                # Calculate distances to expected memories
                for memory_idx in expected_memory_indices:
                    memory_embedding = memory_embeddings[memory_idx]
                    similarity = np.dot(query_embedding, memory_embedding)
                    distance = 1 - similarity
                    
                    query_to_memory_distances.append({
                        "query": query["query"],
                        "query_category": query["category"],
                        "query_type": query["detected_type"],
                        "memory_text": self.all_memories[memory_idx]["text"][:50] + "...",
                        "memory_category": self.all_memories[memory_idx]["category"],
                        "similarity": float(similarity),
                        "distance": float(distance)
                    })
        
        # Analyze distances by query type
        personal_distances = [d for d in query_to_memory_distances if d["query_type"] == "personal"]
        factual_distances = [d for d in query_to_memory_distances if d["query_type"] != "personal"]
        
        avg_personal_similarity = np.mean([d["similarity"] for d in personal_distances]) if personal_distances else 0
        avg_factual_similarity = np.mean([d["similarity"] for d in factual_distances]) if factual_distances else 0
        
        print(f"Average similarity for personal queries: {avg_personal_similarity:.3f}")
        print(f"Average similarity for factual queries: {avg_factual_similarity:.3f}")
        
        # Save results
        with open("diagnostic_output/embedding_analysis.json", "w") as f:
            json.dump({
                "query_to_memory_distances": query_to_memory_distances,
                "avg_personal_similarity": float(avg_personal_similarity),
                "avg_factual_similarity": float(avg_factual_similarity)
            }, f, indent=2)
        
        return query_to_memory_distances
    
    def analyze_threshold_optimization(self):
        """
        Analyze optimal threshold settings for different query types.
        """
        print("\nAnalyzing threshold optimization...")
        
        # Combine all queries
        all_queries = self.personal_queries + self.factual_queries
        
        # Test different threshold values
        thresholds = np.linspace(0.1, 0.9, 9)
        results = []
        
        for threshold in thresholds:
            # Set retriever threshold
            self.retriever.confidence_threshold = threshold
            
            # Test personal queries
            personal_metrics = self._test_queries(self.personal_queries)
            
            # Test factual queries
            factual_metrics = self._test_queries(self.factual_queries)
            
            # Store results
            results.append({
                "threshold": float(threshold),
                "personal": personal_metrics,
                "factual": factual_metrics,
                "overall": {
                    "precision": (personal_metrics["precision"] * len(self.personal_queries) + 
                                 factual_metrics["precision"] * len(self.factual_queries)) / len(all_queries),
                    "recall": (personal_metrics["recall"] * len(self.personal_queries) + 
                              factual_metrics["recall"] * len(self.factual_queries)) / len(all_queries),
                    "f1": (personal_metrics["f1"] * len(self.personal_queries) + 
                          factual_metrics["f1"] * len(self.factual_queries)) / len(all_queries)
                }
            })
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Plot precision
        plt.subplot(3, 1, 1)
        plt.plot(thresholds, [r["personal"]["precision"] for r in results], 'b-', label="Personal")
        plt.plot(thresholds, [r["factual"]["precision"] for r in results], 'r-', label="Factual")
        plt.plot(thresholds, [r["overall"]["precision"] for r in results], 'g-', label="Overall")
        plt.title("Precision vs. Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot recall
        plt.subplot(3, 1, 2)
        plt.plot(thresholds, [r["personal"]["recall"] for r in results], 'b-', label="Personal")
        plt.plot(thresholds, [r["factual"]["recall"] for r in results], 'r-', label="Factual")
        plt.plot(thresholds, [r["overall"]["recall"] for r in results], 'g-', label="Overall")
        plt.title("Recall vs. Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Recall")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot F1
        plt.subplot(3, 1, 3)
        plt.plot(thresholds, [r["personal"]["f1"] for r in results], 'b-', label="Personal")
        plt.plot(thresholds, [r["factual"]["f1"] for r in results], 'r-', label="Factual")
        plt.plot(thresholds, [r["overall"]["f1"] for r in results], 'g-', label="Overall")
        plt.title("F1 Score vs. Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("diagnostic_output/threshold_optimization.png")
        
        # Find optimal thresholds
        personal_f1_values = [r["personal"]["f1"] for r in results]
        factual_f1_values = [r["factual"]["f1"] for r in results]
        overall_f1_values = [r["overall"]["f1"] for r in results]
        
        optimal_personal_idx = np.argmax(personal_f1_values)
        optimal_factual_idx = np.argmax(factual_f1_values)
        optimal_overall_idx = np.argmax(overall_f1_values)
        
        optimal_thresholds = {
            "personal": {
                "threshold": float(thresholds[optimal_personal_idx]),
                "f1": float(personal_f1_values[optimal_personal_idx]),
                "precision": float(results[optimal_personal_idx]["personal"]["precision"]),
                "recall": float(results[optimal_personal_idx]["personal"]["recall"])
            },
            "factual": {
                "threshold": float(thresholds[optimal_factual_idx]),
                "f1": float(factual_f1_values[optimal_factual_idx]),
                "precision": float(results[optimal_factual_idx]["factual"]["precision"]),
                "recall": float(results[optimal_factual_idx]["factual"]["recall"])
            },
            "overall": {
                "threshold": float(thresholds[optimal_overall_idx]),
                "f1": float(overall_f1_values[optimal_overall_idx]),
                "precision": float(results[optimal_overall_idx]["overall"]["precision"]),
                "recall": float(results[optimal_overall_idx]["overall"]["recall"])
            }
        }
        
        print("Optimal thresholds:")
        print(f"Personal queries: {optimal_thresholds['personal']['threshold']:.2f} (F1: {optimal_thresholds['personal']['f1']:.3f})")
        print(f"Factual queries: {optimal_thresholds['factual']['threshold']:.2f} (F1: {optimal_thresholds['factual']['f1']:.3f})")
        print(f"Overall: {optimal_thresholds['overall']['threshold']:.2f} (F1: {optimal_thresholds['overall']['f1']:.3f})")
        
        # Save results
        with open("diagnostic_output/threshold_optimization.json", "w") as f:
            json.dump({
                "results": results,
                "optimal_thresholds": optimal_thresholds
            }, f, indent=2)
        
        return optimal_thresholds
    
    def analyze_failed_queries(self):
        """
        Analyze specific failed queries to understand why they're failing.
        """
        print("\nAnalyzing failed queries...")
        
        # Combine all queries
        all_queries = self.personal_queries + self.factual_queries
        
        # Use a moderate threshold
        self.retriever.confidence_threshold = 0.3
        
        # Test each query and identify failures
        failed_queries = []
        
        for query_idx, query in enumerate(all_queries):
            # Retrieve memories
            retrieved = self.retriever.retrieve_for_context(query["query"], top_k=5)
            retrieved_indices = [item.get("memory_id") for item in retrieved 
                               if isinstance(item.get("memory_id"), int)]
            
            # Find expected memories
            expected_indices = []
            for memory_idx, memory in enumerate(self.all_memories):
                if query["expected"] in memory["text"]:
                    expected_indices.append(memory_idx)
            
            # Check if retrieval failed
            if not any(idx in retrieved_indices for idx in expected_indices):
                # This is a failed query - analyze it
                query_embedding = self.embedding_model.encode(query["query"])
                
                # Calculate similarities with all memories
                similarities = np.dot(self.memory.memory_embeddings, query_embedding)
                
                # Get ranks of expected memories
                expected_ranks = []
                for idx in expected_indices:
                    rank = sum(1 for s in similarities if s > similarities[idx])
                    expected_ranks.append(rank)
                
                # Get top retrieved memories
                top_indices = np.argsort(-similarities)[:5]
                top_memories = [
                    {
                        "memory_id": int(idx),
                        "text": self.all_memories[idx]["text"][:50] + "...",
                        "similarity": float(similarities[idx])
                    }
                    for idx in top_indices
                ]
                
                # Get expected memories
                expected_memories = [
                    {
                        "memory_id": int(idx),
                        "text": self.all_memories[idx]["text"][:50] + "...",
                        "similarity": float(similarities[idx]),
                        "rank": rank
                    }
                    for idx, rank in zip(expected_indices, expected_ranks)
                ]
                
                failed_queries.append({
                    "query": query["query"],
                    "category": query["category"],
                    "detected_type": query["detected_type"],
                    "expected": query["expected"],
                    "top_retrieved": top_memories,
                    "expected_memories": expected_memories,
                    "threshold_used": self.retriever.confidence_threshold
                })
        
        # Print summary
        print(f"Found {len(failed_queries)} failed queries out of {len(all_queries)}")
        
        # Print some examples
        if failed_queries:
            print("\nExample failed queries:")
            for i, item in enumerate(failed_queries[:3]):  # Show first 3
                print(f"{i+1}. Query: \"{item['query']}\" ({item['category']}, detected as {item['detected_type']})")
                print(f"   Expected: {item['expected']}")
                print("   Top retrieved memories:")
                for j, mem in enumerate(item["top_retrieved"]):
                    print(f"     {j+1}. {mem['text']} (sim: {mem['similarity']:.3f})")
                print("   Expected memories:")
                for j, mem in enumerate(item["expected_memories"]):
                    print(f"     {j+1}. {mem['text']} (sim: {mem['similarity']:.3f}, rank: {mem['rank']})")
                print()
        
        # Save results
        with open("diagnostic_output/failed_queries.json", "w") as f:
            json.dump(failed_queries, f, indent=2)
        
        return failed_queries
    
    def _test_queries(self, queries):
        """
        Test a set of queries and calculate metrics.
        
        Args:
            queries: List of query objects to test
            
        Returns:
            Dictionary with precision, recall, and F1 metrics
        """
        if not queries:
            return {"precision": 0, "recall": 0, "f1": 0}
        
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        
        for query in queries:
            # Retrieve memories
            retrieved = self.retriever.retrieve_for_context(query["query"], top_k=5)
            retrieved_indices = [item.get("memory_id") for item in retrieved 
                               if isinstance(item.get("memory_id"), int)]
            
            # Find expected memories
            expected_indices = []
            for memory_idx, memory in enumerate(self.all_memories):
                if query["expected"] in memory["text"]:
                    expected_indices.append(memory_idx)
            
            # Calculate precision and recall
            if retrieved_indices:
                precision = sum(1 for idx in retrieved_indices if idx in expected_indices) / len(retrieved_indices)
            else:
                precision = 0
                
            if expected_indices:
                recall = sum(1 for idx in retrieved_indices if idx in expected_indices) / len(expected_indices)
            else:
                recall = 1  # No expected memories, so perfect recall
            
            # Calculate F1
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
        
        # Calculate averages
        avg_precision = total_precision / len(queries)
        avg_recall = total_recall / len(queries)
        avg_f1 = total_f1 / len(queries)
        
        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1
        }
    
    def run_all_analyses(self):
        """
        Run all diagnostic analyses and generate a comprehensive report.
        """
        print("Running all diagnostic analyses...")
        
        # Run individual analyses
        query_classification_results = self.analyze_query_classification()
        embedding_analysis_results = self.analyze_embedding_space()
        threshold_optimization_results = self.analyze_threshold_optimization()
        failed_query_analysis = self.analyze_failed_queries()
        
        # Generate summary report
        summary = {
            "query_classification": {
                "accuracy": query_classification_results[0],
                "misclassified_count": len(query_classification_results[1])
            },
            "embedding_analysis": {
                "avg_personal_similarity": np.mean([d["similarity"] for d in embedding_analysis_results if d["query_type"] == "personal"]) if embedding_analysis_results else 0,
                "avg_factual_similarity": np.mean([d["similarity"] for d in embedding_analysis_results if d["query_type"] != "personal"]) if embedding_analysis_results else 0
            },
            "threshold_optimization": threshold_optimization_results,
            "failed_queries": {
                "count": len(failed_query_analysis),
                "personal_count": sum(1 for q in failed_query_analysis if q["detected_type"] == "personal"),
                "factual_count": sum(1 for q in failed_query_analysis if q["detected_type"] != "personal")
            }
        }
        
        # Save summary
        with open("diagnostic_output/summary_report.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print key findings
        print("\n=== Key Findings ===")
        print(f"Query classification accuracy: {summary['query_classification']['accuracy']:.2%}")
        print(f"Average similarity for personal queries: {summary['embedding_analysis']['avg_personal_similarity']:.3f}")
        print(f"Average similarity for factual queries: {summary['embedding_analysis']['avg_factual_similarity']:.3f}")
        print(f"Optimal threshold for personal queries: {summary['threshold_optimization']['personal']['threshold']:.2f} (F1: {summary['threshold_optimization']['personal']['f1']:.3f})")
        print(f"Optimal threshold for factual queries: {summary['threshold_optimization']['factual']['threshold']:.2f} (F1: {summary['threshold_optimization']['factual']['f1']:.3f})")
        print(f"Failed queries: {summary['failed_queries']['count']} total ({summary['failed_queries']['personal_count']} personal, {summary['failed_queries']['factual_count']} factual)")
        
        return summary


def main():
    """Run the diagnostic analysis."""
    print("MemoryWeave Diagnostic Analysis")
    print("===============================")
    
    # Initialize diagnostic analysis
    diagnostic = DiagnosticAnalysis()
    
    # Load test data
    diagnostic.load_test_data()
    
    # Run all analyses
    summary = diagnostic.run_all_analyses()
    
    print("\nDiagnostic analysis complete. Results saved to diagnostic_output directory.")


if __name__ == "__main__":
    main()
