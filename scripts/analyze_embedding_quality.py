"""
Analyze Embedding Quality for MemoryWeave

This script analyzes the quality of embeddings by examining how well related memories
cluster together in the embedding space, which is crucial for effective retrieval.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from transformers import AutoTokenizer, AutoModel

from memoryweave.core import ContextualMemory, MemoryEncoder

# Create output directory if it doesn't exist
os.makedirs("diagnostic_output/embedding_quality", exist_ok=True)

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

class EmbeddingQualityAnalyzer:
    """
    Analyzes the quality of embeddings by examining how well related memories cluster together.
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding quality analyzer.
        
        Args:
            model_name: Name of the embedding model to use
        """
        print(f"Loading embedding model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        self.embedding_model = EmbeddingModelWrapper(model, tokenizer)
        self.embedding_dim = model.config.hidden_size
        
        # Initialize memory components
        self.memory = ContextualMemory(embedding_dim=self.embedding_dim)
        self.encoder = MemoryEncoder(self.embedding_model)
        
        # Test data
        self.test_sets = []
        
    def create_test_sets(self):
        """
        Create test sets of related and unrelated memories.
        """
        print("Creating test sets of related and unrelated memories...")
        
        # Test set 1: Personal information
        personal_set = [
            "My name is Alex Thompson",
            "I live in Seattle, Washington",
            "I work as a software engineer",
            "I have a dog named Max",
            "My favorite color is blue"
        ]
        
        # Test set 2: Travel information
        travel_set = [
            "I visited Paris last summer",
            "The Eiffel Tower was amazing",
            "French cuisine is delicious",
            "I stayed in a hotel near the Seine",
            "I want to visit Rome next"
        ]
        
        # Test set 3: Technical information
        tech_set = [
            "Python is my favorite programming language",
            "I'm learning about machine learning algorithms",
            "Neural networks are fascinating",
            "I use TensorFlow for deep learning projects",
            "Data preprocessing is an important step in ML"
        ]
        
        # Test set 4: Mixed topics
        mixed_set = [
            "The weather was nice yesterday",
            "Quantum physics explains particle behavior",
            "I enjoy playing basketball on weekends",
            "The stock market fluctuated this week",
            "Renewable energy is important for sustainability"
        ]
        
        self.test_sets = [
            {"name": "Personal Information", "memories": personal_set, "should_cluster": True},
            {"name": "Travel Information", "memories": travel_set, "should_cluster": True},
            {"name": "Technical Information", "memories": tech_set, "should_cluster": True},
            {"name": "Mixed Topics", "memories": mixed_set, "should_cluster": False}
        ]
        
        # Add all memories to the system
        for test_set in self.test_sets:
            test_set["embeddings"] = []
            test_set["memory_ids"] = []
            
            for memory_text in test_set["memories"]:
                embedding, metadata = self.encoder.encode_concept(
                    concept=test_set["name"],
                    description=memory_text
                )
                
                memory_id = len(self.memory.memory_embeddings)
                self.memory.add_memory(embedding, memory_text, metadata)
                
                test_set["embeddings"].append(embedding)
                test_set["memory_ids"].append(memory_id)
        
        print(f"Created {len(self.test_sets)} test sets with a total of {len(self.memory.memory_embeddings)} memories")
    
    def analyze_intra_set_similarity(self):
        """
        Analyze similarity within each test set.
        """
        print("\nAnalyzing intra-set similarity...")
        
        results = []
        
        for test_set in self.test_sets:
            embeddings = np.array(test_set["embeddings"])
            
            # Calculate pairwise similarities
            similarity_matrix = np.dot(embeddings, embeddings.T)
            
            # Calculate average similarity (excluding self-similarity)
            n = len(embeddings)
            total_similarity = similarity_matrix.sum() - n  # Subtract diagonal (self-similarity)
            avg_similarity = total_similarity / (n * (n - 1))  # Divide by number of pairs
            
            # Calculate min and max similarity (excluding self-similarity)
            np.fill_diagonal(similarity_matrix, -1)  # Replace diagonal with -1
            max_similarity = similarity_matrix.max()
            min_similarity = similarity_matrix[similarity_matrix > -1].min()  # Exclude diagonal
            
            # Store results
            test_set["similarity_analysis"] = {
                "avg_similarity": float(avg_similarity),
                "min_similarity": float(min_similarity),
                "max_similarity": float(max_similarity),
                "similarity_matrix": similarity_matrix.tolist()
            }
            
            results.append({
                "set_name": test_set["name"],
                "should_cluster": test_set["should_cluster"],
                "avg_similarity": float(avg_similarity),
                "min_similarity": float(min_similarity),
                "max_similarity": float(max_similarity)
            })
            
            print(f"{test_set['name']}:")
            print(f"  Average similarity: {avg_similarity:.3f}")
            print(f"  Min similarity: {min_similarity:.3f}")
            print(f"  Max similarity: {max_similarity:.3f}")
            
            # Plot similarity matrix
            plt.figure(figsize=(8, 6))
            plt.imshow(similarity_matrix, cmap='viridis')
            plt.colorbar(label='Similarity')
            plt.title(f"Similarity Matrix: {test_set['name']}")
            plt.tight_layout()
            plt.savefig(f"diagnostic_output/embedding_quality/similarity_{test_set['name'].replace(' ', '_')}.png")
        
        # Save results
        with open("diagnostic_output/embedding_quality/intra_set_similarity.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def analyze_inter_set_similarity(self):
        """
        Analyze similarity between different test sets.
        """
        print("\nAnalyzing inter-set similarity...")
        
        results = []
        
        # Calculate similarity between each pair of test sets
        for i, set1 in enumerate(self.test_sets):
            for j, set2 in enumerate(self.test_sets[i+1:], i+1):
                embeddings1 = np.array(set1["embeddings"])
                embeddings2 = np.array(set2["embeddings"])
                
                # Calculate cross-set similarities
                cross_similarity = np.dot(embeddings1, embeddings2.T)
                
                # Calculate statistics
                avg_similarity = cross_similarity.mean()
                min_similarity = cross_similarity.min()
                max_similarity = cross_similarity.max()
                
                result = {
                    "set1": set1["name"],
                    "set2": set2["name"],
                    "avg_similarity": float(avg_similarity),
                    "min_similarity": float(min_similarity),
                    "max_similarity": float(max_similarity)
                }
                
                results.append(result)
                
                print(f"{set1['name']} vs {set2['name']}:")
                print(f"  Average similarity: {avg_similarity:.3f}")
                print(f"  Min similarity: {min_similarity:.3f}")
                print(f"  Max similarity: {max_similarity:.3f}")
                
                # Plot cross-similarity matrix
                plt.figure(figsize=(8, 6))
                plt.imshow(cross_similarity, cmap='viridis')
                plt.colorbar(label='Similarity')
                plt.title(f"Cross-Similarity: {set1['name']} vs {set2['name']}")
                plt.xlabel(set2["name"])
                plt.ylabel(set1["name"])
                plt.tight_layout()
                plt.savefig(f"diagnostic_output/embedding_quality/cross_similarity_{set1['name'].replace(' ', '_')}_{set2['name'].replace(' ', '_')}.png")
        
        # Save results
        with open("diagnostic_output/embedding_quality/inter_set_similarity.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def analyze_clustering_quality(self):
        """
        Analyze how well memories cluster in the embedding space.
        """
        print("\nAnalyzing clustering quality...")
        
        # Combine all embeddings
        all_embeddings = []
        all_labels = []
        
        for i, test_set in enumerate(self.test_sets):
            all_embeddings.extend(test_set["embeddings"])
            all_labels.extend([i] * len(test_set["embeddings"]))
        
        all_embeddings = np.array(all_embeddings)
        all_labels = np.array(all_labels)
        
        # Try different clustering algorithms
        results = {}
        
        # K-means clustering
        kmeans = KMeans(n_clusters=len(self.test_sets), random_state=42)
        kmeans_labels = kmeans.fit_predict(all_embeddings)
        
        # Calculate clustering metrics
        if len(set(kmeans_labels)) > 1:  # Ensure we have at least 2 clusters
            kmeans_silhouette = silhouette_score(all_embeddings, kmeans_labels)
        else:
            kmeans_silhouette = 0
        
        # Calculate cluster purity (how well clusters match original sets)
        kmeans_purity = self._calculate_purity(all_labels, kmeans_labels)
        
        results["kmeans"] = {
            "silhouette_score": float(kmeans_silhouette),
            "purity": float(kmeans_purity)
        }
        
        print(f"K-means clustering:")
        print(f"  Silhouette score: {kmeans_silhouette:.3f}")
        print(f"  Purity: {kmeans_purity:.3f}")
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan_labels = dbscan.fit_predict(all_embeddings)
        
        # Calculate clustering metrics
        if len(set(dbscan_labels)) > 1 and -1 not in dbscan_labels:  # Ensure we have at least 2 clusters and no noise
            dbscan_silhouette = silhouette_score(all_embeddings, dbscan_labels)
        else:
            dbscan_silhouette = 0
        
        # Calculate cluster purity
        if len(set(dbscan_labels)) > 1:  # Only if we have actual clusters
            dbscan_purity = self._calculate_purity(all_labels, dbscan_labels)
        else:
            dbscan_purity = 0
        
        results["dbscan"] = {
            "silhouette_score": float(dbscan_silhouette),
            "purity": float(dbscan_purity),
            "num_clusters": len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            "noise_points": sum(1 for label in dbscan_labels if label == -1)
        }
        
        print(f"DBSCAN clustering:")
        print(f"  Silhouette score: {dbscan_silhouette:.3f}")
        print(f"  Purity: {dbscan_purity:.3f}")
        print(f"  Number of clusters: {results['dbscan']['num_clusters']}")
        print(f"  Noise points: {results['dbscan']['noise_points']}")
        
        # Save results
        with open("diagnostic_output/embedding_quality/clustering_quality.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _calculate_purity(self, true_labels, cluster_labels):
        """
        Calculate cluster purity (how well clusters match original labels).
        
        Args:
            true_labels: Ground truth labels
            cluster_labels: Cluster assignments
            
        Returns:
            Purity score between 0 and 1
        """
        # Create contingency matrix
        contingency = np.zeros((max(cluster_labels) + 1, max(true_labels) + 1))
        
        for i in range(len(true_labels)):
            contingency[cluster_labels[i], true_labels[i]] += 1
        
        # Calculate purity
        return sum(np.max(contingency, axis=1)) / len(true_labels)
    
    def analyze_query_retrieval(self):
        """
        Analyze how well queries retrieve related memories.
        """
        print("\nAnalyzing query retrieval...")
        
        from memoryweave.core import ContextualRetriever
        
        # Initialize retriever
        retriever = ContextualRetriever(
            memory=self.memory,
            embedding_model=self.embedding_model,
            confidence_threshold=0.3
        )
        
        results = []
        
        # Create test queries for each set
        for test_set in self.test_sets:
            # Create a query that should match this set
            query = f"Tell me about {test_set['name'].lower()}"
            
            # Retrieve memories
            retrieved = retriever.retrieve_for_context(query, top_k=10)
            retrieved_ids = [item.get("memory_id") for item in retrieved 
                           if isinstance(item.get("memory_id"), int)]
            
            # Calculate precision (how many retrieved items are from this set)
            if retrieved_ids:
                precision = sum(1 for idx in retrieved_ids if idx in test_set["memory_ids"]) / len(retrieved_ids)
            else:
                precision = 0
                
            # Calculate recall (how many items from this set were retrieved)
            if test_set["memory_ids"]:
                recall = sum(1 for idx in retrieved_ids if idx in test_set["memory_ids"]) / len(test_set["memory_ids"])
            else:
                recall = 0
                
            # Calculate F1
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            result = {
                "set_name": test_set["name"],
                "query": query,
                "retrieved_count": len(retrieved_ids),
                "expected_count": len(test_set["memory_ids"]),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "retrieved_ids": retrieved_ids,
                "expected_ids": test_set["memory_ids"]
            }
            
            results.append(result)
            
            print(f"Query: \"{query}\"")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1: {f1:.3f}")
            print(f"  Retrieved {len(retrieved_ids)} memories, {sum(1 for idx in retrieved_ids if idx in test_set['memory_ids'])} from expected set")
        
        # Save results
        with open("diagnostic_output/embedding_quality/query_retrieval.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_all_analyses(self):
        """
        Run all embedding quality analyses.
        """
        print("Running all embedding quality analyses...")
        
        # Create test sets
        self.create_test_sets()
        
        # Run analyses
        intra_set_results = self.analyze_intra_set_similarity()
        inter_set_results = self.analyze_inter_set_similarity()
        clustering_results = self.analyze_clustering_quality()
        retrieval_results = self.analyze_query_retrieval()
        
        # Generate summary
        summary = {
            "intra_set_similarity": {
                "avg_similarity_related": np.mean([r["avg_similarity"] for r in intra_set_results if r["should_cluster"]]),
                "avg_similarity_unrelated": np.mean([r["avg_similarity"] for r in intra_set_results if not r["should_cluster"]])
            },
            "inter_set_similarity": {
                "avg_cross_similarity": np.mean([r["avg_similarity"] for r in inter_set_results])
            },
            "clustering_quality": clustering_results,
            "retrieval_quality": {
                "avg_precision": np.mean([r["precision"] for r in retrieval_results]),
                "avg_recall": np.mean([r["recall"] for r in retrieval_results]),
                "avg_f1": np.mean([r["f1"] for r in retrieval_results])
            }
        }
        
        # Save summary
        with open("diagnostic_output/embedding_quality/summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print key findings
        print("\n=== Key Findings ===")
        print(f"Average similarity within related sets: {summary['intra_set_similarity']['avg_similarity_related']:.3f}")
        print(f"Average similarity within unrelated sets: {summary['intra_set_similarity']['avg_similarity_unrelated']:.3f}")
        print(f"Average similarity between different sets: {summary['inter_set_similarity']['avg_cross_similarity']:.3f}")
        print(f"K-means clustering purity: {clustering_results['kmeans']['purity']:.3f}")
        print(f"Average retrieval precision: {summary['retrieval_quality']['avg_precision']:.3f}")
        print(f"Average retrieval recall: {summary['retrieval_quality']['avg_recall']:.3f}")
        print(f"Average retrieval F1: {summary['retrieval_quality']['avg_f1']:.3f}")
        
        return summary


def main():
    """Run the embedding quality analysis."""
    print("MemoryWeave Embedding Quality Analysis")
    print("=====================================")
    
    # Initialize analyzer
    analyzer = EmbeddingQualityAnalyzer()
    
    # Run all analyses
    summary = analyzer.run_all_analyses()
    
    print("\nEmbedding quality analysis complete. Results saved to diagnostic_output/embedding_quality directory.")


if __name__ == "__main__":
    main()
