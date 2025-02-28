"""
Analyze Similarity Distributions for MemoryWeave

This script analyzes the distributions of similarity scores between queries and memories
to identify optimal threshold settings and understand retrieval behavior.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from memoryweave.core import ContextualMemory, MemoryEncoder
from memoryweave.utils.nlp_extraction import NLPExtractor

# Create output directory if it doesn't exist
os.makedirs("diagnostic_output/similarity_distributions", exist_ok=True)


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


class SimilarityDistributionAnalyzer:
    """
    Analyzes the distributions of similarity scores between queries and memories.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the similarity distribution analyzer.

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

        # Test data
        self.personal_queries = []
        self.factual_queries = []
        self.all_memories = []

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
            # Create synthetic test data if file not found
            self._create_synthetic_data()
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
                "type_scores": query_types,
            }

            # Add to appropriate list based on detected type
            if primary_type == "personal":
                self.personal_queries.append(query_item)
            else:
                self.factual_queries.append(query_item)

            # Add expected answer to memories
            embedding, metadata = self.encoder.encode_concept(
                concept=category, description=expected, related_concepts=[primary_type]
            )

            memory_id = len(self.all_memories)
            self.memory.add_memory(embedding, expected, metadata)
            self.all_memories.append(
                {"id": memory_id, "text": expected, "category": category, "embedding": embedding}
            )

        print(
            f"Loaded {len(self.personal_queries)} personal queries and {len(self.factual_queries)} factual queries"
        )
        print(f"Added {len(self.all_memories)} memories to the system")

    def _create_synthetic_data(self):
        """
        Create synthetic test data if no file is provided.
        """
        print("Creating synthetic test data...")

        # Personal queries
        personal_queries = [
            {"query": "What is my name?", "expected": "Your name is Alex Thompson."},
            {"query": "Where do I live?", "expected": "You live in Seattle, Washington."},
            {"query": "What is my favorite color?", "expected": "Your favorite color is blue."},
            {"query": "What is my job?", "expected": "You work as a software engineer."},
            {"query": "Do I have any pets?", "expected": "You have a dog named Max."},
        ]

        # Factual queries
        factual_queries = [
            {
                "query": "What is the capital of France?",
                "expected": "The capital of France is Paris.",
            },
            {"query": "Who wrote Hamlet?", "expected": "William Shakespeare wrote Hamlet."},
            {
                "query": "What is the tallest mountain?",
                "expected": "Mount Everest is the tallest mountain on Earth.",
            },
            {
                "query": "What is the speed of light?",
                "expected": "The speed of light is approximately 299,792,458 meters per second.",
            },
            {
                "query": "When was the Declaration of Independence signed?",
                "expected": "The Declaration of Independence was signed on July 4, 1776.",
            },
        ]

        # Process personal queries
        for item in personal_queries:
            query_types = self.nlp_extractor.identify_query_type(item["query"])
            primary_type = max(query_types.items(), key=lambda x: x[1])[0]

            query_item = {
                "query": item["query"],
                "expected": item["expected"],
                "category": "personal",
                "detected_type": primary_type,
                "type_scores": query_types,
            }

            self.personal_queries.append(query_item)

            # Add to memory
            embedding, metadata = self.encoder.encode_concept(
                concept="personal", description=item["expected"], related_concepts=[primary_type]
            )

            memory_id = len(self.all_memories)
            self.memory.add_memory(embedding, item["expected"], metadata)
            self.all_memories.append(
                {
                    "id": memory_id,
                    "text": item["expected"],
                    "category": "personal",
                    "embedding": embedding,
                }
            )

        # Process factual queries
        for item in factual_queries:
            query_types = self.nlp_extractor.identify_query_type(item["query"])
            primary_type = max(query_types.items(), key=lambda x: x[1])[0]

            query_item = {
                "query": item["query"],
                "expected": item["expected"],
                "category": "factual",
                "detected_type": primary_type,
                "type_scores": query_types,
            }

            self.factual_queries.append(query_item)

            # Add to memory
            embedding, metadata = self.encoder.encode_concept(
                concept="factual", description=item["expected"], related_concepts=[primary_type]
            )

            memory_id = len(self.all_memories)
            self.memory.add_memory(embedding, item["expected"], metadata)
            self.all_memories.append(
                {
                    "id": memory_id,
                    "text": item["expected"],
                    "category": "factual",
                    "embedding": embedding,
                }
            )

        print(
            f"Created {len(self.personal_queries)} personal queries and {len(self.factual_queries)} factual queries"
        )
        print(f"Added {len(self.all_memories)} memories to the system")

    def analyze_similarity_distributions(self):
        """
        Analyze the distributions of similarity scores between queries and memories.
        """
        print("\nAnalyzing similarity distributions...")

        # Combine all queries
        all_queries = self.personal_queries + self.factual_queries

        # Calculate similarity distributions
        similarity_data = {
            "personal": {"relevant": [], "irrelevant": []},
            "factual": {"relevant": [], "irrelevant": []},
        }

        # Process each query
        for query_type, queries in [
            ("personal", self.personal_queries),
            ("factual", self.factual_queries),
        ]:
            for query in queries:
                # Encode query
                query_embedding = self.embedding_model.encode(query["query"])

                # Calculate similarities with all memories
                similarities = np.dot(self.memory.memory_embeddings, query_embedding)

                # Find relevant memories (those containing the expected answer)
                relevant_indices = []
                for memory_idx, memory in enumerate(self.all_memories):
                    if query["expected"] in memory["text"]:
                        relevant_indices.append(memory_idx)

                # Separate relevant and irrelevant similarities
                relevant_similarities = similarities[relevant_indices] if relevant_indices else []
                irrelevant_indices = [
                    i for i in range(len(similarities)) if i not in relevant_indices
                ]
                irrelevant_similarities = (
                    similarities[irrelevant_indices] if irrelevant_indices else []
                )

                # Add to distributions
                similarity_data[query_type]["relevant"].extend(relevant_similarities)
                similarity_data[query_type]["irrelevant"].extend(irrelevant_similarities)

        # Calculate statistics
        stats = {}
        for query_type in ["personal", "factual"]:
            relevant = similarity_data[query_type]["relevant"]
            irrelevant = similarity_data[query_type]["irrelevant"]

            stats[query_type] = {
                "relevant": {
                    "count": len(relevant),
                    "mean": float(np.mean(relevant)) if relevant else 0,
                    "median": float(np.median(relevant)) if relevant else 0,
                    "std": float(np.std(relevant)) if relevant else 0,
                    "min": float(np.min(relevant)) if relevant else 0,
                    "max": float(np.max(relevant)) if relevant else 0,
                },
                "irrelevant": {
                    "count": len(irrelevant),
                    "mean": float(np.mean(irrelevant)) if irrelevant else 0,
                    "median": float(np.median(irrelevant)) if irrelevant else 0,
                    "std": float(np.std(irrelevant)) if irrelevant else 0,
                    "min": float(np.min(irrelevant)) if irrelevant else 0,
                    "max": float(np.max(irrelevant)) if irrelevant else 0,
                },
            }

            # Print statistics
            print(f"{query_type.capitalize()} queries:")
            print(
                f"  Relevant similarities: n={stats[query_type]['relevant']['count']}, mean={stats[query_type]['relevant']['mean']:.3f}, median={stats[query_type]['relevant']['median']:.3f}"
            )
            print(
                f"  Irrelevant similarities: n={stats[query_type]['irrelevant']['count']}, mean={stats[query_type]['irrelevant']['mean']:.3f}, median={stats[query_type]['irrelevant']['median']:.3f}"
            )

        # Plot distributions
        plt.figure(figsize=(12, 10))

        # Personal queries
        plt.subplot(2, 1, 1)
        if similarity_data["personal"]["relevant"]:
            plt.hist(
                similarity_data["personal"]["relevant"],
                bins=20,
                alpha=0.7,
                label="Relevant",
                color="green",
            )
        if similarity_data["personal"]["irrelevant"]:
            plt.hist(
                similarity_data["personal"]["irrelevant"],
                bins=20,
                alpha=0.7,
                label="Irrelevant",
                color="red",
            )
        plt.title("Similarity Distribution for Personal Queries")
        plt.xlabel("Similarity Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Factual queries
        plt.subplot(2, 1, 2)
        if similarity_data["factual"]["relevant"]:
            plt.hist(
                similarity_data["factual"]["relevant"],
                bins=20,
                alpha=0.7,
                label="Relevant",
                color="green",
            )
        if similarity_data["factual"]["irrelevant"]:
            plt.hist(
                similarity_data["factual"]["irrelevant"],
                bins=20,
                alpha=0.7,
                label="Irrelevant",
                color="red",
            )
        plt.title("Similarity Distribution for Factual Queries")
        plt.xlabel("Similarity Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("diagnostic_output/similarity_distributions/similarity_distributions.png")

        # Save data
        with open("diagnostic_output/similarity_distributions/similarity_data.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {
                "personal": {
                    "relevant": [float(x) for x in similarity_data["personal"]["relevant"]],
                    "irrelevant": [float(x) for x in similarity_data["personal"]["irrelevant"]],
                },
                "factual": {
                    "relevant": [float(x) for x in similarity_data["factual"]["relevant"]],
                    "irrelevant": [float(x) for x in similarity_data["factual"]["irrelevant"]],
                },
            }
            json.dump(serializable_data, f, indent=2)

        # Save statistics
        with open("diagnostic_output/similarity_distributions/similarity_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        return stats, similarity_data

    def analyze_optimal_thresholds(self, similarity_data):
        """
        Analyze optimal threshold settings based on similarity distributions.

        Args:
            similarity_data: Dictionary containing similarity distributions

        Returns:
            Dictionary with optimal thresholds and metrics
        """
        print("\nAnalyzing optimal thresholds...")

        results = {}

        for query_type in ["personal", "factual"]:
            relevant = np.array(similarity_data[query_type]["relevant"])
            irrelevant = np.array(similarity_data[query_type]["irrelevant"])

            if len(relevant) == 0 or len(irrelevant) == 0:
                print(f"Skipping {query_type} queries due to insufficient data")
                results[query_type] = {
                    "optimal_threshold": 0.5,  # Default
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "accuracy": 0,
                }
                continue

            # Create combined array with labels
            all_similarities = np.concatenate([relevant, irrelevant])
            all_labels = np.concatenate([np.ones(len(relevant)), np.zeros(len(irrelevant))])

            # Sort by similarity
            sorted_indices = np.argsort(all_similarities)
            sorted_similarities = all_similarities[sorted_indices]
            sorted_labels = all_labels[sorted_indices]

            # Try different thresholds
            thresholds = np.unique(sorted_similarities)
            best_threshold = 0
            best_f1 = 0
            best_metrics = {}

            for threshold in thresholds:
                # Predict using threshold
                predictions = (all_similarities >= threshold).astype(int)

                # Calculate metrics
                true_positives = np.sum((predictions == 1) & (all_labels == 1))
                false_positives = np.sum((predictions == 1) & (all_labels == 0))
                false_negatives = np.sum((predictions == 0) & (all_labels == 1))
                true_negatives = np.sum((predictions == 0) & (all_labels == 0))

                # Precision, recall, F1
                precision = (
                    true_positives / (true_positives + false_positives)
                    if (true_positives + false_positives) > 0
                    else 0
                )
                recall = (
                    true_positives / (true_positives + false_negatives)
                    if (true_positives + false_negatives) > 0
                    else 0
                )
                f1 = (
                    2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                )
                accuracy = (true_positives + true_negatives) / len(all_labels)

                # Check if this is the best F1 score
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_metrics = {
                        "threshold": float(threshold),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "accuracy": float(accuracy),
                        "true_positives": int(true_positives),
                        "false_positives": int(false_positives),
                        "false_negatives": int(false_negatives),
                        "true_negatives": int(true_negatives),
                    }

            results[query_type] = best_metrics

            print(f"{query_type.capitalize()} queries:")
            print(f"  Optimal threshold: {best_threshold:.3f}")
            print(f"  Precision: {best_metrics['precision']:.3f}")
            print(f"  Recall: {best_metrics['recall']:.3f}")
            print(f"  F1: {best_metrics['f1']:.3f}")
            print(f"  Accuracy: {best_metrics['accuracy']:.3f}")

        # Plot precision-recall curves
        plt.figure(figsize=(10, 8))

        for query_type, color in [("personal", "blue"), ("factual", "red")]:
            relevant = np.array(similarity_data[query_type]["relevant"])
            irrelevant = np.array(similarity_data[query_type]["irrelevant"])

            if len(relevant) == 0 or len(irrelevant) == 0:
                continue

            # Create combined array with labels
            all_similarities = np.concatenate([relevant, irrelevant])
            all_labels = np.concatenate([np.ones(len(relevant)), np.zeros(len(irrelevant))])

            # Calculate precision-recall curve
            thresholds = np.linspace(0, 1, 100)
            precision_values = []
            recall_values = []

            for threshold in thresholds:
                predictions = (all_similarities >= threshold).astype(int)

                true_positives = np.sum((predictions == 1) & (all_labels == 1))
                false_positives = np.sum((predictions == 1) & (all_labels == 0))
                false_negatives = np.sum((predictions == 0) & (all_labels == 1))

                precision = (
                    true_positives / (true_positives + false_positives)
                    if (true_positives + false_positives) > 0
                    else 1
                )
                recall = (
                    true_positives / (true_positives + false_negatives)
                    if (true_positives + false_negatives) > 0
                    else 0
                )

                precision_values.append(precision)
                recall_values.append(recall)

            # Plot precision-recall curve
            plt.plot(
                recall_values,
                precision_values,
                color=color,
                label=f"{query_type.capitalize()} Queries",
            )

            # Mark optimal threshold
            optimal_idx = np.abs(thresholds - results[query_type]["threshold"]).argmin()
            plt.scatter(
                recall_values[optimal_idx],
                precision_values[optimal_idx],
                color=color,
                s=100,
                marker="x",
            )
            plt.annotate(
                f"Threshold: {results[query_type]['threshold']:.2f}",
                (recall_values[optimal_idx], precision_values[optimal_idx]),
                xytext=(10, -10),
                textcoords="offset points",
                color=color,
            )

        plt.title("Precision-Recall Curves")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("diagnostic_output/similarity_distributions/precision_recall_curves.png")

        # Save results
        with open("diagnostic_output/similarity_distributions/optimal_thresholds.json", "w") as f:
            json.dump(results, f, indent=2)

        return results

    def analyze_threshold_impact(self, similarity_data):
        """
        Analyze the impact of different threshold settings on retrieval performance.

        Args:
            similarity_data: Dictionary containing similarity distributions

        Returns:
            Dictionary with threshold impact analysis
        """
        print("\nAnalyzing threshold impact...")

        results = {}

        for query_type in ["personal", "factual"]:
            relevant = np.array(similarity_data[query_type]["relevant"])
            irrelevant = np.array(similarity_data[query_type]["irrelevant"])

            if len(relevant) == 0 or len(irrelevant) == 0:
                print(f"Skipping {query_type} queries due to insufficient data")
                continue

            # Create combined array with labels
            all_similarities = np.concatenate([relevant, irrelevant])
            all_labels = np.concatenate([np.ones(len(relevant)), np.zeros(len(irrelevant))])

            # Try different thresholds
            thresholds = np.linspace(0, 1, 21)  # 0.0, 0.05, 0.1, ..., 1.0
            metrics = []

            for threshold in thresholds:
                # Predict using threshold
                predictions = (all_similarities >= threshold).astype(int)

                # Calculate metrics
                true_positives = np.sum((predictions == 1) & (all_labels == 1))
                false_positives = np.sum((predictions == 1) & (all_labels == 0))
                false_negatives = np.sum((predictions == 0) & (all_labels == 1))
                true_negatives = np.sum((predictions == 0) & (all_labels == 0))

                # Precision, recall, F1
                precision = (
                    true_positives / (true_positives + false_positives)
                    if (true_positives + false_positives) > 0
                    else 0
                )
                recall = (
                    true_positives / (true_positives + false_negatives)
                    if (true_positives + false_negatives) > 0
                    else 0
                )
                f1 = (
                    2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                )
                accuracy = (true_positives + true_negatives) / len(all_labels)

                metrics.append(
                    {
                        "threshold": float(threshold),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "accuracy": float(accuracy),
                        "retrieved_count": int(true_positives + false_positives),
                        "relevant_count": int(true_positives + false_negatives),
                    }
                )

            results[query_type] = metrics

        # Plot metrics vs threshold
        plt.figure(figsize=(12, 10))

        for i, metric_name in enumerate(["precision", "recall", "f1"]):
            plt.subplot(3, 1, i + 1)

            for query_type, color, marker in [("personal", "blue", "o"), ("factual", "red", "s")]:
                if query_type not in results:
                    continue

                thresholds = [m["threshold"] for m in results[query_type]]
                metric_values = [m[metric_name] for m in results[query_type]]

                plt.plot(
                    thresholds,
                    metric_values,
                    color=color,
                    marker=marker,
                    label=f"{query_type.capitalize()} Queries",
                )

            plt.title(f"{metric_name.capitalize()} vs. Threshold")
            plt.xlabel("Threshold")
            plt.ylabel(metric_name.capitalize())
            plt.grid(True, alpha=0.3)
            plt.legend()

        plt.tight_layout()
        plt.savefig("diagnostic_output/similarity_distributions/threshold_impact.png")

        # Save results
        with open("diagnostic_output/similarity_distributions/threshold_impact.json", "w") as f:
            json.dump(results, f, indent=2)

        return results

    def run_all_analyses(self):
        """
        Run all similarity distribution analyses.
        """
        print("Running all similarity distribution analyses...")

        # Load test data
        self.load_test_data()

        # Run analyses
        stats, similarity_data = self.analyze_similarity_distributions()
