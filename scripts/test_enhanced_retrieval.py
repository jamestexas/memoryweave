"""
Test script for the enhanced retrieval pipeline in MemoryWeave.

This script tests the two-stage retrieval pipeline with query type adaptation
to validate improvements in retrieval quality.
"""

import json
import os

import torch
from transformers import AutoModel, AutoTokenizer

from memoryweave.core import ContextualMemory, ContextualRetriever, MemoryEncoder
from memoryweave.utils.nlp_extraction import NLPExtractor

# Create output directory if it doesn't exist
os.makedirs("test_output", exist_ok=True)


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


def load_test_data(file_path="datasets/evaluation_queries.json"):
    """Load test data from the evaluation queries file."""
    try:
        with open(file_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []


def populate_memory(memory, encoder, embedding_model, data):
    """Populate memory with test data."""
    for item in data:
        # Add expected answer to memory
        embedding, metadata = encoder.encode_concept(
            concept=item["category"],
            description=item["expected_answer"],
            related_concepts=[item["category"]],
        )
        memory.add_memory(embedding, item["expected_answer"], metadata)

    print(f"Added {len(data)} memories to the system")


def test_retrieval_configurations(data, embedding_model):
    """Test different retrieval configurations and compare results."""
    # Initialize NLP extractor for query classification
    nlp_extractor = NLPExtractor()

    # Separate queries by type
    personal_queries = []
    factual_queries = []

    for item in data:
        query_types = nlp_extractor.identify_query_type(item["query"])
        primary_type = max(query_types.items(), key=lambda x: x[1])[0]

        if primary_type == "personal" or item["category"] == "personal":
            personal_queries.append(item)
        else:
            factual_queries.append(item)

    print(
        f"Testing with {len(personal_queries)} personal queries and {len(factual_queries)} factual queries"
    )

    # Define configurations to test
    configurations = {
        "baseline": {
            "use_two_stage_retrieval": False,
            "query_type_adaptation": False,
            "confidence_threshold": 0.3,
            "adaptive_retrieval": False,
            "semantic_coherence_check": False,
        },
        "two_stage": {
            "use_two_stage_retrieval": True,
            "query_type_adaptation": False,
            "confidence_threshold": 0.3,
            "first_stage_k": 20,
            "adaptive_retrieval": False,
            "semantic_coherence_check": False,
        },
        "query_type_adaptation": {
            "use_two_stage_retrieval": False,
            "query_type_adaptation": True,
            "confidence_threshold": 0.3,
            "adaptive_retrieval": False,
            "semantic_coherence_check": False,
        },
        "combined": {
            "use_two_stage_retrieval": True,
            "query_type_adaptation": True,
            "confidence_threshold": 0.3,
            "first_stage_k": 20,
            "adaptive_retrieval": True,
            "adaptive_k_factor": 0.15,
            "semantic_coherence_check": True,
        },
    }

    results = {}

    # Test each configuration
    for config_name, config in configurations.items():
        print(f"\nTesting configuration: {config_name}")

        # Initialize memory and components
        memory_dim = embedding_model.encode("test").shape[0]
        memory = ContextualMemory(embedding_dim=memory_dim)
        encoder = MemoryEncoder(embedding_model)

        # Populate memory
        populate_memory(memory, encoder, embedding_model, data)

        # Initialize retriever with configuration
        retriever = ContextualRetriever(memory=memory, embedding_model=embedding_model, **config)

        # Test personal queries
        personal_results = test_query_set(retriever, personal_queries)

        # Test factual queries
        factual_results = test_query_set(retriever, factual_queries)

        # Calculate overall results
        overall_precision = (
            personal_results["precision"] * len(personal_queries)
            + factual_results["precision"] * len(factual_queries)
        ) / len(data)
        overall_recall = (
            personal_results["recall"] * len(personal_queries)
            + factual_results["recall"] * len(factual_queries)
        ) / len(data)
        overall_f1 = (
            2 * overall_precision * overall_recall / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0
            else 0
        )

        results[config_name] = {
            "personal": personal_results,
            "factual": factual_results,
            "overall": {"precision": overall_precision, "recall": overall_recall, "f1": overall_f1},
        }

        print(
            f"Personal queries: Precision={personal_results['precision']:.3f}, Recall={personal_results['recall']:.3f}, F1={personal_results['f1']:.3f}"
        )
        print(
            f"Factual queries: Precision={factual_results['precision']:.3f}, Recall={factual_results['recall']:.3f}, F1={factual_results['f1']:.3f}"
        )
        print(
            f"Overall: Precision={overall_precision:.3f}, Recall={overall_recall:.3f}, F1={overall_f1:.3f}"
        )

    # Save results
    with open("test_output/retrieval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def test_query_set(retriever, queries):
    """Test a set of queries and calculate metrics."""
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for query in queries:
        # Retrieve memories
        retrieved = retriever.retrieve_for_context(query["query"], top_k=5)
        retrieved_texts = [
            item.get("text", "") or item.get("content", "") or item.get("description", "")
            for item in retrieved
        ]

        # Check if expected answer is in retrieved texts
        expected = query["expected_answer"]
        found = any(expected in text for text in retrieved_texts)

        # Calculate precision and recall
        precision = 1 / len(retrieved) if found else 0
        recall = 1 if found else 0

        # Calculate F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    # Calculate averages
    avg_precision = total_precision / len(queries) if queries else 0
    avg_recall = total_recall / len(queries) if queries else 0
    avg_f1 = total_f1 / len(queries) if queries else 0

    return {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1}


def main():
    """Run the enhanced retrieval test."""
    print("MemoryWeave Enhanced Retrieval Test")
    print("==================================")

    # Load embedding model
    print("Loading embedding model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embedding_model = EmbeddingModelWrapper(model, tokenizer)

    # Load test data
    data = load_test_data()
    if not data:
        print("No test data available. Exiting.")
        return

    # Test retrieval configurations
    results = test_retrieval_configurations(data, embedding_model)

    # Print summary
    print("\nSummary of Results:")
    print("===================")

    # Find best configuration
    best_config = max(results.items(), key=lambda x: x[1]["overall"]["f1"])
    print(f"Best configuration: {best_config[0]}")
    print(f"Overall F1: {best_config[1]['overall']['f1']:.3f}")
    print(f"Personal F1: {best_config[1]['personal']['f1']:.3f}")
    print(f"Factual F1: {best_config[1]['factual']['f1']:.3f}")

    print("\nTest completed. Results saved to test_output/retrieval_results.json")


if __name__ == "__main__":
    main()
