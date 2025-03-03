#!/usr/bin/env python
"""
Test script for the ANN vector store implementation.

This script creates a large memory store and tests the performance of retrieving memories
using the standard vector store vs. the ANN-based vector store.
"""

import time
import numpy as np
from typing import List

from memoryweave.core.contextual_memory import ContextualMemory
from memoryweave.components.retriever import Retriever

# Use mock embedding model for testing
class MockEmbeddingModel:
    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim
        
    def encode(self, text, batch_size=32):
        """Create a deterministic but unique embedding for a text."""
        if isinstance(text, list):
            return np.array([self._encode_single(t) for t in text])
        return self._encode_single(text)
        
    def _encode_single(self, text):
        # Use hash for deterministic but unique embeddings
        hash_val = hash(text) % 1000000
        np.random.seed(hash_val)
        embedding = np.random.randn(self.embedding_dim)
        return embedding / np.linalg.norm(embedding)  # Normalize

def generate_test_data(num_memories: int, embedding_model) -> List[tuple]:
    """Generate synthetic test memories."""
    print(f"Generating {num_memories} test memories...")
    
    # Create test memories
    memories = []
    topics = ["Python", "Machine Learning", "AI", "Databases", "Cloud Computing"]
    
    for i in range(num_memories):
        topic = topics[i % len(topics)]
        memory_text = f"Memory {i}: Information about {topic} - {i}"
        memory_embedding = embedding_model.encode(memory_text)
        memories.append((memory_text, memory_embedding))
        
    return memories

def test_retrieval_performance(memory_store_size: int):
    """Test the retrieval performance with and without ANN."""
    embedding_model = MockEmbeddingModel()
    
    # Generate test data
    memories = generate_test_data(memory_store_size, embedding_model)
    
    # Test queries
    test_queries = [
        "Tell me about Python",
        "What do you know about AI?",
        "Information about Machine Learning",
        "Cloud Computing details",
        "Database information"
    ]
    query_embeddings = [embedding_model.encode(q) for q in test_queries]
    
    # Test without ANN
    print("\nTesting without ANN:")
    memory_standard = ContextualMemory(
        embedding_dim=768,
        max_memories=memory_store_size + 10,
        use_ann=False,
    )
    
    # Add memories
    for i, (text, embedding) in enumerate(memories):
        memory_standard.add_memory(embedding, text, {"index": i})
    
    # Test retrieval time
    retriever_standard = Retriever(memory=memory_standard, embedding_model=embedding_model)
    standard_times = []
    
    for i, query in enumerate(test_queries):
        start_time = time.time()
        results_standard = memory_standard.retrieve_memories(
            query_embeddings[i], 
            top_k=10, 
            confidence_threshold=0.1
        )
        query_time = time.time() - start_time
        standard_times.append(query_time)
        print(f"Query '{query}' took {query_time:.6f}s, found {len(results_standard)} results")
    
    avg_standard_time = sum(standard_times) / len(standard_times)
    print(f"Average standard retrieval time: {avg_standard_time:.6f}s")
    
    # Test with ANN
    print("\nTesting with ANN:")
    memory_ann = ContextualMemory(
        embedding_dim=768,
        max_memories=memory_store_size + 10,
        use_ann=True,
    )
    
    # Add memories
    for i, (text, embedding) in enumerate(memories):
        memory_ann.add_memory(embedding, text, {"index": i})
    
    # Test retrieval time
    retriever_ann = Retriever(memory=memory_ann, embedding_model=embedding_model)
    ann_times = []
    
    for i, query in enumerate(test_queries):
        start_time = time.time()
        results_ann = memory_ann.retrieve_memories(
            query_embeddings[i], 
            top_k=10, 
            confidence_threshold=0.1
        )
        query_time = time.time() - start_time
        ann_times.append(query_time)
        print(f"Query '{query}' took {query_time:.6f}s, found {len(results_ann)} results")
    
    avg_ann_time = sum(ann_times) / len(ann_times)
    print(f"Average ANN retrieval time: {avg_ann_time:.6f}s")
    
    # Calculate speedup
    speedup = avg_standard_time / avg_ann_time if avg_ann_time > 0 else float('inf')
    print(f"\nPerformance Summary for {memory_store_size} memories:")
    print(f"Standard retrieval: {avg_standard_time:.6f}s")
    print(f"ANN retrieval:      {avg_ann_time:.6f}s")
    print(f"Speedup:            {speedup:.2f}x")

def main():
    """Run the test with different memory store sizes."""
    print("===== Testing ANN Vector Store Performance =====")
    
    # Test with 100 memories (small)
    print("\n----- Testing with 100 memories -----")
    test_retrieval_performance(100)
    
    # Test with 500 memories (medium)
    print("\n----- Testing with 500 memories -----")
    test_retrieval_performance(500)
    
    # Test with 1000 memories (large)
    print("\n----- Testing with 1000 memories -----")
    test_retrieval_performance(1000)
    
    # Test with 5000 memories (very large)
    print("\n----- Testing with 5000 memories -----")
    test_retrieval_performance(5000)

if __name__ == "__main__":
    main()