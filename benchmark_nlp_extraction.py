"""
Simple benchmark for NLP extraction utilities.

This script benchmarks the performance of the extraction utilities
across a variety of test cases.
"""

import time
from memoryweave.utils.nlp_extraction import NLPExtractor

def run_benchmark():
    """Run a simple benchmark of the extraction utilities."""
    # Initialize extractor
    extractor = NLPExtractor()
    
    # Test data
    test_texts = [
        "My name is Alex Thompson and I live in Portland, Oregon.",
        "I have a dog named Rusty who's 5 years old.",
        "My favorite food is Thai curry, especially the spicy kind.",
        "I work as a graphic designer at a marketing agency in downtown.",
        "I enjoy hiking in the mountains on weekends when the weather is nice.",
        "My wife Sarah and I have been married for 7 years now.",
        "I prefer reading science fiction novels before bed.",
        "I'm originally from Chicago but moved to Seattle three years ago.",
        "Blue is definitely my favorite color, especially navy blue.",
        "I studied computer science at Stanford University."
    ]
    
    # Query test data
    test_queries = [
        "What is the capital of France?",
        "Tell me about quantum computing.",
        "What's my name?",
        "Where do I live?",
        "What do you think about climate change?",
        "Write a poem about mountains."
    ]
    
    # Benchmark attribute extraction
    print("\nBenchmarking attribute extraction...")
    start_time = time.time()
    
    for text in test_texts:
        attributes = extractor.extract_personal_attributes(text)
    
    extraction_time = time.time() - start_time
    print(f"Extracted attributes from {len(test_texts)} texts in {extraction_time:.4f} seconds")
    print(f"Average time per text: {extraction_time/len(test_texts):.4f} seconds")
    
    # Benchmark query type identification
    print("\nBenchmarking query type identification...")
    start_time = time.time()
    
    for query in test_queries:
        query_types = extractor.identify_query_type(query)
    
    query_time = time.time() - start_time
    print(f"Identified query types for {len(test_queries)} queries in {query_time:.4f} seconds")
    print(f"Average time per query: {query_time/len(test_queries):.4f} seconds")
    
    # Print summary
    print("\nSummary:")
    print(f"Total benchmark time: {extraction_time + query_time:.4f} seconds")
    print(f"Using spaCy: {extractor.is_spacy_available}")

if __name__ == "__main__":
    run_benchmark()
