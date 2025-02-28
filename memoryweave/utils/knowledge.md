# Utils Module Knowledge

## Module Purpose
The utils module provides utility functions used throughout the MemoryWeave system.

## Key Components
- `similarity.py`: Functions for calculating similarity between embeddings and texts
- `analysis.py`: Tools for analyzing memory retrieval performance and distributions
- `nlp_extraction.py`: NLP-based extraction utilities for personal attributes and query types

## Key Functions

### Similarity Functions
- `cosine_similarity_batched`: Efficient batch calculation of cosine similarity
- `embed_text_batch`: Batch embedding of text
- `fuzzy_string_match`: Fuzzy matching for text comparison

### Analysis Functions
- `analyze_query_similarities`: Analyze similarity distributions for specific queries
- `visualize_memory_categories`: Visualize memory categories and their relationships
- `analyze_retrieval_performance`: Analyze retrieval performance across parameter settings

### NLP Extraction Functions
- `NLPExtractor.extract_personal_attributes`: Extract personal attributes using NLP techniques
- `NLPExtractor.identify_query_type`: Identify query type (factual, personal, opinion, instruction)

## Implementation Details
- Optimized for performance with batch operations
- Uses numpy for efficient vector operations
- Provides both exact and fuzzy matching capabilities
- Includes visualization tools for memory analysis
- Uses spaCy for NLP-based extraction to complement regex patterns

## Hybrid Extraction Approach
The system uses a hybrid approach for extraction:
1. First tries regex patterns for speed and efficiency
2. Falls back to NLP-based extraction when regex confidence is low
3. Combines results from both methods, prioritizing regex for conflicts

Benefits of this approach:
- More flexible matching than regex alone
- Better handling of language variations
- Reduced bias in extraction
- Improved recall while maintaining precision
- Graceful degradation when NLP libraries aren't available

## Usage
These utilities are used internally by the core components but can also be used directly for:
1. Custom similarity calculations
2. Text embedding operations
3. String matching with tolerance for minor differences
4. Analyzing memory retrieval performance
5. Visualizing memory categories and distributions
6. Extracting personal attributes from text
7. Identifying query types for adaptive retrieval

### Example: Analyzing Query Similarities
```python
from memoryweave.utils.analysis import analyze_query_similarities

# Analyze why a specific query might be failing
results = analyze_query_similarities(
    memory_system=memory_system,
    query="What is the capital of France?",
    expected_relevant_indices=[5, 10],  # Indices of memories that should be relevant
    plot=True,
    save_path="query_analysis.png"
)

# Check if relevant memories are below the threshold
for idx, sim, boosted_sim in results["below_threshold"]:
    print(f"Memory {idx} has similarity {sim:.3f} (boosted: {boosted_sim:.3f})")
```

### Example: Using NLP Extraction
```python
from memoryweave.utils.nlp_extraction import NLPExtractor

# Initialize extractor
extractor = NLPExtractor()

# Extract personal attributes
text = "My name is Alex and I live in Seattle. I work as a software engineer."
attributes = extractor.extract_personal_attributes(text)

# Identify query type
query = "What is the capital of France?"
query_types = extractor.identify_query_type(query)
primary_type = max(query_types.items(), key=lambda x: x[1])[0]
```
