# Utils Module Knowledge

## Module Purpose
The utils module provides utility functions used throughout the MemoryWeave system.

## Key Components
- `similarity.py`: Functions for calculating similarity between embeddings and texts

## Key Functions
- `cosine_similarity_batched`: Efficient batch calculation of cosine similarity
- `embed_text_batch`: Batch embedding of text
- `fuzzy_string_match`: Fuzzy matching for text comparison

## Implementation Details
- Optimized for performance with batch operations
- Uses numpy for efficient vector operations
- Provides both exact and fuzzy matching capabilities

## Usage
These utilities are used internally by the core components but can also be used directly for:
1. Custom similarity calculations
2. Text embedding operations
3. String matching with tolerance for minor differences
