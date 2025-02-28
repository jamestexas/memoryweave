"""
Similarity and embedding utility functions.
"""

from difflib import SequenceMatcher
from typing import Any, List, Tuple

import numpy as np


def cosine_similarity_batched(
    query_embedding: np.ndarray, reference_embeddings: np.ndarray, batch_size: int = 1000
) -> np.ndarray:
    """
    Calculate cosine similarity between a query and reference embeddings in batches.

    Args:
        query_embedding: Query embedding vector (1D array)
        reference_embeddings: Reference embedding matrix (2D array)
        batch_size: Batch size for processing large sets of references

    Returns:
        Array of similarity scores
    """
    # Ensure query is normalized
    query_norm = np.linalg.norm(query_embedding)
    if query_norm > 0:
        query_embedding = query_embedding / query_norm

    n_references = reference_embeddings.shape[0]
    similarities = np.zeros(n_references)

    # Process in batches to avoid memory issues with large reference sets
    for i in range(0, n_references, batch_size):
        batch_end = min(i + batch_size, n_references)
        batch = reference_embeddings[i:batch_end]

        # Normalize batch
        batch_norms = np.linalg.norm(batch, axis=1, keepdims=True)
        batch_norms[batch_norms == 0] = 1.0  # Avoid division by zero
        normalized_batch = batch / batch_norms

        # Calculate cosine similarity
        batch_similarities = np.dot(normalized_batch, query_embedding)
        similarities[i:batch_end] = batch_similarities

    return similarities


def embed_text_batch(texts: List[str], embedding_model: Any, batch_size: int = 32) -> np.ndarray:
    """
    Create embeddings for a batch of texts.

    Args:
        texts: list of text strings to embed
        embedding_model: Model to use for creating embeddings
        batch_size: Number of texts to process at once

    Returns:
        Matrix of embeddings
    """
    n_texts = len(texts)

    # Determine embedding dimension from a sample
    if n_texts > 0:
        sample_embedding = embedding_model.encode(texts[0])
        embedding_dim = sample_embedding.shape[0]
    else:
        return np.array([])

    # Initialize results array
    embeddings = np.zeros((n_texts, embedding_dim))

    # Process in batches
    for i in range(0, n_texts, batch_size):
        batch_end = min(i + batch_size, n_texts)
        batch_texts = texts[i:batch_end]

        # This assumes the model can handle batches
        # If not, you would need to encode each text individually
        try:
            batch_embeddings = embedding_model.encode(batch_texts)
            embeddings[i:batch_end] = batch_embeddings
        except Exception as e:
            print(f"Error encoding batch {i}-{batch_end}: {e}")
            # Fallback to individual encoding if batch encoding fails
            for j, text in enumerate(batch_texts):
                embeddings[i + j] = embedding_model.encode(text)

    return embeddings


def fuzzy_string_match(
    query: str, references: List[str], threshold: float = 0.7
) -> List[Tuple[int, float]]:
    """
    Find fuzzy string matches using sequence matching.

    Args:
        query: Query string
        references: list of reference strings to match against
        threshold: Minimum similarity score threshold

    Returns:
        list of (index, score) tuples for matches above threshold
    """
    matches = []

    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()

    for i, ref in enumerate(references):
        # Convert reference to lowercase
        ref_lower = ref.lower()

        # Calculate similarity ratio
        similarity = SequenceMatcher(None, query_lower, ref_lower).ratio()

        if similarity >= threshold:
            matches.append((i, similarity))

    # Sort by similarity score (descending)
    return sorted(matches, key=lambda x: x[1], reverse=True)
