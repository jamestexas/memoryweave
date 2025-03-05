"""Hybrid BM25+Vector search implementation combining lexical and semantic matching."""

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from memoryweave.storage.vector_search.base import IVectorSearchProvider
from memoryweave.storage.vector_search.numpy_search import NumpyVectorSearch

logger = logging.getLogger(__name__)


class HybridBM25VectorSearch(IVectorSearchProvider):
    """
    Hybrid search implementation combining BM25 lexical search with vector semantic search.

    This implementation provides a combined approach that leverages both:
    1. BM25 lexical matching for keyword/phrase sensitivity
    2. Vector similarity for semantic understanding

    The combined approach often outperforms either method alone, particularly for
    queries that contain specific keywords but also require semantic understanding.
    """

    def __init__(
        self,
        dimension: int = 768,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        k1: float = 1.5,
        b: float = 0.75,
        min_df: int = 1,
        **kwargs,
    ):
        """
        Initialize the hybrid BM25+Vector search.

        Args:
            dimension: Dimension of vectors
            bm25_weight: Weight for BM25 score (0-1)
            vector_weight: Weight for vector score (0-1)
            k1: BM25 parameter controlling term frequency scaling
            b: BM25 parameter controlling length normalization
            min_df: Minimum document frequency for terms
            **kwargs: Additional arguments
        """
        self._dimension = dimension
        self._bm25_weight = bm25_weight
        self._vector_weight = vector_weight
        self._k1 = k1
        self._b = b
        self._min_df = min_df

        # Verify weights sum to 1.0
        total_weight = bm25_weight + vector_weight
        if not math.isclose(total_weight, 1.0, abs_tol=1e-10):
            logger.warning(f"Weights don't sum to 1.0 ({total_weight}), normalizing")
            self._bm25_weight = bm25_weight / total_weight
            self._vector_weight = vector_weight / total_weight

        # Use NumpyVectorSearch for vector search
        self._vector_search = NumpyVectorSearch(dimension=dimension)

        # BM25 index state
        self._docs = []  # List of document texts
        self._doc_ids = []  # List of document IDs
        self._term_freqs = []  # Term frequencies per document
        self._doc_lengths = []  # Document lengths in terms
        self._avg_doc_length = 0  # Average document length
        self._vocab = set()  # Vocabulary
        self._idf = {}  # Inverse document frequency
        self._df = Counter()  # Document frequency

        # Check if texts should be provided
        self._texts_required = True

        # Internal state
        self._dirty = True

    def index(self, vectors: np.ndarray, ids: List[Any], texts: Optional[List[str]] = None) -> None:
        """
        Index vectors and texts with associated IDs.

        Args:
            vectors: Matrix of vectors to index (each row is a vector)
            ids: List of IDs corresponding to each vector
            texts: Optional list of document texts for BM25 indexing
        """
        if len(vectors) != len(ids):
            raise ValueError("Number of vectors must match number of IDs")

        # Skip empty input
        if len(vectors) == 0:
            return

        # Index vectors
        self._vector_search.index(vectors, ids)

        # Check if texts are provided
        if texts is None:
            if self._texts_required and len(self._docs) == 0:
                logger.warning("No texts provided for hybrid search, BM25 component will not work")
                self._texts_required = False
            return

        if len(texts) != len(ids):
            raise ValueError("Number of texts must match number of IDs/vectors")

        # Process texts for BM25
        self._doc_ids = list(ids)
        self._docs = list(texts)

        # Tokenize documents
        tokens_per_doc = [self._tokenize(doc) for doc in texts]

        # Calculate document frequencies
        self._vocab = set()
        self._df = Counter()
        for tokens in tokens_per_doc:
            unique_terms = set(tokens)
            self._vocab.update(unique_terms)
            for term in unique_terms:
                self._df[term] += 1

        # Filter terms by min_df
        if self._min_df > 1:
            self._vocab = {term for term in self._vocab if self._df[term] >= self._min_df}

        # Calculate IDF
        N = len(texts)
        self._idf = {}
        for term in self._vocab:
            # Add 1 to denominator to avoid division by zero
            self._idf[term] = math.log((N + 1) / (self._df[term] + 1)) + 1

        # Calculate term frequencies and document lengths
        self._term_freqs = []
        self._doc_lengths = []

        for tokens in tokens_per_doc:
            # Count terms
            term_freq = Counter(tokens)
            self._term_freqs.append(term_freq)

            # Document length (count only terms in vocabulary)
            doc_length = sum(count for term, count in term_freq.items() if term in self._vocab)
            self._doc_lengths.append(doc_length)

        # Calculate average document length
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0
        )

        self._dirty = False

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        threshold: Optional[float] = None,
        query_text: Optional[str] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Search using both BM25 and vector similarity.

        Args:
            query_vector: Query embedding
            k: Number of results to return
            threshold: Optional similarity threshold
            query_text: Optional query text for BM25 matching

        Returns:
            List of (id, score) tuples
        """
        # Get more initial candidates for better hybrid ranking
        candidate_k = min(k * 3, len(self._doc_ids) if hasattr(self, "_doc_ids") else k)

        # Get vector search results
        vector_results = self._vector_search.search(query_vector, candidate_k, None)

        # Early return if no vector results or BM25 not configured
        if not vector_results or not self._texts_required or not query_text:
            return vector_results[:k]

        # Get candidate document IDs from vector search
        candidate_ids = [doc_id for doc_id, _ in vector_results]

        # Map IDs to indices for BM25 scoring
        id_to_index = {id_val: i for i, id_val in enumerate(self._doc_ids)}
        candidate_indices = [
            id_to_index[id_val] for id_val in candidate_ids if id_val in id_to_index
        ]

        # Calculate BM25 scores for candidates
        query_tokens = self._tokenize(query_text)
        bm25_scores = self._calculate_bm25(query_tokens, candidate_indices)

        # Normalize BM25 scores to 0-1 range if not empty
        if bm25_scores:
            max_bm25 = max(bm25_scores)
            if max_bm25 > 0:
                bm25_scores = [score / max_bm25 for score in bm25_scores]

        # Combine vector and BM25 scores
        combined_results = []
        for i, (doc_id, vector_score) in enumerate(vector_results):
            if i < len(bm25_scores):
                bm25_score = bm25_scores[i]
                # Combine scores with weights
                combined_score = self._vector_weight * vector_score + self._bm25_weight * bm25_score
            else:
                # If BM25 score not available, use vector score only
                combined_score = vector_score

            # Apply threshold
            if threshold is None or combined_score >= threshold:
                combined_results.append((doc_id, combined_score))

        # Sort by combined score and limit to k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:k]

    def update(self, vector_id: Any, vector: np.ndarray, text: Optional[str] = None) -> None:
        """
        Update a vector and its text in the index.

        Args:
            vector_id: ID of the vector to update
            vector: New vector
            text: New text (optional)
        """
        # Update vector
        self._vector_search.update(vector_id, vector)

        # Update text if provided
        if text is not None and self._texts_required:
            try:
                # Find the index of the document
                idx = self._doc_ids.index(vector_id)

                # Update document
                self._docs[idx] = text

                # Update BM25 index
                tokens = self._tokenize(text)
                term_freq = Counter(tokens)

                # Update document frequency for terms
                old_unique_terms = set(self._term_freqs[idx].keys())
                new_unique_terms = set(tokens)

                # Remove old terms
                for term in old_unique_terms:
                    if term not in new_unique_terms:
                        self._df[term] -= 1

                # Add new terms
                for term in new_unique_terms:
                    if term not in old_unique_terms:
                        self._df[term] += 1

                # Update term frequencies
                self._term_freqs[idx] = term_freq

                # Update document length
                old_length = self._doc_lengths[idx]
                new_length = sum(count for term, count in term_freq.items() if term in self._vocab)
                self._doc_lengths[idx] = new_length

                # Update average document length
                total_length = sum(self._doc_lengths)
                self._avg_doc_length = total_length / len(self._doc_lengths)

                # Update vocabulary and IDF
                self._vocab = {term for term in self._df if self._df[term] >= self._min_df}
                N = len(self._docs)

                for term in self._vocab:
                    self._idf[term] = math.log((N + 1) / (self._df[term] + 1)) + 1

            except ValueError:
                # ID not found in docs
                logger.warning(f"Document ID {vector_id} not found in BM25 index")

    def delete(self, vector_id: Any) -> None:
        """
        Delete a vector and its text from the index.

        Args:
            vector_id: ID of the vector to delete
        """
        # Delete from vector search
        self._vector_search.delete(vector_id)

        # Delete from BM25 index
        if self._texts_required:
            try:
                # Find the index of the document
                idx = self._doc_ids.index(vector_id)

                # Update document frequency for terms
                unique_terms = set(self._term_freqs[idx].keys())
                for term in unique_terms:
                    self._df[term] -= 1

                # Remove document
                self._docs.pop(idx)
                self._doc_ids.pop(idx)
                self._term_freqs.pop(idx)
                self._doc_lengths.pop(idx)

                # Update average document length
                if self._doc_lengths:
                    self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)
                else:
                    self._avg_doc_length = 0

                # Update vocabulary and IDF
                self._vocab = {term for term in self._df if self._df[term] >= self._min_df}
                N = len(self._docs)

                for term in self._vocab:
                    self._idf[term] = math.log((N + 1) / (self._df[term] + 1)) + 1

            except ValueError:
                # ID not found in docs
                logger.warning(f"Document ID {vector_id} not found in BM25 index")

    def clear(self) -> None:
        """Clear the index."""
        # Clear vector search
        self._vector_search.clear()

        # Clear BM25 index
        self._docs = []
        self._doc_ids = []
        self._term_freqs = []
        self._doc_lengths = []
        self._avg_doc_length = 0
        self._vocab = set()
        self._idf = {}
        self._df = Counter()

        self._dirty = True

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary of statistics
        """
        base_stats = {
            "type": "hybrid_bm25_vector",
            "dimension": self._dimension,
            "vector_weight": self._vector_weight,
            "bm25_weight": self._bm25_weight,
            "size": len(self._doc_ids) if hasattr(self, "_doc_ids") else 0,
            "vocab_size": len(self._vocab) if hasattr(self, "_vocab") else 0,
            "avg_doc_length": self._avg_doc_length if hasattr(self, "_avg_doc_length") else 0,
        }

        # Add vector search stats
        vector_stats = self._vector_search.get_statistics()
        for k, v in vector_stats.items():
            if k not in base_stats and k != "type":
                base_stats[f"vector_{k}"] = v

        return base_stats

    @property
    def dimension(self) -> int:
        """Get the dimension of vectors in the index."""
        return self._dimension

    @property
    def size(self) -> int:
        """Get the number of vectors in the index."""
        return len(self._doc_ids) if hasattr(self, "_doc_ids") else 0

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Replace punctuation with spaces
        text = re.sub(r"[^\w\s]", " ", text)

        # Split into tokens
        tokens = text.split()

        # Remove tokens shorter than 2 characters
        tokens = [token for token in tokens if len(token) >= 2]

        return tokens

    def _calculate_bm25(self, query_tokens: List[str], doc_indices: List[int]) -> List[float]:
        """
        Calculate BM25 scores for documents matching the query.

        Args:
            query_tokens: Tokenized query
            doc_indices: Indices of documents to score

        Returns:
            List of BM25 scores
        """
        scores = []

        for doc_idx in doc_indices:
            # Skip invalid indices
            if doc_idx < 0 or doc_idx >= len(self._term_freqs):
                scores.append(0.0)
                continue

            score = 0.0
            term_freqs = self._term_freqs[doc_idx]
            doc_length = self._doc_lengths[doc_idx]

            # Skip empty documents
            if doc_length == 0:
                scores.append(0.0)
                continue

            # Calculate score for each query term
            for term in query_tokens:
                if term in self._vocab:
                    # Get term frequency in document
                    tf = term_freqs.get(term, 0)

                    # Skip terms not in document
                    if tf == 0:
                        continue

                    # Get inverse document frequency
                    idf = self._idf.get(term, 0)

                    # Calculate BM25 score for term
                    numerator = tf * (self._k1 + 1)
                    denominator = tf + self._k1 * (
                        1 - self._b + self._b * doc_length / self._avg_doc_length
                    )
                    term_score = idf * numerator / denominator

                    score += term_score

            scores.append(score)

        return scores
