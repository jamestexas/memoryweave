"""
Enhanced retrieval strategies for MemoryWeave.

This module implements advanced retrieval strategies that adapt to memory size
and content characteristics for optimal performance.
"""

import logging
import os
import sys
import time
import traceback
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("enhanced_retrieval.log")],
)
logger = logging.getLogger(__name__)

# Log system information
logger.info(f"Python version: {sys.version}")
logger.info(f"NumPy version: {np.__version__}")

from memoryweave.core import ContextualMemory, ContextualRetriever, MemoryEncoder

# Try to import clustering libraries
try:
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors

    SKLEARN_AVAILABLE = True
    logger.info(f"scikit-learn available (version {sklearn.__version__})")
except ImportError as e:
    logger.warning(f"scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
    logger.info(f"HDBSCAN available (version {hdbscan})")
except ImportError as e:
    logger.warning(f"HDBSCAN not available: {e}")
    HDBSCAN_AVAILABLE = False

try:
    import faiss

    FAISS_AVAILABLE = True
    logger.info("FAISS available")
except ImportError:
    logger.warning("FAISS not available")
    FAISS_AVAILABLE = False

# Global spaCy model for reuse
global_nlp = None


class EnhancedRetriever(ContextualRetriever):
    """
    Enhanced version of ContextualRetriever with adaptive strategies for different memory sizes.
    """

    def __init__(self, memory, embedding_model, **kwargs):
        """Initialize with parent class constructor."""
        self.use_clustering = kwargs.pop("use_clustering", True)
        self.clustering_method = kwargs.pop("clustering_method", "kmeans")  # or "hdbscan"
        self.cluster_count = kwargs.pop("cluster_count", 10)
        self.memory_size_threshold = kwargs.pop("memory_size_threshold", 1000)
        self.use_prefiltering = kwargs.pop("use_prefiltering", True)
        self.prefilter_method = kwargs.pop(
            "prefilter_method", "hybrid"
        )  # "keyword", "semantic", or "hybrid"
        self.adaptive_depth = kwargs.pop("adaptive_depth", True)
        self.time_decay = kwargs.pop("time_decay", True)
        self.time_decay_factor = kwargs.pop("time_decay_factor", 0.95)

        logger.info(
            f"Initializing EnhancedRetriever with parameters: clustering={self.use_clustering}({self.clustering_method}), "
            f"prefiltering={self.use_prefiltering}({self.prefilter_method}), "
            f"memory_threshold={self.memory_size_threshold}, adaptive_depth={self.adaptive_depth}"
        )

        try:
            super().__init__(memory, embedding_model, **kwargs)
            logger.info("Parent ContextualRetriever initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing parent ContextualRetriever: {e}")
            logger.error(traceback.format_exc())
            raise

        self.clusters = None
        self.cluster_centers = None
        self.memory_to_cluster = {}

    def build_clusters(self):
        """Build clusters for faster retrieval with large memory sets."""
        if not self.use_clustering or len(self.memory.memory_embeddings) < 100:
            logger.info("Skipping cluster building (use_clustering=False or memory size < 100)")
            return

        logger.info(f"Building memory clusters using {self.clustering_method}...")

        try:
            if self.clustering_method == "hdbscan" and HDBSCAN_AVAILABLE:
                self._build_clusters_hdbscan()
            else:
                self._build_clusters_kmeans()
        except Exception as e:
            logger.error(f"Error building clusters: {e}")
            logger.error(traceback.format_exc())
            self.clusters = None
            self.cluster_centers = None

    def _build_clusters_kmeans(self):
        """Build clusters using K-means."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, unable to perform clustering")
            return

        n_clusters = min(self.cluster_count, max(5, len(self.memory.memory_embeddings) // 20))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.memory_to_cluster = kmeans.fit_predict(self.memory.memory_embeddings)
        self.cluster_centers = kmeans.cluster_centers_

        self.clusters = {}
        for i, cluster_id in enumerate(self.memory_to_cluster):
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(i)

        logger.info(f"Created {n_clusters} memory clusters using K-means")

    def _build_clusters_hdbscan(self):
        """Build clusters using HDBSCAN."""
        if not HDBSCAN_AVAILABLE:
            logger.warning("HDBSCAN not available, falling back to K-means")
            self._build_clusters_kmeans()
            return

        try:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, metric="euclidean")
            self.memory_to_cluster = clusterer.fit_predict(self.memory.memory_embeddings)

            self.clusters = {}
            for i, cluster_id in enumerate(self.memory_to_cluster):
                if cluster_id not in self.clusters:
                    self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)

            logger.info(
                f"Created {len(np.unique(self.memory_to_cluster))} memory clusters using HDBSCAN"
            )
        except Exception as e:
            logger.error(f"Error in HDBSCAN clustering: {e}")
            logger.warning("Falling back to K-means")
            self._build_clusters_kmeans()

    def _initialize_faiss(self):
        """Initialize FAISS index for fast similarity search."""
        if not FAISS_AVAILABLE or len(self.memory.memory_embeddings) == 0:
            return

        try:
            # Get dimension from memory embeddings
            dim = self.memory.memory_embeddings.shape[1]

            # Create HNSW index for approximate nearest neighbor search
            # M parameter controls connections per node (higher = more accurate but more memory)
            # efConstruction controls index building quality (higher = better quality but slower build)
            index = faiss.IndexHNSWFlat(dim, 32)  # 32 connections per node

            # Configure the index
            index.hnsw.efConstruction = 40  # Higher accuracy during construction
            index.hnsw.efSearch = 16  # Higher accuracy during search

            # Add items to the index
            index.add(self.memory.memory_embeddings.astype(np.float32))

            self.faiss_index = index
            print(f"FAISS HNSW index created for {len(self.memory.memory_embeddings)} memories")
        except Exception as e:
            print(f"Error initializing FAISS: {e}")
            self.faiss_index = None

    def update_faiss_index(self):
        """Update the FAISS index after memory changes."""
        if self.use_faiss and FAISS_AVAILABLE:
            self._initialize_faiss()

    def build_clusters(self):
        """Build clusters for faster retrieval with large memory sets."""
        if not self.use_clustering or len(self.memory.memory_embeddings) < 100:
            return

        print(f"Building memory clusters using {self.clustering_method}...")

        if self.clustering_method == "hdbscan" and HDBSCAN_AVAILABLE:
            self._build_clusters_hdbscan()
        else:
            self._build_clusters_kmeans()

    def _build_clusters_kmeans(self):
        """Build clusters using K-means."""
        if not SKLEARN_AVAILABLE:
            print("scikit-learn not available, unable to perform clustering")
            return

        # Determine number of clusters based on memory size
        n_clusters = min(self.cluster_count, max(5, len(self.memory.memory_embeddings) // 20))

        # Fit k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.memory_to_cluster = kmeans.fit_predict(self.memory.memory_embeddings)
        self.cluster_centers = kmeans.cluster_centers_

        # Create cluster lookup
        self.clusters = {}
        for i, cluster_id in enumerate(self.memory_to_cluster):
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(i)

        # If we want overlapping clusters, allow memories to belong to multiple clusters
        if self.overlap_clusters:
            self._create_overlapping_clusters()

        print(f"Created {n_clusters} memory clusters using K-means")

    def _build_clusters_hdbscan(self):
        """Build clusters using HDBSCAN (density-based clustering)."""
        if not HDBSCAN_AVAILABLE:
            print("HDBSCAN not available, falling back to K-means")
            self._build_clusters_kmeans()
            return

        try:
            # Use HDBSCAN for clustering
            # min_cluster_size: minimum size of clusters
            # min_samples: determines the conservativeness of clustering (higher = more conservative)
            # metric: distance metric (cosine is good for embeddings)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=5,
                min_samples=2,
                metric="euclidean",
                core_dist_n_jobs=-1,  # Use all CPU cores
            )

            # Fit the clusterer
            self.memory_to_cluster = clusterer.fit_predict(self.memory.memory_embeddings)

            # Handle outliers (-1 cluster ID in HDBSCAN)
            # Create an additional cluster for outliers
            outlier_indices = np.where(self.memory_to_cluster == -1)[0]
            if len(outlier_indices) > 0:
                print(f"Found {len(outlier_indices)} outliers not assigned to any cluster")
                max_label = np.max(self.memory_to_cluster)
                self.memory_to_cluster[outlier_indices] = max_label + 1

            # Create cluster lookup
            self.clusters = {}
            for i, cluster_id in enumerate(self.memory_to_cluster):
                if cluster_id not in self.clusters:
                    self.clusters[cluster_id] = []
                self.clusters[cluster_id].append(i)

            # Generate cluster centers by taking the mean of each cluster
            unique_clusters = np.unique(self.memory_to_cluster)
            self.cluster_centers = np.zeros((
                len(unique_clusters),
                self.memory.memory_embeddings.shape[1],
            ))

            for i, cluster_id in enumerate(unique_clusters):
                cluster_indices = np.where(self.memory_to_cluster == cluster_id)[0]
                self.cluster_centers[i] = np.mean(
                    self.memory.memory_embeddings[cluster_indices], axis=0
                )

            print(f"Created {len(unique_clusters)} memory clusters using HDBSCAN")

        except Exception as e:
            print(f"Error in HDBSCAN clustering: {e}")
            print("Falling back to K-means")
            self._build_clusters_kmeans()

    def _create_overlapping_clusters(self):
        """Allow memories to belong to multiple clusters based on proximity."""
        if not SKLEARN_AVAILABLE or self.cluster_centers is None:
            return

        # Use nearest neighbors to find closest clusters
        nn = NearestNeighbors(n_neighbors=self.max_clusters_per_memory, metric="cosine")
        nn.fit(self.cluster_centers)

        # For each memory, find the closest clusters
        distances, indices = nn.kneighbors(self.memory.memory_embeddings)

        # Update clusters
        new_clusters = {}
        for i in range(len(self.memory.memory_embeddings)):
            for j in range(1, self.max_clusters_per_memory):  # Skip first one (already assigned)
                if distances[i, j] < 0.3:  # Only add if reasonably close
                    cluster_id = indices[i, j]
                    if cluster_id not in new_clusters:
                        new_clusters[cluster_id] = []
                    new_clusters[cluster_id].append(i)

        # Merge with existing clusters
        for cluster_id, members in new_clusters.items():
            if cluster_id in self.clusters:
                self.clusters[cluster_id].extend(members)
            else:
                self.clusters[cluster_id] = members

    def _prefilter_memories(self, query: str, query_embedding: np.ndarray) -> list[int]:
        """Prefilter memories to reduce the search space."""
        memory_count = len(self.memory.memory_embeddings)

        # Skip prefiltering for small memory sets
        if not self.use_prefiltering or memory_count < 100:
            return list(range(memory_count))

        # Choose prefiltering method based on memory size and requested method
        if self.prefilter_method == "hybrid":
            if memory_count > self.memory_size_threshold:
                # Use a combination of cluster-based and semantic for large memories
                return self._prefilter_hybrid(query, query_embedding)
            else:
                # Use semantic similarity for medium-sized memories
                return self._prefilter_semantic(query_embedding)
        elif self.prefilter_method == "semantic":
            return self._prefilter_semantic(query_embedding)
        elif self.prefilter_method == "cluster" and self.clusters is not None:
            return self._prefilter_clusters(query_embedding)
        elif self.prefilter_method == "keyword":
            return self._prefilter_keywords(query)

        # Default: return all memories
        return list(range(memory_count))

    def _prefilter_clusters(self, query_embedding: np.ndarray) -> list[int]:
        """Prefilter using cluster-based approach."""
        if self.clusters is None or self.cluster_centers is None:
            return list(range(len(self.memory.memory_embeddings)))

        # Find closest clusters
        cluster_similarities = np.dot(self.cluster_centers, query_embedding)

        # Select top clusters (more for larger memory sizes)
        num_clusters = min(3, len(cluster_similarities))
        if len(self.memory.memory_embeddings) > 1000:
            num_clusters = min(5, len(cluster_similarities))

        top_clusters = np.argsort(-cluster_similarities)[:num_clusters]

        # Collect memory indices from top clusters
        candidate_indices = []
        for cluster_id in top_clusters:
            candidate_indices.extend(self.clusters.get(cluster_id, []))

        return candidate_indices

    def _prefilter_semantic(self, query_embedding: np.ndarray) -> list[int]:
        """Prefilter using lightweight semantic similarity."""
        # If FAISS is available, use it for fast approximate nearest neighbors
        if self.use_faiss and self.faiss_index is not None:
            try:
                # Get more candidates than we need to ensure good recall
                k = min(100, len(self.memory.memory_embeddings))
                distances, indices = self.faiss_index.search(
                    query_embedding.reshape(1, -1).astype(np.float32), k
                )
                return indices[0].tolist()
            except Exception as e:
                print(f"FAISS search error: {e}")
                # Fall back to direct similarity

        # Direct similarity calculation for smaller datasets
        similarities = np.dot(self.memory.memory_embeddings, query_embedding)

        # Take top candidates proportional to memory size
        k = min(100, max(20, len(similarities) // 10))
        top_indices = np.argsort(-similarities)[:k]

        return top_indices.tolist()

    def _prefilter_keywords(self, query: str) -> list[int]:
        """Prefilter based on keyword matching."""
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        if not keywords:
            return list(range(len(self.memory.memory_embeddings)))

        # Find memories containing these keywords
        candidate_indices = []
        for i, metadata in enumerate(self.memory.memory_metadata):
            memory_text = ""
            for field in ["text", "content", "description"]:
                if field in metadata:
                    memory_text += " " + str(metadata[field]).lower()

            # Check if any keyword is in the memory text
            if any(keyword in memory_text for keyword in keywords):
                candidate_indices.append(i)

        # If too few candidates, return all memories
        if len(candidate_indices) < max(10, len(self.memory.memory_embeddings) // 100):
            return list(range(len(self.memory.memory_embeddings)))

        return candidate_indices

    def _prefilter_hybrid(self, query: str, query_embedding: np.ndarray) -> list[int]:
        """Use a hybrid prefiltering approach combining multiple methods."""
        # Start with a larger candidate set using semantic similarity
        semantic_candidates = set(self._prefilter_semantic(query_embedding))

        # If using clusters, add candidates from closest clusters
        if self.clusters is not None:
            cluster_candidates = set(self._prefilter_clusters(query_embedding))
            semantic_candidates.update(cluster_candidates)

        # If candidate set is still too large, refine with keyword filtering
        if len(semantic_candidates) > 200:
            keywords = self._extract_keywords(query)
            if keywords:
                refined_candidates = []
                for idx in semantic_candidates:
                    metadata = self.memory.memory_metadata[idx]
                    memory_text = ""
                    for field in ["text", "content", "description"]:
                        if field in metadata:
                            memory_text += " " + str(metadata[field]).lower()

                    # Check if any keyword is in the memory text
                    if any(keyword in memory_text for keyword in keywords):
                        refined_candidates.append(idx)

                # Only use refined if we have enough
                if len(refined_candidates) >= 20:
                    return refined_candidates

        return list(semantic_candidates)

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract important keywords from text."""
        if self.nlp:
            # Use spaCy for better keyword extraction
            doc = self.nlp(text)
            keywords = set()

            # Extract named entities
            for ent in doc.ents:
                keywords.add(ent.text.lower())

            # Extract noun phrases and important nouns
            for chunk in doc.noun_chunks:
                keywords.add(chunk.text.lower())

            # Extract important verbs
            for token in doc:
                if token.pos_ == "VERB" and token.is_alpha and len(token.text) > 3:
                    keywords.add(token.lemma_.lower())

            # Filter out stop words and short words
            keywords = {k for k in keywords if len(k) > 3 and not k.startswith("the ")}
            return keywords
        else:
            # Fallback to basic keyword extraction
            words = text.lower().split()
            return {
                w
                for w in words
                if len(w) > 3
                and w
                not in [
                    "what",
                    "when",
                    "where",
                    "which",
                    "this",
                    "that",
                    "these",
                    "those",
                    "with",
                    "have",
                    "does",
                    "about",
                    "would",
                    "could",
                    "should",
                ]
            }

    def retrieve_for_context(
        self,
        current_input: str,
        conversation_history: Optional[list[dict]] = None,
        top_k: int = 5,
        confidence_threshold: float = None,
    ) -> list[dict]:
        """
        Enhanced retrieve_for_context with adaptive strategy selection.

        Args:
            current_input: The current user input
            conversation_history: Recent conversation history
            top_k: Number of memories to retrieve
            confidence_threshold: Minimum similarity score for memory inclusion

        Returns:
            list of relevant memory entries with metadata
        """
        memory_count = len(self.memory.memory_embeddings)

        # For empty memory, return empty results
        if memory_count == 0:
            return []

        # Build clusters if needed and not already built
        if self.use_clustering and self.clusters is None and memory_count >= 100:
            self.build_clusters()

        # Update FAISS index if needed
        if self.use_faiss and self.faiss_index is None and memory_count >= 100:
            self._initialize_faiss()

        # Update conversation state
        self._update_conversation_state(current_input, conversation_history)

        # Encode the query context
        query_context = self._build_query_context(current_input, conversation_history)
        query_embedding = self.embedding_model.encode(query_context)

        # Extract important keywords for direct reference matching
        important_keywords = self.nlp_extractor.extract_important_keywords(current_input)

        # Extract and update personal attributes
        self._extract_and_update_personal_attributes(current_input, conversation_history)

        # If no confidence threshold is provided, use the default
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        # Determine query type and adjust parameters accordingly
        query_type, adjusted_params = self._adapt_to_query_type(current_input, confidence_threshold)

        # Choose retrieval strategy based on memory size
        if memory_count < self.memory_size_threshold:
            # For smaller memories, use standard retrieval for better accuracy
            memories = self._retrieve_standard(
                query_embedding, query_type, important_keywords, top_k, adjusted_params
            )
        else:
            # For larger memories, use optimized retrieval
            memories = self._retrieve_optimized(
                query_embedding,
                query_type,
                important_keywords,
                top_k,
                adjusted_params,
                current_input,
            )

        # Enhance results with personal attributes relevant to the query
        enhanced_memories = self._enhance_with_personal_attributes(memories, current_input)

        # Apply memory decay if enabled
        if self.memory_decay_enabled:
            self._apply_memory_decay()

        # Track retrieval metrics for dynamic threshold adjustment
        if self.dynamic_threshold_adjustment:
            self._track_retrieval_metrics(query_embedding, enhanced_memories)

        # Ensure we have at least min_results_guarantee results
        if len(enhanced_memories) < self.min_results_guarantee:
            # If we don't have enough results, try again with a lower threshold
            fallback_memories = self._retrieve_fallback(
                query_embedding, query_type, important_keywords, current_input, adjusted_params
            )

            # Add any new memories that weren't already retrieved
            existing_ids = {
                m.get("memory_id") for m in enhanced_memories if isinstance(m.get("memory_id"), int)
            }

            for memory in fallback_memories:
                if memory["memory_id"] not in existing_ids:
                    enhanced_memories.append(memory)
                    if len(enhanced_memories) >= self.min_results_guarantee:
                        break

        return enhanced_memories

    def _extract_and_update_personal_attributes(
        self, current_input: str, conversation_history: Optional[list[dict]]
    ) -> None:
        """Extract and update personal attributes from input and history."""
        # Check conversation history
        if conversation_history:
            for turn in conversation_history[-3:]:  # Look at recent turns
                message = turn.get("message", "")
                response = turn.get("response", "")

                # Extract from message
                extracted_attributes = self.nlp_extractor.extract_personal_attributes(message)
                self._update_personal_attributes(extracted_attributes)

                # Extract from response
                extracted_attributes = self.nlp_extractor.extract_personal_attributes(response)
                self._update_personal_attributes(extracted_attributes)

        # Extract from current input
        extracted_attributes = self.nlp_extractor.extract_personal_attributes(current_input)
        self._update_personal_attributes(extracted_attributes)

    def _retrieve_standard(
        self,
        query_embedding: np.ndarray,
        query_type: str,
        important_keywords: set,
        top_k: int,
        params: dict[str, Any],
    ) -> list[dict]:
        """
        Standard retrieval for smaller memory sets (better accuracy).

        Args:
            query_embedding: Query embedding
            query_type: Type of query (personal, factual, etc.)
            important_keywords: Important keywords from the query
            top_k: Number of results to return
            params: Adjusted parameters for this query type

        Returns:
            list of retrieved memories
        """
        confidence_threshold = params.get("confidence_threshold", self.confidence_threshold)

        # Expand keywords for factual queries if enabled
        expanded_keywords = important_keywords
        if self.enable_keyword_expansion and query_type == "factual":
            expanded_keywords = self._expand_keywords(important_keywords)

        # Apply time-based decay if enabled
        similarities = self._compute_similarities_with_time_decay(query_embedding)

        # Retrieve memories using the hybrid strategy which includes keyword boosting
        return self._retrieve_hybrid(
            query_embedding,
            top_k,
            expanded_keywords,
            confidence_threshold,
            params.get("adaptive_k_factor", self.adaptive_k_factor),
            similarities,
        )

    def _compute_similarities_with_time_decay(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute similarities with optional time-based decay."""
        # Get basic similarity scores
        similarities = np.dot(self.memory.memory_embeddings, query_embedding)

        # Apply time decay if enabled
        if self.time_decay:
            # Normalize temporal markers to [0, 1] range
            max_time = float(self.memory.current_time)
            if max_time > 0:
                # Calculate age factor: newer = higher value, approaching 1
                age_factor = self.memory.temporal_markers / max_time

                # Calculate decay factor: newer = less decay
                # decay_factor = self.time_decay_factor ** (1 - age_factor)
                # More aggressive decay:
                decay_multiplier = 1.0 - ((1.0 - age_factor) * (1.0 - self.time_decay_factor))

                # Apply decay - newer memories get less penalty
                similarities = similarities * decay_multiplier

        return similarities

    def _retrieve_optimized(
        self,
        query_embedding: np.ndarray,
        query_type: str,
        important_keywords: set,
        top_k: int,
        params: dict[str, Any],
        current_input: str,
    ) -> list[dict]:
        """
        Optimized retrieval strategy for larger memory sets.

        Args:
            query_embedding: Query embedding
            query_type: Type of query (personal, factual, etc.)
            important_keywords: Important keywords from the query
            top_k: Number of results to return
            params: Adjusted parameters for this query type
            current_input: Original user input

        Returns:
            list of retrieved memories
        """
        if self.adaptive_depth:
            return self._retrieve_with_adaptive_depth(
                query_embedding, query_type, important_keywords, top_k, params, current_input
            )
        else:
            # First stage: Get candidates through prefiltering
            prefiltered_indices = self._prefilter_memories(current_input, query_embedding)

            # Second stage: Score and rank the candidates
            confidence_threshold = params.get("confidence_threshold", self.confidence_threshold)

            # Expand keywords for factual queries if enabled
            expanded_keywords = important_keywords
            if self.enable_keyword_expansion and query_type == "factual":
                expanded_keywords = self._expand_keywords(important_keywords)

            # Use the prefiltered indices to retrieve memories
            return self._retrieve_from_candidates(
                query_embedding, prefiltered_indices, expanded_keywords, top_k, confidence_threshold
            )

    def _retrieve_with_adaptive_depth(
        self,
        query_embedding: np.ndarray,
        query_type: str,
        important_keywords: set,
        top_k: int,
        params: dict[str, Any],
        current_input: str,
    ) -> list[dict]:
        """
        Retrieve with adaptive search depth - try multiple strategies with increasing depth.

        Args:
            query_embedding: Query embedding
            query_type: Type of query (personal, factual, etc.)
            important_keywords: Important keywords from the query
            top_k: Number of results to return
            params: Adjusted parameters for this query type
            current_input: Original user input

        Returns:
            list of retrieved memories
        """
        confidence_threshold = params.get("confidence_threshold", self.confidence_threshold)

        # Expand keywords for factual queries if enabled
        expanded_keywords = important_keywords
        if self.enable_keyword_expansion and query_type == "factual":
            expanded_keywords = self._expand_keywords(important_keywords)

        # First attempt: Try cluster-based approach (narrowest search)
        if self.clusters is not None:
            cluster_indices = self._prefilter_clusters(query_embedding)
            cluster_results = self._retrieve_from_candidates(
                query_embedding, cluster_indices, expanded_keywords, top_k, confidence_threshold
            )

            # Check if we have good results
            if len(cluster_results) >= self.min_results_guarantee and (
                len(cluster_results) > 0 and cluster_results[0]["relevance_score"] >= 0.5
            ):
                return cluster_results

        # Second attempt: Try hybrid prefiltering (wider search)
        hybrid_indices = self._prefilter_hybrid(current_input, query_embedding)
        hybrid_results = self._retrieve_from_candidates(
            query_embedding,
            hybrid_indices,
            expanded_keywords,
            top_k,
            confidence_threshold * 0.9,  # Lower threshold for wider search
        )

        # Check if we have good results
        if len(hybrid_results) >= self.min_results_guarantee:
            return hybrid_results

        # Third attempt: Try standard retrieval (full search)
        return self._retrieve_standard(
            query_embedding,
            query_type,
            important_keywords,
            top_k,
            {**params, "confidence_threshold": confidence_threshold * 0.8},  # Even lower threshold
        )

    def _retrieve_from_candidates(
        self,
        query_embedding: np.ndarray,
        candidate_indices: list[int],
        important_keywords: set,
        top_k: int,
        confidence_threshold: float,
    ) -> list[dict]:
        """
        Retrieve from a prefiltered set of candidate memories.

        Args:
            query_embedding: Query embedding
            candidate_indices: Indices of candidate memories
            important_keywords: Keywords to boost matching memories
            top_k: Number of results to return
            confidence_threshold: Minimum similarity threshold

        Returns:
            list of retrieved memories
        """
        if not candidate_indices:
            return []

        # Get embeddings for candidate memories
        candidate_embeddings = self.memory.memory_embeddings[candidate_indices]

        # Calculate similarities for candidates
        similarities = np.dot(candidate_embeddings, query_embedding)

        # Apply time decay if enabled
        if self.time_decay:
            # Get temporal markers for candidates
            temporal_markers = self.memory.temporal_markers[candidate_indices]

            # Normalize to [0, 1] range
            max_time = float(self.memory.current_time)
            if max_time > 0:
                # Calculate age factor: newer = higher value
                age_factor = temporal_markers / max_time

                # Calculate decay factor: newer = less decay
                decay_multiplier = 1.0 - ((1.0 - age_factor) * (1.0 - self.time_decay_factor))

                # Apply decay
                similarities = similarities * decay_multiplier

        # Apply activation boost if enabled
        if hasattr(self, "activation_boost") and self.activation_boost:
            activations = self.memory.activation_levels[candidate_indices]
            similarities = similarities * activations

        # Filter by threshold
        valid_mask = similarities >= confidence_threshold
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return []

        valid_similarities = similarities[valid_indices]
        valid_candidates = [candidate_indices[i] for i in valid_indices]

        # Create result objects with keyword boosting
        results = []
        for i, memory_idx in enumerate(valid_candidates):
            metadata = self.memory.memory_metadata[memory_idx]
            score = float(valid_similarities[i])

            # Apply keyword boosting
            boost = 1.0
            if important_keywords:
                boost = self._calculate_keyword_boost(metadata, important_keywords)
                score = score * boost

            results.append({
                "memory_id": int(memory_idx),
                "relevance_score": score,
                "original_score": float(valid_similarities[i]),
                "keyword_boost": boost,
                **metadata,
            })

        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Apply semantic coherence check if enabled
        if self.semantic_coherence_check and len(results) > 1:
            results = self._apply_semantic_coherence(results, query_embedding)

        # Take top-k results
        return results[:top_k]

    def _retrieve_fallback(
        self,
        query_embedding: np.ndarray,
        query_type: str,
        important_keywords: set,
        current_input: str,
        params: dict[str, Any],
    ) -> list[dict]:
        """
        Fallback retrieval method when primary methods don't find enough results.

        Args:
            query_embedding: Query embedding
            query_type: Type of query (personal, factual, etc.)
            important_keywords: Important keywords from the query
            current_input: Original user input
            params: Adjusted parameters for this query type

        Returns:
            list of retrieved memories
        """
        # Use a much lower threshold for fallback
        fallback_threshold = max(
            0.05, params.get("confidence_threshold", self.confidence_threshold) * 0.5
        )

        # Try keyword-based retrieval first (often good for finding specific memories)
        if important_keywords:
            keyword_indices = self._prefilter_keywords(current_input)
            if keyword_indices:
                keyword_results = self._retrieve_from_candidates(
                    query_embedding,
                    keyword_indices,
                    important_keywords,
                    self.min_results_guarantee * 2,  # Get more candidates
                    fallback_threshold,
                )

                if keyword_results:
                    return keyword_results

        # If that fails, use super low threshold on full memory
        return self._retrieve_by_similarity(
            query_embedding,
            self.min_results_guarantee * 2,
            important_keywords,
            fallback_threshold * 0.5,  # Even lower threshold
        )

    def _apply_semantic_coherence(
        self, candidates: list[dict], query_embedding: np.ndarray
    ) -> list[dict]:
        """
        Apply semantic coherence check to ensure retrieved memories form a coherent set.

        Args:
            candidates: Candidate memories
            query_embedding: Query embedding

        Returns:
            Coherent subset of candidates
        """
        # Convert to format expected by memory._apply_coherence_check
        candidate_tuples = [
            (
                c["memory_id"],
                c["relevance_score"],
                {k: v for k, v in c.items() if k not in ["memory_id", "relevance_score"]},
            )
            for c in candidates
        ]

        # Apply coherence check
        coherent_tuples = self.memory._apply_coherence_check(candidate_tuples, query_embedding)

        # Convert back to our format
        coherent_candidates = []
        for memory_id, score, metadata in coherent_tuples:
            coherent_candidates.append({
                "memory_id": memory_id,
                "relevance_score": score,
                **metadata,
            })

        return coherent_candidates


def test_enhanced_retrieval(
    memory_data, embedding_model, test_queries, memory_sizes=[100, 500, 1000, 5000]
):
    """
    Test the enhanced retrieval system at different memory sizes.

    Args:
        memory_data: list of memory items to use
        embedding_model: Embedding model for encoding
        test_queries: Test queries to evaluate
        memory_sizes: Memory sizes to test

    Returns:
        dictionary of results by memory size
    """
    results = {}

    print(f"Testing enhanced retrieval with memory sizes: {memory_sizes}")

    # Try to load spaCy once for all tests
    global global_nlp
    try:
        import spacy

        print("Loading spaCy model once for all tests")
        global_nlp = spacy.load("en_core_web_sm")
    except:
        print("Could not load spaCy model, will use fallback methods")

    for size in memory_sizes:
        if size > len(memory_data):
            print(f"Skipping size {size} (not enough data)")
            continue

        print(f"\nTesting with {size} memories...")

        # Sample memory data
        data_sample = memory_data[:size]

        # Initialize memory and components
        memory_dim = embedding_model.encode("test").shape[0]
        memory = ContextualMemory(embedding_dim=memory_dim)
        encoder = MemoryEncoder(embedding_model)

        # Populate memory
        populate_memory(memory, encoder, data_sample)

        # Initialize standard retriever for comparison
        standard_retriever = ContextualRetriever(
            memory=memory,
            embedding_model=embedding_model,
            use_two_stage_retrieval=True,
            query_type_adaptation=True,
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            personal_query_threshold=0.5,
            factual_query_threshold=0.2,
            adaptive_k_factor=0.15,
        )

        # Initialize enhanced retriever
        enhanced_retriever = EnhancedRetriever(
            memory=memory,
            embedding_model=embedding_model,
            use_two_stage_retrieval=True,
            query_type_adaptation=True,
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            personal_query_threshold=0.5,
            factual_query_threshold=0.2,
            adaptive_k_factor=0.15,
            use_clustering=True,
            clustering_method="hdbscan" if HDBSCAN_AVAILABLE else "kmeans",
            use_prefiltering=True,
            prefilter_method="hybrid",
            memory_size_threshold=500,  # Use optimized methods above this size
            time_decay=True,
            adaptive_depth=True,
        )

        # Test standard retriever
        print("Testing standard retrieval performance...")
        standard_performance = test_retrieval_performance(standard_retriever, test_queries)

        # Test enhanced retriever
        print("Testing enhanced retrieval performance...")
        enhanced_performance = test_retrieval_performance(enhanced_retriever, test_queries)

        # Store results
        results[size] = {
            "standard": standard_performance,
            "enhanced": enhanced_performance,
            "memory_size": size,
            "query_count": len(test_queries),
        }

        # Print summary
        print(f"Results for {size} memories:")
        print(
            f"  Standard: Precision={standard_performance['precision']:.3f}, "
            f"Recall={standard_performance['recall']:.3f}, "
            f"F1={standard_performance['f1']:.3f}, "
            f"Time={standard_performance['avg_retrieval_time']:.3f}s"
        )
        print(
            f"  Enhanced: Precision={enhanced_performance['precision']:.3f}, "
            f"Recall={enhanced_performance['recall']:.3f}, "
            f"F1={enhanced_performance['f1']:.3f}, "
            f"Time={enhanced_performance['avg_retrieval_time']:.3f}s"
        )

        # Clear memory to avoid OOM
        gc.collect()

    return results


def populate_memory(memory, encoder, memory_data):
    """Populate memory with data."""
    print(f"Populating memory with {len(memory_data)} items...")
    for item in tqdm(memory_data):
        embedding, metadata = encoder.encode_concept(
            concept=item["category"], description=item["text"], related_concepts=[item["category"]]
        )
        memory.add_memory(embedding, item["text"], metadata)


def test_retrieval_performance(retriever, queries):
    """Test retrieval performance on a set of queries."""
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    retrieval_times = []

    for query in tqdm(queries):
        # Time the retrieval operation
        start_time = time.time()
        retrieved = retriever.retrieve_for_context(query["query"], top_k=5)
        retrieval_time = time.time() - start_time
        retrieval_times.append(retrieval_time)

        # Extract retrieved texts
        retrieved_texts = [
            item.get("text", "") or item.get("content", "") or item.get("description", "")
            for item in retrieved
        ]

        # Check if expected answer is in retrieved texts
        expected = query["expected"]
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
    query_count = len(queries)
    avg_precision = total_precision / query_count if query_count else 0
    avg_recall = total_recall / query_count if query_count else 0
    avg_f1 = total_f1 / query_count if query_count else 0
    avg_retrieval_time = sum(retrieval_times) / query_count if query_count else 0

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "avg_retrieval_time": avg_retrieval_time,
        "retrieval_times": retrieval_times,
    }


def plot_results(results, output_dir="test_output/enhanced"):
    """Plot enhanced retrieval results."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    sizes = sorted(results.keys())

    standard_f1 = [results[size]["standard"]["f1"] for size in sizes]
    enhanced_f1 = [results[size]["enhanced"]["f1"] for size in sizes]

    standard_recall = [results[size]["standard"]["recall"] for size in sizes]
    enhanced_recall = [results[size]["enhanced"]["recall"] for size in sizes]

    standard_precision = [results[size]["standard"]["precision"] for size in sizes]
    enhanced_precision = [results[size]["enhanced"]["precision"] for size in sizes]

    standard_time = [results[size]["standard"]["avg_retrieval_time"] for size in sizes]
    enhanced_time = [results[size]["enhanced"]["avg_retrieval_time"] for size in sizes]
    # Create figure
    plt.figure(figsize=(15, 12))

    # Plot F1 scores
    plt.subplot(2, 2, 1)
    plt.plot(sizes, standard_f1, "b-o", label="Standard")
    plt.plot(sizes, enhanced_f1, "r-o", label="Enhanced")
    plt.xlabel("Memory Size")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Memory Size")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot precision
    plt.subplot(2, 2, 2)
    plt.plot(sizes, standard_precision, "b-o", label="Standard")
    plt.plot(sizes, enhanced_precision, "r-o", label="Enhanced")
    plt.xlabel("Memory Size")
    plt.ylabel("Precision")
    plt.title("Precision vs. Memory Size")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot recall
    plt.subplot(2, 2, 3)
    plt.plot(sizes, standard_recall, "b-o", label="Standard")
    plt.plot(sizes, enhanced_recall, "r-o", label="Enhanced")
    plt.xlabel("Memory Size")
    plt.ylabel("Recall")
    plt.title("Recall vs. Memory Size")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot retrieval time
    plt.subplot(2, 2, 4)
    plt.plot(sizes, standard_time, "b-o", label="Standard")
    plt.plot(sizes, enhanced_time, "r-o", label="Enhanced")
    plt.xlabel("Memory Size")
    plt.ylabel("Average Retrieval Time (seconds)")
    plt.title("Retrieval Time vs. Memory Size")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/enhanced_results.png")
    plt.close()

    # Save results as JSON
    with open(f"{output_dir}/enhanced_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    import sys

    # Silence warnings
    import warnings

    from transformers import AutoModel, AutoTokenizer

    warnings.filterwarnings("ignore")

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="Test enhanced retrieval strategies")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[100, 500, 1000, 2000], help="Memory sizes to test"
    )
    args = parser.parse_args()

    print("MemoryWeave Enhanced Retrieval Test")
    print("===================================")

    # Initialize embedding model
    print("Loading embedding model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    from scripts.test_optimized_retrieval import (
        EmbeddingModelWrapper,
        generate_synthetic_memories,
        generate_test_queries,
    )

    embedding_model = EmbeddingModelWrapper(model, tokenizer)

    # Generate test data
    print("Generating test data...")
    categories = ["personal", "factual", "technical"]
    memory_data = generate_synthetic_memories(max(args.sizes), categories)
    test_queries = generate_test_queries(memory_data, query_count=20)

    # Run tests
    results = test_enhanced_retrieval(memory_data, embedding_model, test_queries, args.sizes)

    # Plot results
    plot_results(results)

    print("\nEnhanced retrieval test completed. Results saved to test_output/enhanced directory.")
