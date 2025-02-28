"""
Retrieval strategy components for MemoryWeave.
"""

import numpy as np

from .base import Component


class RetrievalStrategy(Component):
    """Base class for retrieval strategies."""
    
    def __init__(self, memory=None):
        """
        Initialize the retrieval strategy.
        
        Args:
            memory: Memory instance to use for retrieval
        """
        self.memory = memory
        self.confidence_threshold = 0.0
    
    def initialize(self, config):
        """
        Initialize the retrieval strategy with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.confidence_threshold = config.get('confidence_threshold', 0.0)
    
    def retrieve(self, query_embedding, top_k, context):
        """
        Retrieve memories based on the query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of memories to retrieve
            context: Additional context (e.g., memory)
            
        Returns:
            List of retrieved memory dicts with memory_id, content, and metadata
        """
        raise NotImplementedError("Subclasses must implement retrieve")
    
    def process_query(self, query, context):
        """
        Process a query to retrieve relevant memories.
        
        This adapter method converts the query to embedding and calls retrieve.
        
        Args:
            query: The query string
            context: Context dictionary containing query_embedding, memory, etc.
            
        Returns:
            Updated context with results
        """
        query_embedding = context.get("query_embedding")
        top_k = context.get("top_k", 5)
        
        # If no query embedding is provided, return empty results
        if query_embedding is None:
            context["results"] = []
            return context
        
        # Use the memory from context or instance
        memory = context.get("memory", self.memory)
        
        # Retrieve memories
        results = self.retrieve(query_embedding, top_k, {"memory": memory})
        
        # Add results to context
        context["results"] = results
        return context


class SimilarityRetrievalStrategy(RetrievalStrategy):
    """Retrieval strategy based on embedding similarity."""
    
    def retrieve(self, query_embedding, top_k, context):
        """
        Retrieve memories based on embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of memories to retrieve
            context: Additional context (e.g., memory)
            
        Returns:
            List of retrieved memory dicts
        """
        memory = context.get("memory", self.memory)
        
        # Get raw results from memory
        raw_results = memory.retrieve_memories(
            query_embedding, top_k=top_k*2
        )
        
        # Format results and apply confidence threshold
        results = []
        for memory_id, similarity, metadata in raw_results:
            if similarity < self.confidence_threshold:
                continue
                
            content = metadata.get("content", "")
            results.append({
                "memory_id": memory_id,
                "content": content,
                "metadata": metadata,
                "similarity": float(similarity),
                "score": float(similarity)
            })
            
            if len(results) >= top_k:
                break
                
        return results


class TemporalRetrievalStrategy(RetrievalStrategy):
    """Retrieval strategy based on recency."""
    
    def retrieve(self, query_embedding, top_k, context):
        """
        Retrieve memories based on recency.
        
        Args:
            query_embedding: Query embedding vector (not used)
            top_k: Number of memories to retrieve
            context: Additional context (e.g., memory)
            
        Returns:
            List of retrieved memory dicts
        """
        memory = context.get("memory", self.memory)
        
        # Get temporal markers and create sorted indices
        temporal_markers = memory.temporal_markers
        sorted_indices = np.argsort(-temporal_markers)[:top_k]
        
        # Format results
        results = []
        for i, idx in enumerate(sorted_indices):
            if i >= top_k:
                break
                
            memory_id = int(idx)
            metadata = memory.memory_metadata[memory_id]
            content = metadata.get("content", "")
            recency = 1.0 - (i / (len(sorted_indices) or 1))
            
            results.append({
                "memory_id": memory_id,
                "content": content,
                "metadata": metadata,
                "recency": float(recency),
                "score": float(recency)
            })
            
        return results


class HybridRetrievalStrategy(RetrievalStrategy):
    """
    Hybrid retrieval strategy combining similarity and recency.
    """
    
    def initialize(self, config):
        """
        Initialize the hybrid retrieval strategy.
        
        Args:
            config: Configuration dictionary containing:
                - relevance_weight: Weight for similarity score
                - recency_weight: Weight for recency score
                - confidence_threshold: Minimum similarity score
        """
        super().initialize(config)
        self.relevance_weight = config.get('relevance_weight', 0.7)
        self.recency_weight = config.get('recency_weight', 0.3)
    
    def retrieve(self, query_embedding, top_k, context):
        """
        Retrieve memories based on weighted combination of similarity and recency.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of memories to retrieve
            context: Additional context (e.g., memory)
            
        Returns:
            List of retrieved memory dicts
        """
        memory = context.get("memory", self.memory)
        
        # Get raw results
        raw_results = memory.retrieve_memories(
            query_embedding, top_k=top_k*3
        )
        
        # Calculate hybrid scores
        hybrid_results = []
        for memory_id, similarity, metadata in raw_results:
            if similarity < self.confidence_threshold:
                continue
                
            # Get recency score (normalized by position in temporal markers)
            recency = 1.0
            if hasattr(memory, 'temporal_markers') and len(memory.temporal_markers) > 0:
                temp_pos = memory.temporal_markers[memory_id]
                latest_pos = max(memory.temporal_markers)
                recency = temp_pos / (latest_pos or 1)
            
            # Calculate hybrid score
            hybrid_score = (
                self.relevance_weight * similarity +
                self.recency_weight * recency
            )
            
            content = metadata.get("content", "")
            hybrid_results.append({
                "memory_id": memory_id,
                "content": content,
                "metadata": metadata,
                "similarity": float(similarity),
                "recency": float(recency),
                "score": float(hybrid_score)
            })
        
        # Sort by hybrid score and take top_k
        hybrid_results.sort(key=lambda x: x["score"], reverse=True)
        return hybrid_results[:top_k]
