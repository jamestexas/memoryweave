"""
Mock models for testing MemoryWeave components.
"""

import numpy as np


class MockEmbeddingModel:
    """
    A deterministic embedding model for testing.
    
    This model generates consistent embeddings based on the input text,
    allowing for reproducible test results.
    """
    
    def __init__(self, embedding_dim=768, seed=42):
        """
        Initialize the mock embedding model.
        
        Args:
            embedding_dim: Dimension of the embeddings
            seed: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.cache = {}  # Cache embeddings for consistency
        
    def encode(self, text):
        """
        Generate a deterministic embedding for the given text.
        
        Args:
            text: Input text to encode
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        # Check cache first for consistency
        if text in self.cache:
            return self.cache[text]
            
        # Generate a deterministic hash from the text
        text_hash = hash(text) % 10000
        
        # Set the random seed based on the text hash for deterministic output
        np.random.seed(self.seed + text_hash)
        
        # Generate a random embedding
        embedding = np.random.randn(self.embedding_dim)
        
        # Normalize to unit length (common for embeddings)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Cache the result
        self.cache[text] = embedding
        
        return embedding


class MockMemory:
    """
    A simplified mock of ContextualMemory for testing components.
    """
    
    def __init__(self, embedding_dim=768):
        """
        Initialize the mock memory.
        
        Args:
            embedding_dim: Dimension of the embeddings
        """
        self.embedding_dim = embedding_dim
        self.memory_embeddings = np.zeros((0, embedding_dim))
        self.memory_metadata = []
        self.activation_levels = np.array([])
        self.temporal_markers = np.array([])
        self.current_time = 0
        
    def add_memory(self, embedding, content, metadata=None):
        """
        Add a memory to the mock memory system.
        
        Args:
            embedding: The embedding vector
            content: The memory content
            metadata: Additional metadata
        """
        if metadata is None:
            metadata = {}
            
        # Add the embedding
        self.memory_embeddings = np.vstack([self.memory_embeddings, embedding.reshape(1, -1)])
        
        # Add metadata
        if "content" not in metadata:
            metadata["content"] = content
        self.memory_metadata.append(metadata)
        
        # Update activation and temporal markers
        self.activation_levels = np.append(self.activation_levels, 1.0)
        self.current_time += 1
        self.temporal_markers = np.append(self.temporal_markers, self.current_time)
        
        return len(self.memory_metadata) - 1  # Return the index of the added memory
        
    def retrieve_memories(self, query_embedding, top_k=5, **kwargs):
        """
        Retrieve memories based on similarity to query embedding.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of memories to retrieve
            **kwargs: Additional arguments
            
        Returns:
            List of (memory_id, score, metadata) tuples
        """
        if len(self.memory_embeddings) == 0:
            return []
            
        # Calculate similarities
        similarities = np.dot(self.memory_embeddings, query_embedding)
        
        # Get top-k indices
        if top_k >= len(similarities):
            top_indices = np.argsort(-similarities)
        else:
            top_indices = np.argpartition(-similarities, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]
            
        # Format results
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(similarities[idx]),
                self.memory_metadata[idx]
            ))
            
        return results
