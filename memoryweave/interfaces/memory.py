"""Memory interface definitions for MemoryWeave.

This module defines the core interfaces for memory storage and retrieval,
including data models, protocols, and base classes for memory components.
"""

from typing import Protocol, TypedDict, List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from enum import Enum, auto

# Type aliases
MemoryID = Union[int, str]
EmbeddingVector = np.ndarray


class MemoryContent(TypedDict):
    """Content of a memory."""
    text: str
    metadata: Dict[str, Any]


@dataclass
class Memory:
    """Data model for a memory in the system."""
    id: MemoryID
    embedding: EmbeddingVector
    content: MemoryContent
    metadata: Dict[str, Any]


class MemoryType(Enum):
    """Types of memories that can be stored."""
    INTERACTION = auto()
    FACT = auto()
    CONCEPT = auto()
    ATTRIBUTE = auto()
    EVENT = auto()


class IMemoryStore(Protocol):
    """Interface for memory storage component."""
    
    def add(self, 
            embedding: EmbeddingVector, 
            content: str, 
            metadata: Optional[Dict[str, Any]] = None) -> MemoryID:
        """Add a memory and return its ID."""
        ...
        
    def get(self, memory_id: MemoryID) -> Memory:
        """Retrieve a memory by ID."""
        ...
        
    def get_all(self) -> List[Memory]:
        """Retrieve all memories."""
        ...
        
    def get_embeddings(self) -> np.ndarray:
        """Get all embeddings as a matrix."""
        ...
        
    def get_ids(self) -> List[MemoryID]:
        """Get all memory IDs."""
        ...
        
    def update_activation(self, memory_id: MemoryID, activation_delta: float) -> None:
        """Update activation level of a memory."""
        ...
        
    def update_metadata(self, memory_id: MemoryID, metadata: Dict[str, Any]) -> None:
        """Update metadata of a memory."""
        ...
        
    def remove(self, memory_id: MemoryID) -> None:
        """Remove a memory from the store."""
        ...
        
    def clear(self) -> None:
        """Clear all memories from the store."""
        ...
        
    def consolidate(self, max_memories: int) -> List[MemoryID]:
        """Consolidate memories to stay within capacity.
        
        Returns:
            List of memory IDs that were removed during consolidation.
        """
        ...


class IVectorStore(Protocol):
    """Interface for vector storage with similarity search capabilities."""
    
    def add(self, id: MemoryID, vector: EmbeddingVector) -> None:
        """Add a vector to the store."""
        ...
        
    def search(self, 
               query_vector: EmbeddingVector, 
               k: int, 
               threshold: Optional[float] = None) -> List[tuple[MemoryID, float]]:
        """Search for similar vectors.
        
        Args:
            query_vector: The query vector to search for
            k: Maximum number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of tuples containing (memory_id, similarity_score)
        """
        ...
        
    def remove(self, id: MemoryID) -> None:
        """Remove a vector from the store."""
        ...
        
    def clear(self) -> None:
        """Clear all vectors from the store."""
        ...


class IActivationManager(Protocol):
    """Interface for managing memory activation levels."""
    
    def update_activation(self, memory_id: MemoryID, activation_delta: float) -> None:
        """Update the activation level of a memory."""
        ...
        
    def get_activation(self, memory_id: MemoryID) -> float:
        """Get the current activation level of a memory."""
        ...
        
    def decay_activations(self, decay_factor: float) -> None:
        """Apply decay to all memory activations."""
        ...
        
    def get_most_active(self, k: int) -> List[tuple[MemoryID, float]]:
        """Get the k most active memories."""
        ...


class ICategoryManager(Protocol):
    """Interface for category management."""
    
    def add_to_category(self, memory_id: MemoryID, embedding: EmbeddingVector) -> int:
        """Add a memory to a category and return the category ID."""
        ...
        
    def get_category(self, memory_id: MemoryID) -> int:
        """Get the category ID for a memory."""
        ...
        
    def get_category_members(self, category_id: int) -> List[MemoryID]:
        """Get all memories in a category."""
        ...
        
    def get_category_prototype(self, category_id: int) -> EmbeddingVector:
        """Get the prototype vector for a category."""
        ...
        
    def consolidate_categories(self, similarity_threshold: float) -> List[int]:
        """Merge similar categories.
        
        Returns:
            List of category IDs that were consolidated.
        """
        ...


class IMemoryEncoder(Protocol):
    """Interface for encoding content into memory embeddings."""
    
    def encode_text(self, text: str) -> EmbeddingVector:
        """Encode text into an embedding vector."""
        ...
        
    def encode_interaction(self, 
                          query: str, 
                          response: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> EmbeddingVector:
        """Encode a query-response interaction into an embedding vector."""
        ...
        
    def encode_concept(self, 
                       concept: str, 
                       definition: str, 
                       examples: Optional[List[str]] = None) -> EmbeddingVector:
        """Encode a concept into an embedding vector."""
        ...