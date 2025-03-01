"""Memory adapter for MemoryWeave.

This module provides adapters that bridge between the old memory architecture
and the new component-based architecture.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from memoryweave.interfaces.memory import IMemoryStore, IVectorStore, IActivationManager, MemoryID
from memoryweave.interfaces.pipeline import IComponent, ComponentType, ComponentID


class LegacyMemoryAdapter(IMemoryStore):
    """
    Adapter that bridges a legacy ContextualMemory to the new IMemoryStore interface.
    
    This adapter allows using a legacy ContextualMemory where a new IMemoryStore
    is expected, providing backward compatibility while migrating to the new architecture.
    """
    
    def __init__(self, legacy_memory, component_id: str = "legacy_memory_adapter"):
        """
        Initialize the legacy memory adapter.
        
        Args:
            legacy_memory: A legacy ContextualMemory object
            component_id: ID for this component
        """
        self._legacy_memory = legacy_memory
        self._component_id = component_id
        self._memory_id_map = {}  # Maps new memory IDs to old memory indices
        self._next_id = 0
    
    def add(self, 
            embedding: np.ndarray, 
            content: str, 
            metadata: Optional[Dict[str, Any]] = None) -> MemoryID:
        """Add a memory and return its ID."""
        # Add to legacy memory
        if metadata is None:
            metadata = {}
        
        # Add text to metadata for legacy compatibility
        metadata_with_text = metadata.copy()
        if 'text' not in metadata_with_text:
            metadata_with_text['text'] = content
        
        # Call legacy add_memory
        legacy_idx = self._legacy_memory.add_memory(embedding, content, metadata_with_text)
        
        # Assign a new memory ID and map it to the legacy index
        memory_id = str(self._next_id)
        self._next_id += 1
        self._memory_id_map[memory_id] = legacy_idx
        
        return memory_id
    
    def get(self, memory_id: MemoryID) -> Dict[str, Any]:
        """Retrieve a memory by ID."""
        if memory_id not in self._memory_id_map:
            raise KeyError(f"Memory with ID {memory_id} not found")
        
        legacy_idx = self._memory_id_map[memory_id]
        
        # Get embedding from legacy memory
        embedding = self._legacy_memory.memory_embeddings[legacy_idx]
        
        # Get metadata from legacy memory
        metadata = self._legacy_memory.memory_metadata[legacy_idx]
        
        # Extract text content from metadata
        text = metadata.get('text', '')
        
        # Construct memory object
        memory = {
            'id': memory_id,
            'embedding': embedding,
            'content': {'text': text, 'metadata': metadata},
            'metadata': metadata
        }
        
        return memory
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve all memories."""
        return [self.get(memory_id) for memory_id in self._memory_id_map.keys()]
    
    def get_embeddings(self) -> np.ndarray:
        """Get all embeddings as a matrix."""
        return self._legacy_memory.memory_embeddings
    
    def get_ids(self) -> List[MemoryID]:
        """Get all memory IDs."""
        return list(self._memory_id_map.keys())
    
    def update_activation(self, memory_id: MemoryID, activation_delta: float) -> None:
        """Update activation level of a memory."""
        if memory_id not in self._memory_id_map:
            raise KeyError(f"Memory with ID {memory_id} not found")
        
        legacy_idx = self._memory_id_map[memory_id]
        
        # Update activation in legacy memory
        if hasattr(self._legacy_memory, '_update_activation'):
            self._legacy_memory._update_activation(legacy_idx)
        elif hasattr(self._legacy_memory, 'core_memory') and hasattr(self._legacy_memory.core_memory, 'update_activation'):
            self._legacy_memory.core_memory.update_activation(legacy_idx)
        else:
            # Fallback if no update_activation method exists
            current_activation = self._legacy_memory.activation_levels[legacy_idx]
            self._legacy_memory.activation_levels[legacy_idx] = current_activation + activation_delta
    
    def update_metadata(self, memory_id: MemoryID, metadata: Dict[str, Any]) -> None:
        """Update metadata of a memory."""
        if memory_id not in self._memory_id_map:
            raise KeyError(f"Memory with ID {memory_id} not found")
        
        legacy_idx = self._memory_id_map[memory_id]
        
        # Update metadata in legacy memory
        self._legacy_memory.memory_metadata[legacy_idx].update(metadata)
    
    def remove(self, memory_id: MemoryID) -> None:
        """Remove a memory from the store."""
        # Legacy memory doesn't support direct removal, so we warn and do nothing
        if memory_id in self._memory_id_map:
            # Just remove from our mapping
            del self._memory_id_map[memory_id]
    
    def clear(self) -> None:
        """Clear all memories from the store."""
        # Legacy memory doesn't support direct clear, so we warn and do nothing
        self._memory_id_map.clear()
        self._next_id = 0
    
    def consolidate(self, max_memories: int) -> List[MemoryID]:
        """Consolidate memories to stay within capacity."""
        # Legacy memory handles consolidation internally
        # Just return an empty list as no explicit consolidation is performed
        return []
    
    # IComponent interface methods
    
    def get_id(self) -> ComponentID:
        """Get the unique identifier for this component."""
        return self._component_id
    
    def get_type(self) -> ComponentType:
        """Get the type of this component."""
        return ComponentType.MEMORY_STORE
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        # No initialization needed, legacy memory is already configured
        pass
    
    def get_dependencies(self) -> List[ComponentID]:
        """Get the IDs of components this component depends on."""
        return []


class LegacyVectorStoreAdapter(IVectorStore):
    """
    Adapter that bridges a legacy ContextualMemory to the new IVectorStore interface.
    
    This adapter allows using a legacy ContextualMemory where a new IVectorStore
    is expected, providing backward compatibility during migration.
    """
    
    def __init__(self, legacy_memory, memory_adapter: LegacyMemoryAdapter, 
                 component_id: str = "legacy_vector_store_adapter"):
        """
        Initialize the legacy vector store adapter.
        
        Args:
            legacy_memory: A legacy ContextualMemory object
            memory_adapter: The memory adapter for ID mapping
            component_id: ID for this component
        """
        self._legacy_memory = legacy_memory
        self._memory_adapter = memory_adapter
        self._component_id = component_id
    
    def add(self, id: MemoryID, vector: np.ndarray) -> None:
        """Add a vector to the store."""
        # Legacy memory adds vectors through add_memory
        # This is a no-op as the vector should already be in the legacy memory
        pass
    
    def search(self, 
               query_vector: np.ndarray, 
               k: int, 
               threshold: Optional[float] = None) -> List[Tuple[MemoryID, float]]:
        """Search for similar vectors."""
        # Use the legacy memory's retrieve_memories method
        legacy_results = self._legacy_memory.retrieve_memories(
            query_embedding=query_vector,
            top_k=k,
            activation_boost=False,  # Don't apply activation boost for pure similarity search
            confidence_threshold=threshold or 0.0
        )
        
        # Convert legacy results to the expected format
        results = []
        for legacy_idx, similarity, _ in legacy_results:
            # Find the new memory ID for this legacy index
            memory_id = None
            for new_id, idx in self._memory_adapter._memory_id_map.items():
                if idx == legacy_idx:
                    memory_id = new_id
                    break
            
            if memory_id is not None:
                results.append((memory_id, similarity))
        
        return results
    
    def remove(self, id: MemoryID) -> None:
        """Remove a vector from the store."""
        # Legacy memory doesn't support direct removal
        pass
    
    def clear(self) -> None:
        """Clear all vectors from the store."""
        # Legacy memory doesn't support direct clear
        pass
    
    # IComponent interface methods
    
    def get_id(self) -> ComponentID:
        """Get the unique identifier for this component."""
        return self._component_id
    
    def get_type(self) -> ComponentType:
        """Get the type of this component."""
        return ComponentType.VECTOR_STORE
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        # No initialization needed, legacy memory is already configured
        pass
    
    def get_dependencies(self) -> List[ComponentID]:
        """Get the IDs of components this component depends on."""
        return [self._memory_adapter.get_id()]


class LegacyActivationManagerAdapter(IActivationManager):
    """
    Adapter that bridges a legacy ContextualMemory to the new IActivationManager interface.
    
    This adapter allows using a legacy ContextualMemory where a new IActivationManager
    is expected, providing backward compatibility during migration.
    """
    
    def __init__(self, legacy_memory, memory_adapter: LegacyMemoryAdapter,
                 component_id: str = "legacy_activation_manager_adapter"):
        """
        Initialize the legacy activation manager adapter.
        
        Args:
            legacy_memory: A legacy ContextualMemory object
            memory_adapter: The memory adapter for ID mapping
            component_id: ID for this component
        """
        self._legacy_memory = legacy_memory
        self._memory_adapter = memory_adapter
        self._component_id = component_id
    
    def update_activation(self, memory_id: MemoryID, activation_delta: float) -> None:
        """Update the activation level of a memory."""
        self._memory_adapter.update_activation(memory_id, activation_delta)
    
    def get_activation(self, memory_id: MemoryID) -> float:
        """Get the current activation level of a memory."""
        if memory_id not in self._memory_adapter._memory_id_map:
            raise KeyError(f"Memory with ID {memory_id} not found")
        
        legacy_idx = self._memory_adapter._memory_id_map[memory_id]
        return float(self._legacy_memory.activation_levels[legacy_idx])
    
    def decay_activations(self, decay_factor: float) -> None:
        """Apply decay to all memory activations."""
        # Legacy memory doesn't have a direct decay method
        # Apply decay manually
        self._legacy_memory.activation_levels *= (1.0 - decay_factor)
    
    def get_most_active(self, k: int) -> List[Tuple[MemoryID, float]]:
        """Get the k most active memories."""
        # Get activations from legacy memory
        activations = self._legacy_memory.activation_levels
        
        # Get top k indices
        if len(activations) <= k:
            top_indices = np.argsort(-activations)
        else:
            top_indices = np.argpartition(-activations, k)[:k]
            top_indices = top_indices[np.argsort(-activations[top_indices])]
        
        # Convert to memory IDs and return
        results = []
        for legacy_idx in top_indices:
            # Find the new memory ID for this legacy index
            memory_id = None
            for new_id, idx in self._memory_adapter._memory_id_map.items():
                if idx == legacy_idx:
                    memory_id = new_id
                    break
            
            if memory_id is not None:
                results.append((memory_id, float(activations[legacy_idx])))
        
        return results
    
    # IComponent interface methods
    
    def get_id(self) -> ComponentID:
        """Get the unique identifier for this component."""
        return self._component_id
    
    def get_type(self) -> ComponentType:
        """Get the type of this component."""
        return ComponentType.MEMORY_STORE  # There's no ACTIVATION_MANAGER type
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        # No initialization needed, legacy memory is already configured
        pass
    
    def get_dependencies(self) -> List[ComponentID]:
        """Get the IDs of components this component depends on."""
        return [self._memory_adapter.get_id()]