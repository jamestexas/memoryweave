"""Retrieval adapter for MemoryWeave.

This module provides adapters that bridge between the old retrieval architecture
and the new component-based architecture.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from memoryweave.interfaces.retrieval import (
    IRetrievalStrategy, RetrievalResult, RetrievalParameters, Query, QueryType
)
from memoryweave.interfaces.pipeline import IComponent, ComponentType, ComponentID
from memoryweave.adapters.memory_adapter import LegacyMemoryAdapter


class LegacyRetrieverAdapter(IRetrievalStrategy):
    """
    Adapter that bridges a legacy ContextualRetriever to the new IRetrievalStrategy interface.
    
    This adapter allows using a legacy retriever where a new IRetrievalStrategy
    is expected, providing backward compatibility during migration.
    """
    
    def __init__(self, legacy_retriever, memory_adapter: LegacyMemoryAdapter,
                 component_id: str = "legacy_retriever_adapter"):
        """
        Initialize the legacy retriever adapter.
        
        Args:
            legacy_retriever: A legacy ContextualRetriever object or MemoryRetriever
            memory_adapter: The memory adapter for ID mapping
            component_id: ID for this component
        """
        self._legacy_retriever = legacy_retriever
        self._memory_adapter = memory_adapter
        self._component_id = component_id
        self._default_params = {
            'similarity_threshold': 0.0,
            'max_results': 5,
            'recency_bias': 0.0,
            'activation_boost': True,
            'keyword_weight': 0.0,
            'min_results': 0
        }
    
    def retrieve(self, 
                query_embedding: np.ndarray, 
                parameters: Optional[RetrievalParameters] = None) -> List[RetrievalResult]:
        """Retrieve memories based on a query embedding."""
        # Merge parameters with defaults
        params = self._default_params.copy()
        if parameters:
            params.update(parameters)
        
        # Extract parameters for legacy retriever
        top_k = params.get('max_results', 5)
        activation_boost = params.get('activation_boost', True)
        confidence_threshold = params.get('similarity_threshold', 0.0)
        
        # Determine if we're using categories
        use_categories = None
        if 'use_categories' in params:
            use_categories = params['use_categories']
        
        # Check the type of legacy retriever
        if hasattr(self._legacy_retriever, 'retrieve_memories'):
            # This is a MemoryRetriever
            legacy_results = self._legacy_retriever.retrieve_memories(
                query_embedding=query_embedding,
                top_k=top_k,
                activation_boost=activation_boost,
                use_categories=use_categories,
                confidence_threshold=confidence_threshold,
                max_k_override=params.get('min_results', 0) > 0
            )
        elif hasattr(self._legacy_retriever, 'retrieve_for_context'):
            # This is a ContextualRetriever
            legacy_results = self._legacy_retriever.retrieve_for_context(
                query_embedding=query_embedding,
                k=top_k,
                threshold=confidence_threshold
            )
        else:
            # Unknown retriever type
            raise TypeError("Unknown legacy retriever type")
        
        # Convert legacy results to RetrievalResult format
        results = []
        for legacy_item in legacy_results:
            # Handle different legacy result formats
            if isinstance(legacy_item, tuple) and len(legacy_item) == 3:
                # Format is (memory_idx, similarity_score, metadata)
                legacy_idx, similarity, metadata = legacy_item
            elif isinstance(legacy_item, dict) and 'memory_idx' in legacy_item:
                # Format is a dictionary with memory_idx, score, and content
                legacy_idx = legacy_item['memory_idx']
                similarity = legacy_item.get('score', 0.0)
                metadata = legacy_item.get('metadata', {})
            else:
                # Unknown format
                continue
            
            # Find the new memory ID for this legacy index
            memory_id = None
            for new_id, idx in self._memory_adapter._memory_id_map.items():
                if idx == legacy_idx:
                    memory_id = new_id
                    break
            
            if memory_id is None:
                # Create a new ID for this memory if not found
                memory_id = str(len(self._memory_adapter._memory_id_map))
                self._memory_adapter._memory_id_map[memory_id] = legacy_idx
            
            # Extract text content
            if 'text' in metadata:
                content = metadata['text']
            else:
                # Try to get content from memory store
                try:
                    memory = self._memory_adapter.get(memory_id)
                    content = memory['content']['text']
                except (KeyError, AttributeError):
                    # If all else fails, use empty string
                    content = ""
            
            # Create RetrievalResult
            result = RetrievalResult(
                memory_id=memory_id,
                content=content,
                metadata=metadata,
                relevance_score=float(similarity)
            )
            
            results.append(result)
        
        return results
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the retrieval strategy."""
        # Update default parameters with configuration
        for key in self._default_params:
            if key in config:
                self._default_params[key] = config[key]
    
    # IComponent interface methods
    
    def get_id(self) -> ComponentID:
        """Get the unique identifier for this component."""
        return self._component_id
    
    def get_type(self) -> ComponentType:
        """Get the type of this component."""
        return ComponentType.RETRIEVAL_STRATEGY
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        self.configure(config)
    
    def get_dependencies(self) -> List[ComponentID]:
        """Get the IDs of components this component depends on."""
        return [self._memory_adapter.get_id()]


class NewToLegacyRetrieverAdapter:
    """
    Adapter that bridges the new retrieval architecture to the legacy interface.
    
    This adapter allows using the new retrieval components with code that
    expects the legacy retriever interface, providing backward compatibility.
    """
    
    def __init__(self, retrieval_strategy: IRetrievalStrategy):
        """
        Initialize the adapter.
        
        Args:
            retrieval_strategy: The new retrieval strategy to adapt
        """
        self._retrieval_strategy = retrieval_strategy
    
    def retrieve_for_context(self, 
                           query_embedding: np.ndarray,
                           k: int = 5,
                           threshold: float = 0.0,
                           **kwargs) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Legacy interface for retrieving memories based on context.
        
        Args:
            query_embedding: The query embedding
            k: Number of results to retrieve
            threshold: Minimum similarity threshold
            **kwargs: Additional parameters
            
        Returns:
            List of (memory_idx, similarity_score, metadata) tuples
        """
        # Convert parameters to new format
        parameters = {
            'max_results': k,
            'similarity_threshold': threshold
        }
        
        # Add additional parameters
        if 'activation_boost' in kwargs:
            parameters['activation_boost'] = kwargs['activation_boost']
        if 'use_categories' in kwargs:
            parameters['use_categories'] = kwargs['use_categories']
        
        # Retrieve using new strategy
        results = self._retrieval_strategy.retrieve(query_embedding, parameters)
        
        # Convert to legacy format
        legacy_results = []
        for result in results:
            # Convert memory_id to int if possible (legacy uses int indices)
            try:
                memory_idx = int(result['memory_id'])
            except (ValueError, TypeError):
                # If conversion fails, use a hash of the string as a unique index
                memory_idx = hash(result['memory_id']) % 10000000
            
            # Create legacy result tuple
            legacy_result = (
                memory_idx,
                result['relevance_score'],
                result['metadata']
            )
            
            legacy_results.append(legacy_result)
        
        return legacy_results