"""Temporal-based retrieval strategy for MemoryWeave.

This module provides implementations of retrieval strategies based on
temporal factors such as recency and frequency of access.
"""

from typing import Dict, List, Any, Optional, Tuple
import time
import numpy as np

from memoryweave.interfaces.memory import IMemoryStore, IActivationManager, MemoryID
from memoryweave.interfaces.retrieval import IRetrievalStrategy, RetrievalResult, RetrievalParameters


class TemporalRetrievalStrategy(IRetrievalStrategy):
    """Retrieval strategy based on memory recency and activation."""
    
    def __init__(self, memory_store: IMemoryStore, activation_manager: IActivationManager):
        """Initialize the temporal retrieval strategy.
        
        Args:
            memory_store: Memory store to retrieve memory content
            activation_manager: Activation manager for memory activations
        """
        self._memory_store = memory_store
        self._activation_manager = activation_manager
        self._default_params = {
            'max_results': 10,
            'recency_window_days': 7.0
        }
    
    def retrieve(self, 
                query_embedding: Optional[EmbeddingVector] = None, 
                parameters: Optional[RetrievalParameters] = None) -> List[RetrievalResult]:
        """Retrieve memories based on temporal factors.
        
        Note:
            The query_embedding is optional for this strategy, as it relies
            on temporal factors rather than vector similarity.
        """
        # Merge parameters with defaults
        params = self._default_params.copy()
        if parameters:
            params.update(parameters)
        
        # Get parameters
        max_results = params.get('max_results', 10)
        recency_window_days = params.get('recency_window_days', 7.0)
        
        # Get most active memories
        active_memories = self._activation_manager.get_most_active(max_results)
        
        # Convert results to RetrievalResult format
        results = []
        for memory_id, activation in active_memories:
            # Retrieve memory content
            memory = self._memory_store.get(memory_id)
            
            # Use creation time as additional metadata
            creation_time = memory.metadata.get('created_at', 0)
            current_time = time.time()
            
            # Convert activation to a relevance score (0-1)
            recency_factor = self._calculate_recency_factor(
                creation_time, current_time, recency_window_days
            )
            
            # Combine activation and recency
            relevance_score = 0.5 * (activation / 10.0 + 0.5) + 0.5 * recency_factor
            
            # Create result
            result = RetrievalResult(
                memory_id=memory_id,
                content=memory.content['text'],
                metadata=memory.metadata,
                relevance_score=float(relevance_score)
            )
            
            results.append(result)
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the retrieval strategy."""
        if 'max_results' in config:
            self._default_params['max_results'] = config['max_results']
        
        if 'recency_window_days' in config:
            self._default_params['recency_window_days'] = config['recency_window_days']
    
    def _calculate_recency_factor(self, 
                                 creation_time: float, 
                                 current_time: float, 
                                 recency_window_days: float) -> float:
        """Calculate a recency factor (0-1) based on memory age."""
        # Calculate age in seconds
        age_seconds = current_time - creation_time
        
        # Convert window to seconds
        window_seconds = recency_window_days * 24 * 60 * 60
        
        # Calculate recency factor
        # 1.0 for brand new, approaching 0.0 as age approaches window
        if age_seconds <= 0:
            return 1.0
        elif age_seconds >= window_seconds:
            return 0.0
        else:
            return 1.0 - (age_seconds / window_seconds)