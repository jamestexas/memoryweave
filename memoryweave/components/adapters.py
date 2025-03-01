"""
Adapters for integrating core components with the pipeline architecture.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from memoryweave.components.base import RetrievalComponent
from memoryweave.core.memory_retriever import MemoryRetriever
from memoryweave.core.contextual_memory import ContextualMemory


class CoreRetrieverAdapter(RetrievalComponent):
    """
    Adapter for using the core MemoryRetriever in the pipeline architecture.
    
    This adapter wraps the core MemoryRetriever and exposes it through the
    component interface defined by the pipeline architecture.
    """
    
    def __init__(
        self,
        memory: ContextualMemory,
        default_top_k: int = 5,
        confidence_threshold: float = 0.0,
    ):
        """
        Initialize the adapter.
        
        Args:
            memory: The memory instance to use for retrieval
            default_top_k: Default number of results to retrieve
            confidence_threshold: Default confidence threshold for retrieval
        """
        self.memory = memory
        self.default_top_k = default_top_k
        self.confidence_threshold = confidence_threshold
        self.activation_boost = True
        self.use_categories = True
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        self.confidence_threshold = config.get('confidence_threshold', self.confidence_threshold)
        self.default_top_k = config.get('top_k', self.default_top_k)
        self.use_categories = config.get('use_categories', self.use_categories)
        self.activation_boost = config.get('activation_boost', self.activation_boost)
        
    def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query to retrieve relevant memories.
        
        Args:
            query: The query string
            context: Context containing query_embedding, etc.
            
        Returns:
            Updated context with results
        """
        # Get query embedding from context
        query_embedding = context.get('query_embedding')
        if query_embedding is None:
            # Try to get embedding model and create embedding
            embedding_model = context.get('embedding_model')
            if embedding_model:
                query_embedding = embedding_model.encode(query)
                
        # If still no embedding, return empty results
        if query_embedding is None:
            return {'results': []}
            
        # Get top_k from context or use default
        top_k = context.get('top_k', self.default_top_k)
        
        # Apply query type adaptation if available
        adapted_params = context.get('adapted_retrieval_params', {})
        confidence_threshold = adapted_params.get(
            'confidence_threshold', self.confidence_threshold
        )
        
        # Retrieve memories
        results = self.memory.retrieve_memories(
            query_embedding=query_embedding,
            top_k=top_k,
            activation_boost=self.activation_boost,
            use_categories=self.use_categories,
            confidence_threshold=confidence_threshold,
        )
        
        # Format results as dictionaries
        formatted_results = []
        for idx, score, metadata in results:
            formatted_results.append({
                'memory_id': idx,
                'relevance_score': score,
                **metadata
            })
            
        return {'results': formatted_results}
