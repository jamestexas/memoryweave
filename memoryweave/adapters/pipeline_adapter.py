"""Pipeline adapter for MemoryWeave.

This module provides adapters that bridge between the old architecture
and the new pipeline-based architecture.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from memoryweave.interfaces.retrieval import (
    IRetrievalStrategy, RetrievalResult, RetrievalParameters, 
    Query, QueryType, QueryContext
)
from memoryweave.interfaces.query import IQueryAnalyzer, IQueryAdapter
from memoryweave.interfaces.pipeline import (
    IComponent, ComponentType, ComponentID, IPipeline, IPipelineStage
)
from memoryweave.adapters.memory_adapter import LegacyMemoryAdapter
from memoryweave.adapters.retrieval_adapter import LegacyRetrieverAdapter
from memoryweave.pipeline.manager import PipelineManager


class LegacyToPipelineAdapter:
    """
    Adapter that bridges the legacy retriever interfaces to the new pipeline architecture.
    
    This adapter allows using the legacy retriever systems with the new component-based
    pipeline architecture, providing backward compatibility during migration.
    """
    
    def __init__(self, 
                legacy_memory, 
                pipeline_manager: Optional[PipelineManager] = None,
                component_id: str = "legacy_pipeline_adapter"):
        """
        Initialize the adapter.
        
        Args:
            legacy_memory: Legacy ContextualMemory object
            pipeline_manager: Optional pipeline manager to use
            component_id: ID for this component
        """
        self._legacy_memory = legacy_memory
        self._component_id = component_id
        
        # Create pipeline manager if none provided
        self._pipeline_manager = pipeline_manager or PipelineManager()
        
        # Create and register memory adapter
        self._memory_adapter = LegacyMemoryAdapter(legacy_memory)
        self._pipeline_manager.register_component(self._memory_adapter)
        
        # Create and register retriever adapter
        if hasattr(legacy_memory, 'memory_retriever'):
            self._retriever_adapter = LegacyRetrieverAdapter(
                legacy_memory.memory_retriever, 
                self._memory_adapter
            )
        else:
            self._retriever_adapter = LegacyRetrieverAdapter(
                legacy_memory, 
                self._memory_adapter
            )
        self._pipeline_manager.register_component(self._retriever_adapter)
        
        # Create default retrieval pipeline
        self._pipeline = self._pipeline_manager.create_pipeline(
            name="default_legacy_pipeline",
            stage_ids=[self._retriever_adapter.get_id()]
        )
    
    def add_memory(self, 
                  embedding: np.ndarray, 
                  text: str, 
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a memory using the new architecture.
        
        Args:
            embedding: Memory embedding vector
            text: Memory text content
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        return self._memory_adapter.add(embedding, text, metadata)
    
    def retrieve_memories(self, 
                         query_embedding: np.ndarray, 
                         top_k: int = 5,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve memories using the pipeline.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to retrieve
            **kwargs: Additional parameters
            
        Returns:
            List of retrieval results
        """
        # Create parameters
        parameters = {
            'max_results': top_k,
            'similarity_threshold': kwargs.get('confidence_threshold', 0.0),
            'activation_boost': kwargs.get('activation_boost', True)
        }
        
        # Create a simple Query object
        query = Query(
            text="",  # We don't have the original text
            embedding=query_embedding,
            query_type=QueryType.UNKNOWN,
            extracted_keywords=[],
            extracted_entities=[]
        )
        
        # Execute pipeline directly with the query embedding and parameters
        results = self._retriever_adapter.retrieve(query_embedding, parameters)
        
        return results
    
    def get_id(self) -> ComponentID:
        """Get the unique identifier for this component."""
        return self._component_id
    
    def get_component(self, component_id: ComponentID) -> Optional[IComponent]:
        """Get a registered component by ID."""
        return self._pipeline_manager.get_component(component_id)
    
    def register_component(self, component: IComponent) -> None:
        """Register a component with the pipeline manager."""
        self._pipeline_manager.register_component(component)
    
    def create_pipeline(self, 
                      name: str, 
                      stage_ids: List[ComponentID]) -> Optional[IPipeline]:
        """Create a new pipeline with the given stages."""
        return self._pipeline_manager.create_pipeline(name, stage_ids)
    
    def execute_pipeline(self, 
                       name: str, 
                       input_data: Any) -> Optional[Any]:
        """Execute a pipeline by name."""
        return self._pipeline_manager.execute_pipeline(name, input_data)


class PipelineToLegacyAdapter:
    """
    Adapter that bridges the new pipeline architecture to the legacy retriever interfaces.
    
    This adapter allows using the new component-based pipeline architecture with
    code that expects the legacy retriever interface, providing backward compatibility.
    """
    
    def __init__(self, 
                pipeline: IPipeline,
                query_analyzer: Optional[IQueryAnalyzer] = None,
                query_adapter: Optional[IQueryAdapter] = None):
        """
        Initialize the adapter.
        
        Args:
            pipeline: The pipeline to use for retrieval
            query_analyzer: Optional query analyzer to use
            query_adapter: Optional query adapter to use
        """
        self._pipeline = pipeline
        self._query_analyzer = query_analyzer
        self._query_adapter = query_adapter
    
    def add_memory(self, 
                  embedding: np.ndarray, 
                  text: str, 
                  metadata: Optional[Dict[str, Any]] = None) -> Any:
        """
        Legacy interface for adding a memory.
        
        This method relies on the first stage of the pipeline being a memory store
        or having a method to add memories.
        
        Args:
            embedding: Memory embedding vector
            text: Memory text content
            metadata: Optional metadata
            
        Returns:
            Memory ID or index
        """
        # Get the first stage of the pipeline
        stages = self._pipeline.get_stages()
        if not stages:
            raise ValueError("Pipeline has no stages")
        
        first_stage = stages[0]
        
        # Check if the stage has an add method
        if hasattr(first_stage, 'add'):
            return first_stage.add(embedding, text, metadata)
        else:
            raise ValueError("First pipeline stage does not support adding memories")
    
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
        # Create a Query object
        query_type = QueryType.UNKNOWN
        if self._query_analyzer and 'query_text' in kwargs:
            query_type = self._query_analyzer.analyze(kwargs['query_text'])
        
        query = Query(
            text=kwargs.get('query_text', ""),
            embedding=query_embedding,
            query_type=query_type,
            extracted_keywords=kwargs.get('keywords', []),
            extracted_entities=kwargs.get('entities', []),
            context=None
        )
        
        # Get parameters
        parameters = None
        if self._query_adapter:
            parameters = self._query_adapter.adapt_parameters(query)
        
        # Override parameters with explicit values
        if parameters is None:
            parameters = {}
        parameters['max_results'] = k
        parameters['similarity_threshold'] = threshold
        
        # Execute the pipeline
        results = self._pipeline.execute(query)
        
        # Convert results to legacy format
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