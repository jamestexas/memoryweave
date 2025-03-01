"""
Factory functions for creating and configuring memory components.
"""

from typing import Dict, Any, Optional

from memoryweave.components.memory_adapter import MemoryAdapter
from memoryweave.components.adapters import CoreRetrieverAdapter
from memoryweave.components.memory_manager import MemoryManager
from memoryweave.core.contextual_memory import ContextualMemory


def create_memory_system(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a complete memory system with all components.
    
    This factory function creates a ContextualMemory instance, wraps it
    with appropriate adapters, and registers everything with a MemoryManager.
    
    Args:
        config: Configuration dictionary for the memory system
        
    Returns:
        Dictionary containing the memory, adapters, and manager
    """
    config = config or {}
    
    # Create the core memory system
    memory_config = config.get('memory', {})
    memory = ContextualMemory(**memory_config)
    
    # Create adapters
    memory_adapter = MemoryAdapter(memory=memory)
    retriever_adapter = CoreRetrieverAdapter(
        memory=memory,
        default_top_k=config.get('default_top_k', 5),
        confidence_threshold=config.get('confidence_threshold', 0.0),
    )
    
    # Create memory manager
    manager = MemoryManager()
    
    # Register components
    manager.register_component('memory', memory_adapter)
    manager.register_component('core_retriever', retriever_adapter)
    
    # Return all created objects
    return {
        'memory': memory,
        'memory_adapter': memory_adapter,
        'retriever_adapter': retriever_adapter,
        'manager': manager,
    }


def configure_memory_pipeline(
    manager: MemoryManager,
    pipeline_type: str = 'standard',
) -> None:
    """
    Configure a memory pipeline with the specified components.
    
    Args:
        manager: The MemoryManager to configure
        pipeline_type: Type of pipeline to configure ('standard', 'advanced', etc.)
    """
    if pipeline_type == 'standard':
        # Configure a basic pipeline with query analysis and retrieval
        pipeline_config = [
            {
                'component': 'query_analyzer',
                'config': {}
            },
            {
                'component': 'core_retriever',
                'config': {
                    'confidence_threshold': 0.3,
                    'top_k': 5,
                    'use_categories': True,
                    'activation_boost': True,
                }
            }
        ]
        manager.build_pipeline(pipeline_config)
        
    elif pipeline_type == 'advanced':
        # Configure an advanced pipeline with more components
        pipeline_config = [
            {
                'component': 'query_analyzer',
                'config': {}
            },
            {
                'component': 'query_adapter',
                'config': {
                    'adaptation_strength': 1.0,
                }
            },
            {
                'component': 'core_retriever',
                'config': {
                    'confidence_threshold': 0.25,
                    'top_k': 10,
                    'use_categories': True,
                    'activation_boost': True,
                }
            },
            {
                'component': 'keyword_boost',
                'config': {
                    'keyword_boost_weight': 0.5,
                }
            },
            {
                'component': 'coherence',
                'config': {
                    'coherence_threshold': 0.2,
                }
            },
            {
                'component': 'adaptive_k',
                'config': {
                    'adaptive_k_factor': 0.3,
                }
            }
        ]
        manager.build_pipeline(pipeline_config)
