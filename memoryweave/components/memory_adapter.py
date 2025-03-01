"""
Memory adapter for integrating core memory with the components architecture.
"""

from typing import Any, Dict, Optional

from memoryweave.components.base import MemoryComponent
from memoryweave.core.contextual_memory import ContextualMemory


class MemoryAdapter(MemoryComponent):
    """
    Adapter for integrating the core memory system with the components architecture.
    
    This class wraps the core memory system and exposes it through the component
    interface, allowing it to be used seamlessly within the pipeline architecture.
    """

    def __init__(
        self,
        memory: Optional[ContextualMemory] = None,
        **memory_kwargs
    ):
        """
        Initialize the memory adapter.
        
        Args:
            memory: Existing ContextualMemory instance to adapt
            **memory_kwargs: Arguments to pass to ContextualMemory constructor if creating a new instance
        """
        self.memory = memory or ContextualMemory(**memory_kwargs)

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        # Apply any configuration updates
        # For now, we don't change any settings after initialization
        pass

    def process(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process memory data with context.
        
        This method handles memory operations like adding or retrieving memories.
        
        Args:
            data: Data to process
            context: Context for processing
            
        Returns:
            Updated context with processing results
        """
        operation = data.get('operation')

        if operation == 'add_memory':
            # Add a memory to the system
            embedding = data.get('embedding')
            text = data.get('text')
            metadata = data.get('metadata', {})

            memory_id = self.memory.add_memory(embedding, text, metadata)
            return {'memory_id': memory_id}

        elif operation == 'retrieve_memories':
            # Retrieve memories from the system
            query_embedding = data.get('query_embedding')
            top_k = data.get('top_k', 5)
            confidence_threshold = data.get('confidence_threshold')

            results = self.memory.retrieve_memories(
                query_embedding=query_embedding,
                top_k=top_k,
                confidence_threshold=confidence_threshold
            )

            # Convert tuple results to dictionaries for easier handling in pipeline
            formatted_results = []
            for idx, score, metadata in results:
                formatted_results.append({
                    'memory_id': idx,
                    'relevance_score': score,
                    **metadata
                })

            return {'results': formatted_results}

        elif operation == 'get_category_statistics':
            # Get statistics about categories
            stats = self.memory.get_category_statistics()
            return {'category_statistics': stats}

        elif operation == 'consolidate_categories':
            # Manually consolidate categories
            threshold = data.get('threshold')
            result = self.memory.consolidate_categories_manually(threshold)
            return {'num_categories': result}

        # Default response if no operation matched
        return {}

    def get_memory(self) -> ContextualMemory:
        """Get the underlying memory instance."""
        return self.memory
