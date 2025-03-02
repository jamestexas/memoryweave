"""
Base class for baseline retrieval implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from memoryweave.interfaces.retrieval import RetrievalResult, Query
from memoryweave.storage.memory_store import Memory


class BaselineRetriever(ABC):
    """Base class for all baseline retrievers.
    
    This abstract class defines the interface for baseline retrieval
    implementations, allowing them to be easily compared with
    MemoryWeave's more sophisticated retrieval strategies.
    """
    
    name: str = "baseline"
    
    @abstractmethod
    def index_memories(self, memories: List[Memory]) -> None:
        """Index a list of memories for retrieval.
        
        Args:
            memories: List of memories to index
        """
        pass
    
    @abstractmethod
    def retrieve(
        self, 
        query: Query, 
        top_k: int = 10, 
        threshold: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Retrieve memories relevant to the query.
        
        Args:
            query: The query to search for
            top_k: Maximum number of results to return
            threshold: Minimum relevance score threshold
            **kwargs: Additional retriever-specific parameters
            
        Returns:
            A RetrievalResult containing retrieved memories and metadata
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics for the baseline.
        
        Returns:
            Dictionary of statistics like index size, feature count, etc.
        """
        pass
    
    def clear(self) -> None:
        """Clear the indexed memories."""
        pass