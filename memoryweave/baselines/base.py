"""
Base class for baseline retrieval implementations.
"""

from abc import ABC, abstractmethod
from typing import Any

from memoryweave.interfaces.memory import Memory
from memoryweave.interfaces.retrieval import Query


class BaselineRetriever(ABC):
    """Base class for all baseline retrievers.

    This abstract class defines the interface for baseline retrieval
    implementations, allowing them to be easily compared with
    MemoryWeave's more sophisticated retrieval strategies.
    """

    name: str = "baseline"

    @abstractmethod
    def index_memories(self, memories: list[Memory]) -> None:
        """Index a list of memories for retrieval.

        Args:
            memories: List of memories to index
        """
        pass

    @abstractmethod
    def retrieve(
        self, query: Query, top_k: int = 10, threshold: float = 0.0, **kwargs
    ) -> dict[str, Any]:
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
    def get_statistics(self) -> dict[str, Any]:
        """Get retrieval statistics for the baseline.

        Returns:
            Dictionary of statistics like index size, feature count, etc.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the indexed memories."""
        pass
