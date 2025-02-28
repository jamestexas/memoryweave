# memoryweave/components/base.py
from abc import ABC, abstractmethod
from typing import Any


class Component(ABC):
    """Base class for all memory components."""

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        pass


class MemoryComponent(Component):
    """Base class for components that operate on memory data."""

    @abstractmethod
    def process(self, data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Process memory data with context."""
        pass


class RetrievalComponent(Component):
    """Base class for components involved in memory retrieval."""

    @abstractmethod
    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Process a query with context."""
        pass


class RetrievalStrategy(RetrievalComponent):
    """Base class for retrieval strategies."""

    @abstractmethod
    def retrieve(
        self, query_embedding: Any, top_k: int, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Retrieve memories based on query embedding."""
        pass


class PostProcessor(RetrievalComponent):
    """Base class for post-processing retrieved memories."""

    @abstractmethod
    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process retrieved results."""
        pass
