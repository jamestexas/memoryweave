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
        self,
        query_embedding: Any,
        top_k: int,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Retrieve memories based on query embedding."""
        pass

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a query to retrieve relevant memories.

        This adapter method converts the query to embedding and calls retrieve.

        Args:
            query: The query string
            context: Context dictionary containing query_embedding, memory, etc.

        Returns:
            Updated context with results
        """
        # Get query embedding from context
        query_embedding = context.get("query_embedding")

        # If no query embedding, try to get embedding model and encode
        if query_embedding is None:
            embedding_model = context.get("embedding_model")
            if embedding_model:
                query_embedding = embedding_model.encode(query)

        # If still no query embedding, return empty results
        if query_embedding is None:
            return {"results": []}

        # Get top_k from context
        top_k = context.get("top_k", 5)

        # Use memory from context or instance
        memory = context.get("memory", getattr(self, "memory", None))

        # Retrieve memories
        results = self.retrieve(query_embedding, top_k, {"memory": memory, **context})

        # Return results
        return {"results": results}


class PostProcessor(RetrievalComponent):
    """Base class for post-processing retrieved memories."""

    @abstractmethod
    def process_results(
        self,
        results: list[dict[str, Any]],
        query: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Process retrieved results."""
        pass

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a query by applying post-processing to results.

        Args:
            query: The query string
            context: Context dictionary containing results, etc.

        Returns:
            Updated context with processed results
        """
        results = context.get("results", [])

        # Process results
        processed_results = self.process_results(
            results=results,
            query=query,
            context=context,
        )

        # Update context with processed results
        return dict(results=processed_results)
