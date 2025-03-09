# memoryweave/components/base.py
import abc
from typing import Any

from pydantic import BaseModel, ConfigDict


class Component(BaseModel):
    """Base component with both Pydantic validation and abstract methods."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @abc.abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"


class MemoryComponent(Component):
    """
    Base class for components that operate on memory data.
    """

    class Config:
        arbitrary_types_allowed = True

    @abc.abstractmethod
    def process(self, data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        Process memory data with context.

        Args:
            data: The memory data dictionary.
            context: The context dictionary.

        Returns:
            A dictionary with processed results.
        """
        raise NotImplementedError


class RetrievalComponent(Component):
    """
    Base class for components involved in memory retrieval.
    """

    @abc.abstractmethod
    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a query with context.

        Args:
            query: The query string.
            context: The context dictionary.

        Returns:
            A dictionary with processed query results.
        """
        raise NotImplementedError


class RetrievalStrategy(RetrievalComponent):
    """
    Base class for retrieval strategies.
    """

    @abc.abstractmethod
    def retrieve(
        self,
        query_embedding: Any,
        top_k: int,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories based on query embedding.

        Args:
            query_embedding: The embedding of the query.
            top_k: The number of top results to return.
            context: The context dictionary (can include memory and other keys).

        Returns:
            A list of dictionaries representing retrieved memories.
        """
        raise NotImplementedError

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a query to retrieve relevant memories.

        This adapter method converts the query to an embedding (if necessary)
        and calls `retrieve`.

        Args:
            query: The query string.
            context: Context dictionary containing keys like `query_embedding`,
                     `embedding_model`, `show_progress_bar`, `top_k`, and optionally `memory`.

        Returns:
            A dictionary with a "results" key holding the retrieved memories.
        """
        query_embedding = context.get("query_embedding")

        if query_embedding is None:
            embedding_model = context.get("embedding_model")
            if embedding_model:
                query_embedding = embedding_model.encode(
                    query,
                    show_progress_bar=context.get("show_progress_bar", False),
                )

        if query_embedding is None:
            return {"results": []}

        top_k = context.get("top_k", 5)
        memory = context.get("memory", getattr(self, "memory", None))
        results = self.retrieve(query_embedding, top_k, {"memory": memory, **context})
        return {"results": results}


class PostProcessor(RetrievalComponent):
    """
    Base class for post-processing retrieved memories.
    """

    @abc.abstractmethod
    def process_results(
        self,
        results: list[dict[str, Any]],
        query: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Process retrieved results.

        Args:
            results: A list of retrieved memory dictionaries.
            query: The original query string.
            context: The context dictionary.

        Returns:
            A list of processed result dictionaries.
        """
        raise NotImplementedError

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a query by applying post-processing to results.

        Args:
            query: The query string.
            context: Context dictionary containing results, etc.

        Returns:
            A dictionary with a "results" key holding the processed results.
        """
        results = context.get("results", [])
        processed_results = self.process_results(
            results=results,
            query=query,
            context=context,
        )
        return {"results": processed_results}
