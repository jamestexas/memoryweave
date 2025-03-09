"""Query processing interface definitions for MemoryWeave.

This module defines the core interfaces for query processing,
including protocols, data models, and base classes for query components.
"""

from typing import Any, Protocol, runtime_checkable

from memoryweave.interfaces.retrieval import Query, QueryType, RetrievalParameters


@runtime_checkable
class IQueryAnalyzer(Protocol):
    """Interface for query analysis."""

    def analyze(self, query_text: str) -> QueryType:
        """Analyze a query to determine its type."""
        ...

    def extract_keywords(self, query_text: str) -> list[str]:
        """Extract keywords from a query."""
        ...

    def extract_entities(self, query_text: str) -> list[str]:
        """Extract entities from a query."""
        ...

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the query analyzer."""
        ...


@runtime_checkable
class IQueryAdapter(Protocol):
    """Interface for adapting query parameters based on query type."""

    def adapt_parameters(self, query: Query) -> RetrievalParameters:
        """Adapt retrieval parameters based on query type."""
        ...

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the query adapter."""
        ...


@runtime_checkable
class IQueryExpander(Protocol):
    """Interface for expanding queries with additional keywords or concepts."""

    def expand(self, query: Query) -> Query:
        """Expand a query with additional keywords or concepts."""
        ...

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the query expander."""
        ...
