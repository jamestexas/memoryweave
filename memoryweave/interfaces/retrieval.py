"""Retrieval interface definitions for MemoryWeave.

This module defines the core interfaces for memory retrieval,
including protocols, data models, and base classes for retrieval components.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol, TypedDict

from memoryweave.interfaces.memory import EmbeddingVector, MemoryID


class QueryType(Enum):
    """types of queries that can be processed."""

    FACTUAL = auto()
    PERSONAL = auto()
    CONCEPTUAL = auto()
    HISTORICAL = auto()
    TEMPORAL = auto()
    UNKNOWN = auto()


@dataclass
class QueryContext:
    """Context information for a query."""

    recent_interactions: list[dict[str, Any]]
    conversation_attributes: dict[str, Any]
    personal_attributes: dict[str, Any]


@dataclass
class Query:
    """Data model for a query in the system."""

    text: str
    embedding: EmbeddingVector
    query_type: QueryType
    extracted_keywords: list[str]
    extracted_entities: list[str]
    context: QueryContext | None = None

    # TODO: Move to dataclasses.asdict() to facilitate dict conversion or pydantic
    def __iter__(self):
        """Make the Query object iterable for dict conversion."""
        yield "text", self.text
        yield "embedding", self.embedding
        yield "query_type", self.query_type
        yield "extracted_keywords", self.extracted_keywords
        yield "extracted_entities", self.extracted_entities
        if self.context is not None:
            yield "context", self.context


class RetrievalResult(TypedDict):
    """Result of a memory retrieval operation."""

    memory_id: MemoryID
    content: str
    metadata: dict[str, Any]
    relevance_score: float


class RetrievalParameters(TypedDict, total=False):
    """Parameters for memory retrieval."""

    similarity_threshold: float
    max_results: int
    recency_bias: float
    activation_boost: float
    keyword_weight: float
    min_results: int
    include_categories: list[int]
    exclude_categories: list[int]


class IRetrievalStrategy(Protocol):
    """Interface for memory retrieval strategies."""

    def retrieve(
        self, query_embedding: EmbeddingVector, parameters: RetrievalParameters
    ) -> list[RetrievalResult]:
        """Retrieve memories based on a query embedding."""
        ...

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the retrieval strategy."""
        ...


class IPostProcessor(Protocol):
    """Interface for post-processing retrieval results."""

    def process(
        self, results: list[RetrievalResult], query: Query, context: QueryContext
    ) -> list[RetrievalResult]:
        """Process retrieval results."""
        ...

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the post-processor."""
        ...


class IRetrievalPipeline(Protocol):
    """Interface for retrieval pipeline."""

    def add_retrieval_strategy(self, strategy: IRetrievalStrategy) -> None:
        """Add a retrieval strategy to the pipeline."""
        ...

    def add_post_processor(self, processor: IPostProcessor) -> None:
        """Add a post-processor to the pipeline."""
        ...

    def retrieve(self, query: Query) -> list[RetrievalResult]:
        """Execute the retrieval pipeline."""
        ...

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the retrieval pipeline."""
        ...
