"""Retrieval interface definitions for MemoryWeave.

This module defines the core interfaces for memory retrieval,
including protocols, data models, and base classes for retrieval components.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, TypedDict

from memoryweave.interfaces.memory import EmbeddingVector, MemoryID


class QueryType(Enum):
    """Types of queries that can be processed."""
    FACTUAL = auto()
    PERSONAL = auto()
    CONCEPTUAL = auto()
    HISTORICAL = auto()
    TEMPORAL = auto()
    UNKNOWN = auto()


@dataclass
class QueryContext:
    """Context information for a query."""
    recent_interactions: List[Dict[str, Any]]
    conversation_attributes: Dict[str, Any]
    personal_attributes: Dict[str, Any]


@dataclass
class Query:
    """Data model for a query in the system."""
    text: str
    embedding: EmbeddingVector
    query_type: QueryType
    extracted_keywords: List[str]
    extracted_entities: List[str]
    context: Optional[QueryContext] = None


class RetrievalResult(TypedDict):
    """Result of a memory retrieval operation."""
    memory_id: MemoryID
    content: str
    metadata: Dict[str, Any]
    relevance_score: float


class RetrievalParameters(TypedDict, total=False):
    """Parameters for memory retrieval."""
    similarity_threshold: float
    max_results: int
    recency_bias: float
    activation_boost: float
    keyword_weight: float
    min_results: int
    include_categories: List[int]
    exclude_categories: List[int]


class IRetrievalStrategy(Protocol):
    """Interface for memory retrieval strategies."""

    def retrieve(self,
                query_embedding: EmbeddingVector,
                parameters: RetrievalParameters) -> List[RetrievalResult]:
        """Retrieve memories based on a query embedding."""
        ...

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the retrieval strategy."""
        ...


class IPostProcessor(Protocol):
    """Interface for post-processing retrieval results."""

    def process(self,
               results: List[RetrievalResult],
               query: Query,
               context: QueryContext) -> List[RetrievalResult]:
        """Process retrieval results."""
        ...

    def configure(self, config: Dict[str, Any]) -> None:
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

    def retrieve(self, query: Query) -> List[RetrievalResult]:
        """Execute the retrieval pipeline."""
        ...

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the retrieval pipeline."""
        ...
