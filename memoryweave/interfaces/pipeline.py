"""Pipeline interface definitions for MemoryWeave.

This module defines the core interfaces for component pipelines,
including protocols, data models, and base classes for pipeline components.
"""

from enum import Enum, auto
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar

T = TypeVar('T')
U = TypeVar('U')
ComponentID = str


class ComponentType(Enum):
    """Types of pipeline components."""
    MEMORY_STORE = auto()
    VECTOR_STORE = auto()
    RETRIEVAL_STRATEGY = auto()
    QUERY_ANALYZER = auto()
    QUERY_ADAPTER = auto()
    POST_PROCESSOR = auto()
    PIPELINE = auto()


class IComponent(Protocol):
    """Base interface for all pipeline components."""

    def get_id(self) -> ComponentID:
        """Get the unique identifier for this component."""
        ...

    def get_type(self) -> ComponentType:
        """Get the type of this component."""
        ...

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        ...

    def get_dependencies(self) -> List[ComponentID]:
        """Get the IDs of components this component depends on."""
        ...


class IComponentRegistry(Protocol):
    """Registry for pipeline components."""

    def register(self, component: IComponent) -> None:
        """Register a component in the registry."""
        ...

    def get_component(self, component_id: ComponentID) -> Optional[IComponent]:
        """Get a component by ID."""
        ...

    def get_components_by_type(self, component_type: ComponentType) -> List[IComponent]:
        """Get all components of a specific type."""
        ...

    def clear(self) -> None:
        """Clear all components from the registry."""
        ...


class IPipelineStage(Generic[T, U], Protocol):
    """Interface for a pipeline stage that processes inputs and produces outputs."""

    def process(self, input_data: T) -> U:
        """Process the input data and return the output."""
        ...

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the pipeline stage."""
        ...


class IPipelineBuilder(Protocol):
    """Interface for building component pipelines."""

    def add_stage(self, stage: IPipelineStage) -> 'IPipelineBuilder':
        """Add a stage to the pipeline."""
        ...

    def build(self) -> 'IPipeline[T, U]':
        """Build the pipeline."""
        ...


class IPipeline(Generic[T, U], Protocol):
    """Interface for a component pipeline."""

    def execute(self, input_data: T) -> U:
        """Execute the pipeline on the input data."""
        ...

    def get_stages(self) -> List[IPipelineStage]:
        """Get all stages in the pipeline."""
        ...
