"""Pipeline builder for MemoryWeave.

This module provides the builder for creating component pipelines,
allowing flexible configuration of processing steps.
"""

import logging
from typing import Generic, List, TypeVar

from memoryweave.interfaces.pipeline import IPipeline, IPipelineBuilder, IPipelineStage

T = TypeVar('T')
U = TypeVar('U')


class Pipeline(Generic[T, U], IPipeline[T, U]):
    """Implementation of a component pipeline."""

    def __init__(self, stages: List[IPipelineStage], name: str = "pipeline"):
        """Initialize the pipeline.
        
        Args:
            stages: List of pipeline stages to execute
            name: Name of the pipeline
        """
        self._stages = stages
        self._name = name
        self._logger = logging.getLogger(__name__)

    def execute(self, input_data: T) -> U:
        """Execute the pipeline on the input data."""
        self._logger.debug(f"Executing pipeline '{self._name}' with {len(self._stages)} stages")

        # Start with the input data
        current_data = input_data

        # Execute each stage in sequence
        for i, stage in enumerate(self._stages):
            self._logger.debug(f"Executing stage {i+1}/{len(self._stages)}")
            current_data = stage.process(current_data)

        # Return the final result
        return current_data

    def get_stages(self) -> List[IPipelineStage]:
        """Get all stages in the pipeline."""
        return self._stages.copy()


class PipelineBuilder(IPipelineBuilder, Generic[T, U]):
    """Builder for creating component pipelines."""

    def __init__(self, name: str = "pipeline"):
        """Initialize the pipeline builder.
        
        Args:
            name: Name of the pipeline
        """
        self._stages: List[IPipelineStage] = []
        self._name = name
        self._logger = logging.getLogger(__name__)

    def add_stage(self, stage: IPipelineStage) -> 'PipelineBuilder':
        """Add a stage to the pipeline."""
        self._stages.append(stage)
        return self

    def build(self) -> IPipeline[T, U]:
        """Build the pipeline."""
        self._logger.debug(f"Building pipeline '{self._name}' with {len(self._stages)} stages")
        return Pipeline[T, U](self._stages, self._name)

    def clear(self) -> 'PipelineBuilder':
        """Clear all stages from the builder."""
        self._stages.clear()
        return self

    def set_name(self, name: str) -> 'PipelineBuilder':
        """Set the name of the pipeline."""
        self._name = name
        return self
