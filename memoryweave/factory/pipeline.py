"""Pipeline component factory for MemoryWeave.

This module provides factories for creating pipeline-related components,
such as pipelines, managers, and executors.
"""

from typing import Any, Dict, List, Optional, TypeVar

from memoryweave.config.options import get_default_config
from memoryweave.config.validation import ConfigValidationError, validate_config
from memoryweave.interfaces.pipeline import IComponent, IPipeline
from memoryweave.interfaces.retrieval import Query, RetrievalResult
from memoryweave.pipeline.executor import PipelineExecutor
from memoryweave.pipeline.manager import PipelineManager

T = TypeVar('T')
U = TypeVar('U')


class PipelineFactory:
    """Factory for creating pipeline-related components."""

    @staticmethod
    def create_pipeline_manager() -> PipelineManager:
        """Create a pipeline manager component.
        
        Returns:
            Configured pipeline manager
        """
        return PipelineManager()

    @staticmethod
    def create_pipeline_executor() -> PipelineExecutor:
        """Create a pipeline executor component.
        
        Returns:
            Configured pipeline executor
        """
        return PipelineExecutor()

    @staticmethod
    def create_pipeline(
        components: List[IComponent],
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[IPipeline]:
        """Create a pipeline from components.
        
        Args:
            components: List of components to include in the pipeline
            config: Optional configuration for the pipeline
            
        Returns:
            Configured pipeline, or None if creation fails
            
        Raises:
            ConfigValidationError: If the configuration is invalid
        """
        # Use default config if none provided
        if config is None:
            config = {}

        # Merge with defaults
        defaults = get_default_config("pipeline")
        merged_config = {**defaults, **config}

        # Validate config
        is_valid, errors = validate_config(merged_config, "pipeline")
        if not is_valid:
            raise ConfigValidationError(errors, "pipeline")

        # Create manager and register components
        manager = PipelineFactory.create_pipeline_manager()
        for component in components:
            manager.register_component(component)

        # Get component IDs from config
        component_ids = merged_config.get("pipeline_stages", [])
        if not component_ids:
            # Use all registered components if none specified
            component_ids = [component.get_id() for component in components]

        # Create pipeline
        pipeline_name = merged_config.get("pipeline_name", "default_pipeline")
        return manager.create_pipeline(pipeline_name, component_ids)

    @staticmethod
    def create_retrieval_pipeline(
        components: List[IComponent],
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[IPipeline[Query, List[RetrievalResult]]]:
        """Create a retrieval pipeline from components.
        
        Args:
            components: List of components to include in the pipeline
            config: Optional configuration for the pipeline
            
        Returns:
            Configured retrieval pipeline, or None if creation fails
            
        Raises:
            ConfigValidationError: If the configuration is invalid
        """
        return PipelineFactory.create_pipeline(components, config)
