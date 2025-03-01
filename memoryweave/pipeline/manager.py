"""Pipeline manager for MemoryWeave.

This module provides the manager for orchestrating component pipelines,
including component registration, pipeline creation, and execution.
"""

from typing import Dict, List, Any, Optional, Generic, TypeVar, Type
import logging

from memoryweave.interfaces.pipeline import (
    IComponent, ComponentType, ComponentID, IComponentRegistry, IPipelineStage, IPipeline
)
from memoryweave.interfaces.retrieval import Query, RetrievalResult
from memoryweave.pipeline.registry import ComponentRegistry
from memoryweave.pipeline.builder import PipelineBuilder


class PipelineManager:
    """Manager for component pipelines."""
    
    def __init__(self):
        """Initialize the pipeline manager."""
        self._registry = ComponentRegistry()
        self._pipelines: Dict[str, IPipeline] = {}
        self._logger = logging.getLogger(__name__)
    
    def register_component(self, component: IComponent) -> None:
        """Register a component with the manager."""
        self._registry.register(component)
    
    def create_pipeline(self, 
                       name: str, 
                       stage_ids: List[ComponentID]) -> Optional[IPipeline]:
        """Create a pipeline from registered components."""
        self._logger.debug(f"Creating pipeline '{name}' with {len(stage_ids)} stages")
        
        # Get pipeline stages
        stages = []
        for stage_id in stage_ids:
            component = self._registry.get_component(stage_id)
            if component is None:
                self._logger.error(f"Component '{stage_id}' not found in registry")
                return None
            
            if not isinstance(component, IPipelineStage):
                self._logger.error(f"Component '{stage_id}' is not a pipeline stage")
                return None
            
            stages.append(component)
        
        # Build pipeline
        pipeline = PipelineBuilder().set_name(name)
        for stage in stages:
            pipeline.add_stage(stage)
        
        # Store and return pipeline
        built_pipeline = pipeline.build()
        self._pipelines[name] = built_pipeline
        
        return built_pipeline
    
    def get_pipeline(self, name: str) -> Optional[IPipeline]:
        """Get a pipeline by name."""
        return self._pipelines.get(name)
    
    def execute_pipeline(self, 
                       name: str, 
                       input_data: Any) -> Optional[Any]:
        """Execute a pipeline by name."""
        pipeline = self.get_pipeline(name)
        if pipeline is None:
            self._logger.error(f"Pipeline '{name}' not found")
            return None
        
        return pipeline.execute(input_data)
    
    def get_component(self, component_id: ComponentID) -> Optional[IComponent]:
        """Get a component by ID."""
        return self._registry.get_component(component_id)
    
    def get_components_by_type(self, 
                              component_type: ComponentType) -> List[IComponent]:
        """Get all components of a specific type."""
        return self._registry.get_components_by_type(component_type)
    
    def clear(self) -> None:
        """Clear all components and pipelines."""
        self._registry.clear()
        self._pipelines.clear()
    
    def list_pipelines(self) -> List[str]:
        """Get names of all registered pipelines."""
        return list(self._pipelines.keys())
    
    def list_components(self) -> List[ComponentID]:
        """Get IDs of all registered components."""
        return self._registry.get_component_ids()
    
    def initialize_component(self, 
                          component_id: ComponentID, 
                          config: Dict[str, Any]) -> bool:
        """Initialize a component with configuration."""
        component = self._registry.get_component(component_id)
        if component is None:
            self._logger.error(f"Component '{component_id}' not found in registry")
            return False
        
        try:
            component.initialize(config)
            return True
        except Exception as e:
            self._logger.error(f"Failed to initialize component '{component_id}': {e}")
            return False