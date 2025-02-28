# memoryweave/components/memory_manager.py
from typing import Any

from memoryweave.components.base import Component


class MemoryManager:
    """
    Coordinates memory components and orchestrates retrieval pipeline.
    """

    def __init__(self):
        self.components = {}
        self.pipeline = []

    def register_component(
        self,
        name: str,
        component: Component,
    ) -> None:
        """Register a component with the memory manager."""
        self.components[name] = component

    def build_pipeline(
        self,
        pipeline_config: list[dict[str, Any]],
    ) -> None:
        """Build a retrieval pipeline from configuration."""
        self.pipeline = []
        for step in pipeline_config:
            component_name = step["component"]
            if component_name in self.components:
                self.pipeline.append({
                    "component": self.components[component_name],
                    "config": step.get("config", {}),
                })
            else:
                raise ValueError(f"Component {component_name} not registered")

    def execute_pipeline(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the retrieval pipeline."""
        pipeline_context = dict(
            query=query,
            original_context=context,
            working_context=context.copy(),
            results=[],
        )
        
        for step in self.pipeline:
            component = step["component"]
            config = step["config"]
            
            # Initialize the component with its configuration
            component.initialize(config)
            
            # Process the query with the component
            step_result = component.process_query(query, pipeline_context)
            
            # Update the pipeline context with the component's results
            if step_result:
                pipeline_context.update(step_result)

        return pipeline_context
