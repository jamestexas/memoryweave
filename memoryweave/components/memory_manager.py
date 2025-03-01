# memoryweave/components/memory_manager.py
from typing import Any

from memoryweave.components.base import Component
from memoryweave.components.pipeline_config import PipelineConfig


class MemoryManager:
    """
    Coordinates memory components and orchestrates retrieval pipeline.
    """

    def __init__(self):
        self.components = {}
        self.pipeline: list[PipelineConfig] = []

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
        # Fix for Pydantic validation - convert list to dict with steps field
        if isinstance(pipeline_config, list):
            config_dict = {"steps": pipeline_config}
            parsed_config = PipelineConfig.model_validate(config_dict)
        else:
            parsed_config = PipelineConfig.model_validate(pipeline_config)

        self.pipeline = []

        for step in parsed_config.steps:
            if step.component in self.components:
                self.pipeline.append(
                    dict(
                        component=self.components[step.component],
                        config=step.config,
                    )
                )
            else:
                raise ValueError(f"Component {step.component} not registered")

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
