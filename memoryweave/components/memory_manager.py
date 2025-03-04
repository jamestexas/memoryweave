# memoryweave/components/memory_manager.py
from typing import Any, Optional

from memoryweave.components.base import Component
from memoryweave.components.pipeline_config import PipelineConfig
from memoryweave.interfaces.memory import Memory
from memoryweave.storage.memory_store import MemoryStore

import logging
logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Coordinates memory components and orchestrates retrieval pipeline.
    """

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self.components = {}
        self.pipeline: list[PipelineConfig] = []
        self.memory_store = memory_store or MemoryStore()

    def get_all_memories(self) -> list[Memory]:
        """Get all memories from the store."""
        return self.memory_store.get_all()

    def register_component(
        self,
        name: str,
        component: Component,
    ) -> None:
        """Register a component with the memory manager."""
        self.components[name] = component

    def register_components(self, components_dict: dict[str, Component]) -> None:
        """Register multiple components at once.

        Args:
            components_dict: Dictionary mapping component names to component instances
        """
        for name, component in components_dict.items():
            if component is not None:
                self.register_component(name, component)

    def build_pipeline(
        self,
        pipeline_config: list[dict[str, Any]],
    ) -> None:
        """Build a retrieval pipeline from configuration."""
        # We don't need to wrap the list in a dictionary
        # Just pass the list directly to the validator
        try:
            parsed_config = PipelineConfig.model_validate(pipeline_config)
            self.pipeline = []

            for step in parsed_config.steps:
                if step.component in self.components:
                    component = self.components[step.component]
                    # Initialize the component with its configuration when building the pipeline
                    component.initialize(step.config)
                    self.pipeline.append(
                        dict(
                            component=component,
                            config=step.config,
                        )
                    )
                else:
                    raise ValueError(f"Component {step.component} not registered")
        except Exception as e:
            # Simpler fallback for tests
            self.pipeline = []
            for step in pipeline_config:
                component_name = step.get("component")
                if component_name in self.components:
                    component = self.components[component_name]
                    # Initialize the component with its configuration when building the pipeline
                    component.initialize(step.get("config", {}))
                    self.pipeline.append({
                        "component": component,
                        "config": step.get("config", {}),
                    })
                else:
                    raise ValueError(f"Component {component_name} not registered") from e

    def execute_pipeline(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the retrieval pipeline.

        This method processes a query through a series of pipeline steps, each of which
        is a component that can modify the pipeline context. The context is updated
        with the results from each component.

        Args:
            query (str): The query string to be processed by the pipeline.
            context (dict[str, Any]): The initial context for the pipeline, which may
                include existing results and other relevant data.

        Returns:
            dict[str, Any]: The final pipeline context after processing the query
            through all components, including any results and modifications made
            by the components.
        """
        """Execute the retrieval pipeline."""
        pipeline_context = dict(
            query=query,
            original_context=context,
            working_context=context.copy(),
            results=[],
        )

        # Copy over any existing results from the input context
        if "results" in context:
            pipeline_context["results"] = context["results"]

        for i, step in enumerate(self.pipeline):
            component = step["component"]
            logger.debug(
                f"[MemoryManager.execute_pipeline] BEFORE step {i} ({component.__class__.__name__}): "
                f"pipeline_context={pipeline_context}"
            )

            # Process the query with the component
            # Note: We no longer reinitialize the component on each query
            # This allows components to maintain state between queries
            step_result = component.process_query(query, pipeline_context)
            logger.debug(
                f"[MemoryManager.execute_pipeline] AFTER step {i} ({component.__class__.__name__}): "
                f"step_result={step_result}"
            )
            # Update the pipeline context with the component's results
            if step_result:
                pipeline_context.update(step_result)
            logger.debug(
                f"[MemoryManager.execute_pipeline] pipeline_context is now: {pipeline_context}"
            )
        return pipeline_context
