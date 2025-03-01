# memoryweave/components/pipeline_config.py
from typing import Any

from pydantic import BaseModel, Field, field_validator

from memoryweave.components.component_names import ComponentName


class PipelineStep(BaseModel):
    component: ComponentName = Field(..., description="The name of the registered component")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for the component",
    )


class PipelineConfig(BaseModel):
    steps: list[PipelineStep] = Field(..., description="list of pipeline steps in order")

    @field_validator("steps", mode="before")
    @classmethod
    def validate_steps(
        cls,
        values: dict[str, list[PipelineStep]] | list[PipelineStep],
    ) -> dict[str, Any]:
        # Allow a plain list of dicts as input
        if isinstance(values, list):
            return dict(steps=values)
        return values
