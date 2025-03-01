# memoryweave/components/dynamic_threshold_adjuster.py
"""
DynamicThresholdAdjuster

A PostProcessor that updates dynamic threshold metrics and ensures a minimum number of results.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from memoryweave.components.base import PostProcessor


class DynamicThresholdConfig(BaseModel):
    window_size: int = Field(5, description="Number of recent metrics to keep")
    adjustment_step: float = Field(0.05, description="Threshold adjustment step")
    min_threshold: float = Field(0.0, description="Minimum allowed threshold")
    max_threshold: float = Field(0.9, description="Maximum allowed threshold")
    min_result_count: int = Field(5, description="Minimum number of results to ensure")
    confidence_threshold: float = Field(0.0, description="Initial confidence threshold")


class DynamicThresholdAdjuster(PostProcessor):
    # Declare the expected configuration model
    config_model = DynamicThresholdConfig

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the dynamic threshold adjuster using the given config."""
        parsed = self.config_model.parse_obj(config)
        self.window_size = parsed.window_size
        self.adjustment_step = parsed.adjustment_step
        self.min_threshold = parsed.min_threshold
        self.max_threshold = parsed.max_threshold
        self.min_result_count = parsed.min_result_count
        self.current_threshold = parsed.confidence_threshold
        self.recent_metrics: list[dict[str, float]] = []

    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Update retrieval metrics and adjust the confidence threshold.
        Also, ensure at least min_result_count results are returned.

        Args:
            results: list of retrieval result dicts (each with a 'relevance_score')
            query: The query string
            context: The context dictionary (which we update with the new threshold)

        Returns:
            Updated list of results.
        """
        # Update metrics for this retrieval
        result_count = len(results)
        avg_score = np.mean([r.get("relevance_score", 0) for r in results]) if results else 0.0
        self.recent_metrics.append({"result_count": result_count, "avg_score": avg_score})
        if len(self.recent_metrics) > self.window_size:
            self.recent_metrics.pop(0)

        # If we have enough metrics, adjust the threshold
        if len(self.recent_metrics) == self.window_size:
            avg_result_count = np.mean([m["result_count"] for m in self.recent_metrics])
            avg_recent_score = np.mean([m["avg_score"] for m in self.recent_metrics])
            # Example adjustment logic:
            if avg_result_count < 1:
                self.current_threshold = max(
                    self.min_threshold, self.current_threshold - self.adjustment_step
                )
            elif avg_result_count > 10 and avg_recent_score < 0.3:
                self.current_threshold = min(
                    self.max_threshold, self.current_threshold + self.adjustment_step
                )
            context["dynamic_confidence_threshold"] = self.current_threshold

        # Ensure minimum result guarantee: if results are too few, append a default entry.
        while len(results) < self.min_result_count:
            results.append(
                {
                    "memory_id": 0,
                    "relevance_score": 0.1,
                    "content": f"No specific information found about: {query}",
                    "type": "default",
                }
            )
        return results
