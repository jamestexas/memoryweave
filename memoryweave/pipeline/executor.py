"""Pipeline executor for MemoryWeave.

This module provides the executor for running component pipelines,
including handling execution context, error handling, and metrics.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar

from memoryweave.interfaces.pipeline import IPipeline


@dataclass
class ExecutionResult:
    """Result of a pipeline execution."""

    success: bool
    result: Any
    error: Optional[Exception] = None
    execution_time: float = 0.0
    metrics: dict[str, Any] = None


T = TypeVar("T")
U = TypeVar("U")


class PipelineExecutor(Generic[T, U]):
    """Executor for pipeline execution with error handling and metrics."""

    def __init__(self):
        """Initialize the pipeline executor."""
        self._logger = logging.getLogger(__name__)
        self._metrics_collectors: list[Callable[[IPipeline, T, U, float], dict[str, Any]]] = []

    def execute(self, pipeline: IPipeline[T, U], input_data: T) -> ExecutionResult:
        """Execute a pipeline with error handling and metrics."""
        start_time = time.time()

        try:
            # Execute pipeline
            result = pipeline.execute(input_data)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Collect metrics
            metrics = self._collect_metrics(pipeline, input_data, result, execution_time)

            # Return successful result
            return ExecutionResult(
                success=True, result=result, execution_time=execution_time, metrics=metrics
            )

        except Exception as e:
            # Log error
            self._logger.exception(f"Error executing pipeline {pipeline.__class__.__name__}: {e}")

            # Calculate execution time
            execution_time = time.time() - start_time

            # Return error result
            return ExecutionResult(
                success=False,
                result=None,
                error=e,
                execution_time=execution_time,
                metrics={"error": str(e)},
            )

    def add_metrics_collector(
        self, collector: Callable[[IPipeline, T, U, float], dict[str, Any]]
    ) -> None:
        """Add a metrics collector function."""
        self._metrics_collectors.append(collector)

    def _collect_metrics(
        self, pipeline: IPipeline[T, U], input_data: T, result: U, execution_time: float
    ) -> dict[str, Any]:
        """Collect metrics from all registered collectors."""
        metrics = {"execution_time": execution_time, "pipeline_stages": len(pipeline.get_stages())}

        # Call each metrics collector
        for collector in self._metrics_collectors:
            try:
                collector_metrics = collector(pipeline, input_data, result, execution_time)
                metrics.update(collector_metrics)
            except Exception as e:
                self._logger.error(f"Error collecting metrics: {e}")

        return metrics


def basic_metrics_collector(
    pipeline: IPipeline, input_data: Any, result: Any, execution_time: float
) -> dict[str, Any]:
    """Basic metrics collector for pipeline execution."""
    metrics = {"execution_time_ms": execution_time * 1000}

    # Add result size metrics if applicable
    if isinstance(result, list):
        metrics["result_count"] = len(result)

    return metrics
