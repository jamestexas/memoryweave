# File: memoryweave/benchmarks/base.py

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

DEFAULT_OUTPUT_FILE: str = "benchmark_results.json"


@dataclass
class BenchmarkConfig:
    """Base configuration for benchmarks."""

    name: str
    description: str
    output_file: str = DEFAULT_OUTPUT_FILE


@dataclass
class BenchmarkResult:
    """Standard benchmark result format."""

    config_name: str
    metrics: dict[str, float]
    start_time: float
    end_time: float
    additional_data: dict[str, Any] = None

    @property
    def duration(self) -> float:
        """Calculate benchmark duration."""
        return self.end_time - self.start_time


class Benchmark(ABC):
    """Base class for all benchmarks."""

    def __init__(self, configs: list[BenchmarkConfig]):
        self.configs = configs
        self.results = []

    @abstractmethod
    def setup(self, config: BenchmarkConfig) -> Any:
        """Set up the benchmark environment."""
        pass

    @abstractmethod
    def run_single_benchmark(self, config: BenchmarkConfig, setup_data: Any) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        pass

    def run_all(self) -> list[BenchmarkResult]:
        """Run all benchmark configurations."""
        results = []
        for config in self.configs:
            setup_data = self.setup(config)
            result = self.run_single_benchmark(config, setup_data)
            results.append(result)

        self.results = results
        return results

    def save_results(
        self,
        output_file: str | None = None,
    ) -> None:
        """Save benchmark results to a file."""
        file_path = (
            output_file or self.configs[0].output_file if self.configs else "benchmark_results.json"
        )
        results_dict = {
            results.config_name: dict(
                metrics=results.metrics,
                duration=results.duration,
                additional_data=results.additional_data,
            )
            for results in self.results
        }

        with open(file_path, "w") as f:
            json.dump(results_dict, f, indent=2)

    @abstractmethod
    def visualize_results(self) -> None:
        """Generate visualizations for benchmark results."""
        pass
