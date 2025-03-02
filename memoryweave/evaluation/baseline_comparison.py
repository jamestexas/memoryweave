"""
Framework for comparing MemoryWeave against baseline retrieval methods.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Type

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel

from memoryweave.baselines.base import BaselineRetriever
from memoryweave.evaluation.coherence_metrics import calculate_semantic_coherence
from memoryweave.components.memory_manager import MemoryManager
from memoryweave.interfaces.retrieval import IRetrievalStrategy, Query, RetrievalResult

# Define a protocol for the memory manager interface
from typing import Protocol
from memoryweave.interfaces.memory import Memory

class IMemoryManager(Protocol):
    """Protocol for memory managers that can be used with baseline comparison."""
    
    def get_all_memories(self) -> list[Memory]:
        """Get all memories from the store."""
        ...


class BaselineConfig(BaseModel):
    """Configuration for a baseline retriever."""

    name: str
    retriever_class: Type[BaselineRetriever]
    parameters: Dict[str, Any] = {}


class ComparisonResult(BaseModel):
    """Results from comparing MemoryWeave with baselines."""

    memoryweave_metrics: Dict[str, Dict[str, Any]]
    baseline_metrics: Dict[str, Dict[str, Dict[str, Any]]]
    query_details: Dict[str, List[Dict[str, Any]]]
    runtime_stats: Dict[str, Dict[str, float]]
    dataset_stats: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True


class BaselineComparison:
    """Framework for comparing MemoryWeave against baseline methods."""

    def __init__(
        self,
        memory_manager: IMemoryManager,
        memoryweave_retriever: IRetrievalStrategy,
        baseline_configs: List[BaselineConfig],
        metrics: List[str] = ["precision", "recall", "f1", "mrr"]
    ):
        """Initialize baseline comparison framework.
        
        Args:
            memory_manager: MemoryWeave memory manager
            memoryweave_retriever: MemoryWeave retrieval strategy
            baseline_configs: List of baseline configurations
            metrics: List of metrics to compute
        """
        self.memory_manager = memory_manager
        self.memoryweave_retriever = memoryweave_retriever
        self.metrics = metrics

        # Initialize baseline retrievers
        self.baseline_retrievers: Dict[str, BaselineRetriever] = {}
        for config in baseline_configs:
            retriever = config.retriever_class(**config.parameters)
            self.baseline_retrievers[config.name] = retriever

    def run_comparison(
        self,
        queries: List[Query],
        relevant_memory_ids: List[List[str]],
        max_results: int = 10,
        threshold: float = 0.0
    ) -> ComparisonResult:
        """Run comparison between MemoryWeave and baselines.
        
        Args:
            queries: List of queries to evaluate
            relevant_memory_ids: List of relevant memory IDs for each query
            max_results: Maximum number of results to retrieve
            threshold: Minimum score threshold
            
        Returns:
            ComparisonResult with metrics for each system
        """
        # Get all memories from memory manager
        all_memories = self.memory_manager.get_all_memories()

        # Initialize baseline retrievers with memories
        for retriever in self.baseline_retrievers.values():
            retriever.index_memories(all_memories)

        # Run queries on MemoryWeave
        memoryweave_results = []
        memoryweave_times = []

        for query in queries:
            start_time = time.time()
            result = self.memoryweave_retriever.retrieve(
                query=query,
                top_k=max_results,
                threshold=threshold
            )
            query_time = time.time() - start_time

            memoryweave_results.append(result)
            memoryweave_times.append(query_time)

        # Run queries on baseline retrievers
        baseline_results: Dict[str, List[RetrievalResult]] = {}
        baseline_times: Dict[str, List[float]] = {}

        for name, retriever in self.baseline_retrievers.items():
            results = []
            times = []

            for query in queries:
                start_time = time.time()
                result = retriever.retrieve(
                    query=query,
                    top_k=max_results,
                    threshold=threshold
                )
                query_time = time.time() - start_time

                results.append(result)
                times.append(query_time)

            baseline_results[name] = results
            baseline_times[name] = times

        # Compute metrics for MemoryWeave
        memoryweave_metrics = self._compute_metrics(
            memoryweave_results,
            queries,
            relevant_memory_ids
        )

        # Compute metrics for baselines
        baseline_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

        for name, results in baseline_results.items():
            baseline_metrics[name] = self._compute_metrics(
                results, queries, relevant_memory_ids
            )

        # Compute runtime statistics
        runtime_stats = {
            "memoryweave": {
                "avg_query_time": np.mean(memoryweave_times),
                "total_time": sum(memoryweave_times),
                "min_time": min(memoryweave_times),
                "max_time": max(memoryweave_times)
            }
        }

        for name, times in baseline_times.items():
            runtime_stats[name] = {
                "avg_query_time": np.mean(times),
                "total_time": sum(times),
                "min_time": min(times),
                "max_time": max(times)
            }

        # Prepare query details
        query_details = {}

        for i, query in enumerate(queries):
            query_id = f"query_{i}"
            query_details[query_id] = []

            # Add MemoryWeave results
            mw_result = memoryweave_results[i]
            mw_memory_ids = [m.id for m in mw_result["memories"]]

            query_details[query_id].append({
                "system": "memoryweave",
                "query_text": query.text,
                "retrieved_ids": mw_memory_ids,
                "relevant_ids": relevant_memory_ids[i],
                "scores": mw_result["scores"],
                "query_time": memoryweave_times[i]
            })

            # Add baseline results
            for name in self.baseline_retrievers.keys():
                bl_result = baseline_results[name][i]
                bl_memory_ids = [m.id for m in bl_result["memories"]]

                query_details[query_id].append({
                    "system": name,
                    "query_text": query.text,
                    "retrieved_ids": bl_memory_ids,
                    "relevant_ids": relevant_memory_ids[i],
                    "scores": bl_result["scores"],
                    "query_time": baseline_times[name][i]
                })

        # Prepare dataset statistics
        dataset_stats = {
            "num_memories": len(all_memories),
            "num_queries": len(queries),
            "avg_query_length": np.mean([len(q.text.split()) for q in queries]),
            "avg_relevant_count": np.mean([len(ids) for ids in relevant_memory_ids])
        }

        # Return comparison results
        return ComparisonResult(
            memoryweave_metrics=memoryweave_metrics,
            baseline_metrics=baseline_metrics,
            query_details=query_details,
            runtime_stats=runtime_stats,
            dataset_stats=dataset_stats
        )

    def _compute_metrics(
        self,
        results: List[RetrievalResult],
        queries: List[Query],
        relevant_memory_ids: List[List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute evaluation metrics for retrieval results.
        
        Args:
            results: List of retrieval results
            queries: List of queries
            relevant_memory_ids: List of relevant memory IDs for each query
            
        Returns:
            Dictionary of metrics
        """
        metrics_by_query = {}

        for i, (result, query, rel_ids) in enumerate(zip(results, queries, relevant_memory_ids)):
            query_id = f"query_{i}"
            metrics_by_query[query_id] = {}

            retrieved_ids = [memory.id for memory in result["memories"]]

            # Calculate precision
            if "precision" in self.metrics:
                precision = self._calculate_precision(retrieved_ids, rel_ids)
                metrics_by_query[query_id]["precision"] = precision

            # Calculate recall
            if "recall" in self.metrics:
                recall = self._calculate_recall(retrieved_ids, rel_ids)
                metrics_by_query[query_id]["recall"] = recall

            # Calculate F1 score
            if "f1" in self.metrics and "precision" in self.metrics and "recall" in self.metrics:
                precision = metrics_by_query[query_id]["precision"]
                recall = metrics_by_query[query_id]["recall"]

                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0

                metrics_by_query[query_id]["f1"] = f1

            # Calculate MRR (Mean Reciprocal Rank)
            if "mrr" in self.metrics:
                mrr = self._calculate_mrr(retrieved_ids, rel_ids)
                metrics_by_query[query_id]["mrr"] = mrr

            # Calculate semantic coherence
            if "coherence" in self.metrics:
                coherence = 0.0
                if len(result["memories"]) >= 2:
                    # Calculate semantic coherence between retrieved memories
                    coherence = calculate_semantic_coherence(result["memories"])

                metrics_by_query[query_id]["coherence"] = coherence

        # Calculate average metrics across all queries
        avg_metrics = {}
        for metric in self.metrics:
            if all(metric in query_metrics for query_metrics in metrics_by_query.values()):
                avg_metrics[metric] = np.mean([
                    query_metrics[metric] for query_metrics in metrics_by_query.values()
                ])

        # Return metrics by query and averages
        return {
            "by_query": metrics_by_query,
            "average": avg_metrics
        }

    def _calculate_precision(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Calculate precision of retrieved results.
        
        Args:
            retrieved_ids: List of retrieved memory IDs
            relevant_ids: List of relevant memory IDs
            
        Returns:
            Precision score (0-1)
        """
        if not retrieved_ids:
            return 0.0

        relevant_count = sum(1 for mem_id in retrieved_ids if mem_id in relevant_ids)
        return relevant_count / len(retrieved_ids)

    def _calculate_recall(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Calculate recall of retrieved results.
        
        Args:
            retrieved_ids: List of retrieved memory IDs
            relevant_ids: List of relevant memory IDs
            
        Returns:
            Recall score (0-1)
        """
        if not relevant_ids:
            return 1.0  # All relevant items retrieved (there are none)

        relevant_count = sum(1 for mem_id in retrieved_ids if mem_id in relevant_ids)
        return relevant_count / len(relevant_ids)

    def _calculate_mrr(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Calculate Mean Reciprocal Rank.
        
        Args:
            retrieved_ids: List of retrieved memory IDs
            relevant_ids: List of relevant memory IDs
            
        Returns:
            MRR score (0-1)
        """
        for i, mem_id in enumerate(retrieved_ids):
            if mem_id in relevant_ids:
                return 1.0 / (i + 1)

        return 0.0

    def visualize_results(
        self,
        comparison_result: ComparisonResult,
        output_path: Optional[str] = None
    ) -> None:
        """Generate visualizations of comparison results.
        
        Args:
            comparison_result: Results from run_comparison
            output_path: Path to save visualizations (optional)
        """
        # Extract metrics for each system
        systems = ["memoryweave"] + list(comparison_result.baseline_metrics.keys())

        metrics = {}
        for metric in self.metrics:
            metrics[metric] = []

            # Add MemoryWeave metrics
            if metric in comparison_result.memoryweave_metrics["average"]:
                metrics[metric].append(
                    comparison_result.memoryweave_metrics["average"][metric]
                )
            else:
                metrics[metric].append(0.0)

            # Add baseline metrics
            for baseline in comparison_result.baseline_metrics.keys():
                if metric in comparison_result.baseline_metrics[baseline]["average"]:
                    metrics[metric].append(
                        comparison_result.baseline_metrics[baseline]["average"][metric]
                    )
                else:
                    metrics[metric].append(0.0)

        # Create subplots for each metric
        num_metrics = len(self.metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))

        if num_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(self.metrics):
            ax = axes[i]
            y_pos = np.arange(len(systems))

            ax.barh(y_pos, metrics[metric], align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(systems)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel(metric.upper())
            ax.set_title(f"{metric.upper()} by System")

            # Add values at the end of each bar
            for j, v in enumerate(metrics[metric]):
                ax.text(v + 0.01, j, f"{v:.3f}", va="center")

        # Create a separate figure for query time comparison
        time_fig, time_ax = plt.subplots(figsize=(10, 4))
        query_times = []
        for system in systems:
            if system == "memoryweave":
                query_times.append(
                    comparison_result.runtime_stats["memoryweave"]["avg_query_time"]
                )
            else:
                query_times.append(
                    comparison_result.runtime_stats[system]["avg_query_time"]
                )

        time_ax.barh(np.arange(len(systems)), query_times, align="center")
        time_ax.set_yticks(np.arange(len(systems)))
        time_ax.set_yticklabels(systems)
        time_ax.invert_yaxis()
        time_ax.set_xlabel("Average Query Time (seconds)")
        time_ax.set_title("Query Performance by System")

        for j, v in enumerate(query_times):
            time_ax.text(v + 0.01, j, f"{v:.5f}s", va="center")

        # Apply tight layout to each figure separately
        plt.figure(fig.number)
        plt.tight_layout()
        
        plt.figure(time_fig.number)
        plt.tight_layout()
        
        # If saving to file, save both figures
        if output_path:
            fig.savefig(output_path)
            
            # Save time figure with a modified filename
            time_output_path = output_path.replace('.png', '_time.png')
            time_fig.savefig(time_output_path)
            
            plt.close(fig)
            plt.close(time_fig)
        else:
            plt.show()

    def save_results(self, comparison_result: ComparisonResult, output_path: str) -> None:
        """Save comparison results to a file.
        
        Args:
            comparison_result: Results from run_comparison
            output_path: Path to save results
        """
        # Convert to dict for JSON serialization
        results_dict = comparison_result.dict()

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

    def generate_html_report(
        self,
        comparison_result: ComparisonResult,
        output_path: str,
        title: str = "MemoryWeave Baseline Comparison"
    ) -> None:
        """Generate an HTML report of comparison results.
        
        Args:
            comparison_result: Results from run_comparison
            output_path: Path to save HTML report
            title: Title for the report
        """
        # Save visualization as PNG for embedding in HTML
        img_path = output_path.replace(".html", "_chart.png")
        self.visualize_results(comparison_result, img_path)

        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    text-align: left;
                    padding: 12px;
                    border: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .system-name {{
                    font-weight: bold;
                }}
                .chart-container {{
                    max-width: 800px;
                    margin: 20px 0;
                }}
                .metric-value {{
                    font-weight: bold;
                }}
                .best-value {{
                    color: #27ae60;
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            
            <h2>Summary</h2>
            <p>
                Dataset: {comparison_result.dataset_stats["num_memories"]} memories, 
                {comparison_result.dataset_stats["num_queries"]} queries
            </p>
            
            <div class="chart-container">
                <h3>Performance Metrics</h3>
                <img src="{os.path.basename(img_path)}" alt="Performance Comparison Chart" style="max-width:100%">
            </div>
            
            <div class="chart-container">
                <h3>Query Performance</h3>
                <img src="{os.path.basename(img_path).replace('.png', '_time.png')}" alt="Query Time Comparison Chart" style="max-width:100%">
            </div>
            
            <h2>Average Metrics</h2>
            <table>
                <tr>
                    <th>System</th>
        """

        # Add metric columns
        for metric in self.metrics:
            html_content += f"<th>{metric.upper()}</th>"

        html_content += "<th>Avg Query Time</th></tr>"

        # Add MemoryWeave metrics
        html_content += "<tr><td class='system-name'>MemoryWeave</td>"

        for metric in self.metrics:
            if metric in comparison_result.memoryweave_metrics["average"]:
                value = comparison_result.memoryweave_metrics["average"][metric]
                html_content += f"<td class='metric-value'>{value:.4f}</td>"
            else:
                html_content += "<td>N/A</td>"

        # Add query time
        query_time = comparison_result.runtime_stats["memoryweave"]["avg_query_time"]
        html_content += f"<td>{query_time:.5f} s</td></tr>"

        # Add baseline metrics
        for baseline in comparison_result.baseline_metrics.keys():
            html_content += f"<tr><td class='system-name'>{baseline}</td>"

            for metric in self.metrics:
                if metric in comparison_result.baseline_metrics[baseline]["average"]:
                    value = comparison_result.baseline_metrics[baseline]["average"][metric]
                    html_content += f"<td class='metric-value'>{value:.4f}</td>"
                else:
                    html_content += "<td>N/A</td>"

            # Add query time
            query_time = comparison_result.runtime_stats[baseline]["avg_query_time"]
            html_content += f"<td>{query_time:.5f} s</td></tr>"

        html_content += """
            </table>
            
            <h2>Dataset Statistics</h2>
            <table>
                <tr>
                    <th>Statistic</th>
                    <th>Value</th>
                </tr>
        """

        # Add dataset statistics
        for stat, value in comparison_result.dataset_stats.items():
            html_content += f"<tr><td>{stat}</td><td>{value}</td></tr>"

        html_content += """
            </table>
            
            <h2>System Details</h2>
            <table>
                <tr>
                    <th>System</th>
                    <th>Total Time</th>
                    <th>Min Query Time</th>
                    <th>Max Query Time</th>
                </tr>
        """

        # Add system details
        for system, stats in comparison_result.runtime_stats.items():
            html_content += f"""
                <tr>
                    <td class='system-name'>{system}</td>
                    <td>{stats['total_time']:.5f} s</td>
                    <td>{stats['min_time']:.5f} s</td>
                    <td>{stats['max_time']:.5f} s</td>
                </tr>
            """

        html_content += """
            </table>
            
            <footer>
                <p>Generated by MemoryWeave Baseline Comparison Framework</p>
            </footer>
        </body>
        </html>
        """

        # Write HTML to file
        with open(output_path, "w") as f:
            f.write(html_content)
