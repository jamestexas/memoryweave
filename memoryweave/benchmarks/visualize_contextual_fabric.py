#!/usr/bin/env python
"""
Visualize the results of the Contextual Fabric benchmark.

This script generates charts comparing the performance of the baseline
retrieval strategy versus the contextual fabric strategy.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Set style for plots
plt.style.use("seaborn-v0_8-whitegrid")


def load_results(results_file: str) -> dict:
    """
    Load benchmark results from JSON file.

    Args:
        results_file: Path to JSON results file

    Returns:
        dictionary of results data
    """
    with open(results_file) as f:
        return json.load(f)


def create_f1_comparison_chart(results: dict, output_file: str = None) -> None:
    """
    Create a bar chart comparing F1 scores across test cases.

    Args:
        results: Benchmark results dictionary
        output_file: Optional output file path
    """
    # Extract test case names and F1 scores
    test_cases = []
    baseline_f1 = []
    fabric_f1 = []

    for test_case in results["test_cases"]:
        test_cases.append(test_case["test_case"])
        baseline_f1.append(test_case["metrics"]["baseline"]["f1"])
        fabric_f1.append(test_case["metrics"]["fabric"]["f1"])

    # Set up plot
    _fig, ax = plt.figure(figsize=(12, 7)), plt.gca()

    # Plot bars
    x = np.arange(len(test_cases))
    width = 0.35

    ax.bar(
        x - width / 2,
        baseline_f1,
        width,
        label="HybridBM25Vector Baseline",
        color="#3498db",
        alpha=0.7,
    )
    ax.bar(x + width / 2, fabric_f1, width, label="Contextual Fabric", color="#2ecc71", alpha=0.7)

    # Add labels and title
    ax.set_xlabel("Test Case")
    ax.set_ylabel("F1 Score")
    ax.set_title("Contextual Fabric vs. HybridBM25Vector Baseline F1 Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(test_cases, rotation=45, ha="right")
    ax.legend()

    # Add value labels on bars
    for i, v in enumerate(baseline_f1):
        ax.text(i - width / 2, v + 0.02, f"{v:.2f}", ha="center")

    for i, v in enumerate(fabric_f1):
        ax.text(i + width / 2, v + 0.02, f"{v:.2f}", ha="center")

    # Add improvement annotations
    for i in range(len(test_cases)):
        improvement = fabric_f1[i] - baseline_f1[i]
        if improvement > 0:
            ax.annotate(
                f"+{improvement:.2f}",
                xy=(i, max(fabric_f1[i], baseline_f1[i]) + 0.05),
                ha="center",
                va="bottom",
                color="green",
                fontweight="bold",
            )

    # Tight layout and save
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Chart saved to {output_file}")
    else:
        plt.show()


def create_summary_chart(results: dict, output_file: str = None) -> None:
    """
    Create a summary chart showing overall performance improvements.

    Args:
        results: Benchmark results dictionary
        output_file: Optional output file path
    """
    # Extract summary metrics
    summary = results["summary"]

    # Set up plot
    _fig, ax = plt.figure(figsize=(10, 6)), plt.gca()

    # Data for bars
    metrics = ["Average F1"]
    baseline = [summary["average_baseline_f1"]]
    fabric = [summary["average_fabric_f1"]]

    # Plot bars
    x = np.arange(len(metrics))
    width = 0.35

    baseline_bars = ax.bar(
        x - width / 2,
        baseline,
        width,
        label="HybridBM25Vector Baseline",
        color="#3498db",
        alpha=0.7,
    )
    fabric_bars = ax.bar(
        x + width / 2, fabric, width, label="Contextual Fabric", color="#2ecc71", alpha=0.7
    )

    # Add labels and title
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Overall Performance: Contextual Fabric vs. HybridBM25Vector Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add value labels on bars
    for bar in baseline_bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    for bar in fabric_bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    # Add improvement annotation
    improvement = summary["average_improvement"]
    if baseline[0] > 0:
        percentage = f"(+{improvement / baseline[0] * 100:.1f}%)"
    else:
        percentage = "(∞%)" if improvement > 0 else "(0%)"

    ax.annotate(
        f"Improvement: +{improvement:.3f} {percentage}",
        xy=(x[0], max(fabric[0], baseline[0]) + 0.05),
        ha="center",
        va="bottom",
        color="green",
        fontweight="bold",
    )

    # Add experiment metadata
    ax.text(
        0.95,
        0.05,
        f"Memories: {summary['num_memories']}\n"
        f"Test Cases: {summary['num_test_cases']}\n"
        f"Date: {results['timestamp'].split('T')[0]}",
        transform=ax.transAxes,
        fontsize=10,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )

    # Tight layout and save
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Chart saved to {output_file}")
    else:
        plt.show()


def main():
    """Main function to generate visualizations."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_contextual_fabric.py <results_file> [output_dir]")
        sys.exit(1)

    results_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "evaluation_charts"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    try:
        results = load_results(results_file)
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)

    # Create charts
    create_f1_comparison_chart(
        results, os.path.join(output_dir, "contextual_fabric_f1_comparison.png")
    )
    create_summary_chart(results, os.path.join(output_dir, "contextual_fabric_summary.png"))

    print("Visualization complete!")


def create_radar_chart(
    data: dict[str, dict[str, float]],
    metrics: list[str],
    title: str,
    output_file: str = None,
) -> plt.Figure:
    """Create a radar chart for comparing multiple metrics across configurations."""
    config_names = list(data.keys())

    # Number of variables
    N = len(metrics)

    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Add each configuration as a polygon
    for _i, config_name in enumerate(config_names):
        values = [data[config_name].get(metric, 0) for metric in metrics]
        values += values[:1]  # Close the loop

        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=config_name)
        ax.fill(angles, values, alpha=0.1)

    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each metric and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.title(title, size=15, color="navy", y=1.1)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


def create_bar_chart(
    data: dict[str, float],
    title: str,
    ylabel: str,
    output_file: str | None = None,
    sort_values: bool = False,
) -> plt.Figure:
    """Create a simple bar chart for benchmark metrics."""
    labels = list(data.keys())
    values = list(data.values())

    if sort_values:
        # Sort by values
        sorted_items = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
        labels, values = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color="royalblue")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


def create_comparison_chart(
    data: dict[str, dict[str, float]],
    metric_name: str,
    title: str,
    ylabel: str,
    output_file: str | None = None,
) -> plt.Figure:
    """Create a comparison chart for a specific metric across configurations."""
    labels = list(data.keys())
    values = [config_data.get(metric_name, 0) for config_data in data.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color="royalblue")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    main()
