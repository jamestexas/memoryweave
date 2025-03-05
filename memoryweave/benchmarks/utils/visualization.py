# File: memoryweave/benchmarks/utils/visualization.py


import matplotlib.pyplot as plt
import numpy as np


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
