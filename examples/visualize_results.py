#!/usr/bin/env python
"""
Visualization script for MemoryWeave evaluation results.

This script generates visualizations from evaluation results to help understand 
the performance characteristics of different configurations.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel

console = Console()

def load_results(file_path):
    """Load results from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_comparison_visualizations(results, output_dir=None):
    """Create comparison visualizations from results."""
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract configuration names and metrics
    configs = list(results.keys())
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame([
        {
            'config': name,
            'retriever_type': results[name]['config'].get('retriever_type', 'unknown'),
            'query_time': results[name]['avg_query_time'],
            'precision': results[name]['precision'],
            'recall': results[name]['recall'],
            'f1_score': results[name]['f1_score'],
            'semantic_similarity': results[name].get('semantic_similarity', 0),
            'overall_score': results[name].get('overall_score', results[name]['f1_score']),
            'avg_results': results[name]['avg_retrieval_count'],
        }
        for name in configs
    ])
    
    # Visualizations
    plt.style.use('fivethirtyeight')
    
    # 1. Performance metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(configs))
    width = 0.18
    
    ax.bar(x - width*2, df['precision'], width, label='Precision')
    ax.bar(x - width, df['recall'], width, label='Recall')
    ax.bar(x, df['f1_score'], width, label='F1 Score')
    ax.bar(x + width, df['semantic_similarity'], width, label='Semantic Similarity')
    ax.bar(x + width*2, df['overall_score'], width, label='Overall Score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Scores')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # 2. Query time vs. result count
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add labels to each point
    for i, row in df.iterrows():
        ax.annotate(row['config'], 
                    (row['query_time'], row['avg_results']),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    # Use different colors for different retriever types
    scatter = ax.scatter(df['query_time'], df['avg_results'], 
                         c=pd.factorize(df['retriever_type'])[0], 
                         s=100, alpha=0.7)
    
    # Add a legend for retriever types
    legend1 = ax.legend(scatter.legend_elements()[0], 
                       df['retriever_type'].unique(),
                       title="Retriever Type",
                       loc="upper left")
    ax.add_artist(legend1)
    
    ax.set_xlabel('Average Query Time (s)')
    ax.set_ylabel('Average Results Count')
    ax.set_title('Speed vs Result Count')
    
    # Add fit line
    if len(df) > 1:
        z = np.polyfit(df['query_time'], df['avg_results'], 1)
        p = np.poly1d(z)
        ax.plot(df['query_time'], p(df['query_time']), "r--", alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'speed_vs_results.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # 3. Precision-Recall tradeoff
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add labels to each point
    for i, row in df.iterrows():
        ax.annotate(row['config'], 
                    (row['precision'], row['recall']),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    # Use different colors for different retriever types
    scatter = ax.scatter(df['precision'], df['recall'], 
                         c=pd.factorize(df['retriever_type'])[0], 
                         s=df['overall_score'] * 200, alpha=0.7)
    
    # Add a legend for retriever types
    legend1 = ax.legend(scatter.legend_elements()[0], 
                       df['retriever_type'].unique(),
                       title="Retriever Type",
                       loc="upper right")
    ax.add_artist(legend1)
    
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_title('Precision-Recall Tradeoff (circle size = overall score)')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    
    # Add F1-score contours
    x = np.linspace(0.01, 1, 100)
    for f1 in [0.2, 0.4, 0.6, 0.8]:
        y = (f1 * x) / (2 * x - f1)
        mask = (y >= 0) & (y <= 1)
        plt.plot(x[mask], y[mask], ':', color='gray', alpha=0.5)
        # Add contour label at rightmost point
        rightmost_idx = np.where(mask)[0][-1] if np.any(mask) else -1
        if rightmost_idx >= 0:
            plt.annotate(f'F1={f1}', xy=(x[rightmost_idx], y[rightmost_idx]),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=8, color='gray')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'precision_recall.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # 4. Performance by query type
    # Extract query type performance if available
    query_type_data = {}
    for config in configs:
        if 'by_query_type' in results[config]:
            for query_type, metrics in results[config]['by_query_type'].items():
                if query_type not in query_type_data:
                    query_type_data[query_type] = []
                
                query_type_data[query_type].append({
                    'config': config,
                    'f1_score': metrics.get('f1_score', 0)
                })
    
    if query_type_data:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate positions
        num_types = len(query_type_data)
        num_configs = len(configs)
        bar_width = 0.7 / num_configs
        
        for i, (query_type, type_data) in enumerate(query_type_data.items()):
            for j, data_point in enumerate(type_data):
                x_pos = i + (j - num_configs/2 + 0.5) * bar_width
                ax.bar(x_pos, data_point['f1_score'], bar_width, 
                     label=data_point['config'] if i == 0 else "")
        
        ax.set_xlabel('Query Type')
        ax.set_ylabel('F1 Score')
        ax.set_title('Performance by Query Type')
        ax.set_xticks(range(num_types))
        ax.set_xticklabels(list(query_type_data.keys()))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / 'query_type_performance.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    # Close all figures to avoid memory leaks
    plt.close('all')
    
    return df

def print_summary(df):
    """Print a text summary of the results."""
    console.print(Panel.fit("[bold green]MemoryWeave Evaluation Summary[/bold green]"))
    
    # Sort by overall score
    top_configs = df.sort_values('overall_score', ascending=False)
    
    console.print("\n[bold]Top Configurations by Overall Score:[/bold]")
    for i, (_, row) in enumerate(top_configs.iterrows()):
        console.print(f"{i+1}. [cyan]{row['config']}[/cyan] (Overall: {row['overall_score']:.4f}, F1: {row['f1_score']:.4f})")
    
    # Calculate performance improvements
    baseline_rows = df[df['config'].str.contains('Baseline')]
    other_rows = df[~df['config'].str.contains('Baseline')]
    
    if not baseline_rows.empty and not other_rows.empty:
        baseline = baseline_rows.iloc[0]
        best_advanced = other_rows.sort_values('overall_score', ascending=False).iloc[0]
        
        f1_improvement = (best_advanced['f1_score'] / baseline['f1_score'] - 1) * 100 if baseline['f1_score'] > 0 else float('inf')
        overall_improvement = (best_advanced['overall_score'] / baseline['overall_score'] - 1) * 100 if baseline['overall_score'] > 0 else float('inf')
        
        console.print("\n[bold]Improvement over Baseline:[/bold]")
        if f1_improvement != float('inf'):
            improvement_color = "green" if f1_improvement > 0 else "red"
            console.print(f"F1 Score: [{improvement_color}]{f1_improvement:.1f}%[/{improvement_color}]")
        
        if overall_improvement != float('inf'):
            improvement_color = "green" if overall_improvement > 0 else "red"
            console.print(f"Overall Score: [{improvement_color}]{overall_improvement:.1f}%[/{improvement_color}]")
    
    # Analyze query response time differences
    fastest = df.loc[df['query_time'].idxmin()]
    slowest = df.loc[df['query_time'].idxmax()]
    
    time_diff = (slowest['query_time'] / fastest['query_time'] - 1) * 100
    
    console.print("\n[bold]Speed Comparison:[/bold]")
    console.print(f"Fastest: [cyan]{fastest['config']}[/cyan] ({fastest['query_time']:.4f}s)")
    console.print(f"Slowest: [cyan]{slowest['config']}[/cyan] ({slowest['query_time']:.4f}s)")
    console.print(f"Difference: {time_diff:.1f}%")
    
    # Highlight any interesting patterns or anomalies
    console.print("\n[bold]Key Observations:[/bold]")
    
    # Check if all advanced configurations have identical scores
    if len(other_rows) > 1:
        f1_scores = other_rows['f1_score'].unique()
        if len(f1_scores) == 1:
            console.print("- [yellow]All advanced configurations have identical F1 scores, suggesting they may be using the same execution path[/yellow]")
    
    # Check for perfect recall
    perfect_recall = df[df['recall'] >= 0.99]
    if not perfect_recall.empty:
        console.print(f"- {len(perfect_recall)} configuration(s) achieved near-perfect recall")
    
    # Check for very low precision
    low_precision = df[df['precision'] < 0.2]
    if not low_precision.empty:
        console.print(f"- [yellow]{len(low_precision)} configuration(s) have precision below 0.2, which may indicate retrieval of too many irrelevant items[/yellow]")
    
    # Analyze semantic similarity if available
    semantic_sim_values = df['semantic_similarity'].unique()
    if len(semantic_sim_values) == 1 and semantic_sim_values[0] == 0:
        console.print("- [yellow]No semantic similarity scores available - consider enabling a real embedding model[/yellow]")
    else:
        best_semantic = df.loc[df['semantic_similarity'].idxmax()]
        console.print(f"- Best semantic relevance: [cyan]{best_semantic['config']}[/cyan] ({best_semantic['semantic_similarity']:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Visualize MemoryWeave evaluation results")
    parser.add_argument("--results", type=str, default="evaluation_results.json", help="Path to results JSON file")
    parser.add_argument("--output-dir", type=str, default="evaluation_charts", help="Directory to save visualizations")
    args = parser.parse_args()
    
    try:
        results = load_results(args.results)
        
        # Create output directory if needed
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # Create visualizations and get DataFrame of results
        df = create_comparison_visualizations(results, args.output_dir)
        
        # Print text summary
        print_summary(df)
        
        if args.output_dir:
            console.print(f"\nVisualizations saved to: [green]{args.output_dir}[/green]")
    
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())