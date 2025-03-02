#!/bin/bash
# Run the Contextual Fabric benchmark with various memory sizes and create visualizations

# Ensure the script is executable with: chmod +x run_contextual_fabric_benchmark.sh

echo "================================================================================"
echo "Note: You will see 'BM25 retrieval failed' warnings during benchmark execution."
echo "These are expected with synthetic test data and don't indicate a problem."
echo "The hybrid strategy will fall back to vector retrieval when BM25 indexing fails."
echo "In real applications with natural language content, BM25 performs much better."
echo "================================================================================"
echo ""

# Create directories if they don't exist
mkdir -p benchmark_results
mkdir -p evaluation_charts

# Run with different memory sizes and preserve results
echo "Running benchmark with 20 memories..."
uv run python -m benchmarks.contextual_fabric_benchmark --memories 20 --output benchmark_results/contextual_fabric_20.json
echo ""

echo "Running benchmark with 100 memories..."
uv run python -m benchmarks.contextual_fabric_benchmark --memories 100 --output benchmark_results/contextual_fabric_100.json
echo ""

echo "Running benchmark with 500 memories..."
uv run python -m benchmarks.contextual_fabric_benchmark --memories 500 --output benchmark_results/contextual_fabric_500.json
echo ""

# Create visualizations for each result
echo "Generating visualizations..."
for file in benchmark_results/contextual_fabric_*.json; do
  size=$(echo $file | grep -o '[0-9]\+' | head -1)
  output_dir="evaluation_charts/contextual_fabric_${size}"
  mkdir -p $output_dir
  uv run python benchmarks/visualize_contextual_fabric.py $file $output_dir
done

echo "Benchmark complete! Results and visualizations are available in benchmark_results/ and evaluation_charts/"