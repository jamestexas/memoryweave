"""
Run Comprehensive Diagnostic Analysis for MemoryWeave

This script runs all diagnostic analyses to identify core issues with memory retrieval
and generates a comprehensive report with findings and recommendations.
"""

import os
import json
import subprocess
import time
import matplotlib.pyplot as plt
from datetime import datetime

# Create main output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"diagnostic_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

def run_analysis(script_name, description):
    """
    Run a diagnostic analysis script and capture its output.
    
    Args:
        script_name: Name of the script to run
        description: Description of the analysis
        
    Returns:
        Dictionary with execution results
    """
    print(f"\n=== Running {description} ===")
    print(f"Executing {script_name}...")
    
    start_time = time.time()
    
    # Run the script and capture output
    result = subprocess.run(
        ["python", f"scripts/{script_name}"],
        capture_output=True,
        text=True
    )
    
    execution_time = time.time() - start_time
    
    # Check if execution was successful
    if result.returncode == 0:
        status = "Success"
    else:
        status = "Failed"
        
    # Save output to file
    output_file = os.path.join(output_dir, f"{script_name.replace('.py', '')}_output.txt")
    with open(output_file, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\nERRORS:\n")
            f.write(result.stderr)
    
    print(f"Execution {status} in {execution_time:.2f} seconds")
    print(f"Output saved to {output_file}")
    
    return {
        "script": script_name,
        "description": description,
        "status": status,
        "execution_time": execution_time,
        "output_file": output_file,
        "return_code": result.returncode
    }

def collect_findings():
    """
    Collect findings from all analysis outputs.
    
    Returns:
        Dictionary with key findings
    """
    print("\n=== Collecting Findings ===")
    
    findings = {}
    
    # Try to load summary files from each analysis
    try:
        # Main diagnostic analysis
        if os.path.exists("diagnostic_output/summary_report.json"):
            with open("diagnostic_output/summary_report.json") as f:
                findings["main_diagnostic"] = json.load(f)
                
        # Embedding quality analysis
        if os.path.exists("diagnostic_output/embedding_quality/summary.json"):
            with open("diagnostic_output/embedding_quality/summary.json") as f:
                findings["embedding_quality"] = json.load(f)
                
        # Similarity distribution analysis
        if os.path.exists("diagnostic_output/similarity_distributions/summary.json"):
            with open("diagnostic_output/similarity_distributions/summary.json") as f:
                findings["similarity_distributions"] = json.load(f)
                
        # Failed queries analysis
        if os.path.exists("diagnostic_output/failed_queries.json"):
            with open("diagnostic_output/failed_queries.json") as f:
                failed_queries = json.load(f)
                findings["failed_queries"] = {
                    "count": len(failed_queries),
                    "personal_count": sum(1 for q in failed_queries if q["detected_type"] == "personal"),
                    "factual_count": sum(1 for q in failed_queries if q["detected_type"] != "personal")
                }
    except Exception as e:
        print(f"Error collecting findings: {e}")
    
    return findings

def generate_report(execution_results, findings):
    """
    Generate a comprehensive report with findings and recommendations.
    
    Args:
        execution_results: List of execution results
        findings: Dictionary with key findings
    """
    print("\n=== Generating Report ===")
    
    report = {
        "timestamp": timestamp,
        "execution_results": execution_results,
        "findings": findings,
        "recommendations": []
    }
    
    # Generate recommendations based on findings
    if findings:
        # Check for query classification issues
        if "main_diagnostic" in findings and "query_classification" in findings["main_diagnostic"]:
            accuracy = findings["main_diagnostic"]["query_classification"]["accuracy"]
            if accuracy < 0.8:
                report["recommendations"].append({
                    "issue": "Query Classification Accuracy",
                    "description": f"Query classification accuracy is low ({accuracy:.2%})",
                    "recommendation": "Improve query classification by enhancing the NLP extraction module with more training data and better patterns."
                })
        
        # Check for threshold optimization
        if "similarity_distributions" in findings and "optimal_thresholds" in findings["similarity_distributions"]:
            personal_threshold = findings["similarity_distributions"]["optimal_thresholds"]["personal"]["threshold"]
            factual_threshold = findings["similarity_distributions"]["optimal_thresholds"]["factual"]["threshold"]
            
            if abs(personal_threshold - factual_threshold) > 0.1:
                report["recommendations"].append({
                    "issue": "Different Optimal Thresholds",
                    "description": f"Personal queries ({personal_threshold:.2f}) and factual queries ({factual_threshold:.2f}) have significantly different optimal thresholds",
                    "recommendation": "Implement query type detection and use different thresholds for different query types."
                })
        
        # Check for embedding quality
        if "embedding_quality" in findings:
            if "intra_set_similarity" in findings["embedding_quality"]:
                related_similarity = findings["embedding_quality"]["intra_set_similarity"]["avg_similarity_related"]
                unrelated_similarity = findings["embedding_quality"]["intra_set_similarity"]["avg_similarity_unrelated"]
                
                if related_similarity < 0.6:
                    report["recommendations"].append({
                        "issue": "Low Intra-Set Similarity",
                        "description": f"Average similarity within related sets is low ({related_similarity:.2f})",
                        "recommendation": "Consider using a more powerful embedding model or fine-tuning the current model on domain-specific data."
                    })
                
                if related_similarity - unrelated_similarity < 0.2:
                    report["recommendations"].append({
                        "issue": "Poor Embedding Separation",
                        "description": f"Small difference between related ({related_similarity:.2f}) and unrelated ({unrelated_similarity:.2f}) similarities",
                        "recommendation": "Enhance embedding quality by using a model with better semantic understanding or implementing contrastive learning."
                    })
        
        # Check for failed queries
        if "failed_queries" in findings:
            if findings["failed_queries"]["count"] > 0:
                personal_ratio = findings["failed_queries"]["personal_count"] / findings["failed_queries"]["count"]
                
                if personal_ratio > 0.7:
                    report["recommendations"].append({
                        "issue": "High Personal Query Failure Rate",
                        "description": f"Most failed queries ({personal_ratio:.0%}) are personal queries",
                        "recommendation": "Implement specialized handling for personal queries, possibly with entity tracking and reference resolution."
                    })
                elif personal_ratio < 0.3:
                    report["recommendations"].append({
                        "issue": "High Factual Query Failure Rate",
                        "description": f"Most failed queries ({1-personal_ratio:.0%}) are factual queries",
                        "recommendation": "Implement advanced semantic matching for factual queries, possibly with query expansion and two-stage retrieval."
                    })
    
    # If no specific recommendations could be generated, add general ones
    if not report["recommendations"]:
        report["recommendations"] = [
            {
                "issue": "General Retrieval Performance",
                "description": "Overall retrieval performance needs improvement",
                "recommendation": "Implement query type detection and specialized retrieval pipelines for different query types."
            },
            {
                "issue": "Embedding Quality",
                "description": "Embedding quality may be affecting retrieval performance",
                "recommendation": "Consider using a more powerful embedding model or fine-tuning on domain-specific data."
            },
            {
                "issue": "Threshold Optimization",
                "description": "Threshold settings may not be optimal for all query types",
                "recommendation": "Implement dynamic threshold adjustment based on query characteristics."
            }
        ]
    
    # Save report to file
    report_file = os.path.join(output_dir, "diagnostic_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MemoryWeave Diagnostic Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .section {{ margin-bottom: 30px; }}
            .finding {{ margin-bottom: 15px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }}
            .recommendation {{ margin-bottom: 15px; padding: 10px; background-color: #e6f7ff; border-radius: 5px; }}
            .success {{ color: green; }}
            .failure {{ color: red; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>MemoryWeave Diagnostic Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="section">
            <h2>Execution Summary</h2>
            <table>
                <tr>
                    <th>Analysis</th>
                    <th>Status</th>
                    <th>Execution Time</th>
                </tr>
    """
    
    for result in execution_results:
        status_class = "success" if result["status"] == "Success" else "failure"
        html_report += f"""
                <tr>
                    <td>{result["description"]}</td>
                    <td class="{status_class}">{result["status"]}</td>
                    <td>{result["execution_time"]:.2f} seconds</td>
                </tr>
        """
    
    html_report += """
            </table>
        </div>
        
        <div class="section">
            <h2>Key Findings</h2>
    """
    
    # Add findings to HTML report
    if "main_diagnostic" in findings:
        html_report += """
            <h3>Query Classification</h3>
        """
        if "query_classification" in findings["main_diagnostic"]:
            accuracy = findings["main_diagnostic"]["query_classification"]["accuracy"]
            html_report += f"""
            <div class="finding">
                <p>Query classification accuracy: {accuracy:.2%}</p>
            </div>
            """
    
    if "similarity_distributions" in findings:
        html_report += """
            <h3>Similarity Distributions</h3>
        """
        if "optimal_thresholds" in findings["similarity_distributions"]:
            personal = findings["similarity_distributions"]["optimal_thresholds"]["personal"]
            factual = findings["similarity_distributions"]["optimal_thresholds"]["factual"]
            html_report += f"""
            <div class="finding">
                <p>Optimal threshold for personal queries: {personal["threshold"]:.2f} (F1: {personal["f1"]:.3f})</p>
                <p>Optimal threshold for factual queries: {factual["threshold"]:.2f} (F1: {factual["f1"]:.3f})</p>
            </div>
            """
    
    if "embedding_quality" in findings:
        html_report += """
            <h3>Embedding Quality</h3>
        """
        if "intra_set_similarity" in findings["embedding_quality"]:
            related = findings["embedding_quality"]["intra_set_similarity"]["avg_similarity_related"]
            unrelated = findings["embedding_quality"]["intra_set_similarity"]["avg_similarity_unrelated"]
            html_report += f"""
            <div class="finding">
                <p>Average similarity within related sets: {related:.3f}</p>
                <p>Average similarity within unrelated sets: {unrelated:.3f}</p>
            </div>
            """
    
    html_report += """
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
    """
    
    # Add recommendations to HTML report
    for recommendation in report["recommendations"]:
        html_report += f"""
            <div class="recommendation">
                <h3>{recommendation["issue"]}</h3>
                <p><strong>Description:</strong> {recommendation["description"]}</p>
                <p><strong>Recommendation:</strong> {recommendation["recommendation"]}</p>
            </div>
        """
    
    html_report += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    html_file = os.path.join(output_dir, "diagnostic_report.html")
    with open(html_file, "w") as f:
        f.write(html_report)
    
    print(f"Report saved to {report_file}")
    print(f"HTML report saved to {html_file}")
    
    return report_file, html_file

def main():
    """Run the comprehensive diagnostic analysis."""
    print("MemoryWeave Comprehensive Diagnostic Analysis")
    print("============================================")
    print(f"Output directory: {output_dir}")
    
    # List of analyses to run
    analyses = [
        {"script": "diagnostic_analysis.py", "description": "Main Diagnostic Analysis"},
        {"script": "analyze_embedding_quality.py", "description": "Embedding Quality Analysis"},
        {"script": "analyze_similarity_distributions.py", "description": "Similarity Distribution Analysis"}
    ]
    
    # Run each analysis
    execution_results = []
    for analysis in analyses:
        result = run_analysis(analysis["script"], analysis["description"])
        execution_results.append(result)
    
    # Collect findings
    findings = collect_findings()
    
    # Generate report
    report_file, html_file = generate_report(execution_results, findings)
    
    print("\nDiagnostic analysis complete.")
    print(f"Report saved to {report_file}")
    print(f"HTML report saved to {html_file}")
    print("\nNext steps:")
    print("1. Review the diagnostic report for findings and recommendations")
    print("2. Implement the recommended changes to improve retrieval performance")
    print("3. Re-run the diagnostic analysis after implementing changes to measure improvement")


if __name__ == "__main__":
    main()
