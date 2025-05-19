import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
import re
import os
from typing import Dict, List, Optional, Tuple, Union

class JobPerformanceAnalyzer:
    # [Keep all your existing code here...]
    # Only adding the new AI chat functionality below

    def chat_interface(self):
        """
        Provides a simple chat interface to query job performance data using natural language.
        """
        print("\n=== Databricks Job Performance AI Assistant ===")
        print("Ask questions about your job performance data or type 'exit' to quit.")
        print("Examples:")
        print(" - Which job has the highest variance in runtime?")
        print(" - What's the optimal cluster size for performance?")
        print(" - Show me jobs with performance issues")
        print(" - What recommendations do you have for job 102?")
        
        while True:
            user_input = input("\nYour question: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Exiting chat. Goodbye!")
                break
                
            response = self._process_chat_query(user_input)
            print("\nAnswer:")
            print(response)
    
    def _process_chat_query(self, query: str) -> str:
        """
        Process a natural language query about job performance.
        
        Args:
            query: Natural language query about job performance
            
        Returns:
            Response to the query
        """
        query = query.lower()
        
        # Get data ready
        summary = self.get_job_summary()
        cluster_analysis = self.analyze_cluster_configurations()
        issues = self.identify_performance_issues()
        recommendations = self.recommend_optimizations()
        
        # Handle queries about job stats
        if any(keyword in query for keyword in ['highest runtime', 'longest runtime', 'slowest']):
            if len(summary) > 0:
                max_runtime = summary['avg_runtime'].max()
                slowest_job = summary[summary['avg_runtime'] == max_runtime]['job_id'].values[0]
                return f"Job {slowest_job} has the highest average runtime at {max_runtime:.2f} minutes."
            return "No job runtime data available."
        
        if any(keyword in query for keyword in ['lowest runtime', 'shortest runtime', 'fastest']):
            if len(summary) > 0:
                min_runtime = summary['avg_runtime'].min()
                fastest_job = summary[summary['avg_runtime'] == min_runtime]['job_id'].values[0]
                return f"Job {fastest_job} has the lowest average runtime at {min_runtime:.2f} minutes."
            return "No job runtime data available."
        
        if any(keyword in query for keyword in ['variance', 'highest variance', 'most variable']):
            if len(summary) > 0 and 'std_runtime' in summary.columns:
                # Filter out rows with NaN std_runtime
                valid_summary = summary.dropna(subset=['std_runtime'])
                if len(valid_summary) > 0:
                    # Calculate coefficient of variation for a fair comparison
                    valid_summary['cv'] = valid_summary['std_runtime'] / valid_summary['avg_runtime']
                    max_var_job = valid_summary.loc[valid_summary['cv'].idxmax()]['job_id']
                    max_cv = valid_summary['cv'].max()
                    return f"Job {max_var_job} has the highest runtime variance with a coefficient of variation of {max_cv:.2f}."
                return "Not enough data to calculate variance for any jobs."
            return "No standard deviation data available."
        
        # Handle queries about cluster configurations
        if any(keyword in query for keyword in ['optimal cluster', 'best cluster', 'optimal worker', 'best worker']):
            if len(cluster_analysis) > 0:
                mean_col = cluster_analysis.columns[cluster_analysis.columns.str.contains('mean')][0]
                min_runtime = cluster_analysis[mean_col].min()
                best_worker_count = cluster_analysis.loc[cluster_analysis[mean_col] == min_runtime, 'num_workers'].values[0]
                return f"The optimal cluster configuration appears to be {best_worker_count} workers, with an average runtime of {min_runtime:.2f} minutes."
            return "No cluster analysis data available."
        
        # Handle queries about specific jobs
        if 'job' in query and any(str(job_id) in query for job_id in summary['job_id'].astype(str).values):
            # Extract job ID from query
            job_ids = [job_id for job_id in summary['job_id'].astype(str).values if job_id in query]
            if job_ids:
                job_id = job_ids[0]
                job_summary = summary[summary['job_id'].astype(str) == job_id]
                job_issues = [issue for issue in issues if str(issue['job_id']) == job_id]
                job_recommendations = [rec for rec in recommendations if str(rec['job_id']) == job_id]
                
                response = f"Information for Job {job_id}:\n"
                if not job_summary.empty:
                    response += f"- Average runtime: {job_summary['avg_runtime'].values[0]:.2f} minutes\n"
                    response += f"- Minimum runtime: {job_summary['min_runtime'].values[0]:.2f} minutes\n"
                    response += f"- Maximum runtime: {job_summary['max_runtime'].values[0]:.2f} minutes\n"
                    if not pd.isna(job_summary['std_runtime'].values[0]):
                        response += f"- Runtime std dev: {job_summary['std_runtime'].values[0]:.2f} minutes\n"
                
                if job_issues:
                    response += "\nIdentified issues:\n"
                    for issue in job_issues:
                        response += f"- {issue['description']} (Severity: {issue['severity']})\n"
                
                if job_recommendations:
                    response += "\nRecommendations:\n"
                    for rec in job_recommendations:
                        response += f"- {rec['description']}\n"
                        response += f"  Action: {rec['action']}\n"
                
                return response
            
        # Handle queries about issues
        if any(keyword in query for keyword in ['issues', 'problem', 'error']):
            if issues:
                response = f"Found {len(issues)} issues across {len(set([issue['job_id'] for issue in issues]))} jobs:\n\n"
                for issue in issues:
                    response += f"Job {issue['job_id']}: {issue['description']} (Severity: {issue['severity']})\n"
                return response
            return "No issues have been identified in the job runs."
        
        # Handle queries about recommendations
        if any(keyword in query for keyword in ['recommendation', 'suggest', 'improve', 'optimize']):
            if recommendations:
                response = f"Here are {len(recommendations)} recommendations for {len(set([rec['job_id'] for rec in recommendations]))} jobs:\n\n"
                for rec in recommendations:
                    response += f"Job {rec['job_id']}: {rec['description']}\n"
                    response += f"Action: {rec['action']}\n\n"
                return response
            return "No recommendations available based on the current data."
        
        # General summary when query doesn't match specific patterns
        response = "Job Performance Summary:\n\n"
        
        if len(summary) > 0:
            avg_runtime = summary['avg_runtime'].mean()
            response += f"- Average runtime across all jobs: {avg_runtime:.2f} minutes\n"
            response += f"- Total job runs analyzed: {summary['run_count'].sum()}\n"
            response += f"- Number of unique jobs: {len(summary)}\n"
        
        if issues:
            response += f"- Found {len(issues)} issues in {len(set([issue['job_id'] for issue in issues]))} jobs\n"
        
        if recommendations:
            response += f"- Generated {len(recommendations)} recommendations\n"
        
        if len(cluster_analysis) > 0:
            mean_col = cluster_analysis.columns[cluster_analysis.columns.str.contains('mean')][0]
            min_runtime = cluster_analysis[mean_col].min()
            best_worker_count = cluster_analysis.loc[cluster_analysis[mean_col] == min_runtime, 'num_workers'].values[0]
            response += f"- Optimal worker count appears to be: {best_worker_count}\n"
        
        response += "\nTry asking more specific questions about jobs, performance, issues, or recommendations."
        return response


# Add this at the end of your script to launch the chat interface
if __name__ == "__main__":
    # Initialize the analyzer with your dataset
    csv_path = 'C:/Users/dasin/OneDrive/Documents/hckthn/repo/testdbx/data/databricks_job_runs_realistic.csv'
    
    analyzer = JobPerformanceAnalyzer(csv_path)
    
    # Run the standard analysis first
    print("\n--- JOB SUMMARY ---")
    summary = analyzer.get_job_summary()
    print(summary)
    
    print("\n--- CLUSTER ANALYSIS ---")
    cluster_analysis = analyzer.analyze_cluster_configurations()
    print(cluster_analysis)
    
    print("\n--- PERFORMANCE ISSUES ---")
    issues = analyzer.identify_performance_issues()
    for issue in issues:
        print(f"Job ID: {issue['job_id']}, Issue: {issue['description']}, Severity: {issue['severity']}")
    
    print("\n--- OPTIMIZATION RECOMMENDATIONS ---")
    recommendations = analyzer.recommend_optimizations()
    for rec in recommendations:
        print(f"Job ID: {rec['job_id']}, Recommendation: {rec['description']}")
        print(f"   Action: {rec['action']}")
    
    # Start the chat interface
    analyzer.chat_interface()
