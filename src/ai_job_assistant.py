import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
import re
import os
from typing import Dict, List, Optional, Tuple, Union

class AIJobAssistant:
    """
    AI assistant that can analyze Databricks job performance and engage in general conversation.
    """
    
    def __init__(self, job_analyzer=None):
        """
        Initialize the AI assistant with an optional job analyzer.
        
        Args:
            job_analyzer: Optional JobPerformanceAnalyzer instance
        """
        self.job_analyzer = job_analyzer
        self.context = []  # Maintain some conversation context
    
    def chat(self):
        """Starts an interactive chat session with the AI assistant."""
        print("\n=== Databricks Performance AI Assistant ===")
        print("I can help with your Databricks job performance and answer general questions.")
        print("Examples:")
        print(" - Which job has the highest variance in runtime?")
        print(" - What's the optimal cluster size for performance?")
        print(" - What are best practices for optimizing Spark jobs?")
        print(" - How does Databricks autoscaling work?")
        print(" - Type 'exit' to end our conversation")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nAI: Goodbye! Have a great day.")
                break
                
            response = self.process_query(user_input)
            print(f"\nAI: {response}")
    
    def process_query(self, query: str) -> str:
        """
        Process a user query, either about job data or general topics.
        
        Args:
            query: User's query
            
        Returns:
            Response to the query
        """
        # Store recent context
        self.context.append(query)
        if len(self.context) > 5:
            self.context.pop(0)
        
        # Check if query is about job performance data
        is_job_related = self._is_job_performance_query(query)
        
        if is_job_related and self.job_analyzer:
            try:
                # Process job performance related query
                return self._process_job_query(query)
            except Exception as e:
                return f"I encountered an issue analyzing your job data: {str(e)}. Feel free to ask me something else or try rephrasing your question."
        else:
            # Handle general knowledge queries
            return self._process_general_query(query)
    
    def _is_job_performance_query(self, query: str) -> bool:
        """Determine if a query is related to job performance analysis."""
        query = query.lower()
        
        # Keywords that strongly indicate job performance questions
        job_keywords = [
            'job', 'runtime', 'cluster', 'performance', 'databricks', 
            'worker', 'execution', 'spark', 'duration', 'run time', 
            'timeout', 'error', 'failure', 'recommendation', 'optimize'
        ]
        
        # Check for job-specific query indicators
        if any(keyword in query for keyword in job_keywords):
            return True
            
        # Check if query appears to reference specific job IDs from our data
        if self.job_analyzer:
            summary = self.job_analyzer.get_job_summary()
            if not summary.empty and hasattr(summary, 'job_id'):
                if any(f"job {str(job_id)}" in query for job_id in summary['job_id']):
                    return True
        
        # If recent context was about jobs, this might be a follow-up
        if len(self.context) > 1:
            previous_query = self.context[-2].lower() if len(self.context) >= 2 else ""
            if any(keyword in previous_query for keyword in job_keywords):
                # This is likely a follow-up to a job-related question
                return True
        
        return False
    
    def _process_job_query(self, query: str) -> str:
        """Process a query specifically about job performance data."""
        if not self.job_analyzer:
            return "I don't have access to job performance data right now. I can answer general questions about Databricks, Spark, or data processing instead."
        
        query = query.lower()
        
        # Get data ready
        summary = self.job_analyzer.get_job_summary()
        cluster_analysis = self.job_analyzer.analyze_cluster_configurations()
        issues = self.job_analyzer.identify_performance_issues()
        recommendations = self.job_analyzer.recommend_optimizations()
        
        # Count total jobs
        if any(keyword in query for keyword in ['how many job', 'total job', 'count job']):
            if not summary.empty:
                job_count = len(summary)
                total_runs = summary['run_count'].sum()
                return f"There are {job_count} unique jobs in the current dataset with a total of {total_runs} job runs."
            return "No job data available."
        
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
            if not cluster_analysis.empty:
                # Handle MultiIndex columns properly
                if isinstance(cluster_analysis.columns, pd.MultiIndex):
                    # Find the mean column in MultiIndex
                    mean_col = None
                    for col in cluster_analysis.columns:
                        if 'mean' in col[1].lower():
                            mean_col = col
                            break
                    
                    if mean_col:
                        min_runtime = cluster_analysis[mean_col].min()
                        best_worker_row = cluster_analysis[cluster_analysis[mean_col] == min_runtime]
                        if not best_worker_row.empty:
                            best_worker_count = best_worker_row.iloc[0]['num_workers']
                            return f"The optimal cluster configuration appears to be {best_worker_count} workers, with an average runtime of {min_runtime:.2f} minutes."
                else:
                    # For non-MultiIndex columns
                    mean_col = [col for col in cluster_analysis.columns if 'mean' in col.lower()]
                    if mean_col:
                        mean_col = mean_col[0]
                        min_runtime = cluster_analysis[mean_col].min()
                        best_worker_count = cluster_analysis.loc[cluster_analysis[mean_col] == min_runtime, 'num_workers'].values[0]
                        return f"The optimal cluster configuration appears to be {best_worker_count} workers, with an average runtime of {min_runtime:.2f} minutes."
            
            return "No cluster analysis data available or couldn't determine optimal configuration."
        
        # Handle queries about specific jobs
        job_id_match = None
        if 'job' in query:
            # Try to extract job ID from query
            for job_id in summary['job_id'].astype(str).values:
                if job_id in query or f"job {job_id}" in query:
                    job_id_match = job_id
                    break
                    
        if job_id_match:
            job_summary = summary[summary['job_id'].astype(str) == job_id_match]
            job_issues = [issue for issue in issues if str(issue['job_id']) == job_id_match]
            job_recommendations = [rec for rec in recommendations if str(rec['job_id']) == job_id_match]
            
            response = f"Information for Job {job_id_match}:\n"
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
        
        # Safely extract optimal worker count information
        if not cluster_analysis.empty:
            try:
                if isinstance(cluster_analysis.columns, pd.MultiIndex):
                    # Find the mean column in MultiIndex
                    for col in cluster_analysis.columns:
                        if 'mean' in col[1].lower():
                            mean_col = col
                            min_runtime = cluster_analysis[mean_col].min()
                            best_worker_row = cluster_analysis[cluster_analysis[mean_col] == min_runtime]
                            if not best_worker_row.empty:
                                best_worker_count = best_worker_row.iloc[0]['num_workers']
                                response += f"- Optimal worker count appears to be: {best_worker_count}\n"
                            break
            except Exception:
                # If any error occurs when trying to determine optimal worker count, just skip it
                pass
        
        response += "\nTry asking more specific questions about jobs, performance, issues, or recommendations."
        return response
    
    def _process_general_query(self, query: str) -> str:
        """Process a general knowledge or conversation query."""
        query = query.lower()
        
        # Databricks/Spark optimization best practices responses
        if any(term in query for term in ['best practice', 'best way', 'optimize', 'improve performance']):
            if 'databricks' in query or 'spark' in query:
                return """Here are some Databricks/Spark job optimization best practices:

1. Right-size your clusters - Match cluster size to your workload requirements
2. Use appropriate file formats - Parquet or Delta are preferred for performance
3. Optimize data partitioning - Choose partition keys wisely based on query patterns
4. Caching - Use cache() or persist() for reused data
5. Minimize data shuffling - Reduce operations like groupBy, join, and repartition
6. Use broadcast joins for small tables - When joining a small table with a large one
7. Set appropriate parallelism - Number of partitions should be 2-3x the number of cores
8. Use efficient UDFs - Prefer Pandas UDFs over regular Python UDFs
9. Optimize storage - Consider data skipping with Z-ordering or data compaction

Would you like me to elaborate on any of these techniques?"""
            
        # Knowledge about Databricks autoscaling
        elif 'autoscaling' in query and ('databricks' in query or 'cluster' in query):
            return """Databricks autoscaling automatically adjusts the number of worker nodes in your cluster based on workload:

1. How it works: Databricks monitors the load and adds/removes nodes to maintain optimal performance
2. Configuration: Set min and max workers when creating your cluster
3. Benefits: Cost savings by using only necessary resources and improved job efficiency
4. Metrics used: Spark task scheduling, HDFS I/O, and system metrics
5. Best for: Workloads with varying resource demands

Autoscaling works best for jobs that have periods of high computation followed by lower demand.
For consistent workloads, fixed-size clusters might be more appropriate."""
        
        # Response about Databricks job scheduling
        elif any(term in query for term in ['schedule', 'cron', 'trigger', 'automate']):
            if 'job' in query and ('databricks' in query or 'spark' in query):
                return """Databricks jobs can be scheduled using several methods:

1. Time-based scheduling: Using cron expressions (e.g., "0 0 * * *" for daily at midnight)
2. Triggered via REST API: For programmatic job control
3. Event-based triggers: Such as new files arriving in cloud storage
4. Dependent job scheduling: Jobs can be set to run after other jobs complete
5. Webhook triggers: External systems can trigger jobs via webhooks

The scheduling options are accessible in the job configuration UI or via the Jobs API.
For complex workflows, you might consider using Databricks Workflows to orchestrate multiple jobs."""

        # Response about comparing Spark and MapReduce
        elif ('spark' in query and 'mapreduce' in query) or ('spark' in query and 'hadoop' in query):
            if any(term in query for term in ['compare', 'difference', 'better', 'faster']):
                return """Comparing Spark vs. MapReduce:

Spark advantages:
- In-memory processing (up to 100x faster than MapReduce)
- Supports iterative algorithms through DAG execution model
- Rich API with SQL, DataFrame, ML, graph processing
- Interactive mode (shell) for exploration
- Unified platform for batch, streaming, and ML

MapReduce advantages:
- Can handle larger-than-memory datasets with fewer resources
- Simpler architecture for basic use cases
- Mature ecosystem and tooling

Spark has largely superseded MapReduce for most modern big data workloads due to its performance and ease of use, though MapReduce may still be appropriate for some specific use cases where memory constraints are significant."""

        # General greeting
        elif any(term in query for term in ['hello', 'hi ', 'hey', 'greetings']):
            return "Hello! I'm your Databricks Job Performance AI Assistant. I can help you analyze your job performance data or answer general questions about Databricks, Spark, or data processing. How can I assist you today?"

        # Introduction
        elif any(term in query for term in ['who are you', 'what can you do', 'introduce yourself', 'what are you']):
            return """I'm your Databricks Job Performance AI Assistant. I can:

1. Analyze your Databricks job performance data
2. Identify performance issues and optimization opportunities
3. Provide recommendations to improve job efficiency
4. Answer questions about Databricks, Spark, and big data processing
5. Share best practices for data engineering

Feel free to ask me about your job metrics, cluster configurations, or general knowledge about Databricks and Spark!"""

        # Fallback response
        else:
            return """I can help you with:

1. Analyzing your Databricks job performance data
2. Identifying optimization opportunities
3. Answering questions about Databricks and Spark
4. Providing big data processing best practices

Try asking something like:
- "Which job has the highest runtime?"
- "What's the optimal cluster size?"
- "What are best practices for optimizing Spark jobs?"
- "How does Databricks autoscaling work?"
"""


# Integration code to connect the AI assistant with your job analyzer
if __name__ == "__main__":
    # Initialize the job analyzer with your dataset
    csv_path = 'C:/Users/dasin/OneDrive/Documents/hckthn/repo/testdbx/data/databricks_job_runs_realistic.csv'
    
    # Import your original JobPerformanceAnalyzer class
    from job_perf2 import JobPerformanceAnalyzer  # Adjust import as needed
    
    # Initialize your job analyzer
    job_analyzer = JobPerformanceAnalyzer(csv_path)
    
    # Run the standard analysis first
    print("\n--- JOB SUMMARY ---")
    summary = job_analyzer.get_job_summary()
    print(summary)
    
    print("\n--- CLUSTER ANALYSIS ---")
    cluster_analysis = job_analyzer.analyze_cluster_configurations()
    print(cluster_analysis)
    
    print("\n--- PERFORMANCE ISSUES ---")
    issues = job_analyzer.identify_performance_issues()
    for issue in issues:
        print(f"Job ID: {issue['job_id']}, Issue: {issue['description']}, Severity: {issue['severity']}")
    
    print("\n--- OPTIMIZATION RECOMMENDATIONS ---")
    recommendations = job_analyzer.recommend_optimizations()
    for rec in recommendations:
        print(f"Job ID: {rec['job_id']}, Recommendation: {rec['description']}")
        print(f"   Action: {rec['action']}")
    
    # Create the AI assistant with the job analyzer
    ai_assistant = AIJobAssistant(job_analyzer)
    
    # Start the chat interface
    ai_assistant.chat()
