import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
import re
import os
import requests
from typing import Dict, List, Optional, Tuple, Union

class AIJobAgent:
    """
    AI agent that connects to a real AI model API and augments it with Databricks job performance data.
    """
    
    def __init__(self, job_analyzer=None, api_key=None):
        """
        Initialize the AI agent with a job analyzer and API key for AI service.
        
        Args:
            job_analyzer: JobPerformanceAnalyzer instance
            api_key: API key for the AI service (Claude, OpenAI, etc.)
        """
        self.job_analyzer = job_analyzer
        self.api_key = api_key
        self.conversation_history = []
        
        # If no API key provided, check environment variable
        if not self.api_key:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    
    def chat(self):
        """Starts an interactive chat session with the AI agent."""
        print("\n=== Databricks Performance AI Agent ===")
        print("I can help with your Databricks job performance and answer general questions.")
        print("Type 'exit' to end our conversation")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nAI Agent: Goodbye! Have a great day.")
                break
                
            # Capture job data information to augment the AI model's knowledge
            job_data_context = self._get_job_data_context(user_input)
            
            # Send to AI model with context
            response = self._get_ai_response(user_input, job_data_context)
            print(f"\nAI Agent: {response}")
    
    def _get_job_data_context(self, query):
        """
        Get relevant job data context based on the query to augment the AI model.
        
        Args:
            query: User's query
            
        Returns:
            String containing relevant job data context
        """
        if not self.job_analyzer:
            return "No job data available."
            
        context = []
        
        # Determine which job data is relevant to the query
        query_lower = query.lower()
        
        # Include job summary if relevant
        if any(keyword in query_lower for keyword in ['job', 'runtime', 'performance', 'summary']):
            try:
                summary = self.job_analyzer.get_job_summary()
                if not summary.empty:
                    context.append(f"Job Summary: {summary.to_string()}")
            except Exception as e:
                context.append(f"Error getting job summary: {str(e)}")
        
        # Include cluster analysis if relevant
        if any(keyword in query_lower for keyword in ['cluster', 'worker', 'optimal', 'configuration']):
            try:
                cluster_analysis = self.job_analyzer.analyze_cluster_configurations()
                if not cluster_analysis.empty:
                    context.append(f"Cluster Analysis: {cluster_analysis.to_string()}")
            except Exception as e:
                context.append(f"Error getting cluster analysis: {str(e)}")
        
        # Include issues if relevant
        if any(keyword in query_lower for keyword in ['issue', 'problem', 'error']):
            try:
                issues = self.job_analyzer.identify_performance_issues()
                if issues:
                    issues_str = "\n".join([f"Job {issue['job_id']}: {issue['description']} (Severity: {issue['severity']})" for issue in issues])
                    context.append(f"Identified Issues:\n{issues_str}")
            except Exception as e:
                context.append(f"Error getting issues: {str(e)}")
        
        # Include recommendations if relevant
        if any(keyword in query_lower for keyword in ['recommendation', 'suggest', 'improve', 'optimize']):
            try:
                recommendations = self.job_analyzer.recommend_optimizations()
                if recommendations:
                    recs_str = "\n".join([f"Job {rec['job_id']}: {rec['description']} - Action: {rec['action']}" for rec in recommendations])
                    context.append(f"Optimization Recommendations:\n{recs_str}")
            except Exception as e:
                context.append(f"Error getting recommendations: {str(e)}")
        
        # If query is about a specific job ID, include that job's details
        if 'job' in query_lower:
            try:
                summary = self.job_analyzer.get_job_summary()
                job_id_match = None
                
                for job_id in summary['job_id'].astype(str).values:
                    if job_id in query_lower or f"job {job_id}" in query_lower:
                        job_id_match = job_id
                        break
                        
                if job_id_match:
                    job_info = summary[summary['job_id'].astype(str) == job_id_match]
                    context.append(f"Information for Job {job_id_match}: {job_info.to_string()}")
            except Exception as e:
                context.append(f"Error getting specific job info: {str(e)}")
        
        # Join all relevant context
        if context:
            return "\n\n".join(context)
        else:
            return "No specific job data context available for this query."
    
    def _get_ai_response(self, query, job_data_context):
        """Get a response from OpenAI's models with fallback options."""
        
        if not self.api_key:
            return "API key not found. Please set OPENAI_API_KEY environment variable or provide API key during initialization."
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        try:
            # OpenAI API integration with newer client library
            from openai import OpenAI
            
            # Set up the client with the API key
            client = OpenAI(api_key=self.api_key)
            
            # Create system message with job data context
            system_message = f"""You are an AI assistant specializing in Databricks job performance analysis and general data engineering knowledge.
            You have access to the user's Databricks job performance data. Answer their question using this data when relevant.
            
            Here is the relevant job data context for this query:
            
            {job_data_context}"""
            
            # Prepare messages for the conversation
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
            
            # Try models in order of preference with fallback
            models_to_try = ["gpt-4.1", "gpt-4", "gpt-3.5-turbo"]
            
            for model in models_to_try:
                try:
                    # Call the OpenAI API
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=1000
                    )
                    
                    ai_response = response.choices[0].message.content
                    self.conversation_history.append({"role": "assistant", "content": ai_response})
                    return ai_response
                except Exception as model_error:
                    # If this model fails, try the next one
                    print(f"Model {model} failed: {str(model_error)}. Trying next model...")
                    continue
            
            # If we get here, all models failed
            return "Unable to get a response from any available AI model. Please check your API access and try again."
            
        except Exception as e:
            return f"Error getting AI response: {str(e)}. Please ensure your API key is correct and the OpenAI service is available."


# Integration code to connect the AI agent with your job analyzer
if __name__ == "__main__":
    # Initialize the job analyzer with your dataset
    csv_path = 'C:/Users/dasin/OneDrive/Documents/hckthn/repo/testdbx/data/databricks_job_runs_realistic.csv'
    
    # Import your original JobPerformanceAnalyzer class
    from job_perf2 import JobPerformanceAnalyzer  # Adjust import as needed
    
    # Initialize your job analyzer
    job_analyzer = JobPerformanceAnalyzer(csv_path)
    
    # # Run the standard analysis first
    # print("\n--- JOB SUMMARY ---")
    # summary = job_analyzer.get_job_summary()
    # print(summary)
    
    # print("\n--- CLUSTER ANALYSIS ---")
    # cluster_analysis = job_analyzer.analyze_cluster_configurations()
    # print(cluster_analysis)
    
    # print("\n--- PERFORMANCE ISSUES ---")
    # issues = job_analyzer.identify_performance_issues()
    # for issue in issues:
    #     print(f"Job ID: {issue['job_id']}, Issue: {issue['description']}, Severity: {issue['severity']}")
    
    # print("\n--- OPTIMIZATION RECOMMENDATIONS ---")
    # recommendations = job_analyzer.recommend_optimizations()
    # for rec in recommendations:
    #     print(f"Job ID: {rec['job_id']}, Recommendation: {rec['description']}")
    #     print(f"   Action: {rec['action']}")
    
    # Check if API key is available from environment variable
    api_key = input("Enter your OpenAI API Key: ")
    
    if not api_key:
        print("\nWARNING: No AI API key found in environment variables.")
        print("For full AI capabilities, set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
        print("Continuing with simulated AI responses.")
    
    # Create the AI agent with the job analyzer and API key
    ai_agent = AIJobAgent(job_analyzer, api_key)
    
    # Start the chat interface
    ai_agent.chat()
