import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
import re
from typing import Dict, List, Optional, Tuple, Union


class JobPerformanceAnalyzer:
    """
    Class for analyzing Databricks job performance metrics and providing optimization recommendations.
    """
    
    def __init__(self, job_data_path: str):
        """
        Initialize the analyzer with the path to the job performance data.
        
        Args:
            job_data_path: Path to the CSV or JSON file containing job performance data
        """
        self.job_data_path = job_data_path
        self.job_data = None
        self.load_data()
        self.runtime_col = None
        self.job_id_col = None
        self.cluster_cols = None
        self.output_cols = None
        self.start_time_col = None
        self._identify_columns()
        
    def load_data(self) -> None:
        """Load the job performance data from CSV or JSON file."""
        try:
            if self.job_data_path.endswith('.csv'):
                self.job_data = pd.read_csv(self.job_data_path)
            elif self.job_data_path.endswith('.json'):
                self.job_data = pd.read_json(self.job_data_path)
            else:
                # Try to infer format based on content
                try:
                    self.job_data = pd.read_csv(self.job_data_path)
                except:
                    try:
                        self.job_data = pd.read_json(self.job_data_path)
                    except Exception as e:
                        raise ValueError(f"Unsupported file format. Error: {str(e)}")
            
            print(f"Successfully loaded data with {len(self.job_data)} job runs.")
        except Exception as e:
            raise Exception(f"Failed to load job data: {str(e)}")
    
    def _identify_columns(self):
        """Identify key columns in the dataset."""
        if self.job_data is None:
            raise ValueError("No data loaded")
            
        # Print all columns to debug
        print("Available columns:", self.job_data.columns.tolist())
        
        # Find runtime column
        for col in ['runtime', 'runtime_minutes', 'duration', 'execution_time', 'duration_seconds']:
            if col in self.job_data.columns:
                self.runtime_col = col
                print(f"Using '{col}' as runtime column")
                break
                
        # Find job ID column
        for col in ['job_id', 'job_name', 'job', 'id', 'name']:
            if col in self.job_data.columns:
                self.job_id_col = col
                print(f"Using '{col}' as job ID column")
                break
        
        # Find cluster info columns
        self.cluster_cols = [col for col in self.job_data.columns 
                            if any(term in col.lower() for term in ['cluster', 'worker', 'node'])]
        if self.cluster_cols:
            print(f"Found cluster columns: {self.cluster_cols}")
        else:
            print("No cluster columns found")
            
        # Find output/log columns
        self.output_cols = [col for col in self.job_data.columns 
                           if any(term in col.lower() for term in ['output', 'log', 'job_output'])]
        if self.output_cols:
            print(f"Found output columns: {self.output_cols}")
        else:
            print("No output/log columns found")
            
        # Find start time column
        for col in ['start_time', 'start_date', 'start', 'run_time_utc']:
            if col in self.job_data.columns:
                self.start_time_col = col
                print(f"Using '{col}' as start time column")
                break
        
        # Convert runtime to minutes if in seconds
        if self.runtime_col and 'second' in self.runtime_col.lower():
            print(f"Converting {self.runtime_col} from seconds to minutes")
            self.job_data['runtime_minutes'] = self.job_data[self.runtime_col] / 60.0
            self.runtime_col = 'runtime_minutes'
    
    def get_schema(self) -> Dict:
        """
        Return the schema of the job data.
        
        Returns:
            Dict containing column names and their data types
        """
        if self.job_data is None:
            raise ValueError("No data loaded")
            
        schema = {
            "columns": list(self.job_data.columns),
            "dtypes": {col: str(dtype) for col, dtype in zip(self.job_data.columns, self.job_data.dtypes)}
        }
        return schema
    
    def get_job_summary(self) -> pd.DataFrame:
        """
        Get a summary of all jobs in the dataset.
        
        Returns:
            DataFrame with job summary statistics
        """
        if self.job_data is None:
            raise ValueError("No data loaded")
        
        if self.job_id_col is None:
            print("Warning: Could not find job identifier column")
            return pd.DataFrame()
        
        # Group by job identifier and calculate statistics
        if self.runtime_col:
            summary = self.job_data.groupby(self.job_id_col).agg({
                self.runtime_col: ['count', 'mean', 'min', 'max', 'std']
            })
            summary.columns = ['run_count', 'avg_runtime', 'min_runtime', 'max_runtime', 'std_runtime']
            return summary.reset_index()
        else:
            # Just count runs per job if no runtime info is available
            return self.job_data.groupby(self.job_id_col).size().reset_index(name='run_count')
    
    def analyze_cluster_configurations(self) -> pd.DataFrame:
        """
        Analyze cluster configurations and their impact on job runtime.
        
        Returns:
            DataFrame with cluster configurations and performance metrics
        """
        if self.job_data is None:
            raise ValueError("No data loaded")
        
        if not self.cluster_cols or not self.runtime_col:
            print("Warning: Could not find cluster configuration or runtime columns")
            return pd.DataFrame()
        
        # Check for direct worker count column
        if 'num_workers' in self.job_data.columns:
            print("Using 'num_workers' column directly")
            worker_count_col = 'num_workers'
        else:
            # Extract cluster size information
            print("Extracting worker count from cluster configuration")
            def extract_cluster_size(row):
                for col in self.cluster_cols:
                    if pd.notna(row[col]):
                        if isinstance(row[col], str):
                            # Try to extract worker count from JSON or text
                            if 'num_workers' in row[col].lower():
                                match = re.search(r'num_workers["\s:]+(\d+)', row[col])
                                if match:
                                    return int(match.group(1))
                            elif 'worker' in row[col].lower() and 'count' in row[col].lower():
                                match = re.search(r'worker.*count["\s:]+(\d+)', row[col])
                                if match:
                                    return int(match.group(1))
                return None
            
            try:
                self.job_data['worker_count'] = self.job_data.apply(extract_cluster_size, axis=1)
                worker_count_col = 'worker_count'
            except Exception as e:
                print(f"Error extracting worker count: {str(e)}")
                return pd.DataFrame()
        
        # Group by worker count and get runtime stats
        try:
            if worker_count_col in self.job_data.columns and self.job_data[worker_count_col].notna().any():
                return self.job_data.groupby(worker_count_col).agg({
                    self.runtime_col: ['count', 'mean', 'min', 'max']
                }).reset_index()
        except Exception as e:
            print(f"Error analyzing cluster configurations: {str(e)}")
        
        return pd.DataFrame()
    
    def identify_performance_issues(self) -> List[Dict]:
        """
        Identify potential performance issues in the job runs.
        
        Returns:
            List of dictionaries containing identified issues
        """
        if self.job_data is None:
            raise ValueError("No data loaded")
        
        issues = []
        
        if self.runtime_col:
            # Identify jobs with high runtime variance
            if self.job_id_col:
                job_stats = self.job_data.groupby(self.job_id_col)[self.runtime_col].agg(['mean', 'std']).reset_index()
                # Calculate coefficient of variation (relative standard deviation)
                job_stats['cv'] = job_stats['std'] / job_stats['mean'].replace(0, float('nan'))
                
                # Jobs with high variance in runtime
                high_variance_jobs = job_stats[job_stats['cv'] > 0.5].dropna()
                for _, row in high_variance_jobs.iterrows():
                    issues.append({
                        'job_id': row[self.job_id_col],
                        'issue_type': 'high_runtime_variance',
                        'description': f"Job has high runtime variance (CV: {row['cv']:.2f})",
                        'severity': 'medium'
                    })
            
            # Identify outlier runtimes (significantly longer than average)
            overall_mean = self.job_data[self.runtime_col].mean()
            overall_std = self.job_data[self.runtime_col].std()
            outliers = self.job_data[self.job_data[self.runtime_col] > overall_mean + 2 * overall_std]
            
            for _, row in outliers.iterrows():
                job_id = row[self.job_id_col] if self.job_id_col else "Unknown"
                issues.append({
                    'job_id': job_id,
                    'issue_type': 'runtime_outlier',
                    'description': f"Runtime ({row[self.runtime_col]:.2f}) significantly higher than average ({overall_mean:.2f})",
                    'severity': 'high'
                })
        
        # Check logs/output for common error patterns
        if self.output_cols:
            output_col = self.output_cols[0]
            
            # Define error patterns to look for
            error_patterns = [
                ('memory_error', r'out\s+of\s+memory|memory\s+exceed|MemoryError'),
                ('timeout', r'timeout|timed\s+out'),
                ('resource_constraint', r'resource\s+constraint|no\s+available\s+resource|not\s+enough\s+resource'),
                ('disk_space', r'no\s+space\s+left|disk\s+full|not\s+enough\s+disk'),
                ('data_skew', r'data\s+skew|skewed\s+data|partition\s+skew'),
                ('shuffle_error', r'shuffle\s+error|shuffle\s+failed'),
                ('connection_error', r'connection.*timeout')
            ]
            
            # Search for patterns in output logs
            for _, row in self.job_data.iterrows():
                if pd.notna(row.get(output_col)) and isinstance(row[output_col], str):
                    job_id = row[self.job_id_col] if self.job_id_col else "Unknown"
                    
                    for error_type, pattern in error_patterns:
                        if re.search(pattern, row[output_col], re.IGNORECASE):
                            issues.append({
                                'job_id': job_id,
                                'issue_type': error_type,
                                'description': f"Found {error_type.replace('_', ' ')} pattern in job output",
                                'severity': 'high'
                            })
        
        return issues
    
    def recommend_optimizations(self) -> List[Dict]:
        """
        Provide optimization recommendations based on job performance analysis.
        
        Returns:
            List of dictionaries containing optimization recommendations
        """
        if self.job_data is None:
            raise ValueError("No data loaded")
        
        recommendations = []
        issues = self.identify_performance_issues()
        
        # Group issues by job_id
        issues_by_job = {}
        for issue in issues:
            job_id = issue['job_id']
            if job_id not in issues_by_job:
                issues_by_job[job_id] = []
            issues_by_job[job_id].append(issue)
        
        # Generate recommendations based on issues
        for job_id, job_issues in issues_by_job.items():
            # Check for memory issues
            if any(issue['issue_type'] == 'memory_error' for issue in job_issues):
                recommendations.append({
                    'job_id': job_id,
                    'recommendation_type': 'increase_memory',
                    'description': 'Increase memory allocation for the job',
                    'action': 'Update cluster configuration with more memory per node'
                })
            
            # Check for data skew issues
            if any(issue['issue_type'] == 'data_skew' for issue in job_issues):
                recommendations.append({
                    'job_id': job_id,
                    'recommendation_type': 'optimize_partitioning',
                    'description': 'Optimize data partitioning to reduce skew',
                    'action': 'Review partition keys and consider repartition operations'
                })
            
            # Check for timeout issues
            if any(issue['issue_type'] == 'timeout' for issue in job_issues) or \
               any(issue['issue_type'] == 'connection_error' for issue in job_issues):
                recommendations.append({
                    'job_id': job_id,
                    'recommendation_type': 'increase_timeout',
                    'description': 'Increase timeout setting for the job',
                    'action': 'Update job configuration with longer timeout period'
                })
            
            # Check for resource constraints
            if any(issue['issue_type'] == 'resource_constraint' for issue in job_issues):
                recommendations.append({
                    'job_id': job_id,
                    'recommendation_type': 'increase_cluster_size',
                    'description': 'Increase cluster size for the job',
                    'action': 'Update cluster configuration with more worker nodes'
                })
            
            # Check for high variance in runtime
            if any(issue['issue_type'] == 'high_runtime_variance' for issue in job_issues):
                recommendations.append({
                    'job_id': job_id,
                    'recommendation_type': 'investigate_variability',
                    'description': 'Investigate causes of runtime variability',
                    'action': 'Check for data volume changes, resource contention, or inconsistent input sources'
                })
            
            # Check for outlier runtimes
            if any(issue['issue_type'] == 'runtime_outlier' for issue in job_issues):
                recommendations.append({
                    'job_id': job_id,
                    'recommendation_type': 'optimize_performance',
                    'description': 'Optimize job for better performance',
                    'action': 'Review job code and execution plan for optimization opportunities'
                })
        
        return recommendations
    
    def visualize_job_performance(self, job_id=None):
        """
        Create visualizations of job performance metrics.
        
        Args:
            job_id: Optional job_id to filter for specific job
        
        Returns:
            Dictionary with plot figures
        """
        if self.job_data is None:
            raise ValueError("No data loaded")
        
        if not self.runtime_col:
            raise ValueError("Could not find runtime column in the data")
        
        # Filter for specific job if requested
        data = self.job_data
        if job_id and self.job_id_col and job_id in data[self.job_id_col].values:
            data = data[data[self.job_id_col] == job_id]
        
        # Create runtime distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(data[self.runtime_col], bins=20)
        plt.xlabel('Runtime (minutes)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Job Runtimes')
        runtime_dist_fig = plt.gcf()
        plt.close()
        
        # Timeline plot
        timeline_fig = None
        if self.start_time_col:
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(data[self.start_time_col]):
                    data[self.start_time_col] = pd.to_datetime(data[self.start_time_col])
                
                # Sort by start time
                data = data.sort_values(self.start_time_col)
                
                plt.figure(figsize=(12, 6))
                plt.scatter(data[self.start_time_col], data[self.runtime_col])
                plt.xlabel('Start Time')
                plt.ylabel('Runtime (minutes)')
                plt.title('Job Runtime Timeline')
                plt.xticks(rotation=45)
                plt.tight_layout()
                timeline_fig = plt.gcf()
                plt.close()
            except Exception as e:
                print(f"Could not create timeline plot: {str(e)}")
        
        # Cluster performance plot
        cluster_fig = None
        if 'num_workers' in data.columns:
            try:
                plt.figure(figsize=(10, 6))
                data.groupby('num_workers')[self.runtime_col].mean().plot(kind='bar')
                plt.xlabel('Worker Count')
                plt.ylabel('Average Runtime (minutes)')
                plt.title('Average Runtime by Cluster Size')
                cluster_fig = plt.gcf()
                plt.close()
            except Exception as e:
                print(f"Could not create cluster performance plot: {str(e)}")
        elif 'worker_count' in data.columns and data['worker_count'].notna().any():
            try:
                plt.figure(figsize=(10, 6))
                data.groupby('worker_count')[self.runtime_col].mean().plot(kind='bar')
                plt.xlabel('Worker Count')
                plt.ylabel('Average Runtime (minutes)')
                plt.title('Average Runtime by Cluster Size')
                cluster_fig = plt.gcf()
                plt.close()
            except Exception as e:
                print(f"Could not create cluster performance plot: {str(e)}")
        
        return {
            'runtime_distribution': runtime_dist_fig,
            'timeline': timeline_fig,
            'cluster_performance': cluster_fig
        }
    
    def generate_performance_report(self) -> Dict:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary containing performance report data
        """
        if self.job_data is None:
            raise ValueError("No data loaded")
        
        return {
            'summary': self.get_job_summary().to_dict(orient='records'),
            'cluster_analysis': self.analyze_cluster_configurations().to_dict(orient='records'),
            'issues': self.identify_performance_issues(),
            'recommendations': self.recommend_optimizations()
        }

# Initialize the analyzer with your dataset
analyzer = JobPerformanceAnalyzer('C:/Users/dasin/OneDrive/Documents/hckthn/repo/testdbx/data/databricks_job_runs_realistic.csv')

# Get job summary statistics
summary = analyzer.get_job_summary()
print("\n--- JOB SUMMARY ---")
print(summary)

# Analyze how cluster configurations impact performance
cluster_analysis = analyzer.analyze_cluster_configurations()
print("\n--- CLUSTER ANALYSIS ---")
print(cluster_analysis)

# Identify performance issues
issues = analyzer.identify_performance_issues()
print("\n--- PERFORMANCE ISSUES ---")
for issue in issues:
    print(f"Job ID: {issue['job_id']}, Issue: {issue['description']}, Severity: {issue['severity']}")

# Get optimization recommendations
recommendations = analyzer.recommend_optimizations()
print("\n--- OPTIMIZATION RECOMMENDATIONS ---")
for rec in recommendations:
    print(f"Job ID: {rec['job_id']}, Recommendation: {rec['description']}")
    print(f"   Action: {rec['action']}")

# Generate visualizations
print("\n--- GENERATING VISUALIZATIONS ---")
visualizations = analyzer.visualize_job_performance()
print("Visualizations created but not displayed (would need a display interface)")

# Save the visualizations
try:
    for name, fig in visualizations.items():
        if fig:
            fig_path = f"{name}.png"
            fig.savefig(fig_path)
            print(f"Saved visualization to {fig_path}")
except Exception as e:
    print(f"Error saving visualizations: {str(e)}")

# Generate a comprehensive report
print("\n--- GENERATING PERFORMANCE REPORT ---")
report = analyzer.generate_performance_report()
print("Report generated with the following sections:")
for section, data in report.items():
    if isinstance(data, list):
        print(f"- {section}: {len(data)} items")
    else:
        print(f"- {section}")

print("\nAnalysis complete!")
