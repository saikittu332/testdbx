import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data directly
data_path = 'C:/Users/dasin/OneDrive/Documents/hckthn/repo/testdbx/data/databricks_job_runs_realistic.csv'
job_data = pd.read_csv(data_path)

print("Data loaded successfully with", len(job_data), "rows")
print("Columns:", job_data.columns.tolist())

# Convert seconds to minutes
job_data['runtime_minutes'] = job_data['duration_seconds'] / 60.0

# Create a simple summary
summary = job_data.groupby('job_id').agg({
    'runtime_minutes': ['count', 'mean', 'min', 'max', 'std']
})
summary.columns = ['run_count', 'avg_runtime', 'min_runtime', 'max_runtime', 'std_runtime']
summary = summary.reset_index()

print("\n--- JOB SUMMARY ---")
print(summary)

# Check for issues
print("\n--- CHECKING FOR ISSUES ---")
overall_mean = job_data['runtime_minutes'].mean()
overall_std = job_data['runtime_minutes'].std()
outliers = job_data[job_data['runtime_minutes'] > overall_mean + 2 * overall_std]

print(f"Mean runtime: {overall_mean:.2f} minutes")
print(f"Runtime std: {overall_std:.2f} minutes")
print(f"Found {len(outliers)} outlier runs")

if len(outliers) > 0:
    print("\nOutlier Runs:")
    for _, row in outliers.iterrows():
        print(f"Job ID: {row['job_id']}, Runtime: {row['runtime_minutes']:.2f} minutes")

# Create a simple plot
print("\n--- CREATING VISUALIZATION ---")
plt.figure(figsize=(10, 6))
plt.hist(job_data['runtime_minutes'], bins=10)
plt.xlabel('Runtime (minutes)')
plt.ylabel('Frequency')
plt.title('Distribution of Job Runtimes')
plt.savefig('runtime_distribution_debug.png')
print("Saved plot to runtime_distribution_debug.png")

print("\nScript completed successfully")
