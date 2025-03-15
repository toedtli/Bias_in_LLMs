import pandas as pd
from scipy.stats import ttest_ind_from_stats
import itertools
import os

# --- Configuration ---
# File paths for the three runs
file_run1 = "explizite_Analyse/results/results_run_1/results_run_1.csv"
file_run2 = "explizite_Analyse/results/results_run_2/results_run_2.csv"
file_run3 = "explizite_Analyse/results/results_run_3/results_run_3.csv"

# Significance threshold
alpha = 0.05

# --- Helper function using aggregated stats ---
def perform_ttest_from_aggregates(mean1, sem1, n1, mean2, sem2, n2, alpha=0.05):
    """
    Performs Welch's t-test using summary statistics for two groups.
    It computes the standard deviation from the SEM and sample size (std = sem * sqrt(n)),
    and then uses ttest_ind_from_stats.
    """
    std1 = sem1 * (n1 ** 0.5)
    std2 = sem2 * (n2 ** 0.5)
    t_stat, p_value = ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=n1,
                                           mean2=mean2, std2=std2, nobs2=n2,
                                           equal_var=False)
    significance = "significant" if p_value < alpha else "not significant"
    return t_stat, p_value, significance

# --- Read CSV files ---
df1 = pd.read_csv(file_run1)
df2 = pd.read_csv(file_run2)
df3 = pd.read_csv(file_run3)

# Add a column to indicate the run for each dataframe
df1["run"] = "run1"
df2["run"] = "run2"
df3["run"] = "run3"

# --- Combine the data ---
# Assumes each CSV contains: "Model", "Group", "Axis Name", "mean", "sem", and "n"
df_all = pd.concat([df1, df2, df3], ignore_index=True)

# --- Run pairwise t-tests for every Model, Group, Axis Name combination ---
group_columns = ["Model", "Group", "Axis Name"]
results = []

# Group by the combination of Model, Group, and Axis Name
for keys, group in df_all.groupby(group_columns):
    # Get the unique runs present for this group
    runs = group["run"].unique()
    # Only proceed if at least two runs are available for comparison
    if len(runs) < 2:
        continue
    # For every pair of runs, perform the t-test
    for run_a, run_b in itertools.combinations(runs, 2):
        # Extract the aggregated values for run_a and run_b.
        # Here we assume that there is only one row per combination.
        row_a = group[group["run"] == run_a].iloc[0]
        row_b = group[group["run"] == run_b].iloc[0]
        
        # Extract mean, SEM, and sample size from each run
        mean_a, sem_a, n_a = row_a["mean"], row_a["SEM"], row_a["count"]
        mean_b, sem_b, n_b = row_b["mean"], row_b["SEM"], row_b["count"]
        
        # Perform the t-test using the aggregated statistics
        t_stat, p_value, significance = perform_ttest_from_aggregates(
            mean_a, sem_a, n_a, mean_b, sem_b, n_b, alpha
        )
        
        results.append({
            "Model": keys[0],
            "Group": keys[1],
            "Axis Name": keys[2],
            "Run A": run_a,
            "Run B": run_b,
            "Mean A": mean_a,
            "SEM A": sem_a,
            "n A": n_a,
            "Mean B": mean_b,
            "SEM B": sem_b,
            "n B": n_b,
            "t_stat": t_stat,
            "p_value": p_value,
            "significance": significance
        })

# --- Save the t-test results ---
output_dir = "explizite_Analyse/t-test/per_run"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "t_test_results_by_run.csv")
results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

print(f"T-test results saved to {output_path}")