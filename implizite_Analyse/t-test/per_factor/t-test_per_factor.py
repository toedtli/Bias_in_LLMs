#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import itertools

def main():
    # Define file paths.
    input_csv = "implizite_Analyse/data/scoring_processed/scoring_processed_run_1.csv"
    output_csv = "implizite_Analyse/t-test/per_factor/all_ttests_results.csv"
    alpha = 0.05  # significance level

    # Load the CSV file.
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    print("Columns in dataset:", df.columns.tolist())

    # Check if the dependent variable 'Score' exists.
    if "Alternative Scoring" not in df.columns:
        print("The 'Alternative Scoring' column is missing from the CSV.")
        return
    
    # All factors to consider: all columns except ...
    factors = [col for col in df.columns if col not in ["Alternative Scoring", "Score", "Question ID", "Choice Set", "Choices Used"]]    
    print("Factors available for grouping and comparison:", factors)
        
    results = []

    # For each column, use it as a grouping factor.
    for group_factor in factors:
        unique_levels = df[group_factor].dropna().unique()
        #print(f"\nGrouping by {group_factor} with levels: {unique_levels}")
        for level in unique_levels:
            # Subset data for the current grouping level.
            sub_df = df[df[group_factor] == level]
            # For each other factor (excluding group_factor and Score), perform t-tests.
            for comp_factor in [f for f in factors if f != group_factor]:
                comp_levels = sub_df[comp_factor].dropna().unique()
                if len(comp_levels) < 2:
                    continue  # need at least two groups to compare
                # For every pair of levels in the compared factor:
                for lvl1, lvl2 in itertools.combinations(comp_levels, 2):
                    data1 = sub_df[sub_df[comp_factor] == lvl1]["Score"]
                    data2 = sub_df[sub_df[comp_factor] == lvl2]["Score"]
                    if data1.empty or data2.empty:
                        continue
                    t_stat, p_value = ttest_ind(data1, data2, nan_policy="omit")
                    significance = "Significant" if p_value < alpha else "Not significant"
                    results.append({
                        "Grouping Factor": group_factor,
                        "Grouping Level": level,
                        "Compared Factor": comp_factor,
                        "Group 1": lvl1,
                        "Group 2": lvl2,
                        "Mean 1": data1.mean(),
                        "Mean 2": data2.mean(),
                        "n1": data1.count(),
                        "n2": data2.count(),
                        "t-statistic": t_stat,
                        "p-value": p_value,
                        "Significance": significance
                    })
                   # print(f"{group_factor}={level} | {comp_factor}: {lvl1} vs {lvl2} -> t = {t_stat:.3f}, p = {p_value:.3e}, {significance}")
    
    # Convert results to a DataFrame and save to CSV.
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\nAll t-test results saved to {output_csv}")

if __name__ == "__main__":
    main()