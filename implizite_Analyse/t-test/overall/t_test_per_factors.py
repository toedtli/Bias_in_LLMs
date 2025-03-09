#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import itertools

def main():
    # Define file paths.
    input_csv = "implizite_Analyse/data/scoring_processed/scoring_processed_run_1.csv"
    output_csv = "implizite_Analyse/t-test/overall/overall_significance_results.csv"
    alpha = 0.05  # significance level

    # Load the CSV file.
    df = pd.read_csv(input_csv)
    print("Columns in dataset:", df.columns.tolist())

    # Define the dependent variable and the factors to test.
    response_var = "Score"
    potential_factors = ["Language", "Choice Set", "Formulation Key", "Scorerer Model", "Source Model"]
    # Use only those factors that are present in the CSV.
    factors = [f for f in potential_factors if f in df.columns]
    print("Testing differences for factors:", factors)

    # List to store t-test results.
    results = []

    # Loop over each factor.
    for factor in factors:
        groups = df[factor].dropna().unique()
        if len(groups) < 2:
            print(f"Not enough groups to compare for factor {factor}.")
            continue

        # For each pair of groups within the factor, perform a Welch t-test.
        for g1, g2 in itertools.combinations(groups, 2):
            data1 = df[df[factor] == g1][response_var]
            data2 = df[df[factor] == g2][response_var]
            t_stat, p_value = ttest_ind(data1, data2, nan_policy='omit')
            significance = "Significant" if p_value < alpha else "Not significant"
            results.append({
                "Factor": factor,
                "Group1": g1,
                "Group2": g2,
                "t-statistic": t_stat,
                "p-value": p_value,
                "Result": significance
            })
            print(f"Factor {factor}: {g1} vs {g2} -> t = {t_stat:.3f}, p = {p_value:.3e}, {significance}")

    # Convert results to a DataFrame and save to CSV.
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"All t-test results saved to {output_csv}")

if __name__ == "__main__":
    main()