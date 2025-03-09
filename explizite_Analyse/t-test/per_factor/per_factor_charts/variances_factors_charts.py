#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway
import itertools

def cohen_d(group1, group2):
    """Compute Cohen's d for two groups."""
    m1, m2 = group1.mean(), group2.mean()
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    n1, n2 = group1.count(), group2.count()
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)) / (n1 + n2 - 2))
    return (m1 - m2) / pooled_std if pooled_std != 0 else 0

def main():
    # Define paths.
    file_path = "explizite_Analyse/data/processed/scoring_processed_run_1.csv"
    results_dir = "explizite_Analyse/t-test/per_factor"
    charts_dir = os.path.join(results_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    # Load the dataset.
    data = pd.read_csv(file_path)
    print("Columns in dataset:", data.columns.tolist())
    
    # Define the dependent variable and factors.
    response_var = "Score"  
    potential_factors = ["Language", "Choice Set", "Formulation Key", "Question ID", "Model"]
    
    # Use only factors present in the data.
    available_factors = [factor for factor in potential_factors if factor in data.columns]
    print("Available factors:", available_factors)
    
    # We want to always group by Model.
    if "Model" not in available_factors:
        print("Error: The dataset must include a 'Model' column.")
        return

    # For analysis, consider all factors except Model (which we use for grouping).
    analysis_factors = [factor for factor in available_factors if factor != "Model"]
    
    # Create combined charts: For each analysis factor, compare all models in one grouped bar chart.
    for factor in analysis_factors:
        print("\nCreating combined chart for factor:", factor)
        # Group by factor and Model, computing mean, std and count.
        stats = data.groupby([factor, "Model"])[response_var].agg(["mean", "std", "count"]).reset_index()
        stats["sem"] = stats["std"] / np.sqrt(stats["count"])
        groups = sorted(stats[factor].unique(), key=lambda x: (0, int(x.replace("question", ""))) if x.startswith("question") else (1, x))
        models_sorted = sorted(stats["Model"].unique())
        
        x = np.arange(len(groups))
        width = 0.8 / len(models_sorted)  # width of each bar
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, model in enumerate(models_sorted):
            model_data = stats[stats["Model"] == model]
            # Ensure the values are ordered by group.
            means = []
            sems = []
            for group in groups:
                row = model_data[model_data[factor] == group]
                if not row.empty:
                    means.append(row["mean"].iloc[0])
                    sems.append(row["sem"].iloc[0])
                else:
                    means.append(np.nan)
                    sems.append(0)
            ax.bar(x + i * width, means, width, yerr=sems, capsize=4, label=model)
        
        ax.set_xlabel(factor)
        ax.set_ylabel(response_var)
        ax.set_title(f"Comparison of {response_var} by {factor} Across Models")
        ax.set_xticks(x + width*(len(models_sorted)-1)/2)
        ax.set_xticklabels(groups, rotation=45)
        ax.legend(title="Model")
        fig.tight_layout()
        
        chart_filename = f"Combined_{factor}_mean_{response_var}.png".replace(" ", "_")
        chart_path = os.path.join(charts_dir, chart_filename)
        plt.savefig(chart_path)
        plt.close()
        print(f"Combined chart saved to {chart_path}")


if __name__ == "__main__":
    main()