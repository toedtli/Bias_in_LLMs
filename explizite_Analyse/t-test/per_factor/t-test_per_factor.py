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
    pooled_std = np.sqrt(((n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)) / (n1 + n2 - 2))
    return (m1 - m2) / pooled_std if pooled_std != 0 else 0

def main():
    # Define paths.
    file_path = "explizite_Analyse/data/processed/scoring_processed_run_1.csv"
    results_dir = "explizite_Analyse/t-test/per_factor/"
    charts_dir = os.path.join(results_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    alpha = 0.05
    
    # Load the dataset.
    data = pd.read_csv(file_path)
    print("Columns in dataset:", data.columns.tolist())
    
    # Define dependent variable and factors.
    response_var = "Score"
    # We will use these factors in our tests.
    potential_factors = ["Language", "Choice Set", "Formulation Key", "Question ID", "Model"]
    available_factors = [factor for factor in potential_factors if factor in data.columns]
    print("Available factors:", available_factors)
    
    # We now group by both "Question ID" and "Model".
    if "Question ID" not in available_factors or "Model" not in available_factors:
        print("Error: The dataset must include both 'Question ID' and 'Model' columns.")
        return

    # For analysis, we will test differences across the following factors.
    analysis_factors = ["Language", "Choice Set", "Formulation Key"]
    
    # Lists to store test results.
    ttest_results = []
    anova_results = []
    
    # Outer loops: iterate over each Question ID and Model combination.
    questions = data["Question ID"].dropna().unique()
    for question in questions:
        subset_q = data[data["Question ID"] == question]
        models_in_q = subset_q["Model"].dropna().unique()
        for model in models_in_q:
            print("\n" + "="*50)
            print(f"Analyzing for Question: {question} and Model: {model}")
            subset = subset_q[subset_q["Model"] == model]
            
            # For each factor, perform statistical tests.
            for factor in analysis_factors:
                print("\n" + "-"*40)
                print(f"Analysis for factor: {factor} within Question: {question}, Model: {model}")
                groups = subset[factor].dropna().unique()
                print("Groups found:", groups)
                
                # Print variances for each group.
                group_variances = subset.groupby(factor)[response_var].var()
                print("Variance of", response_var, "by", factor)
                print(group_variances)
                
                # Only perform tests if there are at least two groups.
                if len(groups) >= 2:
                    # If exactly two groups, perform one t-test.
                    if len(groups) == 2:
                        group1 = subset[subset[factor] == groups[0]][response_var]
                        group2 = subset[subset[factor] == groups[1]][response_var]
                        t_stat, p_value = ttest_ind(group1, group2, nan_policy='omit')
                        d_value = cohen_d(group1, group2)
                        g1_avg = group1.mean()
                        g2_avg = group2.mean()
                        g1_count = group1.count()
                        g2_count = group2.count()
                        significance = "significant" if p_value < alpha else "not"
                        ttest_results.append({
                            "Question ID": question,
                            "Model": model,
                            "Factor": factor,
                            "Group1": groups[0],
                            "Group2": groups[1],
                            "Group1 avg": g1_avg,
                            "Group1 count": g1_count,
                            "Group2 avg": g2_avg,
                            "Group2 count": g2_count,
                            "t-statistic": t_stat,
                            "p-value": p_value,
                            "Cohen_d": d_value,
                            "Result": significance
                        })
                        print(f"T-test between {groups[0]} and {groups[1]}: t = {t_stat:.3f}, p = {p_value:.3e}, "
                              f"Cohen's d = {d_value:.3f} -> {significance}")
                        print(f"  {groups[0]}: mean = {g1_avg:.3f}, count = {g1_count}; {groups[1]}: mean = {g2_avg:.3f}, count = {g2_count}")
                    # For more than two groups, perform pairwise t-tests.
                    else:
                        print("Pairwise t-tests:")
                        for (g1, g2) in itertools.combinations(groups, 2):
                            group1 = subset[subset[factor] == g1][response_var]
                            group2 = subset[subset[factor] == g2][response_var]
                            t_stat, p_value = ttest_ind(group1, group2, nan_policy='omit')
                            d_value = cohen_d(group1, group2)
                            g1_avg = group1.mean()
                            g2_avg = group2.mean()
                            g1_count = group1.count()
                            g2_count = group2.count()
                            significance = "significant" if p_value < alpha else "not"
                            ttest_results.append({
                                "Question ID": question,
                                "Model": model,
                                "Factor": factor,
                                "Group1": g1,
                                "Group2": g2,
                                "Group1 avg": g1_avg,
                                "Group1 count": g1_count,
                                "Group2 avg": g2_avg,
                                "Group2 count": g2_count,
                                "t-statistic": t_stat,
                                "p-value": p_value,
                                "Cohen_d": d_value,
                                "Result": significance
                            })
                            print(f"Comparing {g1} vs {g2}: t = {t_stat:.3f}, p = {p_value:.3e}, "
                                  f"Cohen's d = {d_value:.3f} -> {significance}")
                            print(f"  {g1}: mean = {g1_avg:.3f}, count = {g1_count}; {g2}: mean = {g2_avg:.3f}, count = {g2_count}")
                else:
                    print("Not enough groups to perform t-tests or ANOVA.")
    
    # Save t-test results to CSV.
    ttest_results_df = pd.DataFrame(ttest_results)
    ttest_csv_path = os.path.join(results_dir, "ttest_results_by_question_model.csv")
    ttest_results_df.to_csv(ttest_csv_path, index=False)
    print(f"\nT-test results saved to {ttest_csv_path}")
    
    
    # Create combined charts: For each analysis factor, create one grouped bar chart comparing all (Question ID, Model) combinations.
    for factor in analysis_factors:
        print("\nCreating combined chart for factor:", factor)
        # Group by the factor and the two grouping variables.
        stats = data.groupby([factor, "Question ID", "Model"])[response_var].agg(["mean", "std", "count"]).reset_index()
        stats["sem"] = stats["std"] / np.sqrt(stats["count"])
        groups = sorted(stats[factor].unique())
        
        # Create a label for each (Question ID, Model) combination.
        stats["QM"] = stats["Question ID"].astype(str) + " | " + stats["Model"].astype(str)
        qm_combinations = sorted(stats["QM"].unique())
        
        x = np.arange(len(groups))
        width = 0.8 / len(qm_combinations)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, qm in enumerate(qm_combinations):
            qm_data = stats[stats["QM"] == qm]
            # Ensure the order is according to the factor levels.
            means = []
            sems = []
            for group in groups:
                row = qm_data[qm_data[factor] == group]
                if not row.empty:
                    means.append(row["mean"].iloc[0])
                    sems.append(row["sem"].iloc[0])
                else:
                    means.append(np.nan)
                    sems.append(0)
            ax.bar(x + i*width, means, width, yerr=sems, capsize=4, label=qm)
        
        ax.set_xlabel(factor)
        ax.set_ylabel(response_var)
        ax.set_title(f"Comparison of {response_var} by {factor}\nGrouped by Question ID & Model")
        ax.set_xticks(x + width*(len(qm_combinations)-1)/2)
        ax.set_xticklabels(groups, rotation=45)
        ax.legend(title="Question | Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()
        
        chart_filename = f"Combined_{factor}_mean_{response_var}_by_Question_Model.png".replace(" ", "_")
        chart_path = os.path.join(charts_dir, chart_filename)
        plt.savefig(chart_path)
        plt.close()
        print(f"Combined chart saved to {chart_path}")
    
    # Final scientific summary statement.
    sig_ttests = ttest_results_df[ttest_results_df["Result"] == "significant"].shape[0]
    total_ttests = ttest_results_df.shape[0]
    summary_text = (f"Out of {total_ttests} pairwise comparisons across Question and Model combinations, "
                    f"{sig_ttests} showed statistically significant differences (p < {alpha}). "
                    "This indicates that the combination of Question ID and Model can modulate how factors such as "
                    "Language, Choice Set, and Formulation Key affect the Score. Future work should further examine "
                    "these interactions with larger sample sizes and adjust for potential confounders.")
    print("\nScientific Summary:")
    print(summary_text)

if __name__ == "__main__":
    main()