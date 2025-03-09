import os
import pandas as pd
import math
import itertools
from scipy.stats import t

def welch_ttest_from_summary(mean1, std1, n1, mean2, std2, n2):
    """
    Computes Welch's t-test (unpaired, unequal variances) from summary statistics.
    Returns (t_stat, p_value) for a two-tailed test.
    """
    if n1 < 2 or n2 < 2:
        return None, None

    numerator = mean1 - mean2
    denom = math.sqrt((std1**2 / n1) + (std2**2 / n2))
    if denom == 0:
        return None, None
    t_stat = numerator / denom

    # Welch-Satterthwaite degrees of freedom
    v1 = std1**2 / n1
    v2 = std2**2 / n2
    df_num = (v1 + v2)**2
    df_den = (v1**2)/(n1 - 1) + (v2**2)/(n2 - 1)
    if df_den == 0:
        return t_stat, None
    df = df_num / df_den

    # Two-tailed p-value
    p_value = (1 - t.cdf(abs(t_stat), df)) * 2
    return t_stat, p_value

def create_overall_model_differences_csv(
    csv_path="explizite_Analyse/statistics/group_axis_statistics_overall.csv",
    output_csv="explizite_Analyse/statistics/t-test/per_model/model_differences.csv",
    alpha=0.05
):
    """
    Reads a CSV containing overall average scoring per model.
    
    First, for each row, if the grouping column (either "Group" or "Axis Name")
    indicates bedrohungswahrnehmung (case-insensitive), the mean score is transformed
    by subtracting it from 100. This adjustment is done before any aggregation.
    
    Then the script aggregates the data by group and model:
      - The aggregated mean is the average of the adjusted means ("adj_mean").
      - The aggregated std is approximated as the average of "std" values.
      - The total count is the sum of "count" values.
    
    Finally, for each group, all pairwise comparisons between models are performed using
    Welch's t-test (based on the aggregated summary statistics: adj_mean, std, count).
    The results are saved in one CSV with columns:
      Group, Model A, Model B, Mean A, Mean B, Std A, Std B, n_A, n_B, t-statistic, p-value, alpha, Significance.
    """
    # Read the CSV file.
    df = pd.read_csv(csv_path)
    
    # Determine grouping column: use "Group" if available, otherwise "Axis Name".
    group_col = "Group" if "Group" in df.columns else "Axis Name"
    
    # Check required columns.
    required_cols = {group_col, "Model", "mean", "std", "count"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV file must contain the following columns: {required_cols}")
    
    if "Axis Name" in df.columns:
        df["adj_mean"] = df.apply(
            lambda row: 100 - row["mean"] if str(row["Axis Name"]).strip().lower() == "bedrohungswahrnehmung" else row["mean"],
            axis=1
        )
    else:
        df["adj_mean"] = df["mean"]
        
    # Now aggregate by group and model using the adjusted mean.
    agg_df = df.groupby([group_col, "Model"]).agg({
        "adj_mean": "mean",   # average of adjusted means
        "std": "mean",        # approximate std by averaging
        "count": "sum"        # total count
    }).reset_index()
    
    # List to store all pairwise comparison results.
    results = []
    
    # Process each group.
    groups = agg_df[group_col].unique()
    for grp in groups:
        grp_df = agg_df[agg_df[group_col].str.lower() == str(grp).lower()]
        models = grp_df["Model"].unique()
        if len(models) < 2:
            continue
        # For every pair of models within this group:
        for model_a, model_b in itertools.combinations(models, 2):
            row_a = grp_df[grp_df["Model"] == model_a].iloc[0]
            row_b = grp_df[grp_df["Model"] == model_b].iloc[0]
            
            mean_a = row_a["adj_mean"]
            mean_b = row_b["adj_mean"]
            std_a = float(row_a["std"])
            std_b = float(row_b["std"])
            n_a = int(row_a["count"])
            n_b = int(row_b["count"])
            
            t_stat, p_val = welch_ttest_from_summary(mean_a, std_a, n_a, mean_b, std_b, n_b)
            if t_stat is None or p_val is None:
                continue
            
            significance = "Significant" if p_val < alpha else "Not significant"
            
            results.append({
                "Group": grp,
                "Model A": model_a,
                "Model B": model_b,
                "Mean A": mean_a,
                "Mean B": mean_b,
                "Std A": std_a,
                "Std B": std_b,
                "n_A": n_a,
                "n_B": n_b,
                "t-statistic": round(t_stat, 4),
                "p-value": p_val,
                "alpha": alpha,
                "Significance": significance
            })
    
    # Convert the results to a DataFrame and save to CSV.
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"Overall model differences CSV saved to {output_csv}")

if __name__ == "__main__":
    create_overall_model_differences_csv()