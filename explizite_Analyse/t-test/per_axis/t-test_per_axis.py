import os
import math
import itertools
import pandas as pd
from scipy.stats import t

def welch_ttest_from_summary(mean1, std1, n1, mean2, std2, n2):
    """
    Computes Welch's t-test (unpaired, unequal variances) from summary statistics:
      mean, std, n for each of the two samples.
    
    Returns:
      (t_stat, p_value) as floats.
      If there's a division-by-zero or invalid scenario, returns (None, None).
    """
    # If either sample size < 2, we cannot compute variance properly
    if n1 < 2 or n2 < 2:
        return None, None

    # If either std is NaN or zero in a way that breaks the formula, handle gracefully
    if pd.isna(std1) or pd.isna(std2) or std1 < 0 or std2 < 0:
        return None, None

    # Welch's t statistic
    numerator = mean1 - mean2
    denom = math.sqrt((std1**2 / n1) + (std2**2 / n2))
    if denom == 0:
        return None, None
    t_stat = numerator / denom

    # Welch–Satterthwaite degrees of freedom
    v1 = (std1**2) / n1
    v2 = (std2**2) / n2
    df_num = (v1 + v2)**2
    df_den = (v1**2)/(n1 - 1) + (v2**2)/(n2 - 1)
    if df_den == 0:
        return t_stat, None
    df = df_num / df_den

    # Two-sided p-value
    # survival function (1 - cdf) of t at abs(t_stat) then double it
    p_value = (1 - t.cdf(abs(t_stat), df)) * 2
    return t_stat, p_value


def test_model_differences_per_group(
    csv_path="explizite_Analyse/statistics/group_axis_statistics_overall.csv",
    alpha=0.05,
    output_dir="explizite_Analyse/statistics/t-test/per_axis"
):
    """
    For each Axis in the CSV, performs pairwise (unpaired) hypothesis tests (Welch's t-test)
    between models *within each group*, using summary stats:
        mean, std, count
    from the CSV.

    The CSV must have these columns at minimum:
      - 'Axis Name'
      - 'Group'
      - 'Model'
      - 'mean'
      - 'std'
      - 'count'

    It writes one CSV file per axis to `output_dir`. The CSV contains rows:
        Group, Model A, Model B, t-statistic, p-value, alpha, Significance, n_A, n_B, ...
    """

    # 1. Read the CSV
    df = pd.read_csv(csv_path)

    # 2. Verify we have the expected columns
    required_cols = {"Model", "Group", "Axis Name", "mean", "SEM", "count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"The CSV is missing required columns: {missing}")

    # 3. Get all unique axes
    unique_axes = sorted(df["Axis Name"].unique())

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 4. For each axis, do pairwise tests *within each group*
    for axis in unique_axes:
        axis_df = df[df["Axis Name"] == axis].copy()

        # Unique groups within this axis
        unique_groups = sorted(axis_df["Group"].unique())

        # We'll store all results for this axis in a list of dicts
        axis_results = []

        for group in unique_groups:
            group_df = axis_df[axis_df["Group"] == group]

            # Identify the distinct models for that group
            models_in_group = sorted(group_df["Model"].unique())
            # If there's only 1 model in this group, no comparisons to do
            if len(models_in_group) < 2:
                continue

            # Check all pairwise model comparisons
            for model_a, model_b in itertools.combinations(models_in_group, 2):
                # Get the row(s) for model A and model B
                row_a = group_df[group_df["Model"] == model_a]
                row_b = group_df[group_df["Model"] == model_b]

                # If multiple rows appear for the same Model, Group, Axis Name, 
                # you'd need to decide how to handle that. 
                # We'll assume there's exactly 1 row per combination:
                if len(row_a) != 1 or len(row_b) != 1:
                    # If there's not exactly one row per (Axis,Group,Model), skip
                    continue

                # Extract summary stats for model A
                mean_a = float(row_a["mean"].values[0])
                std_a  = float(row_a["std"].values[0])
                n_a    = int(row_a["count"].values[0])

                # Extract summary stats for model B
                mean_b = float(row_b["mean"].values[0])
                std_b  = float(row_b["std"].values[0])
                n_b    = int(row_b["count"].values[0])

                # Run Welch's t-test from summary stats
                t_stat, p_val = welch_ttest_from_summary(mean_a, std_a, n_a, mean_b, std_b, n_b)

                # If invalid or insufficient data
                if t_stat is None or p_val is None:
                    # Provide a debug message or skip
                    # print(f"Skipping {model_a} vs. {model_b} in Group '{group}', insufficient data.")
                    continue

                # Determine significance
                significance = "Significant" if p_val < alpha else "Not significant"

                axis_results.append({
                    "Group": group,
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

        # Write out a CSV if we have comparisons for this axis
        if axis_results:
            results_df = pd.DataFrame(axis_results)
            safe_axis_name = str(axis).replace("/", "_").replace("\\", "_")
            out_path = os.path.join(output_dir, f"{safe_axis_name}_test_results.csv")
            results_df.to_csv(out_path, index=False)
            print(f"[Axis: {axis}] Saved {len(axis_results)} comparisons to {out_path}")
        else:
            print(f"[Axis: {axis}] No valid comparisons found.")

if __name__ == "__main__":
    test_model_differences_per_group()