import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_significance_by_group(
    csv_path="explizite_Analyse/statistics/hypothesis_test/overall/overall_model_differences.csv",
    output_path="explizite_Analyse/statistics/hypothesis_test/overall/overall_charts/overall_significant_differences_barchart.png"
):
    """
    Reads the CSV file (overall_model_differences.csv) which has one row per
    group-model pair comparison. For each group, computes:
      - total number of comparisons
      - number of significant comparisons
      - proportion of significant comparisons
    Then creates a bar chart showing the proportion of significant comparisons per group.
    """
    df = pd.read_csv(csv_path)
    
    # Check if "Group" column exists
    if "Group" not in df.columns:
        print("No 'Group' column found in the CSV. Cannot plot significance by group.")
        return
    
    # Group by 'Group' and compute stats
    group_stats = df.groupby("Group").agg(
        total_comparisons=("p-value", "count"),
        significant_comparisons=("Significance", lambda x: (x.str.lower() == "significant").sum())
    ).reset_index()
    group_stats["proportion_significant"] = group_stats["significant_comparisons"] / group_stats["total_comparisons"]
    
    print("Significance stats by group:")
    print(group_stats)
    
    # Sort groups by proportion_significant descending for a nice chart
    group_stats = group_stats.sort_values("proportion_significant", ascending=False)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(group_stats["Group"], group_stats["proportion_significant"], color="skyblue")
    plt.xlabel("Gruppe")
    plt.ylabel("Anteil signifikanter Vergleiche")
    plt.title("Proportion signifikanter Unterschiede pro Gruppe")
    plt.ylim(0, 1)
    
    # Annotate bars with the proportion
    for bar, proportion in zip(bars, group_stats["proportion_significant"]):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{proportion:.2f}",
            ha="center",
            va="bottom"
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Bar chart saved to {output_path}")

if __name__ == "__main__":
    plot_significance_by_group()