import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_significant_differences_scatter(
    csv_path="explizite_Analyse/statistics/hypothesis_test/overall/overall_model_differences.csv",
    output_path="explizite_Analyse/statistics/hypothesis_test/overall/overall_charts/overall_significant_differences.png"
):
    """
    Reads the CSV file containing pairwise model differences (one row per group-model pair comparison).
    Aggregates across all groups by model pair, then creates a scatter plot of:
      - x-axis: average t-statistic
      - y-axis: average absolute difference in means
      - marker size: total number of comparisons
      - marker color: proportion of significant comparisons
    Each point is annotated with the model pair label.
    """
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Create a standardized model_pair label (so "A vs B" equals "B vs A")
    def model_pair_label(row):
        models = sorted([str(row["Model A"]), str(row["Model B"])])
        return f"{models[0]} vs {models[1]}"
    
    df["model_pair"] = df.apply(model_pair_label, axis=1)
    
    # Compute absolute mean difference for each comparison
    df["mean_diff"] = (df["Mean A"] - df["Mean B"]).abs()
    
    # Group by model pair across all groups
    grouped = df.groupby("model_pair").agg(
        total_comparisons=("p-value", "count"),
        significant_comparisons=("Significance", lambda x: (x.str.lower() == "significant").sum()),
        avg_t_stat=("t-statistic", "mean"),
        avg_mean_diff=("mean_diff", "mean")
    ).reset_index()
    
    # Proportion of significant comparisons
    grouped["proportion_significant"] = grouped["significant_comparisons"] / grouped["total_comparisons"]
    
    print("Aggregated results by model pair (across all groups):")
    print(grouped)
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    
    # Marker size scaled by total comparisons
    sizes = grouped["total_comparisons"] * 40  # Adjust the multiplier for better visibility
    
    scatter = plt.scatter(
        grouped["avg_t_stat"],
        grouped["avg_mean_diff"],
        s=sizes,
        c=grouped["proportion_significant"],
        cmap="viridis",
        alpha=0.7,
        edgecolor="k"
    )
    
    plt.xlabel("Durchschnittlicher t-Wert")
    plt.ylabel("Durchschnittliche absolute Mittelwertdifferenz")
    plt.title("Gesamtübersicht signifikanter Unterschiede zwischen Modellen\n(über alle Gruppen aggregiert)")
    
    # Add colorbar for the proportion of significant comparisons
    cbar = plt.colorbar(scatter)
    cbar.set_label("Proportion Signifikant")
    
    # Annotate each point with the model pair label
    for i, row in grouped.iterrows():
        plt.annotate(
            row["model_pair"],
            (row["avg_t_stat"], row["avg_mean_diff"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            va="bottom",
            fontsize=8
        )
    
    plt.tight_layout()
    
    # Save and close
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Scatter plot saved to {output_path}")

if __name__ == "__main__":
    plot_significant_differences_scatter()