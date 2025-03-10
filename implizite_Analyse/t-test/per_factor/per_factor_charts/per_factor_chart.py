#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Define file paths.
    input_csv = "implizite_Analyse/t-test/per_factor/all_ttests_results.csv"
    output_dir = "implizite_Analyse/t-test/per_factor/per_factor_charts/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the CSV file.
    df = pd.read_csv(input_csv)
    print("Columns in CSV:", df.columns.tolist())
    
    # Normalize the "Significance" column for easier grouping.
    df["Significance"] = df["Significance"].str.lower()
    
    # Group by Grouping Factor and compute summary statistics.
    grouped = df.groupby("Grouping Factor").agg(
        total_tests=("p-value", "count"),
        significant_tests=("Significance", lambda x: (x == "significant").sum()),
        avg_t_stat=("t-statistic", "mean"),
        avg_p_value=("p-value", "mean")
    ).reset_index()
    grouped["proportion_significant"] = grouped["significant_tests"] / grouped["total_tests"]
    
    print("Grouped summary by Grouping factor:")
    print(grouped)
    
    # Chart 1: Bar Chart of Proportion of Significant Tests by Grouping Factor.
    plt.figure(figsize=(10, 6))
    bars = plt.bar(grouped["Grouping Factor"], grouped["proportion_significant"], color="skyblue")
    plt.xlabel("Grouping Factor", fontsize=12)
    plt.ylabel("Proportion of Significant Tests", fontsize=12)
    plt.title("Proportion of Significant Differences per Grouping Factor", fontsize=14)
    plt.ylim(0, 1)
    
    # Annotate each bar with the proportion and absolute significant test count.
    for bar, prop, sig, total in zip(bars, grouped["proportion_significant"], grouped["significant_tests"], grouped["total_tests"]):
        height = bar.get_height()
        # Format: "0.50 - 3/6 Signifikant"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{prop:.2f} - {sig}/{total} Signifikant",
            ha="center",
            va="bottom",
            fontsize=10
        )
    
    plt.tight_layout()
    output_bar_chart = os.path.join(output_dir, "proportion_significant_by_grouping_factor.png")
    plt.savefig(output_bar_chart, dpi=300)
    plt.close()
    print(f"Bar chart saved to {output_bar_chart}")
    
    # Chart 2: Scatter Plot of Average t-statistic vs. Average p-value per Grouping Factor.
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        grouped["avg_t_stat"],
        grouped["avg_p_value"],
        s=grouped["total_tests"] * 20,  # marker size scaled by total tests
        c=grouped["proportion_significant"],
        cmap="viridis",
        alpha=0.8,
        edgecolor="k"
    )
    plt.xlabel("Average t-statistic", fontsize=12)
    plt.ylabel("Average p-value", fontsize=12)
    plt.title("Comparison: Average t-statistic vs. p-value per Grouping Factor", fontsize=14)
    
    # Add a colorbar to show the proportion of significant tests.
    cbar = plt.colorbar(scatter)
    cbar.set_label("Proportion of Significant Tests", fontsize=12)
    
    # Annotate each point with its Grouping factor label.
    for i, row in grouped.iterrows():
        plt.annotate(row["Grouping Factor"], (row["avg_t_stat"], row["avg_p_value"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=10)
    
    plt.tight_layout()
    output_scatter_chart = os.path.join(output_dir, "avg_t_vs_avg_p_by_grouping_factor.png")
    plt.savefig(output_scatter_chart, dpi=300)
    plt.close()
    print(f"Scatter plot saved to {output_scatter_chart}")

if __name__ == "__main__":
    main()