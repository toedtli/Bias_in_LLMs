#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load the t-test results CSV file.
    file_path = "explizite_Analyse/variance_factors/ttest_results_by_question_model.csv"
    df = pd.read_csv(file_path)
    
    # Inspect the columns and a few rows.
    print("Data columns:", df.columns.tolist())
    print("First few rows:\n", df.head())
    
    # Group the results by Factor and calculate summary statistics.
    summary = df.groupby("Factor").agg(
        total_comparisons=("Result", "count"),
        significant_comparisons=("Result", lambda x: (x == "significant").sum()),
        avg_t_stat=("t-statistic", "mean"),
        avg_cohen_d=("Cohen_d", "mean")
    ).reset_index()
    
    summary["proportion_significant"] = summary["significant_comparisons"] / summary["total_comparisons"]
    
    print("\nSummary statistics by Factor:")
    print(summary)
    
    # Create subdirectory for saving CSV and plots if it doesn't exist.
    subanalysis_dir = "explizite_Analyse/variance_factors/subanalysis_variance_factors"
    os.makedirs(subanalysis_dir, exist_ok=True)
    
    # Save the summary statistics as a CSV.
    summary_csv_path = os.path.join(subanalysis_dir, "summary_statistics_by_factor.csv")
    summary.to_csv(summary_csv_path, index=False)
    print(f"Summary statistics saved to {summary_csv_path}")
    
    # Combined plot: Stacked bar chart for total vs. significant comparisons.
    # Compute the non-significant comparisons.
    summary["non_significant"] = summary["total_comparisons"] - summary["significant_comparisons"]
    
    # Set up the plot.
    fig, ax = plt.subplots(figsize=(10, 6))
    ind = np.arange(len(summary))
    width = 0.6
    
    # Plot non-significant comparisons (bottom segment).
    p1 = ax.bar(ind, summary["non_significant"], width, color="lightgray", label="Nicht Signifikant")
    # Plot significant comparisons (top segment).
    p2 = ax.bar(ind, summary["significant_comparisons"], width, bottom=summary["non_significant"], color="salmon", label="Signifikant")
    
    # Set the x-axis.
    ax.set_xticks(ind)
    ax.set_xticklabels(summary["Factor"], rotation=45)
    ax.set_ylabel("Anzahl")
    ax.set_title("Aggregierter Vergleich signifikanter Unterschiede nach Faktor\n(Über alle Frage-Modell-Kombinationen)")
    ax.legend()
    
    # Annotate each bar with the proportion and additional stats.
    for i, row in summary.iterrows():
        total = row["total_comparisons"]
        sig = row["significant_comparisons"]
        prop = row["proportion_significant"]
        avg_t = row["avg_t_stat"]
        avg_d = row["avg_cohen_d"]
        # Place the annotation in the center of the significant segment.
        y_pos = row["non_significant"] + row["significant_comparisons"] / 2
        annotation = f"{sig}/{total} ({prop:.0%})"
        ax.text(ind[i], y_pos, annotation, ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    combined_plot_path = os.path.join(subanalysis_dir, "combined_significance_by_factor.png")
    plt.savefig(combined_plot_path)
    plt.show()
    print(f"Combined plot saved to {combined_plot_path}")
    
    # Scientific summary statement.
    print("\nScientific Summary:")
    for _, row in summary.iterrows():
        print(f"Factor '{row['Factor']}': {row['significant_comparisons']} out of {row['total_comparisons']} comparisons "
              f"({row['proportion_significant']:.2%}) were significant. "
              f"Average t-statistic: {row['avg_t_stat']:.3f}, Average Cohen's d: {row['avg_cohen_d']:.3f}.")
    
    print("\nOverall, factors with a higher proportion of significant comparisons indicate more pronounced differences between their levels. "
          "This suggests that these factors (e.g., Language, Choice Set, or Formulation Key) may have a meaningful impact on the Score. "
          "Further research with larger samples and controlled designs is recommended to validate these findings and explore potential interactions with Model characteristics.")

if __name__ == "__main__":
    main()