import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_mean_diff_by_group_model_pair(
    csv_path="explizite_Analyse/statistics/hypothesis_test/overall/overall_model_differences.csv",
    output_path="explizite_Analyse/statistics/hypothesis_test/overall/overall_charts/overall_mean_diff_barchart.png"
):
    """
    Reads overall_model_differences.csv, which has one row per group-model pair comparison.
    Creates a grouped bar chart showing the average absolute mean difference (|Mean A - Mean B|)
    for each model pair across all groups.

    - X-axis: each Group
    - Each group has multiple bars, one for each model pair
    - Bar height: average absolute mean difference for that group-model_pair
    """
    df = pd.read_csv(csv_path)

    # Check if 'Group' column exists
    if "Group" not in df.columns:
        print("No 'Group' column found in the CSV. Cannot plot group-based bar chart.")
        return

    # Create a standardized 'model_pair' label (e.g. "gpt vs gemini")
    def model_pair_label(row):
        models = sorted([str(row["Model A"]), str(row["Model B"])])
        return f"{models[0]} vs {models[1]}"
    
    df["model_pair"] = df.apply(model_pair_label, axis=1)

    # Compute absolute mean difference for each row
    df["mean_diff"] = (df["Mean A"] - df["Mean B"]).abs()

    # Group by [Group, model_pair] and take the average of mean_diff
    grouped = df.groupby(["Group", "model_pair"])["mean_diff"].mean().reset_index()

    # Pivot so that each row is a Group and each column is a model_pair
    pivot_df = grouped.pivot(index="Group", columns="model_pair", values="mean_diff").fillna(0)

    # Create a grouped bar chart
    plt.figure(figsize=(12, 6))
    ax = pivot_df.plot(kind="bar", figsize=(12,6), colormap="viridis", edgecolor="black")

    plt.xlabel("Gruppe")
    plt.ylabel("Durchschnittliche absolute Mittelwertdifferenz")
    plt.title("Durchschnittliche absolute Mittelwertdifferenz pro Gruppe und Modellpaar")
    plt.legend(title="Modellpaar", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Grouped bar chart saved to {output_path}")

if __name__ == "__main__":
    plot_mean_diff_by_group_model_pair()