import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_scores(csv_path: str):
    """
    Reads the CSV and aggregates scores by (Model, Group, Axis Name).
    Returns:
      - axis_scores: nested dict with structure: axis_scores[model][group][axis_name] = avg_score
      - axis_names: list of unique axis names
      - models: list of unique model names
      - groups: list of unique group names
    """
    df = pd.read_csv(csv_path)

    # Ensure the necessary columns exist
    required_cols = {"Model", "Group", "Axis Name", "mean"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}. Found: {list(df.columns)}"
        )

    # Identify unique identifiers
    models = sorted(df["Model"].unique())
    groups = sorted(df["Group"].unique())
    axis_names = sorted(df["Axis Name"].unique())

    # Prepare nested data structure: axis_scores[model][group][axis_name] = list_of_scores
    axis_scores = {}
    for model in models:
        axis_scores[model] = {}
        for group in groups:
            axis_scores[model][group] = {}
            for axis_name in axis_names:
                axis_scores[model][group][axis_name] = []

    # Populate the structure with raw scores from the CSV
    for _, row in df.iterrows():
        model = row["Model"]
        group = row["Group"]
        axis_name = row["Axis Name"]
        score_val = row["mean"]
        axis_scores[model][group][axis_name].append(score_val)

    # Convert the lists of scores to their average
    for model in models:
        for group in groups:
            for axis_name in axis_names:
                scores_list = axis_scores[model][group][axis_name]
                if len(scores_list) == 0:
                    axis_scores[model][group][axis_name] = 0.0
                else:
                    axis_scores[model][group][axis_name] = sum(scores_list) / len(scores_list)

    return axis_scores, axis_names, models, groups


def plot_radar_charts_single_figure(axis_scores, axis_names, models, groups, output_folder="Charts"):
    """
    Creates exactly ONE figure containing multiple radar-chart subplots:
      - One subplot per Group
      - All models (e.g. 4) on each subplot, each with its own polygon
      - A single legend indicating which color corresponds to which Model
    Up to 6 groups (2x3 layout). If more than 6 groups exist, the rest are skipped.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    number_of_axes = len(axis_names)

    # Create a single figure with up to 6 subplots (2x3)
    fig, axes = plt.subplots(nrows=2, 
                             ncols=3, 
                             figsize=(5 * 3, 5 * 2), 
                             subplot_kw={"polar": True})
    axes = axes.flatten()

    # Generate angles for the radar chart (one per axis)
    angles = np.linspace(0, 2 * math.pi, number_of_axes, endpoint=False)

    # We will keep track of line objects from the first subplot to build a global legend
    lines_for_legend = []

    for idx, group in enumerate(groups):
        # If there are more than 6 groups, skip the extras
        if idx >= 6:
            print(f"WARNING: More than 6 groups detected. Skipping group: {group}")
            break

        ax = axes[idx]

        # Plot one polygon per model
        for model_idx, model in enumerate(models):
            # Gather average scores for this (model, group)
            scores = [axis_scores[model][group][axis_name] for axis_name in axis_names]

            # Close the radar polygon by repeating the first value at the end
            scores_cycle = scores + [scores[0]]
            angle_cycle = list(angles) + [angles[0]]

            # Plot the line
            (line,) = ax.plot(angle_cycle, scores_cycle, label=model)
            # Fill under the curve with some transparency
            ax.fill(angle_cycle, scores_cycle, alpha=0.25)

            # Collect the line object for legend from the first subplot only
            if idx == 0:
                lines_for_legend.append(line)

        # Radar chart styling
        ax.set_theta_zero_location("N")   # Start from the top
        ax.set_theta_direction(-1)        # Go clockwise
        ax.set_xticks(angles)
        ax.set_xticklabels(axis_names, fontsize=9)
        ax.set_title(f"\n{group}\n", fontsize=14, pad=20)
        ax.set_ylim(bottom=0)  # sets minimum radial axis value to 0

    # Hide any unused subplots if there are fewer than 6 groups
    if len(groups) < 6:
        for unused_idx in range(len(groups), 6):
            axes[unused_idx].set_visible(False)

    # Create one legend for the entire figure
    # Use the handles (lines) and labels from the first subplot
    # We do this outside the loop so it's displayed only once
    handles, labels = lines_for_legend, [line.get_label() for line in lines_for_legend]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=len(models))

    fig.suptitle("Durchschnittliches Scoring über alle Kombinationen", fontsize=16, y=0.96)
    # Adjust to make room for the legend at the bottom
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    output_path = os.path.join(output_folder, "combined_radar_chart.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Radar charts saved to: {output_path}")


def main():
    csv_path = "explizite_Analyse/results/combined_results/scoring_combined.csv"  # Adjust path if needed
    axis_scores, axis_names, models, groups = load_scores(csv_path)
    plot_radar_charts_single_figure(axis_scores, axis_names, models, groups, 
                                    output_folder="explizite_Analyse/radar_charts")


if __name__ == "__main__":
    main()