import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_heatmap(csv_file_path: str, output_path: str = "final_heatmap.png"):
    """
    Reads a CSV file containing columns: 'Scorer Model', 'Source Model', and 'Score',
    then creates a heatmap comparing scorer models (rows) vs source models (columns),
    showing the average score ± standard deviation in each cell.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the input CSV file.
    output_path : str
        Filename for the resulting heatmap image.
    """
    # 1. Load CSV
    df = pd.read_csv(csv_file_path)
    
    # Make sure we have the required columns
    required_cols = {"Scorer Model", "Source Model", "Score"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required_cols}, but has {df.columns.tolist()}")

    # 2. Group by (Scorer Model, Source Model) and compute mean and std
    grouped = df.groupby(["Scorer Model", "Source Model"])["Score"].agg(["mean", "sem"]).reset_index()

    # 3. Pivot to form a matrix (rows = scorer, columns = source)
    #    We'll have two pivoted DataFrames: one for mean, one for std
    mean_df = grouped.pivot(index="Scorer Model", columns="Source Model", values="mean")
    std_df = grouped.pivot(index="Scorer Model", columns="Source Model", values="sem")
    
    # Fill NaNs with 0.0 for clarity (in case some pairs don't exist in the CSV)
    mean_df = mean_df.fillna(0.0)
    std_df = std_df.fillna(0.0)

    # 4. Create annotation DataFrame: "avg ± std"
    annot_df = mean_df.copy()
    for row in annot_df.index:
        for col in annot_df.columns:
            avg_val = mean_df.loc[row, col]
            std_val = std_df.loc[row, col]
            annot_df.loc[row, col] = f"{avg_val:.2f} ± {std_val:.2f}"

    row_avgs = mean_df.mean(axis=1)
    col_avgs = mean_df.mean(axis=0)

    # 5. Plot the heatmap
    plt.figure(figsize=(8, 3))
    # Convert mean_df to numeric for the color scale
    ax = sns.heatmap(mean_df.astype(float), 
                annot=annot_df,      # string annotations with avg ± std
                fmt="",              # no numeric formatting (strings used)
                cmap="Reds",     # or any preferred colormap
                cbar=True)

    plt.title("Durchschnitt aller Bewertungen der Beschreibungen über alle Gruppen", fontsize=14)
    plt.xlabel("Source Model")
    plt.ylabel("Scorer Model")
    # Modify ytick labels to include row averages
    new_y_labels = [f"{label}\navg: {row_avgs[label]:.2f}" for label in mean_df.index]
    # Modify xtick labels to include column averages
    new_x_labels = [f"{label}\navg: {col_avgs[label]:.2f}" for label in mean_df.columns]
    ax.set_yticklabels(new_y_labels, rotation=0, fontsize=8)
    ax.set_xticklabels(new_x_labels, rotation=0, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 1])

    # 6. Save and show
    plt.savefig(output_path, dpi=300)

if __name__ == "__main__":
    # Replace 'path_to_your_data.csv' with your actual CSV file path
    csv_path = "implizite_Analyse/data/scoring_processed/scoring_processed_run_1.csv"
    create_heatmap(csv_path, "implizite_Analyse/scoring_heatmaps/final_heatmap.png")