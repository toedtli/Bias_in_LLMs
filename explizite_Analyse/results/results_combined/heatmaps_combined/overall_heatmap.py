import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_heatmap(input_csv, output_dir):
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Example: If needed, subtract Bedrohungswahrnehmung from 100
    df.loc[df['Axis Name'] == 'Bedrohungswahrnehmung', 'mean'] = 100 - df['mean']

    # Group data by Model and Group, averaging across all Axis Names
    grouped = df.groupby(['Model', 'Group'], as_index=False).agg({'mean': 'mean', 'SEM': 'mean', 'count': 'sum'})

    # Pivot the data so that 'Model' becomes rows and 'Group' becomes columns for the mean
    heatmap_mean = grouped.pivot(index='Model', columns='Group', values='mean').round(0).astype(int)

    # Pivot the data for the SEM
    heatmap_sem = grouped.pivot(index='Model', columns='Group', values='SEM').round(0).astype(int)
    heatmap_count = grouped.pivot(index='Model', columns='Group', values='count').round(0).astype(int)

    # Create a text DataFrame for displaying "mean±SEM" in each cell
    heatmap_text = heatmap_mean.copy()
    for idx in heatmap_text.index:
        for col in heatmap_text.columns:
            m_val = heatmap_mean.loc[idx, col]
            s_val = heatmap_sem.loc[idx, col]
            c_val = heatmap_count.loc[idx, col]
            heatmap_text.loc[idx, col] = f"{m_val} ± {s_val} \n n={c_val}"

    # --- Compute overall mean±SEM per Model (row) ---
    model_stats = grouped.groupby('Model', as_index=True).agg({'mean': 'mean', 'SEM': 'mean', 'count': 'sum'})

    # --- Compute overall mean±SEM per Group (column) ---
    group_stats = grouped.groupby('Group', as_index=True).agg({'mean': 'mean', 'SEM': 'mean', 'count': 'sum'})

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "heatmap_combined.png")

    # Create a figure with a wide-but-short aspect
    plt.figure(figsize=(10, 4))

    # Plot the numeric means, but use the "mean±SEM" strings as the annotations
    ax = sns.heatmap(
        heatmap_mean,
        cmap="Blues",
        annot=heatmap_text,
        fmt="",        # We already have our own string format
        vmin=75,
        cbar=False
    )

    # ----- Build custom row labels (Models) with overall mean±SEM -----
    row_labels = []
    for model in heatmap_mean.index:
        # If you want more decimal places, adjust the .1f below
        m_avg = model_stats.loc[model, 'mean'].round(0).astype(int)
        s_avg = model_stats.loc[model, 'SEM'].round(0).astype(int)
        c_sum = model_stats.loc[model, 'count'].round(0).astype(int)
        row_label = f"{model}\n{m_avg} ± {s_avg} \n n={c_sum}"
        row_labels.append(row_label)

    # ----- Build custom column labels (Groups) with overall mean±SEM -----
    col_labels = []
    for group in heatmap_mean.columns:
        m_avg = group_stats.loc[group, 'mean'].round(0).astype(int)
        s_avg = group_stats.loc[group, 'SEM'].round(0).astype(int)
        c_sum = group_stats.loc[group, 'count'].round(0).astype(int)
        col_label = f"{group}\n{m_avg} ± {s_avg} \n n={c_sum}"
        col_labels.append(col_label)

    # Apply the new row and column labels
    ax.set_yticklabels(row_labels, rotation=0, fontsize=8)  # Keep them horizontal
    ax.set_xticklabels(col_labels, rotation=0, fontsize=8)  # Keep them horizontal

    # Add a German title
    plt.title("Durchschnittlicher Zustimmungs-Score über alle Dimensionen pro Gruppe \n (Zustimmungs-Score ± Std. Abw. vom Mittelwert, n=Beantwortete Fragenkombinationen)", pad=15)

    # Add axis labels in German
    plt.xlabel("Gruppe")
    plt.ylabel("Modelle")


    # Make room at the bottom for the caption
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved to {output_path}")

if __name__ == "__main__":
    input_csv = "explizite_Analyse/results/results_combined/scoring_combined.csv"  # Update if needed
    output_dir = "explizite_Analyse/results/results_combined/heatmaps_combined"
    generate_heatmap(input_csv, output_dir)