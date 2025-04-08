import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # For seaborn-based heatmaps

def create_axis_heatmaps():
    run = 'run_3'
    # Path to the input CSV file
    input_csv = f"explizite_Analyse/results/results_{run}/results_{run}.csv"
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Check if the Axis column exists
    if 'Axis Name' not in df.columns:
        raise ValueError("The CSV file does not contain an 'Axis Name' column.")
    unique_axes = sorted(df['Axis Name'].unique())
    
    # Ensure the needed columns exist
    for col in ['Model', 'Group', 'mean', 'SEM']:
        if col not in df.columns:
            raise ValueError(f"The CSV file must contain a '{col}' column.")
    
    # Create output directories
    output_dir = f"explizite_Analyse/results/results_{run}"
    output_dir_heatmaps = os.path.join(output_dir, f"heatmaps_{run}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_heatmaps, exist_ok=True)
    
    for axis in unique_axes:
        # Filter rows for the current axis
        axis_df = df[df['Axis Name'] == axis].copy()
        
        # Create a new column 'Mean±SEM' with integer values as text.
        # Convert each value to int and then to string.
        axis_df['Mean±SEM'] = (
            axis_df['mean'].round(0).astype(int).astype(str) + " ± " +
            axis_df['SEM'].round(0).astype(int).astype(str) + "\n n=" + axis_df['count'].round(0).astype(int).astype(str)
        )
        
        # Pivot for the text annotations (Mean±SEM)
        pivot_df = axis_df.pivot(index='Model', columns='Group', values='Mean±SEM')
        
        # Pivot for the numeric heatmap data (mean values)
        pivot_numeric_df = axis_df.pivot(index='Model', columns='Group', values='mean')
        
        pivot_numeric_df_count = axis_df.pivot(index='Model', columns='Group', values='count')
        
        pivot_numeric_df = pivot_numeric_df.round(0).astype(int)
        
        # Calculate column and row averages and standard deviations (all as int)
        col_avgs = pivot_numeric_df.mean(axis=0).round(0).astype(int)
        col_stds = pivot_numeric_df.sem(axis=0).round(0).astype(int)
        col_counts = pivot_numeric_df_count.sum(axis=0).round(0).astype(int)

        row_avgs = pivot_numeric_df.mean(axis=1).round(0).astype(int)
        row_stds = pivot_numeric_df.sem(axis=1).round(0).astype(int)
        row_counts = pivot_numeric_df_count.sum(axis=1).round(0).astype(int)
        
        # Determine figure dimensions so that text fits nicely
        fig_width = max(10, pivot_numeric_df.shape[1] * 1.2)
        fig_height = max(4, pivot_numeric_df.shape[0] * 0.5)
        plt.figure(figsize=(fig_width, fig_height))
        
        # Choose colormap and vmin depending on the axis name
        if axis == "Bedrohungswahrnehmung":
            cmap = 'Reds'
            vmin = 25
        else:
            cmap = 'Blues'
            vmin = 75
        
        # Create the heatmap using seaborn.
        # The numeric data is used to color the cells, while pivot_df provides text annotations.
        ax = sns.heatmap(
            pivot_numeric_df,
            cmap=cmap,
            vmin=vmin,
            annot=pivot_df,
            fmt="",
            linecolor="white",
            # cbar_kws={"shrink": 0.8, "label": "Score"}
            cbar=False
        )
        
        # Update tick labels: include custom labels with column/row averages and std.
        new_xticklabels = [f"{col}\nØ {col_avgs[col]} ± {col_stds[col]} \n n={col_counts[col]}" 
                           for col in pivot_numeric_df.columns]
        new_yticklabels = [f"{model}\nØ {row_avgs[model]} ± {row_stds[model]} \n n={row_counts[model]}" 
                           for model in pivot_numeric_df.index]
        ax.set_xticklabels(new_xticklabels, rotation=0, ha='center', fontsize=8)
        ax.set_yticklabels(new_yticklabels, rotation=0, fontsize=8)
        ax.set_xlabel("Gruppen")
        ax.set_ylabel("Befragte Modelle")
        ax.set_title(f"{axis} (Durchschnitts-Score ± Stand. Abw. vom Mittelwert)", pad=10)
        

        
        plt.tight_layout()
        
        # Save the heatmap image
        heatmap_file = os.path.join(output_dir_heatmaps, f"{axis}_heatmap.png")
        plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap for Axis Name '{axis}' as: {heatmap_file}")

if __name__ == "__main__":
    create_axis_heatmaps()