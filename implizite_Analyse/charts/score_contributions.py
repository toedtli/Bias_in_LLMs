#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # File path for the descriptive statistics CSV.
    csv_file = "implizite_Analyse/statistics/descriptive_statistics.csv"
    # Create an output directory for charts.
    output_dir = "implizite_Analyse/charts"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file and trim any whitespace from column names.
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    
    # Print out the columns and first few rows for verification.
    print("Columns found:", df.columns.tolist())
    print("First few rows of the data:")
    print(df.head())
    
    # Check that the essential columns exist.
    required_columns = ["Source Model", "Group"]
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV.")
            return

    # Determine if we have descriptive statistic columns.
    has_mean = "mean" in df.columns
    has_std = "sem" in df.columns
    has_count = "count" in df.columns



    # ---------------------------------------------
    # Visualization 1: Grouped Bar Chart for Mean Values
    # ---------------------------------------------
    if has_mean:
        # Pivot the data: rows = Groups, columns = Source Models, values = Mean.
        pivot_mean = df.pivot(index="Group", columns="Source Model", values="mean")
        
        # If Std is available, create a pivot for it (to use as error bars).
        if has_std:
            pivot_std = df.pivot(index="Group", columns="Source Model", values="sem")
        else:
            pivot_std = None
        
        # Plot the grouped bar chart.
        ax = pivot_mean.plot(kind='bar', figsize=(10, 6), yerr=pivot_std, capsize=4)
        ax.set_xlabel("Groups", fontsize=12)
        ax.set_ylabel("Mean Value", fontsize=12)
        ax.set_title("Mean Value by Groups and Source Models", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, "grouped_bar_mean_by_group_and_source_model.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved chart: {output_path}")
    else:
        print("Column 'Mean' not found in CSV. Skipping the grouped bar chart for mean values.")


if __name__ == "__main__":
    main()