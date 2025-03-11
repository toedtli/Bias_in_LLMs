import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def extract_mean(cell):
    """
    Helper function to extract the numeric mean from a "mean±SEM" string.
    Returns NaN if the cell is "NaN" or if an error occurs.
    """
    try:
        if cell == "NaN" or pd.isnull(cell):
            return np.nan
        return float(cell.split("±")[0])
    except:
        return np.nan

def main():
    # 1. Read in the two CSV files
    df1 = pd.read_csv("explizite_Analyse/statistics/group_axis_statistics_overall_run_1.csv")
    df2 = pd.read_csv("explizite_Analyse/statistics/group_axis_statistics_overall_run_2.csv")

    # Concatenate the two dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    print("Columns in df1:", df1.columns.tolist())
    print("Columns in df2:", df2.columns.tolist())
    print("Columns in concatenated df:", df.columns.tolist())
    
    # Ensure the required 'Axis Name' column exists
    if "Axis Name" not in df.columns:
        raise KeyError("Expected column 'Axis Name' not found in the data!")

    # 2. Create a merged dataframe by grouping by Model, Group, and Axis Name,
    #    and computing the mean for 'mean' and 'SEM'
    merged_df = df.groupby(["Model", "Group", "Axis Name"], as_index=False).agg({"mean": "mean", "SEM": "mean"})

    # Save the merged csv file
    output_dir = "explizite_Analyse/results/combined_results"
    os.makedirs(output_dir, exist_ok=True)
    merged_csv_filename = os.path.join(output_dir, "scoring_cmobined.csv")
    merged_df.to_csv(merged_csv_filename, index=False)
    print(f"Saved merged CSV: {merged_csv_filename}")

    # Get the list of unique axis names from the merged dataframe
    axis_names = merged_df["Axis Name"].unique()
    print("Found Axis Names:", axis_names)

    # Define the order of groups for the final table
    groups = ["Katalanen", "Kurden", "Palästinenser", "Rohingya", "Tibeter", "Uiguren"]

    # Loop over each axis to create per-axis CSVs and heatmaps
    for axis in axis_names:
        print(f"\nProcessing Axis Name: {axis}")
        # Filter the merged dataframe for the current axis
        axis_df = merged_df[merged_df["Axis Name"] == axis]

        # Pivot to wide format so that each group becomes its own column.
        pivoted = axis_df.pivot(index="Model", columns="Group", values=["mean", "SEM"])
        print("Pivoted columns (flattened):", pivoted.columns.to_flat_index().tolist())

        # Reindex the pivoted columns so that groups appear in the desired order
        pivoted = pivoted.reindex(groups, axis=1, level=1)

        # Create a new DataFrame that combines mean and SEM into a single string "mean±SEM"
        final_df = pd.DataFrame(index=pivoted.index)
        for group in groups:
            final_df[group] = pivoted.apply(
                lambda row: f"{row[('mean', group)]:.2f}±{row[('SEM', group)]:.2f}" 
                            if pd.notnull(row[('mean', group)]) and pd.notnull(row[('SEM', group)])
                            else "NaN",
                axis=1
            )

        # Restore the 'Model' column from the index
        final_df.reset_index(inplace=True)
        final_df.columns.name = None  # Clean up column names

        # Save to a new CSV file with the Axis Name in its filename.
        safe_axis = "".join(c if c.isalnum() else "_" for c in str(axis))
        csv_filename = os.path.join(output_dir, f"combined_statistics_{safe_axis}.csv")
        final_df.to_csv(csv_filename, index=False)
        print(f"Saved CSV for Axis Name '{axis}' as: {csv_filename}")

        # ----- Create Heatmap for the current CSV -----
        # Set 'Model' as index for heatmap plotting
        heat_df = final_df.copy()
        if "Model" in heat_df.columns:
            heat_df.set_index("Model", inplace=True)
        
        # Create a numeric DataFrame by extracting the numeric part (mean) from "mean±SEM"
        df_numeric = heat_df.applymap(extract_mean)
        df_text = heat_df.copy()  # For overlaying the text

        # Calculate figure dimensions based on data size
        fig_width = max(4.0, df_numeric.shape[1] * 1.2)
        fig_height = max(1.0, df_numeric.shape[0] * 0.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Choose colormap and vmin based on the axis name (example condition)
        if axis == "Bedrohungswahrnehmung":
            cax = ax.imshow(df_numeric, cmap='Reds', aspect='auto', vmin=25)
        else:
            cax = ax.imshow(df_numeric, cmap='Blues', aspect='auto', vmin=75)
        
        # Add a colorbar
        cb = fig.colorbar(cax, ax=ax, fraction=0.05, pad=0.05)
        cb.set_label('Durchschnittswert', rotation=90)
        
        # Set tick labels for columns and rows
        ax.set_xticks(range(len(df_numeric.columns)))
        ax.set_xticklabels(df_numeric.columns, rotation=0, ha='center', fontsize=8)
        ax.set_yticks(range(len(df_numeric.index)))
        ax.set_yticklabels(df_numeric.index, fontsize=8)
        
        # Title for the heatmap
        ax.set_title(f"{axis} (Durchschnitts-Score ± Stand. Abw. vom Mittelwert)", pad=10)
        
        # Overlay each cell with its "mean±SEM" text
        for i in range(df_numeric.shape[0]):      # for each row (Model)
            for j in range(df_numeric.shape[1]):  # for each column (Group)
                text_val = df_text.iat[i, j]
                ax.text(j, i, text_val,
                        ha='center', va='center', color='black', fontsize=8)
        
        plt.tight_layout()
        
        # Create an output directory for the heatmaps inside the CSV directory
        heatmap_output_dir = os.path.join(output_dir, "heatmaps")
        os.makedirs(heatmap_output_dir, exist_ok=True)
        heatmap_file = os.path.join(heatmap_output_dir, f"final_combined_heatmap_{safe_axis}.png")
        plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap for Axis Name '{axis}' as: {heatmap_file}")

if __name__ == "__main__":
    main()