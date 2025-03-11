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

    # Debug: Print columns in df1 and df2
    print("Columns in df1:", df1.columns.tolist())
    print("Columns in df2:", df2.columns.tolist())
    
    # 2. Concatenate them
    df = pd.concat([df1, df2], ignore_index=True)
    print("Columns in combined df:", df.columns.tolist())
    
    # Check for the required 'Axis Name' column
    if "Axis Name" not in df.columns:
        raise KeyError("Expected column 'Axis Name' not found in the data!")

    # Get the list of unique axis names
    axis_names = df["Axis Name"].unique()
    print("Found Axis Names:", axis_names)

    # Define the order of groups for the final table
    groups = ["Katalanen", "Kurden", "Palästinenser", "Rohingya", "Tibeter", "Uiguren"]

    # Ensure the output directory exists for CSV files
    output_dir = "explizite_Analyse/statistics/combined_runs"
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each axis
    for axis in axis_names:
        print(f"\nProcessing Axis Name: {axis}")
        # Filter the DataFrame for the current axis
        axis_df = df[df["Axis Name"] == axis]

        # 3. Group by Model and Group, then compute the average of 'mean' and 'SEM'
        grouped = (
            axis_df.groupby(["Model", "Group"], as_index=False)
                   .agg({"mean": "mean", "SEM": "mean"})
        )
        print("Grouped columns:", grouped.columns.tolist())

        # 4. Pivot to wide format so that each group becomes its own column.
        pivoted = grouped.pivot(index="Model", columns="Group")
        print("Pivoted columns (flattened):", pivoted.columns.to_flat_index().tolist())

        # Reindex the pivoted columns so that groups appear in the desired order
        pivoted = pivoted.reindex(groups, axis=1, level=1)

        # 5. Create a new DataFrame that combines mean and SEM into a single string "mean±SEM"
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

        # 6. Save to a new CSV file with the Axis Name in its filename.
        safe_axis = "".join(c if c.isalnum() else "_" for c in str(axis))
        csv_filename = os.path.join(output_dir, f"final_group_axis_statistics_{safe_axis}.csv")
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
        heatmap_file = os.path.join(heatmap_output_dir, f"final_heatmap_{safe_axis}.png")
        plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap for Axis Name '{axis}' as: {heatmap_file}")

if __name__ == "__main__":
    main()