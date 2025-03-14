import pandas as pd
import os

# Define file paths
file_paths = [
    "implizite_Analyse/data/scoring_processed/scoring_processed_run_1.csv",
    "implizite_Analyse/data/scoring_processed/scoring_processed_run_2.csv",
    "implizite_Analyse/data/scoring_processed/scoring_processed_run_3.csv",
]

# Load and concatenate CSV files
dfs = []
for file in file_paths:
    if os.path.exists(file):
        dfs.append(pd.read_csv(file))
    else:
        print(f"Warning: {file} not found.")

# Concatenate all available data
if dfs:
    concatenated_df = pd.concat(dfs, ignore_index=True)
    
    # Save concatenated CSV
    output_path = "implizite_Analyse/data/scoring_processed/scoring_processed_combined.csv"
    concatenated_df.to_csv(output_path, index=False)
    print(f"Concatenated CSV saved to {output_path}")
else:
    print("No files were found. Please check the file paths.")