import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
    
# 1. Read in the two CSV files
df1 = pd.read_csv("implizite_Analyse/results/run_1/descriptive_statistics_run_1.csv")
df2 = pd.read_csv("implizite_Analyse/results/run_2/descriptive_statistics_run_2.csv")

# Concatenate the two dataframes
df = pd.concat([df1, df2], ignore_index=True)
print("Columns in df1:", df1.columns.tolist())
print("Columns in df2:", df2.columns.tolist())
print("Columns in concatenated df:", df.columns.tolist())


# 2. Create a merged dataframe by grouping by Model, Group, and Axis Name,
#    and computing the mean for 'mean' and 'SEM'
merged_df = df.groupby(["Source Model", "Group"], as_index=False).agg({"count": "sum", "mean": "mean", "sem": "mean"})

# Save the merged csv file
output_dir = "implizite_Analyse/results/combined"
os.makedirs(output_dir, exist_ok=True)
merged_csv_filename = os.path.join(output_dir, "combined_model_group.csv")
merged_df.to_csv(merged_csv_filename, index=False)
print(f"Saved merged CSV: {merged_csv_filename}")
