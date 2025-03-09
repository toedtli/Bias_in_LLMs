#!/usr/bin/env python3
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths
csv_path = "implizite_Analyse/data/scoring_processed/scoring_processed_run_1.csv"
output_dir = "implizite_Analyse/scoring_heatmaps"
os.makedirs(output_dir, exist_ok=True)

# Set a scientific plot style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 8)})

# Load the CSV data
df = pd.read_csv(csv_path)

# Ensure the required columns exist
required_columns = {"Scorer Model", "Source Model", "Score", "Group", "Language"}
if not required_columns.issubset(df.columns):
    raise ValueError("The CSV file must contain the following columns: " + ", ".join(required_columns))

# Check for duplicate entries (for the combination of scorer, source, group, and language)
dup_count = df.duplicated(subset=["Scorer Model", "Source Model", "Group", "Language"]).sum()
if dup_count > 0:
    print(f"Warning: Found {dup_count} duplicate entries. They will be aggregated using the mean.")

# Iterate over each language
languages = df["Language"].unique()
for lang in languages:
    # For each language, retrieve unique groups
    groups_in_lang = df[df["Language"] == lang]["Group"].unique()
    for group in groups_in_lang:
        # Filter the DataFrame for the current language and group
        subset = df[(df["Language"] == lang) & (df["Group"] == group)]
        if subset.empty:
            continue
        
        # Group by Scorer Model and Source Model; average over the choice sets
        agg_scores = subset.groupby(["Scorer Model", "Source Model"])["Score"].mean().reset_index()
        
        # Pivot the table so that rows are Scorer Models and columns are Source Models
        pivot = agg_scores.pivot(index="Scorer Model", columns="Source Model", values="Score")
        
        # Plot the heatmap
        plt.figure()
        sns.heatmap(pivot, annot=True, cmap="Reds", fmt=".2f", cbar_kws={'label': 'Mean Score'}, vmin=0)
        plt.title(f"Evaluation for {group} in {lang}")
        plt.ylabel("Scorer Model")
        plt.xlabel("Source Model")
        
        # Save the heatmap to the output directory
        heatmap_filename = f"heatmap_{lang}_{group}.png"
        heatmap_path = os.path.join(output_dir, heatmap_filename)
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()

print("Heatmaps generated and saved in:", output_dir)