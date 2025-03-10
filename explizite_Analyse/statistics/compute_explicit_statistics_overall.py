import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")

# Define the output directory and ensure it exists
output_dir = "explizite_Analyse/statistics/"
os.makedirs(output_dir, exist_ok=True)

# Load the CSV data
df = pd.read_csv("explizite_Analyse/data/processed/scoring_processed_run_2.csv")


def compute_statistics(csv_path, 
                       output_stats_csv="explizite_Analyse/statistics/group_axis_statistics_overall.csv"):
    """
    Reads the scoring CSV and computes summary statistics (count, mean, std, median, min, max, 25th percentile,
    75th percentile, IQR, and outlier count) for each combination of Model, Language, Group, and Axis Name.
    Also performs a Kruskal–Wallis test (non-parametric ANOVA) to compare scores across groups for each Model, Language,
    and Axis combination. The results are saved to two CSV files.
    """
    df = pd.read_csv(csv_path)
    # Mapping von englischen zu deutschen Gruppennamen
    df['Group'] = df['Group'].str.replace(r'Rohingya\s*\(.*\)', 'Rohingya', regex=True)

    mapping = {
        "Kurds": "Kurden",
        "Palestinians": "Palästinenser",
        "Uyghurs": "Uiguren",
        "Catalans": "Katalanen",
        "Tibetans": "Tibeter",
        "Rohingya": "Rohingya"  # bleibt gleich
    }

    # Ersetze die Werte in der 'Group'-Spalte
    df['Group'] = df['Group'].replace(mapping)
    # Group by Model, Language, Group, and Axis Name
    grouped = df.groupby(['Model', 'Group', 'Axis Name'])
    
    # Compute basic statistics
    stats_df = grouped['Score'].agg(['count', 'std', 'mean', 'median', 'min', 'max']).reset_index()

    # Compute standard deviation of the average (Standard Error of the Mean, SEM)
    stats_df['SEM'] = grouped['Score'].std().values / np.sqrt(stats_df['count'])

    # Compute percentiles and IQR
    stats_df['25%'] = grouped['Score'].quantile(0.25).values
    stats_df['75%'] = grouped['Score'].quantile(0.75).values
    stats_df['IQR'] = stats_df['75%'] - stats_df['25%']

    # Define a helper to count outliers
    def outlier_count(x):
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return ((x < lower_bound) | (x > upper_bound)).sum()

    # Compute outlier count per group
    stats_df['outliers'] = grouped['Score'].apply(outlier_count).values
    
    # Ensure output directories exist for CSVs
    os.makedirs(os.path.dirname(output_stats_csv), exist_ok=True)
    
    # Save statistics CSV
    stats_df.to_csv(output_stats_csv, index=False)
    print(f"Statistics CSV saved to {output_stats_csv}")

    return stats_df, df

def main():
    # Adjust the path to your CSV file if needed.
    csv_path = "explizite_Analyse/data/processed/scoring_processed_run_2.csv"
    stats_df, df = compute_statistics(csv_path)

if __name__ == "__main__":
    main()