import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_descriptive_statistics(df):
    """
    Compute descriptive statistics by grouping the DataFrame by 'Source Model' and 'Group'.
    """
    summary = df.groupby(['Source Model', 'Group'])['Score'].agg(
        count='count', 
        mean='mean', 
        std='std', 
        sem='sem', 
        median='median', 
        min='min', 
        max='max'
    ).reset_index()
    return summary

def plot_bar(df, output_dir):
    """
    Generate a stacked bar plot showing the score contribution (mean score) of each group for each model.
    """
    plt.figure(figsize=(12, 8))
    # Aggregate using mean, then unstack to create a DataFrame with groups as columns.
    grouped_data = df.groupby(['Source Model', 'Group'])['Score'].sum().unstack()
    grouped_data.plot(kind='bar', figsize=(12, 8))
    plt.title('Score Contribution by Group for Each Model')
    plt.xlabel('Source Model')
    plt.xticks(rotation=0)
    plt.ylabel('Average Score')
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    file_path = os.path.join(output_dir, 'score_contribution_by_group.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    print(f"Saved stacked bar plot to {file_path}")


def main():
    # Hardcoded paths for CSV file and output directory.
    csv_file = "implizite_Analyse/data/scoring_processed_run_1.csv"
    output_dir = "implizite_Analyse/statistics"
    
  
    # Read CSV file.
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at {csv_file} is empty.")
        return
    except pd.errors.ParserError:
        print(f"Error: Could not parse CSV file at {csv_file}. Check file format.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV: {e}")
        return

    # Verify that the CSV contains the necessary columns.
    required_columns = {'Source Model', 'Group', 'Score'}
    if not required_columns.issubset(df.columns):
        print(f"CSV file must contain the following columns: {required_columns}")
        return

    # Ensure that the 'Score' column is numeric.
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

    # Compute descriptive statistics and save to CSV.
    stats_df = compute_descriptive_statistics(df)
    stats_csv_path = os.path.join(output_dir, 'descriptive_statistics.csv')
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Descriptive statistics saved to {stats_csv_path}")

    # Generate individual visualizations.
    plot_bar(df, output_dir)


if __name__ == '__main__':
    main()