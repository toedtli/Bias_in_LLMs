import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

run = 'run_2_3'

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
    

def main():
    # Hardcoded paths for CSV file and output directory.
    csv_file = f"implizite_Analyse/data/scoring_processed/scoring_processed_{run}.csv"
    output_dir = f"implizite_Analyse/results/{run}"
    
    
  
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

    df['Group'] = df['Group'].str.replace(r'Rohingya\s*\(.*\)', 'Rohingya', regex=True)
    mapping = {
        "Kurds": "Kurden",
        "Palestinians": "Palästinenser",
        "Uyghurs": "Uiguren",
        "Catalans": "Katalanen",
        "Tibetans": "Tibeter",
        "Rohingya": "Rohingya"
    }

    # Ersetze die Werte in der 'Group'-Spalte
    df['Group'] = df['Group'].replace(mapping)
    
    # Verify that the CSV contains the necessary columns.
    required_columns = {'Source Model', 'Group', 'Score'}
    if not required_columns.issubset(df.columns):
        print(f"CSV file must contain the following columns: {required_columns}")
        return

    # Ensure that the 'Alternative Scoring' column is numeric.
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

    # Compute descriptive statistics and save to CSV.
    stats_df = compute_descriptive_statistics(df)
    stats_csv_path = os.path.join(output_dir, 'scoring_model_group.csv')
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Descriptive statistics saved to {stats_csv_path}")


if __name__ == '__main__':
    main()