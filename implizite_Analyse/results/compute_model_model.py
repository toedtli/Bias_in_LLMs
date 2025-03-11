#!/usr/bin/env python3
import pandas as pd

def main():
    # Define the input and output file paths.
    input_file = 'implizite_Analyse/data/scoring_processed/scoring_processed_run_1.csv'
    output_file = 'implizite_Analyse/results/run_1/scoring_run_1.csv'
    
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(input_file)
    
    # Group by the scorer and source models, computing the mean and SEM of the "Score" column.
    grouped_df = df.groupby(['Scorer Model', 'Source Model'], as_index=False).agg(
        Score=('Score', 'mean'),
        SEM=('Score', 'sem'),
        Count=('Score', 'count')
    )
    
    # Rename the "Scorer Model" column to "Score Model" to match the requested output.
    grouped_df.rename(columns={'Scorer Model': 'Scorer Model'}, inplace=True)
    
    # Save the resulting DataFrame to a new CSV file.
    grouped_df.to_csv(output_file, index=False)
    print(f"Grouped CSV file saved as: {output_file}")

if __name__ == "__main__":
    main()