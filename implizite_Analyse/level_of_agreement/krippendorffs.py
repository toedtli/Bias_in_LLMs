#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import krippendorff

def main():
    # Read the CSV file (update the file path if needed)
    df = pd.read_csv('implizite_Analyse/data/scoring_processed/scoring_processed_combined.csv')
    df = df.dropna(subset=['Score'])
    
    # Ensure required columns are present
    required_cols = ['Source Model', 'Scorer Model', 'Score', 'Model Response']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the data.")
    
    # Create a unique text order within each Source Model, Scorer Model group.
    # This assumes that for a given Source Model, all scorers rate the texts in the same order.
    df['text_order'] = df.groupby(['Source Model', 'Scorer Model']).cumcount()
    
    overall_results = []
    pairwise_results = []
    
    # Group by Source Model so that we compare ratings for the same generated texts.
    grouping_cols = ['Source Model']
    for group_key, group_df in df.groupby(grouping_cols):
        source_model = group_key  # since grouping_cols is a single column
        
        # Pivot the data: rows are texts (ordered by 'text_order') and columns are Scorer Models.
        pivot = group_df.pivot_table(
            index='text_order', 
            columns='Scorer Model', 
            values='Score', 
            aggfunc='first'
        )
        data_array = pivot.to_numpy()
        
        # Compute overall (multi-rater) Krippendorff's alpha if there are at least 2 texts and 2 scorers.
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            overall_alpha = np.nan
        else:
            overall_alpha = krippendorff.alpha(reliability_data=data_array, level_of_measurement='ordinal')
        
        overall_results.append({
            'Source Model': source_model,
            'Type': 'Overall',
            'Scorer Model 1': 'All',
            'Scorer Model 2': '',
            'Krippendorff Alpha': overall_alpha,
            'Num_Texts': pivot.shape[0],
            'Num_Scorers': pivot.shape[1]
        })
        
        # Compute pairwise Krippendorff's alpha for each pair of Scorer Models
        scorers = pivot.columns.tolist()
        for i in range(len(scorers)):
            for j in range(i+1, len(scorers)):
                scorer1 = scorers[i]
                scorer2 = scorers[j]
                # Only consider texts where both scorers provided a rating
                sub_df = pivot[[scorer1, scorer2]].dropna()
                if sub_df.shape[0] < 2:
                    pair_alpha = np.nan
                else:
                    sub_array = sub_df.to_numpy()
                    pair_alpha = krippendorff.alpha(reliability_data=sub_array, level_of_measurement='ordinal')
                pairwise_results.append({
                    'Source Model': source_model,
                    'Type': 'Pairwise',
                    'Scorer Model 1': scorer1,
                    'Scorer Model 2': scorer2,
                    'Krippendorff Alpha': pair_alpha,
                    'Num_Texts': sub_df.shape[0]
                })
    
    # Combine overall and pairwise results into a single DataFrame
    results_df = pd.DataFrame(overall_results + pairwise_results)
    
    # Save the results into the specified directory
    output_dir = 'implizite_Analyse/krippendorffs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'krippendorffs_2.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()