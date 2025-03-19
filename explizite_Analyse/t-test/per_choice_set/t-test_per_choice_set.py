#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Define input and output paths
input_csv = 'explizite_Analyse/data/processed/scoring_processed_combined.csv'
output_dir = 'explizite_Analyse/t-test/per_choice_set'
os.makedirs(output_dir, exist_ok=True)

# Load the CSV file
df = pd.read_csv(input_csv)

# Check that necessary columns exist (adjust these if your CSV has different names)
required_columns = ['Question ID', 'Formulation Key', 'Language', 'Group', 'Model', 'Choice Set', 'Score']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in CSV. Please adjust the column names accordingly.")

# This list will collect all t-test results
results = []

# Group the data by the factors of interest (including Language so that comparisons are only within the same language)
grouped = df.groupby(['Question ID', 'Formulation Key', 'Language', 'Group', 'Model'])

for group_keys, group_df in grouped:
    qid, formulation, language, group_val, model = group_keys

    # Get unique choice sets within the group
    choice_sets = group_df['Choice Set'].unique()

    # Only proceed if there are at least two choice sets to compare
    if len(choice_sets) < 2:
        continue

    # Perform pairwise comparisons among all choice sets in the same group
    for i in range(len(choice_sets)):
        for j in range(i + 1, len(choice_sets)):
            cs1 = choice_sets[i]
            cs2 = choice_sets[j]

            # Select score data for each choice set and drop missing values
            data1 = group_df[group_df['Choice Set'] == cs1]['Score'].dropna()
            data2 = group_df[group_df['Choice Set'] == cs2]['Score'].dropna()

            # Only proceed if both groups have data
            if len(data1) == 0 or len(data2) == 0:
                continue

            # Perform Welch's t-test (does not assume equal variances)
            t_stat, p_value = ttest_ind(data1, data2, equal_var=False)

            # Calculate basic statistics
            mean1 = data1.mean()
            mean2 = data2.mean()
            n1 = len(data1)
            n2 = len(data2)

            # Save the t-test result into the results list
            results.append({
                'question_id': qid,
                'formulation': formulation,
                'language': language,
                'group': group_val,
                'model': model,
                'choice1': cs1,
                'choice2': cs2,
                'n1': n1,
                'n2': n2,
                'mean1': mean1,
                'mean2': mean2,
                't_stat': t_stat,
                'p_value': p_value
            })

# Convert the results to a DataFrame and save to one CSV file
results_df = pd.DataFrame(results)
results_csv = os.path.join(output_dir, 't_test_results.csv')
results_df.to_csv(results_csv, index=False)

# Create one bar chart showing the proportion of significant tests over all t-tests
total_tests = len(results_df)
if total_tests > 0:
    # Define significance threshold (e.g., p < 0.05)
    sig_count = results_df[results_df['p_value'] < 0.05].shape[0]
    non_sig_count = total_tests - sig_count

    # Calculate percentages
    proportions = [(sig_count / total_tests) * 100, (non_sig_count / total_tests) * 100]
    categories = ['Significant', 'Not Significant']

    # Create the bar chart
    fig, ax = plt.subplots()
    ax.bar(categories, proportions, capsize=5)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Proportion of Significant T-Tests')
    
    # Save the chart as a PNG file
    chart_filepath = os.path.join(output_dir, 'significance_proportion.png')
    plt.savefig(chart_filepath, bbox_inches='tight')
    plt.close()
else:
    print("No t-test results available to create a bar chart.")

print("All t-test results have been stored in one CSV file and one summary bar chart has been created.")