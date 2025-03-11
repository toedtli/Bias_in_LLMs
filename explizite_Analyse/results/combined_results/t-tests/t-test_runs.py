import pandas as pd
import numpy as np
from scipy.stats import ttest_ind_from_stats

def main():
    # Read the CSV files from the specified paths
    df1 = pd.read_csv("explizite_Analyse/results/results_run_1/results_run_1.csv")
    df2 = pd.read_csv("explizite_Analyse/results/results_run_2/results_run_2.csv")
    
    # Merge the dataframes on Model, Group, and Axis.
    # Here we assume that the CSVs have a column named 'Axis'. 
    # The merged dataframe will have columns from both runs with suffixes.
    merged = pd.merge(df1, df2, on=["Model", "Group", "Axis Name"], suffixes=[" Run 1", " Run 2"])
    
    # Rename columns for clarity (and rename 'Axis' to 'Axis Name' per specification)
    merged.rename(columns={
        "Axis": "Axis Name",
        "mean Run 1": "Mean Run 1", 
        "mean Run 2": "Mean Run 2",
        "sem Run 1": "SEM Run 1", 
        "sem Run 2": "SEM Run 2",
        "count Run 1": "Count Run 1", 
        "count Run 2": "Count Run 2"
    }, inplace=True)
    
    # Function to compute p-value from the summary stats for one merged row.
    def compute_p_value(row):
        # Derive standard deviations from SEM and count.
        std1 = row["SEM Run 1"] * np.sqrt(row["Count Run 1"])
        std2 = row["SEM Run 2"] * np.sqrt(row["Count Run 2"])
        
        # Run the t-test using summary statistics. 
        # Using equal_var=False to perform Welch's t-test.
        t_stat, p_val = ttest_ind_from_stats(
            mean1=row["Mean Run 1"], std1=std1, nobs1=row["Count Run 1"],
            mean2=row["Mean Run 2"], std2=std2, nobs2=row["Count Run 2"],
            equal_var=False
        )
        return p_val

    # Apply the function to each row to get the p-value
    merged["p-value"] = merged.apply(compute_p_value, axis=1)
    
    # Create a significance column (using p < 0.05 as threshold)
    merged["significance"] = merged["p-value"].apply(lambda p: "Significant" if p < 0.05 else "Not Significant")
    
    # Select and order the final columns as specified
    final_columns = [
        "Model", "Group", "Axis Name",
        "Count Run 1", "Count Run 2",
        "Mean Run 1", "Mean Run 2",
        "SEM Run 1", "SEM Run 2",
        "p-value", "significance"
    ]
    result_df = merged[final_columns]
    
    # Save the results to a new CSV file
    result_df.to_csv("explizite_Analyse/results/combined_results/t-tests/t_test_results.csv", index=False)
    print("T-test results saved to t_test_results.csv")

if __name__ == "__main__":
    main()