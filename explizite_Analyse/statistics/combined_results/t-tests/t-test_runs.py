import pandas as pd
from scipy import stats
import os

def main():
    # 1. Read in the two CSV files
    df1 = pd.read_csv("explizite_Analyse/statistics/group_axis_statistics_overall_run_1.csv")
    df2 = pd.read_csv("explizite_Analyse/statistics/group_axis_statistics_overall_run_2.csv")
    
    # Print columns for debugging
    print("Columns in df1:", df1.columns.tolist())
    print("Columns in df2:", df2.columns.tolist())
    
    # 2. Add a column to identify the run
    df1["run"] = 1
    df2["run"] = 2

    # 3. Concatenate the two DataFrames
    df = pd.concat([df1, df2], ignore_index=True)

    # Determine which grouping columns to use.
    # Always group by "Model" and "Group"; include "Axis Name" if available.
    grouping_columns = ["Model", "Group"]
    if "Axis Name" in df.columns:
        grouping_columns.append("Axis Name")
    
    print("Grouping by:", grouping_columns)

    results = []

    # 4. For each unique combination, perform a t-test comparing the scores between runs
    for name, group_df in df.groupby(grouping_columns):
        # Extract the score values (assuming the column is named "mean") for each run
        data_run1 = group_df[group_df["run"] == 1]["mean"].dropna()
        data_run2 = group_df[group_df["run"] == 2]["mean"].dropna()
        
        # Compute the mean score for each run (if available)
        mean_run1 = data_run1.mean() if not data_run1.empty else None
        mean_run2 = data_run2.mean() if not data_run2.empty else None

        # Run the t-test if both groups have data; otherwise, leave p_value as None
        if len(data_run1) > 0 and len(data_run2) > 0:
            t_stat, p_value = stats.ttest_ind(data_run1, data_run2, equal_var=False)
        else:
            t_stat, p_value = None, None

        # Determine significance based on p-value (threshold 0.05)
        if p_value is not None and p_value < 0.05:
            decision = "significant"
        else:
            decision = "not significant"

        # Record the results for this grouping.
        # If grouping_columns has multiple keys, 'name' is a tuple.
        if isinstance(name, tuple):
            result_entry = list(name) + [mean_run1, mean_run2, p_value, decision]
        else:
            result_entry = [name, mean_run1, mean_run2, p_value, decision]
        results.append(result_entry)

    # 5. Define the output column names
    if "Axis Name" in grouping_columns:
        col_names = ["Model", "Group", "Axis Name", "mean_run1", "mean_run2", "p_value", "decision"]
    else:
        col_names = ["Model", "Group", "mean_run1", "mean_run2", "p_value", "decision"]

    # Create a DataFrame with all the t-test results
    results_df = pd.DataFrame(results, columns=col_names)

    # 6. Ensure the output directory exists
    output_dir = "explizite_Analyse/statistics/combined_results/t-tests"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "ttest_results.csv")
    
    # 7. Save the results to a CSV file
    results_df.to_csv(output_file, index=False)
    print(f"T-test results saved to {output_file}")

if __name__ == "__main__":
    main()