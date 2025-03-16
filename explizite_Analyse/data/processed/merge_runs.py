import pandas as pd
import matplotlib.pyplot as plt

def merge_append_and_plot():
    # Read the three CSV files and add a Run identifier
    df1 = pd.read_csv("explizite_Analyse/data/processed/scoring_processed_run_1.csv")
    df1["Run"] = "Run 1"
    df2 = pd.read_csv("explizite_Analyse/data/processed/scoring_processed_run_2.csv")
    df2["Run"] = "Run 2"
    df3 = pd.read_csv("explizite_Analyse/data/processed/scoring_processed_run_3.csv")
    df3["Run"] = "Run 3"
    
    # Combine them into one DataFrame
    merged_df = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Verify necessary columns exist
    required_columns = ["Choice Set", "Language", "Model", "Group", "Score"]
    for col in required_columns:
        if col not in merged_df.columns:
            raise ValueError(f"Column '{col}' not found in the data.")
    
    # Append the language value to the end of the Choice Set
    merged_df["Choice Set"] = merged_df["Choice Set"].astype(str) + "(" + merged_df["Language"].astype(str) + ")"
    
    # Save the merged dataframe to a new CSV file
    merged_csv_path = "explizite_Analyse/data/processed/scoring_processed_combined.csv"
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"Merged CSV created successfully: {merged_csv_path}")


if __name__ == "__main__":
    merge_append_and_plot()