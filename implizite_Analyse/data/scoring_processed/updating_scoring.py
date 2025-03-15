#!/usr/bin/env python3
import os
import pandas as pd

def main():
    # Define input and output file paths.
    input_csv = "implizite_Analyse/data/scoring_processed/scoring_processed_run_2_3.csv"
    output_csv = "implizite_Analyse/data/scoring_processed/scoring_processed_run_2_3_test.csv"
    
    # Read the CSV file.
    df = pd.read_csv(input_csv)
    
    # Trim column names (remove leading/trailing whitespace).
    df.columns = df.columns.str.strip()
    
    # If a "Language" column exists, append the appropriate suffix to the "Choice Set" column.
    if "Language" in df.columns and "Choice Set" in df.columns:
        # Ensure the Language values are trimmed and lowercase.
        df["Language"] = df["Language"].str.strip().str.lower()
        df["Choice Set"] = df.apply(
            lambda row: f"{row['Choice Set']} (de)" if row["Language"] == "de" \
                        else (f"{row['Choice Set']} (en)" if row["Language"] == "en" \
                        else row["Choice Set"]), 
            axis=1
        )
    
    # Add to the "Group" column: if the value is "Rohingya", append (de) or (en) based on Language.
    if "Group" in df.columns and "Language" in df.columns:
        df["Group"] = df.apply(
            lambda row: f"{row['Group']} (de)" if row["Group"] == "Rohingya" and row["Language"] == "de"
                        else (f"{row['Group']} (en)" if row["Group"] == "Rohingya" and row["Language"] == "en"
                        else row["Group"]),
            axis=1
        )
        
    # Add the "Alternative Scoring" column based on the "Score" values.
    # Mapping: score = 100 → 3, 75 → 2, 25 → 1, 0 → 0.
    if "Score" in df.columns:
        score_map = {100: 4, 75: 3, 25: 2, 10: 1, 0: 0}
        df["Alternative Scoring"] = df["Score"].map(score_map)
        
    # Create output directory if it doesn't exist.
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save the processed DataFrame to the output CSV.
    df.to_csv(output_csv, index=False)
    print(f"Processed file saved to {output_csv}")

if __name__ == "__main__":
    main()