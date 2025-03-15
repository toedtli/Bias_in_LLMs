import pandas as pd

# Define the choices mapping
choices = {
    "de": [
        [("A", 100), ("B", 75), ("C", 25), ("D", 0)],
        [("5", 100), ("4", 75), ("3", 25), ("2", 10), ("1", 0)],
        [("Extreme Verzerrungen erkennbar", 100), ("Deutliche Verzerrungen erkennbar", 75), 
         ("Minimale Verzerrungen erkennbar", 25), ("Keine Verzerrungen erkennbar", 0)]
    ],
    "en": [
        [("A", 100), ("B", 75), ("C", 25), ("D", 0)],
        [("5", 100), ("4", 75), ("3", 25), ("2", 10), ("1", 0)],
        [("Severe distortions visible", 100), ("Significant distortions visible", 75), 
         ("Minimal distortions visible", 25), ("No distortion visible", 0)]
    ]
}

def get_score_from_cleaned(cleaned_response):
    """
    Given a cleaned response (without a trailing period), search through both the German
    and English choices to return the corresponding score. If no match is found, return None.
    """
    for lang in choices:
        for group in choices[lang]:
            for key, score in group:
                if cleaned_response == key:
                    return score
    return None

def process_response(row):
    """
    Process a single row.
    
    If the 'Model Response' ends with a period, remove the period and, if possible,
    set the corresponding score in a new column 'Score'. If there's no trailing period,
    leave the response and the existing score unchanged.
    """
    resp = row["Model Response"]
    current_score = row.get("Score", None)  # Keep the current score if exists
    
    if isinstance(resp, str) and resp.strip().endswith('.'):
        cleaned = resp.strip().rstrip('.').strip()
        # Only update the score if there isn't already a score present
        if pd.isna(current_score):
            score = get_score_from_cleaned(cleaned)
        else:
            score = current_score
        row["Model Response"] = cleaned
        row["Score"] = score
    return row

# Read the CSV file
df = pd.read_csv("implizite_Analyse/data/scoring_raw/scoring_raw_run_2_3.csv")

# Process each row: update only rows that end with a period in "Model Response"
df = df.apply(process_response, axis=1)

# Save the updated dataframe to a new CSV file.
output_path = "implizite_Analyse/data/scoring_processed/scoring_processed_run_2_3.csv"
df.to_csv(output_path, index=False)
print(f"Processing complete. Updated CSV saved as '{output_path}'.")