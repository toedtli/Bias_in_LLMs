import pandas as pd

# Define the file paths
input_file = 'explizite_Analyse/data/raw/scoring_run_2.csv'
output_file = 'explizite_Analyse/data/processed/scoring_processed_run_2.csv'

# Load the CSV file
df = pd.read_csv(input_file)

# Define mapping choices
choices = {
    "de": [
        [("Ja", 100), ("Eher Ja", 75), ("Eher Nein", 25), ("Nein", 0)],
        [("1", 100), ("2", 75), ("3", 25), ("4", 0)],
        [("Stimme zu", 100), ("Stimme eher zu", 75), ("Stimme eher nicht zu", 25), ("Stimme nicht zu", 0)]
    ],
    "en": [
        [("Yes", 100), ("Rather Yes", 75), ("Rather No", 25), ("No", 0)],
        [("1", 100), ("2", 75), ("3", 25), ("4", 0)],
        [("Agree", 100), ("Rather agree", 75), ("Rather disagree", 25), ("Disagree", 0)]
    ]
}

# Flatten the choices dictionary into a single mapping dictionary
mapping = {}
for lang_choices in choices.values():
    for choice_group in lang_choices:
        for text, score in choice_group:
            normalized_text = text.lower().replace('.', '').strip()
            mapping[normalized_text] = (text, score)

# Function to normalize and map responses
def normalize_and_score(response):
    if pd.isnull(response):
        return response, None
    response_norm = response.lower().replace('.', '').strip()
    if response_norm in mapping:
        return mapping[response_norm]
    else:
        return response, None

# Apply normalization and scoring
df[['response_normalized', 'Score']] = df.apply(
    lambda row: pd.Series(normalize_and_score(row['Response'])),
    axis=1
)

# Update 'model_response' with normalized response
df['Response'] = df['response_normalized']
df.drop(columns=['response_normalized'], inplace=True)

# Save updated DataFrame to CSV
df.to_csv(output_file, index=False)

print(f"Updated file saved to: {output_file}")