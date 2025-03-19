import json
import os
import pandas as pd
import csv

# ---------------------------
# Step 1. Define constants.
# ---------------------------
models = ["gpt", "gemini", "qwen", "deepseek"]
languages = ["de", "en"]

# For groups, use the actual names per language.
group_names = {
    "de": ["Kurden", "Palästinenser", "Uiguren", "Katalanen", "Tibeter", "Rohingya"],
    "en": ["Kurds", "Palestinians", "Uyghurs", "Catalans", "Tibetans", "Rohingya"]
}

# For choices, we have three choice sets per language.
# The values are lists of (text, score) tuples.
choice_set_labels = {
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

# ---------------------------
# Step 2. Load and check the fragenkatalog JSON.
# ---------------------------
with open('explizite_Analyse/data/fragenkatalog_extended.json', 'r', encoding='utf-8') as f:
    fragenkatalog = json.load(f)

# (Optional) Check that each question has two formulations per expected language.
for question_id, question_data in fragenkatalog.items():
    for lang, formulations in question_data.get('questions', {}).items():
        if lang not in languages:
            print(f"Warning: language '{lang}' in {question_id} is not in expected languages {languages}.")
        if len(formulations.keys()) != 2:
            print(f"Warning: {question_id} in language '{lang}' has {len(formulations.keys())} formulations (expected 2).")

# ---------------------------
# Step 3. Build full expected combinations.
# ---------------------------
# For each model, question, language, formulation, each choice set (using its actual texts),
# and for each group (specific to the language) we create one entry.
full_combinations = []
for model in models:
    for question_id, question_data in fragenkatalog.items():
        for lang, formulations in question_data.get('questions', {}).items():
            for formulation in formulations.keys():
                # For language 'en', we want the choice sets in a specific order.
                # Desired order for 'en' is: second, third, then first.
                if lang == 'en':
                    order = [1, 2, 0]
                else:
                    order = list(range(len(choice_set_labels[lang])))
                for idx in order:
                    cs = choice_set_labels[lang][idx]
                    # Extract just the choice texts.
                    choices_text = [choice for choice, score in cs]
                    for group in group_names[lang]:
                        full_combinations.append({
                            'model': model,
                            'group': group,
                            'language': lang,
                            'question_id': question_id,
                            'formulation': formulation,
                            'choice_set': str(choices_text)
                        })

print("Total full combinations (all models):", len(full_combinations))  # Should be 4320.

# Build a set of tuples from full_combinations.
full_set = set(
    (entry['model'], entry['group'], entry['language'], entry['question_id'], entry['formulation'], entry['choice_set'])
    for entry in full_combinations
)

# ---------------------------
# Step 4. Load scored combinations from CSV.
# ---------------------------
# The CSV is assumed to have columns: "Model", "Group", "Language", "Question ID", "Formulation Key", "Choice Set"
df_scored = pd.read_csv('explizite_Analyse/data/processed/scoring_processed_combined.csv')
print("Total scored combinations loaded:", len(df_scored))

# Build the scored_set as a set of tuples.
scored_set = set(
    tuple(row) for row in df_scored[['Model', 'Group', 'Language', 'Question ID', 'Formulation Key', 'Choice Set']].values
)

# ---------------------------
# Step 5. Print heads of both sets for comparison.
# ---------------------------
# For a consistent sample, filter to a specific subset.
# We choose model 'deepseek', group 'Catalans', language 'en', and question_id 'question1'.
sample_full = [item for item in full_set]
print("Head of full_set:")
for item in sample_full[:5]:
    print(item)

sample_scored = [item for item in scored_set]
print("\nHead of scored_set:")
for item in sample_scored[:5]:
    print(item)

    
# Compute missing combinations as the set difference and then convert to a sorted list in descending order.
missing_combinations = set(full_set) - set(scored_set)
missing_list = sorted(list(missing_combinations), reverse=True)

print("\nHead of missing_combinations:")
for item in missing_list[:5]:
    print(item)
    
    
# ---------------------------
# Step 6. Save missing combinations to file.
# ---------------------------
output_path = os.path.join('explizite_Analyse/missing_combinations/missing_combinations_combined.csv')
# Write the missing_list to the CSV file.
with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Write header row.
    writer.writerow(['Model', 'Group', 'Language', 'Question ID', 'Formulation', 'Choice Set'])
    # Write each missing combination row.
    for item in missing_list:
         writer.writerow(item)

print(f"Missing combinations saved to {output_path}")