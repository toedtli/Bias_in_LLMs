import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file (adjust the path if necessary)
csv_file = "implizite_Analyse/data/scoring_processed/scoring_processed_run_2.csv"
df = pd.read_csv(csv_file)

# Identify the column for score contributions.
score_col = None
for col in df.columns:
    if col.lower() in ["score", "score contribution"]:
        score_col = col
        break

if score_col is None:
    raise ValueError("Could not find a 'Score' or 'Score Contribution' column in the CSV file.")

# Ensure the CSV contains a "Models" column to compare differences between models.
if "Source Model" not in df.columns:
    raise ValueError("CSV does not contain a 'Models' column required for comparing models.")

# Group the data by each factor and "Models", then unstack to show each model separately.
group_model_scores = df.groupby(["Group", "Source Model"])[score_col].sum().unstack(fill_value=0)
language_model_scores = df.groupby(["Language", "Source Model"])[score_col].sum().unstack(fill_value=0)
question_model_scores = df.groupby(["Question ID", "Source Model"])[score_col].sum().unstack(fill_value=0)
choice_set_model_scores = df.groupby(["Choice Set", "Source Model"])[score_col].sum().unstack(fill_value=0)

# Print the numerical summaries.
print("Score contributions by Group and Models:")
print(group_model_scores, "\n")

print("Score contributions by Language and Models:")
print(language_model_scores, "\n")

print("Score contributions by Question ID and Models:")
print(question_model_scores, "\n")

print("Score contributions by Choice Set and Models:")
print(choice_set_model_scores, "\n")

# Set up the directory to save charts.
save_dir = "implizite_Analyse/factors_analysis/run_2"
os.makedirs(save_dir, exist_ok=True)

# Plotting function with saving capability.
def plot_bar(data, title, save_path, xlabel="Category", ylabel="Total Score"):
    plt.figure(figsize=(10, 7))
    data.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title="Models")
    plt.tight_layout()
    # Save the chart to the specified path.
    plt.savefig(save_path)
    plt.close()

# Create and save bar charts for each factor.
plot_bar(group_model_scores, 
         "Score Contributions by Group (with Models)", 
         os.path.join(save_dir, "score_contributions_by_group.png"))
plot_bar(language_model_scores, 
         "Score Contributions by Language (with Models)", 
         os.path.join(save_dir, "score_contributions_by_language.png"))
plot_bar(question_model_scores, 
         "Score Contributions by Question ID (with Models)", 
         os.path.join(save_dir, "score_contributions_by_question_id.png"))
plot_bar(choice_set_model_scores, 
         "Score Contributions by Choice Set (with Models)", 
         os.path.join(save_dir, "score_contributions_by_choice_set.png"))