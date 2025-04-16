import os
import pandas as pd
import matplotlib.pyplot as plt

run = 'combined'
# Load the CSV file (adjust the path if necessary)
csv_file = f"implizite_Analyse/data/scoring_processed/scoring_processed_{run}.csv"
save_dir = f"implizite_Analyse/factors_analysis/{run}"

df = pd.read_csv(csv_file)

# Identify the column for score contributions.
score_col = None
for col in df.columns:
    if col.lower() in ["score", "score contribution"]:
        score_col = col
        break

if score_col is None:
    raise ValueError("Could not find a 'Score' or 'Score Contribution' column in the CSV file.")

# Ensure the CSV contains a "Source Model" column to compare differences between models.
if "Source Model" not in df.columns:
    raise ValueError("CSV does not contain a 'Source Model' column required for comparing models.")

# Group the data by each factor and "Source Model", then unstack to show each model separately.
group_model_scores = df.groupby(["Group", "Source Model"])[score_col].mean().unstack(fill_value=0)
language_model_scores = df.groupby(["Language", "Source Model"])[score_col].mean().unstack(fill_value=0)
question_model_scores = df.groupby(["Question ID", "Source Model"])[score_col].mean().unstack(fill_value=0)
choice_set_model_scores = df.groupby(["Choice Set", "Source Model"])[score_col].mean().unstack(fill_value=0)

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
os.makedirs(save_dir, exist_ok=True)

# Define the mapping for "Choice Set"
choice_set_mapping = {
    "(5,4,3,2,1) (de)": "5 → 1 (de)",
    "(A,B,C,D) (de)": "A → D (de)",
    "Extreme Verzerrungen erkennbar; Deutliche Verzerrungen erkennbar; Minimale Verzerrungen erkennbar; Keine Verzerrungen erkennbar (de)": "Extreme → Keine",
    "(5,4,3,2,1) (en)": "5 → 1 (en)",
    "(A,B,C,D) (en)": "A → D (en)",
    "Severe distortions visible; Significant distortions visible; Minimal distortions visible; No distortion visible (en)": "Severe → No"
}

# Function to adjust x-ticks dynamically
def adjust_xticks(ax, data, factor):
    if factor == "Choice Set":
        current_labels = data.index.tolist()
        new_labels = [choice_set_mapping.get(label, label) for label in current_labels]

        ax.set_xticks(range(len(new_labels)))  
        ax.set_xticklabels(new_labels, rotation=45)

# Angepasste Plot-Funktion mit übersetzten Beschriftungen
def plot_bar(data, title, save_path, xlabel, ylabel, factor=None):
    plt.figure(figsize=(10, 7))
    ax = data.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title="Modelle")

    # Apply x-tick adjustments if relevant
    if factor == "Choice Set":
        adjust_xticks(ax, data, factor)
    else:
        plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Erstelle und speichere die Diagramme mit den deutschen Beschriftungen.

plot_bar(group_model_scores, 
         "Bias-Score nach Gruppe (Modellvergleich)", 
         os.path.join(save_dir, "score_contributions_by_group.png"),
         xlabel="Gruppe",
         ylabel="Durchschnittlicher Bias-Score",
         factor="Group")

plot_bar(language_model_scores, 
         "Bias-Score nach Sprache (Modellvergleich)", 
         os.path.join(save_dir, "score_contributions_by_language.png"),
         xlabel="Sprache",
         ylabel="Durchschnittlicher Bias-Score",
         factor="Language")

plot_bar(question_model_scores, 
         "Bias-Score nach Frage-ID (Modellvergleich)", 
         os.path.join(save_dir, "score_contributions_by_question_id.png"),
         xlabel="Frage-ID",
         ylabel="Durchschnittlicher Bias-Score",
         factor="Question ID")

plot_bar(choice_set_model_scores, 
         "Bias-Score nach Antwortmöglichkeiten (Modellvergleich)", 
         os.path.join(save_dir, "score_contributions_by_choice_set.png"),
         xlabel="Antwortmöglichkeiten",
         ylabel="Durchschnittlicher Bias-Score",
         factor="Choice Set")

print(f"Charts saved in: {save_dir}")