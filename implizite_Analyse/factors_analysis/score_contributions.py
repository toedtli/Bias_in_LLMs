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

# Define helper to calculate both mean and SEM
def get_grouped_scores_and_sems(factor):
    grouped = df.groupby([factor, "Source Model"])[score_col].agg(['mean', 'std', 'count'])
    grouped['sem'] = grouped['std'] / grouped['count']**0.5
    means = grouped['mean'].unstack(fill_value=0)
    sems = grouped['sem'].unstack(fill_value=0).fillna(0)
    return means, sems

# Get data and SEMs
group_model_scores, group_model_sems = get_grouped_scores_and_sems("Group")
language_model_scores, language_model_sems = get_grouped_scores_and_sems("Language")
question_model_scores, question_model_sems = get_grouped_scores_and_sems("Question ID")
choice_set_model_scores, choice_set_model_sems = get_grouped_scores_and_sems("Choice Set")

# Print the numerical summaries
print("Score contributions by Group and Models:")
print(group_model_scores, "\n")

print("Score contributions by Language and Models:")
print(language_model_scores, "\n")

print("Score contributions by Question ID and Models:")
print(question_model_scores, "\n")

print("Score contributions by Choice Set and Models:")
print(choice_set_model_scores, "\n")

# Calculate and print mean SEMs
def print_mean_sem(sems, name):
    mean_sem = sems.mean().mean()
    print(f"Mean SEM by {name}: {mean_sem:.4f}")

print_mean_sem(group_model_sems, "Group")
print_mean_sem(language_model_sems, "Language")
print_mean_sem(question_model_sems, "Question ID")
print_mean_sem(choice_set_model_sems, "Choice Set")
print()

# Set up the directory to save charts
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

# Modified plot function with error bars
def plot_bar(data, sems, title, save_path, xlabel, ylabel, factor=None):
    plt.figure(figsize=(10, 7))
    ax = data.plot(kind="bar", yerr=sems.values.T, capsize=4, legend=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title="Modelle")

    if factor == "Choice Set":
        adjust_xticks(ax, data, factor)
    else:
        plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Create and save the charts
plot_bar(group_model_scores, group_model_sems,
         "Bias-Score nach Gruppe (Modellvergleich)",
         os.path.join(save_dir, "score_contributions_by_group.png"),
         xlabel="Gruppe",
         ylabel="Durchschnittlicher Bias-Score",
         factor="Group")

plot_bar(language_model_scores, language_model_sems,
         "Bias-Score nach Sprache (Modellvergleich)",
         os.path.join(save_dir, "score_contributions_by_language.png"),
         xlabel="Sprache",
         ylabel="Durchschnittlicher Bias-Score",
         factor="Language")

plot_bar(question_model_scores, question_model_sems,
         "Bias-Score nach Frage-ID (Modellvergleich)",
         os.path.join(save_dir, "score_contributions_by_question_id.png"),
         xlabel="Frage-ID",
         ylabel="Durchschnittlicher Bias-Score",
         factor="Question ID")

plot_bar(choice_set_model_scores, choice_set_model_sems,
         "Bias-Score nach Antwortmöglichkeiten (Modellvergleich)",
         os.path.join(save_dir, "score_contributions_by_choice_set.png"),
         xlabel="Antwortmöglichkeiten",
         ylabel="Durchschnittlicher Bias-Score",
         factor="Choice Set")

print(f"Charts saved in: {save_dir}")