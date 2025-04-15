import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_combined_chart(csv_file, output_chart):
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_file)
    df.loc[df["Axis Name"] == "Bedrohungswahrnehmung", "Score"] = 100 - df["Score"]

    # Group by "Question ID", "Model", and "Formulation Key" to get the average score
    aggregated = df.groupby(["Question ID", "Model", "Formulation Key"])["Score"].mean().reset_index()

    # Pivot the aggregated data so that the index is "Question ID" and columns are a MultiIndex (Model, Formulation Key)
    pivot_table = aggregated.pivot_table(index="Question ID", columns=["Model", "Formulation Key"], values="Score")
    
    # Helper function: extract numeric parts from "Question ID" (e.g., "question1", "question2", etc.)
    def extract_numeric(qid):
        match = re.search(r'\d+', str(qid))
        return int(match.group()) if match else 0
    
    # Sort the pivot table by Question ID numerically
    pivot_table = pivot_table.sort_index(key=lambda x: x.map(extract_numeric))
    
    # Determine default colors from matplotlib's color cycle for each unique model
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    unique_models = sorted(set([model for model, formulation in pivot_table.columns]), key=str.lower)
    model_color_mapping = {model: default_colors[i % len(default_colors)] for i, model in enumerate(unique_models)}
    
    # Map formulation keys to hatch patterns (to differentiate between formulations within the same model)
    unique_form_keys = sorted(aggregated["Formulation Key"].unique())
    available_patterns = ['', '///', 'xx', '...']  # Extend if more than four formulations exist
    formulation_hatches = {k: available_patterns[i % len(available_patterns)] for i, k in enumerate(unique_form_keys)}
    
    # Set up the overall bar chart.
    questions = pivot_table.index.tolist()
    n_questions = len(questions)
    n_bars = len(pivot_table.columns)
    
    # Define bar parameters: each group spans 0.8 units on the x-axis
    bar_width = 0.8 / n_bars
    x = np.arange(n_questions)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # To avoid duplicate legend entries, track which labels have been added.
    added_labels = {}
    
    # Loop through each column (each is a tuple: (Model, Formulation Key))
    for i, (model, formulation_key) in enumerate(pivot_table.columns):
        # Calculate offset for the bar in the group
        offset = (i - n_bars/2) * bar_width + bar_width/2
        scores = pivot_table.iloc[:, i].values
        
        # Use the default color for the model
        color = model_color_mapping.get(model, None)
        hatch = formulation_hatches.get(formulation_key, '')
        
        label = f"{model} - {formulation_key}"
        if label in added_labels:
            label = None
        else:
            added_labels[label] = True
        
        ax.bar(x + offset, scores, width=bar_width, color=color, edgecolor='black', hatch=hatch, label=label)
    
    # Set titles and labels
    ax.set_title("Average Score per Question ID by Model and Formulation Key", fontsize=14)
    ax.set_xlabel("Question ID", fontsize=12)
    ax.set_ylabel("Durchschnittliche Punktzahl", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(questions, rotation=90, ha='center')
    
    # Create two legends:
    # Legend for models (colors) using the model_color_mapping
    model_legend_elements = [Patch(facecolor=model_color_mapping[model], edgecolor='black', label=model)
                             for model in unique_models]
    # Legend for formulation keys (hatch patterns)
    formulation_legend_elements = [Patch(facecolor='white', edgecolor='black', hatch=formulation_hatches[form_key],
                                         label=form_key)
                                   for form_key in unique_form_keys]
    
    first_legend = ax.legend(handles=model_legend_elements, title="Model", loc='upper left')
    second_legend = ax.legend(handles=formulation_legend_elements, title="Formulation Key", loc='upper right')
    ax.add_artist(first_legend)
    
    plt.tight_layout()
    plt.savefig(output_chart)
    plt.close()
    print(f"Saved combined chart to {output_chart}")

if __name__ == "__main__":
    csv_file = "explizite_Analyse/data/processed/scoring_processed_combined.csv"
    output_chart = "explizite_Analyse/factors_analysis/formulation/combined_formulation_differences.png"
    plot_combined_chart(csv_file, output_chart)