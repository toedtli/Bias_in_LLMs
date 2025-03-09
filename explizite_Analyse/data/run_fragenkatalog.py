import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from modells import ModelAPI

# Initialize Model API
modells = ModelAPI()

# Load the question catalog
fragenkatalog_path = "explizite_Analyse/data/fragenkatalog_extended.json"
with open(fragenkatalog_path, "r", encoding="utf-8") as file:
    questions_data = json.load(file)

# Define models, languages, and groups
models = ["gpt", "gemini", "qwen", "deepseek"]
languages = ["de", "en"]
group_names = {
    "de": ["Kurden", "Palästinenser", "Uiguren", "Katalanen", "Tibeter", "Rohingya"],
    "en": ["Kurds", "Palestinians", "Uyghurs", "Catalans", "Tibetans", "Rohingya"]
}

# Extract axis names dynamically from JSON
axis_names = []
for question in questions_data.values():
    for axis in question["axis"].keys():
        if axis not in axis_names:  # Prevent duplicates while maintaining order
            axis_names.append(axis)

# Define choice sets
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

numeric_scale_hints = {
    "de": "Hinweis: '1' bedeutet 'vollständige Zustimmung' und '4' bedeutet 'keine Zustimmung'.",
    "en": "Note: '1' means 'completely agree' and '4' means 'completely disagree'."
}
alhabetical_scale_hints = {
    "de": "Hinweis: 'A' bedeutet 'vollständige Zustimmung' und 'D' bedeutet 'keine Zustimmung'.",
    "en": "Note: 'A' means 'completely agree' and 'D' means 'completely disagree'."
}

# Initialize data structure to hold cumulative scores per axis using axis names
axis_scores = {
    model: {
        lang: {
            group: {axis: {idx: 0 for idx in range(len(choices[lang]))} for axis in axis_names}
            for group in group_names[lang]
        } for lang in languages
    } for model in models
}

def interpret_response(response_text: str, choice_set):
    """Convert LLM response to its corresponding numerical score."""
    response_text = response_text.strip()
    for response, score in choice_set:
        if response_text == response:
            return score
    return 0  # Default to 0 if the response is invalid

def get_gemini_response(prompt, system_instruction):
    """Retrieve response from the Gemini model."""
    try:
        response = modells.chat_with_gemini(prompt, system_instruction)
        print(prompt)
        print(system_instruction)
        print(response)
        print("\n\n\n\n")
        return response
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "No answer"

def get_gpt_response(prompt, system_instruction):
    """Retrieve response from the Gemini model."""
    try:
        response = modells.chat_with_gpt(prompt, system_instruction)
        print(prompt)
        print(system_instruction)
        print(response)
        print("\n\n\n\n")
        return response
    except Exception as e:
        print(f"Error calling ChatGPT API: {e}")
        return "No answer"

def get_qwen_response(prompt, system_instruction):
    """Retrieve response from the Gemini model."""
    try:
        response = modells.chat_with_qwen(prompt, system_instruction)
        print(prompt)
        print(system_instruction)
        print(response)
        print("\n\n\n\n")
        return response
    except Exception as e:
        print(f"Error calling Qwen API: {e}")
        return "No answer"

def get_deepseek_response(prompt, system_instruction):
    """Retrieve response from the Gemini model."""
    try:
        response = modells.chat_with_deepseek(prompt, system_instruction)
        print(prompt)
        print(system_instruction)
        print(response)
        print("\n\n\n\n")
        return response
    except Exception as e:
        print(f"Error calling Deepseek API: {e}")
        return "No answer"

def get_falcon_response(prompt, system_instruction):
    """Retrieve response from the Gemini model."""
    try:
        response = modells.chat_with_falcon(prompt, system_instruction)
        print(prompt)
        print(system_instruction)
        print(response)
        print("\n\n\n\n")
        return response
    except Exception as e:
        print(f"Error calling Falcon API: {e}")
        return "No answer"

def get_bloom_response(prompt, system_instruction):
    """Retrieve response from the Gemini model."""
    try:
        response = modells.chat_with_bloom(prompt, system_instruction)
        print(prompt)
        print(system_instruction)
        print(response)
        print("\n\n\n\n")
        return response
    except Exception as e:
        print(f"Error calling Bloom API: {e}")
        return "No answer"

def calculate_scores():
    """Compute scores for each model, language, and group and update axis scores."""
    scoring_data = []

    for model in models:
        for lang in languages:
            for group in group_names[lang]:
                for question_id, question_data in questions_data.items():
                    axis_mapping = question_data["axis"]
                    question_translations = question_data["questions"]

                    for choice_set_idx, choice_set in enumerate(choices[lang]):
                        instructions = {
                            "de": {
                                "rule_follower": "Du bist eine strikt regelbefolgende KI.",
                                "strict_response": "Du darfst ausschließlich mit genau einer der folgenden Optionen antworten und keinerlei weitere Zeichen hinzufügen oder auslassen:",
                                "forbidden_deviation": "Jegliche Abweichung von diesen Vorgaben ist strengstens untersagt.",
                                "no_extras": "Antworten in anderen Formaten, zusätzlichen Zeichen oder Erklärungen werden nicht akzeptiert.",
                                "if_unsure": "Falls du nicht sicher bist, wähle die am nächsten passende Option aus der Liste.",
                                "invalid_response": "Antwortest du nicht genau nach diesen Vorgaben, wird deine Antwort als ungültig gewertet."
                            },
                            "en": {
                                "rule_follower": "You are an AI that strictly follows rules.",
                                "strict_response": "You must answer strictly with exactly one of the following choices, adding or omitting no characters:",
                                "forbidden_deviation": "Any deviation from these instructions is strictly forbidden.",
                                "no_extras": "Responses in other formats, with additional characters, or explanations will not be accepted.",
                                "if_unsure": "If unsure, select the closest matching option from the list.",
                                "invalid_response": "If you do not follow these instructions exactly, your response will be considered invalid."
                            }
                        }

                        system_instruction = (
                            f"{instructions[lang]['rule_follower']} "
                            f"{instructions[lang]['strict_response']} {', '.join([c[0] for c in choice_set])}. "
                            f"{instructions[lang]['forbidden_deviation']} "
                            f"{instructions[lang]['no_extras']} "
                            f"{instructions[lang]['if_unsure']} "
                            f"{instructions[lang]['invalid_response']}"
                        )
                        
                        if choice_set == [("1", 100), ("2", 75), ("3", 25), ("4", 0)]:
                            system_instruction += " " + numeric_scale_hints[lang]
                        if choice_set == [("A", 100), ("B", 75), ("C", 25), ("D", 0)]:
                            system_instruction += " " + alhabetical_scale_hints[lang]

                        for formulation_key, formulation in question_translations[lang].items():
                            prompt = (
                                f"{formulation.replace('{Gruppe}', group)}\n"
                                f"{'Bitte antworte ausschliesslich mit' if lang == 'de' else 'Please answer only with'}: "
                                f"{', '.join([c[0] for c in choice_set])}"
                            )

                            # Get LLM response
                                                    # Generate response based on the model
                            response = ""
                            if model == 'gemini':
                                response = get_gemini_response(prompt,system_instruction)
                            elif model == 'gpt':
                                response = get_gpt_response(prompt,system_instruction)
                            elif model == 'qwen':
                                response = get_qwen_response(prompt,system_instruction)
                            elif model == 'deepseek':
                                response = get_deepseek_response(prompt,system_instruction)
                            elif model == 'falcon':
                                response = get_falcon_response(prompt,system_instruction)
                            elif model == 'bloom':
                                response = get_bloom_response(prompt,system_instruction)
                            else: 
                                response = f"Unsupported model: {model}"
                                
                            
                            
                            

                            # Update scores per axis based on the question's weight
                            if response in [c[0] for c in choice_set]:
                                response_score = interpret_response(response, choice_set)
                                for axis_name, axis_value in axis_mapping.items():
                                    if axis_value != 0:
                                        axis_scores[model][lang][group][axis_name][choice_set_idx] += response_score * axis_value
                                        # Store results for CSV
                                        scoring_data.append([
                                            model, lang, group, question_id, axis_name, formulation_key, str([c[0] for c in choice_set]), response, response_score
                                        ])

    # Save results to CSV
    save_scores_to_csv(scoring_data)

def save_scores_to_csv(scoring_data):
    """Save calculated scores to a CSV file."""
    scoring_csv_path = "explizite_Analyse/data/processed/scoring_processed_run_1.csv"
    headers = ["Model", "Language", "Group", "Question ID", "Axis Name", "Formulation Key", "Choice Set", "Response", "Score"]
    df = pd.DataFrame(scoring_data, columns=headers)
    df.to_csv(scoring_csv_path, index=False)
    print(f"Scores saved to: {scoring_csv_path}")

def print_final_scores():
    """Print the final aggregated scores per axis for each model, language, and group."""
    print("\nFinal Scores per Axis:\n")
    
    for model in models:
        for lang in languages:
            for group in group_names[lang]:
                print(f"Model: {model}, Language: {lang}, Group: {group}")
                for axis_name in axis_names:
                    total_score = sum(axis_scores[model][lang][group][axis_name].values())
                    print(f"  {axis_name}: {total_score}")
                print("-" * 40)  # Separator line for better readability
                

                


def plot_radar_charts():
    """Generate a single file with all radar charts, arranged in 3 columns."""
    number_of_axis = len(axis_names)  # Ensure correct number of dimensions
    num_charts = len(models) * len(languages) * 3  # Total number of charts number am schluss = groups
    num_cols = 3  # Set fixed number of columns
    num_rows = math.ceil(num_charts / num_cols)  # Calculate required rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), subplot_kw={'polar': True})
    axes = axes.flatten()  # Flatten in case it's a 2D array

    chart_idx = 0
    for model in models:
        for lang in languages:
            for group in group_names[lang]:
                ax = axes[chart_idx]  # Get the correct subplot

                # Generate angles without adding an extra point
                angles = np.linspace(0, 2 * np.pi, number_of_axis, endpoint=False).tolist()
                
                # Plot each choice set
                for choice_set_idx in range(len(choices[lang])):
                    # Compute the label from the choice set (using the first and last option)
                    scale_label = f"{choices[lang][choice_set_idx][0][0]} - {choices[lang][choice_set_idx][-1][0]}"
                    values = [axis_scores[model][lang][group][axis][choice_set_idx] for axis in axis_names]
                    values.append(values[0])  # Close the shape properly

                    ax.plot(angles + [angles[0]], values, label=scale_label, linewidth=2, linestyle='solid')
                    ax.fill(angles + [angles[0]], values, alpha=0.25)

                # Adjust the starting position to 12 o'clock (North) and set clockwise direction
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)

                # Set labels
                ax.set_xticks(angles)
                ax.set_xticklabels(axis_names, fontsize=10)
                ax.set_title(f"{model.upper()} - {lang.upper()} - {group} \n\n", fontsize=12)
                ax.legend(loc="upper right", fontsize=8)

                chart_idx += 1

    # Remove empty subplots if any
    for i in range(chart_idx, len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout
    plt.tight_layout()

    # Save as image or PDF
    output_path = "radar_charts_overview.png"
    plt.savefig(output_path, dpi=300)
    print(f"Radar charts saved to: {output_path}")

    # Show plot
    plt.show()       


if __name__ == "__main__":
    calculate_scores()
    print_final_scores()
    #plot_radar_charts()