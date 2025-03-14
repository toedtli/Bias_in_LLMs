import os
import csv
import json
import sys

# Ensure the project root is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from modells import ModelAPI

# Initialize model API
modells = ModelAPI()

# Load the JSON file with the descriptions
with open('implizite_Analyse/data/beschreibungs_generierung_extended.json', 'r', encoding='utf-8') as file:
    questions = json.load(file)

# Define the my_models
my_models = ["gpt", "gemini", "qwen", "deepseek"] 
languages = ["de", "en"]

# Define groups and their translations
groups = {
    "de": ["Kurden", "Palästinenser", "Uiguren", "Katalanen", "Tibeter", "Rohingya"],
    "en": ["Kurds", "Palestinians", "Uyghurs", "Catalans", "Tibetans", "Rohingya"]
}

# Function to generate text using Gemini
def generate_gemini(prompt, system_instruction=None):
    return modells.chat_with_gemini(prompt, system_instruction)

# Function to generate text using GPT (OpenAI)
def generate_gpt(prompt, system_instruction=None):
    return modells.chat_with_gpt(prompt, system_instruction)

# Function to generate text using Falcon via HuggingFace API
def generate_qwen(prompt, system_instruction=None):
    return modells.chat_with_qwen(prompt, system_instruction)

# Function to generate text using Bloom via HuggingFace API
def generate_deepseek(prompt, system_instruction=None):
    return modells.chat_with_deepseek(prompt, system_instruction)
 
# Function to create nested CSV files per model
def create_group_csv(questions, groups, my_models):
    base_path = "implizite_Analyse/data/descriptions_2"
    
    # Pfad und Datei für die aggregierte CSV erstellen
    csv_file_path_all = os.path.join(base_path, "all_descriptions.csv")
    os.makedirs(base_path, exist_ok=True)
    
    with open(csv_file_path_all, 'w', encoding='utf-8', newline='') as all_csv_file:
        all_csv_writer = csv.writer(all_csv_file)
        all_csv_writer.writerow(["Model", "Group", "Language", "Question ID", "Question", "Answer"])
        # Iterate over my_models
        for model in my_models:
            for lang in languages:
                # Define the folder path for the current model and language
                lang_folder = os.path.join(base_path, model, lang)
                os.makedirs(lang_folder, exist_ok=True)

                # Iterate over groups
                for group in groups[lang]:
                    csv_file_path = os.path.join(lang_folder, f"{group.lower()}.csv")
                    csv_file_path_all = os.path.join(base_path, "all_descriptions.csv")

                    with open(csv_file_path, 'w', encoding='utf-8', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(["Model", "Group", "Language", "Question ID", "Question", "Answer"])  # Updated header

                        # Iterate over questions
                        for question_id, translations in questions.items():
                            question_text = translations[lang].replace("{Gruppe}", group)

                            # Generate response based on the model
                            response = ""
                            if model == 'gemini':
                                response = generate_gemini(question_text)
                            elif model == 'gpt':
                                response = generate_gpt(question_text)
                            elif model == 'qwen':
                                response = generate_qwen(question_text)
                            elif model == 'deepseek':
                                response = generate_deepseek(question_text)
                            else: 
                                response = f"Unsupported model: {model}"

                            # Write row
                            csv_writer.writerow([model, group, lang, question_id, question_text, response])
                            # Write row in aggregierte CSV Datei
                            all_csv_writer.writerow([model, group, lang, question_id, question_text, response])
                

                    print(f"Created: {csv_file_path}")

# Run the script
if __name__ == "__main__":
    create_group_csv(questions, groups, my_models)