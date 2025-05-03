import re
import csv

def rewrite_question_to_statement(text):
    pattern = re.compile(r'\b(question|Question)(\d+)\b')
    
    def replace(match):
        prefix = match.group(1)
        number = match.group(2)
        new_prefix = 'Statement' if prefix[0].isupper() else 'statement'
        return f"{new_prefix}{number}"
    
    return pattern.sub(replace, text)


def rewrite_header(header_text):
    return header_text.replace("Question ID", "Statement ID")

# === Set your file paths here ===
input_file_path = "explizite_Analyse/data/raw/scoring_run_3.csv"   # Replace with your actual input file path
output_file_path = "explizite_Analyse/data/raw/scoring_run_3_updated.csv" # Replace with desired output file path

# Read and process the input CSV
with open(input_file_path, mode='r', encoding='utf-8', newline='') as infile, \
     open(output_file_path, mode='w', encoding='utf-8', newline='') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Process the header row separately
    header = next(reader)
    header = [rewrite_header(cell) for cell in header]
    writer.writerow([rewrite_question_to_statement(cell) for cell in header])

    # Process the remaining rows
    for row in reader:
        new_row = [rewrite_question_to_statement(cell) for cell in row]
        writer.writerow(new_row)

print(f"CSV transformation complete. Output written to: {output_file_path}")