import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_bar(df, output_dir):
    """
    Generate a stacked bar plot showing the score contribution (mean score) of each group for each model.
    """
    plt.figure(figsize=(12, 8))
    
    # Aggregate using sum, then unstack to create a DataFrame with groups as columns.
    grouped_data = df.groupby(['Scorer Model', ' Choice Set'])['Score'].sum().unstack()

    # Plot
    ax = grouped_data.plot(kind='bar', figsize=(12, 8), width=0.7, stacked=True)

    plt.title('Aufsummierte Punkte pro Antwortset')
    plt.xlabel('Scorer Model')
    plt.ylabel('Average Score')
    plt.xticks(rotation=0)

    # Place the legend *below* the plot, centered
    # Adjust ncol to place items horizontally
    plt.legend(
        title='Antwort Sets',
        bbox_to_anchor=(0.5, -0.15),  # (x, y) position in axes coordinates
        loc='upper center',
        ncol=1  # number of legend columns (adjust as you wish)
    )

    # Use tight_layout with enough bottom margin for the legend
    plt.tight_layout(rect=[0, 0, 1, 1])
    
    # Save the figure
    file_path = os.path.join(output_dir, 'score_contribution_by_cs.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    print(f"Saved stacked bar plot to {file_path}")


def main():
    # Hardcoded paths for CSV file and output directory.
    csv_file = "implizite_Analyse/2_analysis/scoring_updated.csv"
    output_dir = "implizite_Analyse/2_analysis/statistics"
    
    # Read CSV file.
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at {csv_file} is empty.")
        return
    except pd.errors.ParserError:
        print(f"Error: Could not parse CSV file at {csv_file}. Check file format.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV: {e}")
        return

    # Verify that the CSV contains the necessary columns.
    required_columns = {'Scorer Model', ' Choice Set', 'Score'}
    if not required_columns.issubset(df.columns):
        print(f"CSV file must contain the following columns: {required_columns}")
        return

    # Ensure that the 'Score' column is numeric.
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

    # Generate the visualization.
    plot_bar(df, output_dir)


if __name__ == '__main__':
    main()