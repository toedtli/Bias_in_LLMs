import pandas as pd
import matplotlib.pyplot as plt

def create_bar_chart(df, index_col, columns_col, value_col, title, xlabel, ylabel, legend_title, output_path):
    """
    Creates and saves a bar chart from a pivoted dataframe.

    Parameters:
        df (DataFrame): The original dataframe.
        index_col (str): Column name to use as the index after pivoting.
        columns_col (str): Column name to use as columns after pivoting.
        value_col (str): Column name with values to plot.
        title (str): Title of the chart.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        legend_title (str): Title for the legend.
        output_path (str): Path where the figure will be saved.
    """
    pivot_df = df.pivot(index=index_col, columns=columns_col, values=value_col)
    ax = pivot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=0)
    ax.legend(title=legend_title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Load the CSV file; adjust the path if needed
    df = pd.read_csv("implizite_Analyse/results/run_1/scoring_model_group.csv")
    
    # Create bar chart: Model Scores by Group
    create_bar_chart(
        df=df,
        index_col="Group",
        columns_col="Source Model",
        value_col="mean",
        title="Model Scores by Group",
        xlabel="Group",
        ylabel="Score",
        legend_title="Model",
        output_path="implizite_Analyse/results/run_1/model_scores_by_group.png"
    )
    
    # Create bar chart: Group Scores by Model
    create_bar_chart(
        df=df,
        index_col="Source Model",
        columns_col="Group",
        value_col="mean",
        title="Group Scores by Model",
        xlabel="Source Model",
        ylabel="Score",
        legend_title="Group",
        output_path="implizite_Analyse/results/run_1/group_scores_by_model.png"
    )

if __name__ == "__main__":
    main()