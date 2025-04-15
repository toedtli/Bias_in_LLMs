import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def create_charts_from_csv(csv_file, output_folder):
    # 1. Read the CSV into a DataFrame
    df = pd.read_csv(csv_file)
    
    # 2. Create a column that shows "first->last" from the 'Choice Set' column
    df["Choice Set"] = df["Choice Set"].astype(str)
    df["choice_split"] = df["Choice Set"].str.split(",")
    df["choice_label"] = df["choice_split"].apply(
        lambda x: x[0] + "->" + x[-1] if len(x) > 1 else x[0]
    )
    df.drop(columns=["choice_split"], inplace=True)
    
    # 3. Map group names from English to German
    group_mapping = {
        "Kurds": "Kurden",
        "Palestinians": "Palästinenser",
        "Uyghurs": "Uiguren",
        "Catalans": "Katalanen",
        "Tibetans": "Tibeter",
        "Rohingya": "Rohingya"
    }
    df["Group"] = df["Group"].replace(group_mapping)

    # 4. Map Choice Set strings to shorter labels
    choice_set_mapping = {
        "['1', '2', '3', '4'](de)": "1 → 4 (de)",
        "['Ja', 'Eher Ja', 'Eher Nein', 'Nein'](de)": "Ja → Nein",
        "['Stimme zu', 'Stimme eher zu', 'Stimme eher nicht zu', 'Stimme nicht zu'](de)": "Stimme zu → Stimme nicht zu",
        "['1', '2', '3', '4'](en)": "1 → 4 (en)",
        "['Yes', 'Rather Yes', 'Rather No', 'No'](en)": "Yes → No",
        "['Agree', 'Rather agree', 'Rather disagree', 'Disagree'](en)": "Agree → Disagree"
    }
    df["Choice Set"] = df["Choice Set"].replace(choice_set_mapping)

    # 5. Reverse the score if the axis is "Bedrohungswahrnehmung"
    df.loc[df["Axis Name"] == "Bedrohungswahrnehmung", "Score"] = 100 - df["Score"]

    # 6. Ensure key columns exist in DataFrame. Adjust if needed.
    required_columns = ["Score", "Language", "Question ID", "Group", "Model", "Choice Set", "Formulation Key"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame. Check your CSV headers.")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    #############################################################
    # FUNCTION: Plot the bar charts by Model (in German)
    #############################################################
    def plot_bar_chart_by_model(group_column):
        """
        Creates a bar chart of average Score grouped by:
            - The specified group_column and Model.
        The figure layout is fixed into three rows:
            Row 1: Title (always at the top)
            Row 2: Legend (directly below the title)
            Row 3: Bar chart (automatically takes the remaining height)
        """
        # Group and pivot data
        grouped = df.groupby([group_column, "Model"])["Score"].mean().reset_index()
        pivoted = grouped.pivot(index=group_column, columns="Model", values="Score")

        # Sort numerically if using Question IDs
        if group_column == "Question ID":
            import re
            def extract_numeric(val):
                match = re.search(r'(\d+)', str(val))
                return int(match.group(1)) if match else 0
            pivoted = pivoted.reindex(sorted(pivoted.index, key=extract_numeric))

        # Map axis label to German if available
        axis_label_map = {
            "Group": "Gruppe",
            "Language": "Sprache",
            "Question ID": "Question ID",
            "Choice Set": "Auswahlset",
            "Formulation Key": "Formulation Key"
        }
        x_axis_label = axis_label_map.get(group_column, group_column)

        # Create figure with constrained layout
        fig = plt.figure(constrained_layout=True, figsize=(8, 6))
        gs = gridspec.GridSpec(nrows=3, ncols=1, figure=fig, height_ratios=[0.1, 0.1, 1])
        
        # Row 1: Title
        ax_title = fig.add_subplot(gs[0, 0])
        ax_title.text(
            0.5, 0,
            f"Durchschnittliche Punktzahl nach {x_axis_label} (pro Modell)",
            ha='center', va='center', fontsize=12
        )
        ax_title.set_axis_off()

        # Row 2: Legend (will be filled after plotting)
        ax_legend = fig.add_subplot(gs[1, 0])
        ax_legend.set_axis_off()

        # Row 3: Bar Chart
        ax_chart = fig.add_subplot(gs[2, 0])
        pivoted.plot(kind="bar", ax=ax_chart, legend=False)
        ax_chart.set_ylabel("Durchschnittliche Punktzahl")
        ax_chart.set_xlabel(x_axis_label)
        ax_chart.set_ylim(0, 100)

        # Grab legend handles and labels from the chart
        handles, labels = ax_chart.get_legend_handles_labels()
        # Limit the legend columns to a maximum of 4 to prevent excessive width
        max_ncol = 4
        ncol = len(labels) if len(labels) <= max_ncol else max_ncol
        ax_legend.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=ncol)

        # Save figure
        filename = f"factor_{group_column}_chart.png".replace(" ", "_")
        file_path = os.path.join(output_folder, filename)
        plt.savefig(file_path)
        plt.close()
        print(f"Saved chart: {file_path}")

    #############################################################
    # FUNCTION: Plot the bar chart with Models on the x-axis (in German)
    #############################################################
    def plot_bar_chart_with_model_on_x(group_column):
        """
        Creates a bar chart of average Score grouped by Model and another category.
        In this example, the data is grouped by Model and the specified group_column.
        This chart puts the Models on the x-axis.
        """


        # Group and pivot data by Model and the specified group_column
        grouped = df.groupby(["Model", group_column])["Score"].mean().reset_index()
        pivoted = grouped.pivot(index="Model", columns=group_column, values="Score")
        
        # Sort the rows (Models) alphabetically and the columns (e.g., Question ID) numerically
        pivoted = pivoted.sort_index()
        import re
        def extract_numeric(val):
            match = re.search(r'(\d+)', str(val))
            return int(match.group(1)) if match else 0
        pivoted = pivoted.reindex(columns=sorted(pivoted.columns, key=extract_numeric))
        
        axis_label_map = {
            "Group": "Gruppe",
            "Language": "Sprache",
            "Question ID": "Question ID",
            "Choice Set": "Auswahlset",
            "Formulation Key": "Formulation Key"
        }
        x_axis_label = axis_label_map.get(group_column, group_column)
        # Create figure with constrained layout
        fig = plt.figure(constrained_layout=True, figsize=(8, 6))
        gs = gridspec.GridSpec(nrows=3, ncols=1, figure=fig, height_ratios=[0.2, 0.2, 1])
        
        # Row 1: Title
        ax_title = fig.add_subplot(gs[0, 0])
        ax_title.text(
            0.5, 0.5,
            f"Durchschnittliche Punktzahl pro {x_axis_label} (Modelle auf der x-Achse)",
            ha='center', va='center', fontsize=12
        )
        ax_title.set_axis_off()

        # Row 2: Legend (filled after plotting)
        ax_legend = fig.add_subplot(gs[1, 0])
        ax_legend.set_axis_off()

        # Row 3: Bar Chart with Model on the x-axis
        ax_chart = fig.add_subplot(gs[2, 0])
        pivoted.plot(kind="bar", ax=ax_chart, legend=False)
        ax_chart.set_ylabel("Durchschnittliche Punktzahl")
        ax_chart.set_xlabel(x_axis_label)
        ax_chart.set_ylim(0, 100)

        # Grab legend handles and labels from the chart
        handles, labels = ax_chart.get_legend_handles_labels()
        max_ncol = 4
        ncol = len(labels) if len(labels) <= max_ncol else max_ncol
        ax_legend.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=ncol)

        # Save figure
        filename = f"model_{group_column}_chart.png"
        file_path = os.path.join(output_folder, filename)
        plt.savefig(file_path)
        plt.close()
        print(f"Saved chart: {file_path}")

    #############################################################
    # Create charts for Group, Language, Question ID, and Choice Set
    #############################################################
    plot_bar_chart_by_model("Group")
    plot_bar_chart_by_model("Language")
    plot_bar_chart_by_model("Question ID")
    plot_bar_chart_by_model("Choice Set")
    plot_bar_chart_by_model("Formulation Key")
    
    #############################################################
    # Create additional charts with Models on the x-axis for various groupings
    #############################################################
    plot_bar_chart_with_model_on_x("Group")
    plot_bar_chart_with_model_on_x("Language")
    plot_bar_chart_with_model_on_x("Question ID")
    plot_bar_chart_with_model_on_x("Choice Set")
    plot_bar_chart_with_model_on_x("Formulation Key")

if __name__ == "__main__":
    create_charts_from_csv(
        csv_file="explizite_Analyse/data/processed/scoring_processed_combined.csv",
        output_folder="explizite_Analyse/factors_analysis"
    )