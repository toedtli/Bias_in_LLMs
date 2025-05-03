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
    required_columns = ["Score", "Language", "Statement ID", "Group", "Model", "Choice Set", "Formulation Key"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame. Check your CSV headers.")
    
    os.makedirs(output_folder, exist_ok=True)

    # Helper function to extract SEM data
    def get_means_and_sems(group_cols):
        grouped = df.groupby(group_cols)["Score"].agg(['mean', 'std', 'count'])
        grouped['sem'] = grouped['std'] / grouped['count']**0.5
        means = grouped['mean'].unstack()
        sems = grouped['sem'].unstack().fillna(0)
        return means, sems

    def plot_bar_chart_by_model(group_column):
        means, sems = get_means_and_sems([group_column, "Model"])

        # Sort numerically if Statement ID
        if group_column == "Statement ID":
            import re
            def extract_numeric(val):
                match = re.search(r'(\d+)', str(val))
                return int(match.group(1)) if match else 0
            means = means.reindex(sorted(means.index, key=extract_numeric))
            sems = sems.reindex(means.index)

        axis_label_map = {
            "Group": "Gruppe",
            "Language": "Sprache",
            "Statement ID": "Statement ID",
            "Choice Set": "Auswahlset",
            "Formulation Key": "Formulierungsschlüssel"
        }
        x_axis_label = axis_label_map.get(group_column, group_column)

        fig = plt.figure(constrained_layout=True, figsize=(8, 6))
        gs = gridspec.GridSpec(nrows=3, ncols=1, figure=fig, height_ratios=[0.1, 0.1, 1])
        
        ax_title = fig.add_subplot(gs[0, 0])
        ax_title.text(
            0.5, 0,
            f"Durchschnittliche Punktzahl nach {x_axis_label} (pro Modell)",
            ha='center', va='center', fontsize=12
        )
        ax_title.set_axis_off()

        ax_legend = fig.add_subplot(gs[1, 0])
        ax_legend.set_axis_off()

        ax_chart = fig.add_subplot(gs[2, 0])
        means.plot(kind="bar", ax=ax_chart, yerr=sems.values.T, capsize=4, legend=False)
        ax_chart.set_ylabel("Durchschnittliche Punktzahl")
        ax_chart.set_xlabel(x_axis_label)
        ax_chart.set_ylim(0, 100)

        handles, labels = ax_chart.get_legend_handles_labels()
        max_ncol = 4
        ncol = len(labels) if len(labels) <= max_ncol else max_ncol
        ax_legend.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=ncol)

        filename = f"factor_{group_column}_chart.png".replace(" ", "_")
        file_path = os.path.join(output_folder, filename)
        plt.savefig(file_path)
        plt.close()
        print(f"Saved chart: {file_path}")

    def plot_bar_chart_with_model_on_x(group_column):
        means, sems = get_means_and_sems(["Model", group_column])

        means = means.sort_index()
        sems = sems.reindex(means.index)

        import re
        def extract_numeric(val):
            match = re.search(r'(\d+)', str(val))
            return int(match.group(1)) if match else 0
        means = means.reindex(columns=sorted(means.columns, key=extract_numeric))
        sems = sems.reindex(columns=means.columns)

        axis_label_map = {
            "Group": "Gruppe",
            "Language": "Sprache",
            "Statement ID": "Statement ID",
            "Choice Set": "Auswahlset",
            "Formulation Key": "Formulierungsschlüssel"
        }
        x_axis_label = axis_label_map.get(group_column, group_column)

        fig = plt.figure(constrained_layout=True, figsize=(8, 6))
        gs = gridspec.GridSpec(nrows=3, ncols=1, figure=fig, height_ratios=[0.2, 0.2, 1])
        
        ax_title = fig.add_subplot(gs[0, 0])
        ax_title.text(
            0.5, 0.5,
            f"Durchschnittliche Punktzahl pro {x_axis_label} (Modelle auf der x-Achse)",
            ha='center', va='center', fontsize=12
        )
        ax_title.set_axis_off()

        ax_legend = fig.add_subplot(gs[1, 0])
        ax_legend.set_axis_off()

        ax_chart = fig.add_subplot(gs[2, 0])
        means.plot(kind="bar", ax=ax_chart, yerr=sems.values.T, capsize=4, legend=False)
        ax_chart.set_ylabel("Durchschnittliche Punktzahl")
        ax_chart.set_xlabel("Modell")
        ax_chart.set_ylim(0, 100)

        handles, labels = ax_chart.get_legend_handles_labels()
        max_ncol = 4
        ncol = len(labels) if len(labels) <= max_ncol else max_ncol
        ax_legend.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=ncol)

        filename = f"model_{group_column}_chart.png"
        file_path = os.path.join(output_folder, filename)
        plt.savefig(file_path)
        plt.close()
        print(f"Saved chart: {file_path}")

    # Generate charts for all groupings
    for group in ["Group", "Language", "Statement ID", "Choice Set", "Formulation Key"]:
        plot_bar_chart_by_model(group)
        plot_bar_chart_with_model_on_x(group)

if __name__ == "__main__":
    create_charts_from_csv(
        csv_file="explizite_Analyse/data/processed/scoring_processed_combined.csv",
        output_folder="explizite_Analyse/factors_analysis"
    )