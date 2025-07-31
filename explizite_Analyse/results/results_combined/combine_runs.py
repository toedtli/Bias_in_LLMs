import pandas as pd
import ipdb
import sys,os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Added for seaborn-based heatmap

def extract_mean(cell):
    """
    Helper function to extract the numeric mean from a "mean±SEM" string.
    Returns NaN if the cell is "NaN" or if an error occurs.
    """
    try:
        if cell == "NaN" or pd.isnull(cell):
            return np.nan
        return float(cell.split("±")[0])
    except:
        return np.nan

def main(output_format = '.png',group_language='de'):
    # 1. Read in the three CSV files
    df1 = pd.read_csv("explizite_Analyse/results/results_run_1/results_run_1.csv")
    df2 = pd.read_csv("explizite_Analyse/results/results_run_2/results_run_2.csv")
    df3 = pd.read_csv("explizite_Analyse/results/results_run_3/results_run_3.csv")
    if group_language=='en':
        d = {'Uiguren':'Uyghurs','Katalanen':'Catalans','Kurden':'Curds','Tibeter':'Tibetans','Palästinenser':'Palestinians','Rohingya':'Rohingya'}
        df1['Group'] = df1['Group'].map(d)  
        df2['Group'] = df2['Group'].map(d)
        df3['Group'] = df3['Group'].map(d)

    # Concatenate the dataframes
    df = pd.concat([df1, df2, df3], ignore_index=True)
    print("Columns in df1:", df1.columns.tolist())
    print("Columns in df2:", df2.columns.tolist())
    print("Columns in df3:", df3.columns.tolist())
    print("Columns in concatenated df:", df.columns.tolist())
    
    # Ensure the required 'Axis Name' column exists
    if "Axis Name" not in df.columns:
        raise KeyError("Expected column 'Axis Name' not found in the data!")

    # 2. Create a merged dataframe by grouping by Model, Group, and Axis Name,
    #    and computing the mean for 'mean' and 'SEM'
    merged_df = df.groupby(["Model", "Group", "Axis Name"], as_index=False).agg({"mean": "mean", "SEM": "mean", "count": "sum"})

    # Save the merged csv file
    output_dir = "explizite_Analyse/results/results_combined"
    os.makedirs(output_dir, exist_ok=True)
    merged_csv_filename = os.path.join(output_dir, "scoring_combined.csv")
    merged_df.to_csv(merged_csv_filename, index=False)
    print(f"Saved merged CSV: {merged_csv_filename}")

    # Get the list of unique axis names from the merged dataframe
    axis_names = merged_df["Axis Name"].unique()
    print("Found Axis Names:", axis_names)

    # Define the order of groups for the final table
    groups_de = ["Katalanen", "Kurden", "Palästinenser", "Rohingya", "Tibeter", "Uiguren"]
    if group_language=='de':
        groups = ["Katalanen", "Kurden", "Palästinenser", "Rohingya", "Tibeter", "Uiguren"]
    elif group_language=='en':
        groups = ["Catalans", "Curds", "Palestinians", "Rohingya", "Tibetans", "Uyghurs"]
        d_axis = {'Bedrohungswahrnehmung':'Perception of Threat', 'Empathie':'Empathy', 'Kulturelle Wertschätzung':'Cultural Appreciation','Politische Anerkennung':'Political Recognition', 'Unterstützungsbedarf':'Need for Support'} 
        axis_names_en = list(d_axis.values())  
        axis_names_de = list(d_axis.keys())  
#        axis_names = list(d_axis.values())  
#        axis_names = pd.Series(axis_names)
    else:
        raise ValueError('Unknown group language')

    # Loop over each axis to create per-axis CSVs and heatmaps
    df_numeric_list= []
    df_text_list= []
    col_labels_list=[]
    row_labels_list=[]
    vmin_list=[]
    cmap_list=[]
    for axis_de,axis_en in zip(axis_names_de,axis_names_en):
        print(f"\nProcessing Axis Name: {axis_de}")
        # Filter the merged dataframe for the current axis
        axis_df = merged_df[merged_df["Axis Name"] == axis_de]

        # Pivot to wide format so that each group becomes its own column.
        pivoted = axis_df.pivot(index="Model", columns="Group", values=["mean", "SEM", 'count'])
        print("Pivoted columns (flattened):", pivoted.columns.to_flat_index().tolist())

        # Reindex the pivoted columns so that groups appear in the desired order
        pivoted = pivoted.reindex(groups, axis=1, level=1)
        
        # -- Calculate numeric averages from the pivoted data --
        # Create separate DataFrames for mean and SEM from pivoted.
        df_mean_pivot = pivoted.xs("mean", axis=1, level=0)
        df_sem_pivot = pivoted.xs("SEM", axis=1, level=0)
        df_count_pivot = pivoted.xs("count", axis=1, level=0)
        # Compute average values for rows and columns.
        # Compute average values for rows and columns (no decimal point)
        row_mean_avgs = df_mean_pivot.mean(axis=1).round(0).astype(int)
        row_sem_avgs = df_sem_pivot.mean(axis=1).round(0).astype(int)
        row_counts = df_count_pivot.sum(axis=1).round(0).astype(int)
        col_mean_avgs = df_mean_pivot.mean(axis=0).round(0).astype(int)
        col_sem_avgs = df_sem_pivot.mean(axis=0).round(0).astype(int)
        col_counts = df_count_pivot.sum(axis=0).round(0).astype(int)

        # Create a new DataFrame that combines mean and SEM into a single string "mean±SEM"
        final_df = pd.DataFrame(index=pivoted.index)
        for group in groups:
            final_df[group] = pivoted.apply(
                lambda row: f"{int(round(row[('mean', group)]))} ± {int(round(row[('SEM', group)]))}\n n={int(round(row[('count', group)]))}" 
                            if pd.notnull(row[('mean', group)]) and pd.notnull(row[('SEM', group)])
                            else "NaN",
                axis=1
            )

        # ----- Create Heatmap for the current CSV using Seaborn -----
        # Set 'Model' as index for heatmap plotting
        heat_df = final_df.copy()
        if "Model" in heat_df.columns:
            heat_df.set_index("Model", inplace=True)
        
        # Create a numeric DataFrame by extracting the numeric part (mean) from "mean±SEM"
        df_numeric = heat_df.applymap(extract_mean)
        # Use the text in heat_df as the annotations
        df_text = heat_df.copy()

        # Create custom row and column labels that include both the average mean and SEM.
        new_row_labels = [f"{idx}\n{row_mean_avgs[idx]} ± {row_sem_avgs[idx]}\n n={row_counts[idx]}" 
                          for idx in df_mean_pivot.index]
        new_col_labels = [f"{col}\n{col_mean_avgs[col]} ± {col_sem_avgs[col]}\n n={col_counts[col]}" 
                          for col in df_mean_pivot.columns]

        # Determine figure dimensions (can be adjusted for your needs)
        fig_width = max(10, df_numeric.shape[1] * 1.2)
        fig_height = max(4, df_numeric.shape[0] * 0.5)

        df_numeric_list.append(df_numeric)
        df_text_list.append(df_text)
        col_labels_list.append(new_col_labels) 
        row_labels_list.append(new_row_labels) 

        plt.figure(figsize=(fig_width, fig_height))
        
        # Generate the heatmap using Seaborn
        # Choose colormap and vmin based on the axis name (example condition)
        if axis_de == "Bedrohungswahrnehmung":
            cmap = 'Reds'
            vmin = 25
        else:
            cmap = 'Blues'
            vmin = 75
        
        vmin_list.append(vmin)
        cmap_list.append(cmap)
        ax = sns.heatmap(
            df_numeric,
            cmap=cmap,
            vmin=vmin,
            annot=df_text,
            fmt="",
            linecolor="white",
            # cbar_kws={"shrink": 0.8, "label": "Score"}
            cbar=True

        )
        
        # Update tick labels with custom labels that include averages for both mean and SEM.
        ax.set_xticklabels(new_col_labels, rotation=0, ha="center", fontsize=8)
        ax.set_yticklabels(new_row_labels, rotation=0, fontsize=8)
        if group_language=='de':
            axis=axis_de
            ax.set_title(f"{axis}\n(Durchschnittlicher Zustimmungs-Score ± Stand. Abw. vom Mittelwert, n = Beantwortete Kombinationen)", pad=15)
            ax.set_xlabel("Gruppen")
            ax.set_ylabel("Modelle")
        elif group_language=='en':
            axis=axis_en
            ax.set_title(f"{axis}\n(Mean Consent Score ± Std Dev. of the mean, n = number of answered questions)", pad=15)
            ax.set_xlabel("Group")
            ax.set_ylabel("Model")
            #iax.set_title(f"{axis}\n(Durchschnittlicher Zustimmungs-Score ± Stand. Abw. vom Mittelwert, n = Beantwortete Kombinationen)", pad=15)

        plt.tight_layout(h_pad=30)
        
        # Save the generated heatmap image
        heatmap_output_dir = os.path.join(output_dir, "heatmaps_combined")
        os.makedirs(heatmap_output_dir, exist_ok=True)
        heatmap_file = os.path.join(heatmap_output_dir, f"heatmap_{axis}.png")
        plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')

        plt.close()
        print(f"Saved heatmap for Axis Name '{axis}' as: {heatmap_file}")
        print('getcwd:',os.getcwd())
        print('argv:',sys.argv)

    fig, axs = plt.subplots(3, 2, figsize=(30, 20))
    axs = axs.flatten()
    print(len(axs),len(df_numeric_list),len(df_text_list),len(vmin_list),len(cmap_list),len(row_labels_list),len(axis_names) )
    ipdb.set_trace()
    for df_numeric,df_text,vmin,cmap,new_row_labels,new_col_labels,ax in zip(df_numeric_list, df_text_list, vmin_list,cmap_list, row_labels_list, col_labels_list,axs[:-1]):
        sns.heatmap(
            df_numeric,
            cmap=cmap,
            vmin=vmin,
            annot=df_text,
            fmt="",
            linecolor="white",
            # cbar_kws={"shrink": 0.8, "label": "Score"}
            cbar=True,
            ax=ax
        )
    
        # Update tick labels with custom labels that include averages for both mean and SEM.
        ax.set_xticklabels(new_col_labels, rotation=0, ha="center", fontsize=8)
        ax.set_yticklabels(new_row_labels, rotation=0, fontsize=8)
        ax.set_xlabel("Gruppen")
        ax.set_ylabel("Modelle")
        ax.set_title(f"{axis}\n(Durchschnittlicher Zustimmungs-Score ± Stand. Abw. vom Mittelwert, n = Beantwortete Kombinationen)", pad=15)
    fig.delaxes(axs[-1])
    fig.subplots_adjust(hspace=0.3,wspace=-0.0)
    plt.savefig('test.svg')


if __name__ == "__main__":
    #main()
    main(output_format='.svg',group_language='en')
    #main(output_format='.png',group_language='en')
