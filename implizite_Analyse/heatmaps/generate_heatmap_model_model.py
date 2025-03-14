import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

run = 'run_2_1'

# 0. Create output directory if it doesn't exist
output_dir = f"implizite_Analyse/heatmaps/{run}"
os.makedirs(output_dir, exist_ok=True)

# 1. Read the CSV
df = pd.read_csv(f"implizite_Analyse/results/{run}/scoring_model_model.csv")

# 2. Pivot the data so that rows = Source Model, columns = Scorer Model
df_mean = df.pivot(index="Source Model", columns="Scorer Model", values="Score")
df_std = df.pivot(index="Source Model", columns="Scorer Model", values="SEM")

# 4. Prepare text annotations of the form "mean ± SEM"
annot_matrix = df_mean.copy()
for source in df_mean.index:
    for scorer in df_mean.columns:
        m = df_mean.loc[source, scorer]
        s = df_std.loc[source, scorer]
        annot_matrix.loc[source, scorer] = f"{m:.2f} ± {s:.2f}"

# 5. Create custom row and column labels that include average values
row_avgs = df_mean.mean(axis=1).round(2)  # Average for each Source Model (rows)
col_avgs = df_mean.mean(axis=0).round(2)  # Average for each Scorer Model (columns)

new_row_labels = [f"{idx}\nØ {row_avgs[idx]}" for idx in df_mean.index]
new_col_labels = [f"{col}\nØ {col_avgs[col]}" for col in df_mean.columns]

# 7. Plot the heatmap
plt.figure(figsize=(8, 3))
ax = sns.heatmap(
    df_mean,
    cmap="Reds",
    vmin=5,
    vmax=35,
    annot=annot_matrix,
    fmt="",
    linecolor="white",
    cbar_kws={"shrink": 0.8, "label": "Score"}
)

# 8. Tidy up labels and title
ax.set_xticklabels(new_col_labels, rotation=0, ha="center")
ax.set_yticklabels(new_row_labels, rotation=0)
ax.set_xlabel("Scorer Model")
ax.set_ylabel("Source Model")
ax.set_title("Durchschnitt aller Bewertungen der Beschreibungen über alle Gruppen", pad=15)

plt.tight_layout()

# 9. Save the figure
output_path = os.path.join(output_dir, f"heatmap_model_model_{run}.png")
plt.savefig(output_path, dpi=300)
plt.close()