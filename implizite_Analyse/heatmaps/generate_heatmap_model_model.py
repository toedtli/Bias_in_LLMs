import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

run = 'combined'

# 0. Create output directory if it doesn't exist
output_dir = f"implizite_Analyse/heatmaps/{run}"
os.makedirs(output_dir, exist_ok=True)

# 1. Read the CSV
df = pd.read_csv(f"implizite_Analyse/results/{run}/scoring_model_model.csv")

# 2. Pivot the data so that rows = Scorer Model, columns = Source Model
df_mean = df.pivot(index="Scorer Model", columns="Source Model", values="Score")
df_std = df.pivot(index="Scorer Model", columns="Source Model", values="SEM")


# 4. Prepare text annotations of the form "mean ± SEM"
annot_matrix = df_mean.copy()
for scorer in df_mean.index:
    for source in df_mean.columns:
        m = df_mean.loc[scorer, source]
        s = df_std.loc[scorer, source]
        annot_matrix.loc[scorer, source] = f"{m:.2f} ± {s:.2f}"

# 5. Create custom row and column labels that include average values
row_avgs = df_mean.mean(axis=1).round(2)
col_avgs = df_mean.mean(axis=0).round(2)

new_row_labels = [f"{idx}\nØ {row_avgs[idx]}" for idx in df_mean.index]
new_col_labels = [f"{col}\nØ {col_avgs[col]}" for col in df_mean.columns]

# 7. Plot the heatmap
plt.figure(figsize=(8, 3))  # Adjust for a similar aspect ratio as your example
ax = sns.heatmap(
    df_mean,
    cmap="Reds",
    vmin=0,  # Adjust if your data range is different
    vmax=35,  # Adjust if your data range is different
    annot=annot_matrix,
    fmt="",  # We've pre-formatted the annotation strings
    linecolor="white",
    cbar_kws={"shrink": 0.8, "label": "Score"}
)

# 8. Tidy up labels and title
ax.set_xticklabels(new_col_labels, rotation=0, ha="center")
ax.set_yticklabels(new_row_labels, rotation=0)
ax.set_xlabel("Source Model")
ax.set_ylabel("Scorer Model")
ax.set_title("Durchschnitt aller Bewertungen der Beschreibungen über alle Gruppen", pad=15)

plt.tight_layout()

# 9. Save the figure
output_path = os.path.join(output_dir, f"heatmap_model_model_{run}.png")
plt.savefig(output_path, dpi=300)
plt.close()