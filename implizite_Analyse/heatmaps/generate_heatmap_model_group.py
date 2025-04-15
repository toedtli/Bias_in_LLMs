import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

run = 'run_2_3'

# 0. Create output directory if it doesn't exist
output_dir = f"implizite_Analyse/heatmaps/{run}"
os.makedirs(output_dir, exist_ok=True)

# 1. Read the CSV
df = pd.read_csv(f"implizite_Analyse/results/{run}/scoring_model_group.csv")

# 2. Pivot the data so that rows = Scorer Model, columns = Source Model
df_mean = df.pivot(index="Source Model", columns="Group", values="mean").round(0).astype(int)
df_std = df.pivot(index="Source Model", columns="Group", values="sem").round(0).astype(int)
df_count = df.pivot(index="Source Model", columns="Group", values="count")


# 4. Prepare text annotations of the form "mean ± SEM"
annot_matrix = df_mean.copy()
for scorer in df_mean.index:
    for source in df_mean.columns:
        m = df_mean.loc[scorer, source]
        s = df_std.loc[scorer, source]
        c = df_count.loc[scorer, source]
        annot_matrix.loc[scorer, source] = f"{m} ± {s} \n n={c}"

# 5. Create custom row and column labels that include average values
row_avgs = df_mean.mean(axis=1).round(0).astype(int)
col_avgs = df_mean.mean(axis=0).round(0).astype(int)
row_sems = df_std.mean(axis=1).round(0).astype(int)
col_sems = df_std.mean(axis=0).round(0).astype(int)
row_count = df_count.sum(axis=1).round(0).astype(int)
col_count = df_count.sum(axis=0).round(0).astype(int)

new_row_labels = [f"{idx}\nØ {row_avgs[idx]} ± {row_sems[idx]}\n n={row_count[idx]}" for idx in df_mean.index]
new_col_labels = [f"{col}\nØ {col_avgs[col]} ± {col_sems[col]}\n n={col_count[col]}" for col in df_mean.columns]

# 7. Plot the heatmap
plt.figure(figsize=(9, 4))  # Adjust for a similar aspect ratio as your example
ax = sns.heatmap(
    df_mean,
    cmap="Reds",
    vmin=5,  # Adjust if your data range is different
    vmax=35,  # Adjust if your data range is different
    annot=annot_matrix,
    fmt="",  # We've pre-formatted the annotation strings
    linecolor="white",
    cbar=False
    # cbar_kws={"shrink": 0.8, "label": "Bias-Score"}
)

# 8. Tidy up labels and title
ax.set_xticklabels(new_col_labels, rotation=0, ha="center", fontsize=8)
ax.set_yticklabels(new_row_labels, rotation=0, fontsize=8)
ax.set_xlabel("Gruppe", fontsize=9)
ax.set_ylabel("Beschreibende Modelle", fontsize=9)
ax.set_title("Bias-Score der Beschreibungen pro Gruppe \n(Durchschnittlicher Bias-Score ± Stand. Abw. vom Mittelwert, n= Anzahl bewertete Beschreibungen)", pad=15)

plt.tight_layout()

# 9. Save the figure
output_path = os.path.join(output_dir, f"heatmap_model_group_{run}.png")
plt.savefig(output_path, dpi=300)
plt.close()