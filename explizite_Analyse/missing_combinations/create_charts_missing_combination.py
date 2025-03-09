import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Load the missing combinations CSV.
# ---------------------------
missing_csv_path = 'explizite_Analyse/missing_combinations/missing_combinations.csv'
df_missing = pd.read_csv(missing_csv_path)

# ---------------------------
# Aggregate counts by Choice Set, Group, and Model.
# ---------------------------
agg_data = df_missing.groupby(['Choice Set', 'Group', 'Model']).size().reset_index(name='Count')

# Map Choice Set and Group to categorical positions.
choice_sets = sorted(agg_data['Choice Set'].unique())
groups = sorted(agg_data['Group'].unique())
models = sorted(agg_data['Model'].unique())

x_mapping = {cs: i for i, cs in enumerate(choice_sets)}
y_mapping = {g: i for i, g in enumerate(groups)}

# Define colors for each model (using a colormap).
color_mapping = {}
colors = plt.get_cmap('tab10').colors
for i, model in enumerate(models):
    color_mapping[model] = colors[i % len(colors)]

# ---------------------------
# Create a bubble chart.
# ---------------------------
plt.figure(figsize=(12, 8))

# Plot each bubble with size proportional to its count.
for _, row in agg_data.iterrows():
    cs = row['Choice Set']
    grp = row['Group']
    mod = row['Model']
    count = row['Count']
    x = x_mapping[cs]
    y = y_mapping[grp]
    # Adjust bubble size factor if necessary.
    bubble_size = count * 50
    plt.scatter(x, y, s=bubble_size, color=color_mapping[mod], 
                alpha=0.6, edgecolor='black', label=mod)

# Create a deduplicated legend for Models.
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), title='Model')

# Set axis ticks and labels.
plt.xticks(list(x_mapping.values()), list(x_mapping.keys()), rotation=45, ha='right')
plt.yticks(list(y_mapping.values()), list(y_mapping.keys()))

plt.xlabel('Choice Set')
plt.ylabel('Group')
plt.title('Missing Combinations by Choice Set, Group, and Model')

plt.tight_layout()

# ---------------------------
# Save the plot to the specified file.
# ---------------------------
output_dir = 'explizite_Analyse/missing_combinations'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'overall_missing_combinations.png')
plt.savefig(output_path, dpi=300)
plt.close()