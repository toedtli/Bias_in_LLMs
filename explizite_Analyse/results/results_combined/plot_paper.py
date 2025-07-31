import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
def make_plot(df_numeric_list,df_text_list,vmin_list,cmap_list,row_labels_list,col_labels_list,axis_names_en,num_heatmaps = 5):


    fig, axs = plt.subplots(3, 2, figsize=(30, 20))
    axs_flat = list(axs.flatten()) # Flatten the 2D array of axes into a 1D list

    # Hide the last two subplots (bottom row, left and right)
    axs_flat[-1].set_visible(False)
    axs_flat[-2].set_visible(False)

    # --- Calculate position for the centered 5th plot ---
    bbox_bottom_left = axs_flat[4].get_position()
    bbox_bottom_right = axs_flat[5].get_position()

    center_x_of_row = (bbox_bottom_left.x0 + bbox_bottom_right.x1) / 2
    plot_width = bbox_bottom_left.width +0.095

    new_left = center_x_of_row - (plot_width / 2)

    # --- ADJUSTMENT HERE: Calculate new_bottom to add vertical spacing ---
    # We need to explicitly lower the new_bottom position of the central plot.
    # You can try adjusting this value (e.g., subtracting a small number like 0.05 or 0.1)
    # or derive it from an existing plot's bottom and desired spacing.

    # Let's take the bottom of the previous row (axs_flat[2] or axs_flat[3])
    # and add some padding.
    # The `y0` of a subplot is its bottom edge. The `height` is its vertical extent.
    # The space between rows is implicitly handled by subplots.
    # If `new_bottom` is `bbox_bottom_left.y0`, it means it's exactly at the bottom of the grid row.
    # To prevent overlap with plots above, we need to push it down or increase figure's vertical spacing.

    # Let's try to infer a reasonable 'bottom' for the new axis based on the figure's bottom margin
    # and a desired vertical spacing from the row above.

    # This part is a bit tricky with mixed `subplots` and `add_axes`
    # Let's try to simply decrease the `new_bottom` to push it down.
    # The amount to subtract will depend on your figsize and desired spacing.
    vertical_offset = 0.05 # Adjust this value (e.g., 0.02, 0.05, 0.1) as needed for spacing
    new_bottom = bbox_bottom_left.y0 - vertical_offset
    new_height = bbox_bottom_left.height # Keep the height the same

    # Make sure new_bottom isn't negative or too low for the labels
    # If `new_bottom` becomes too small, it will push text outside the figure.
    if new_bottom < 0.05: # Arbitrary small value for minimum bottom margin
        new_bottom = 0.05

    # Add the new axis with calculated position
    center_ax = fig.add_axes([new_left, new_bottom, plot_width, new_height])

    # --- Assemble the list of axes to plot on ---
    axes_to_fill = axs_flat[:4] + [center_ax]

    # --- Verification (Good practice to keep these) ---
    print(f"Number of axes to fill: {len(axes_to_fill)}")
    print(f"Number of data sets available: {len(df_numeric_list)}")

    if len(axes_to_fill) != len(df_numeric_list):
        print("WARNING: Mismatch between number of axes and number of data sets!")
        print("This could lead to empty plots or missing data.")

    # Plotting loop
    for i, (df_numeric, df_text, vmin, cmap, new_row_labels, new_col_labels, ax, axis_en) in enumerate( zip(df_numeric_list, df_text_list, vmin_list, cmap_list, row_labels_list, col_labels_list, axes_to_fill, axis_names_en)):
        print(f"Plotting data set {i+1} on axis: {axis_en}") # Trace which data goes where
        sns.heatmap(
        df_numeric,
        cmap=cmap,
        vmin=vmin,
        annot=df_text,
        fmt="",
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Score"},
        annot_kws={"fontsize": 18},
        cbar=True,
        ax=ax
        )
        ax.set_xticklabels(new_col_labels, rotation=0, ha="center", fontsize=15)
        ax.set_yticklabels(new_row_labels, rotation=0, fontsize=15)
        ax.set_title(f"{axis_en}\n(Mean Consent Score ± Std Dev., n = number of answered questions)", pad=15, fontsize=18)
        ax.set_xlabel("Group", fontsize=15)
        ax.set_ylabel("Model", fontsize=15)

    # --- IMPORTANT: Call plt.tight_layout() and fig.subplots_adjust() ONLY ONCE, AT THE END ---
    # Remove plt.tight_layout from the loop!

    # Apply tight_layout first to try and get a good default spacing for the grid plots
    # We use rect to define the area for tight_layout, leaving space for the custom subplot below if needed
    # If you use tight_layout, it might override subplots_adjust for the grid plots.
    # For custom axes, it's often better to *only* use subplots_adjust or manage margins manually.

    # Let's remove the global tight_layout if you're explicitly using subplots_adjust.
    # The conflict with tight_layout and manually added axes is common.

    # Adjust overall subplot parameters
    # Values below are a starting point, you might need to fine-tune them.
    # The 'bottom' value here should accommodate your new_bottom of center_ax
    fig.subplots_adjust(
    hspace=0.5,   # Vertical space between subplots in grid (horizontal space between columns)
    wspace=0.1,   # Horizontal space between subplots in grid (vertical space between rows)
    left=0.08,    # Left margin
    right=0.98,   # Right margin
    top=0.92,     # Top margin
    bottom=0.08   # Bottom margin (ensure this is low enough for your custom plot's labels)
    )

    # You might still need to adjust 'vertical_offset' for 'new_bottom' of center_ax
    # and 'bottom' in subplots_adjust to get the perfect fit.
    # If titles overlap, increase 'hspace' or decrease 'top'.
    # If column labels overlap, increase 'wspace' or decrease 'right'.

    plt.savefig('paper_plot.svg')
    plt.savefig('paper_plot.png', dpi=300)
    plt.show()

if __name__=='__main__':
    # --- Dummy Data for Demonstration ---
    # Ensure this list has 5 elements for 5 plots!
    num_heatmaps=5
    df_numeric_list = [pd.DataFrame(np.random.rand(3, 3) * (i + 1)) for i in range(num_heatmaps)] # Vary data to see clear differences
    df_text_list = [df_numeric.round(2).astype(str) for df_numeric in df_numeric_list]
    vmin_list = [0 for _ in range(num_heatmaps)]
    cmap_list = ["viridis" for _ in range(num_heatmaps)]
    row_labels_list = [[f"Row {r+1}" for r in range(3)] for _ in range(num_heatmaps)]
    col_labels_list = [[f"Col {c+1}" for c in range(3)] for _ in range(num_heatmaps)]
    axis_names_en = [f"Plot {i+1}" for i in range(num_heatmaps)]
    # --- End Dummy Data ---
    make_plot(df_numeric_list,df_text_list,vmin_list,cmap_list,row_labels_list,col_labels_list,axis_names_en,num_heatmaps = num_heatmaps)
