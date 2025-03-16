#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

def krippendorff_alpha_two(rater1, rater2):
    """
    Compute Krippendorff's alpha for interval data for two raters.
    rater1 and rater2 are 1D numpy arrays of scores (for the same set of items).
    For two raters, one common formulation is:
    
         alpha = 1 - (mean squared difference between raters) / (variance of all ratings)
    
    If variance is zero, returns 1.0 (perfect agreement).
    """
    if len(rater1) < 2:
        return np.nan  # Not enough items to compute an estimate
    # observed disagreement: average squared difference between the two raters
    diff = np.mean((rater1 - rater2) ** 2)
    # expected disagreement: variance of all ratings from both raters
    combined = np.concatenate([rater1, rater2])
    var = np.var(combined, ddof=0)
    if var == 0:
        return 1.0  # perfect agreement if there is no variation at all
    return 1 - (diff / var)

def compute_pairwise_alpha(pivot_df):
    """
    Given a pivoted DataFrame with rows as items and columns as scorer models,
    compute the pairwise Krippendorff's alpha for every pair of scorer models,
    using only the items (rows) where both provided a rating.
    
    Returns a dict with keys as tuples (scorer_i, scorer_j) and the computed alpha.
    """
    scorers = pivot_df.columns
    pairwise = {}
    for i in range(len(scorers)):
        for j in range(i+1, len(scorers)):
            scorer_i = scorers[i]
            scorer_j = scorers[j]
            # Select only the items where both scorers have provided a rating.
            common = pivot_df[[scorer_i, scorer_j]].dropna()
            if len(common) >= 2:
                alpha_val = krippendorff_alpha_two(common[scorer_i].values, common[scorer_j].values)
            else:
                alpha_val = np.nan
            pairwise[(scorer_i, scorer_j)] = alpha_val
    return pairwise

def average_alpha_for_scorer(pairwise_dict, scorer):
    """
    For a given scorer (rater), average the pairwise alphas computed
    (across all pairs that include that scorer).
    """
    values = []
    for (s1, s2), alpha_val in pairwise_dict.items():
        if scorer in (s1, s2) and not np.isnan(alpha_val):
            values.append(alpha_val)
    if values:
        return np.mean(values)
    else:
        return np.nan

def main():
    # Read the CSV file.
    df = pd.read_csv("implizite_Analyse/data/scoring_processed/scoring_processed_combined.csv")
    # Exclude rows with missing Score.
    df = df.dropna(subset=["Score"])
    
    # Create an identifier for each text.
    # If you have a dedicated text ID column, you can use that.
    # Here we combine "Source Model" and "Model Response" (cast to string) as a proxy.
    df["item_id"] = df["Source Model"].astype(str) + "_" + df["Model Response"].astype(str)
    
    results = []
    # We assume that for each Source Model, many texts were rated by several Scorer Models.
    for source_model, group in df.groupby("Source Model"):
        # Create a pivot table with rows as items and columns as Scorer Models; 
        # the Score column provides the ratings.
        pivot = group.pivot_table(index="item_id", columns="Scorer Model", values="Score", aggfunc="mean")
        
        # Compute pairwise Krippendorff's alpha between every two scorers.
        pairwise = compute_pairwise_alpha(pivot)
        
        # For each scorer in this source model, average its pairwise alphas.
        for scorer in pivot.columns:
            avg_alpha = average_alpha_for_scorer(pairwise, scorer)
            results.append({
                "Source Model": source_model,
                "Scorer Model": scorer,
                "Average Pairwise Alpha": avg_alpha
            })
    
    # Create a DataFrame for the results.
    results_df = pd.DataFrame(results)
    
    # Ensure the output directory exists.
    output_dir = "implizite_Analyse"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "krippendorff.csv")
    
    # Save the results to CSV.
    results_df.to_csv(output_path, index=False)
    print(f"Krippendorff's alpha values saved to {output_path}")

if __name__ == "__main__":
    main()