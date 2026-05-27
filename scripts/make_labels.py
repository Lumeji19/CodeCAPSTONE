
import pandas as pd
from pathlib import Path


STABILITY_RANK_COLS = [
    "output_stability_rank",
    "structural_total_rank",
    "hyp_C_geom_std_rank",
    "hyp_C_mode_frequency_rank",
]
PERFORMANCE_RANK_COLS = [
    "accuracy_mean_rank",
    "accuracy_worst_rank",
    "auc_mean_rank",
    "auc_worst_rank",
]
ALL_RANK_COLS = STABILITY_RANK_COLS + PERFORMANCE_RANK_COLS

# Weight for stability metrics in the stability-weighted scheme.
STABILITY_WEIGHT = 2.0
PERFORMANCE_WEIGHT = 1.0

def pick_single_metric_winners(results_df, metric_rank_col):
  
    rows = []
    for dataset_name in results_df["dataset"].unique():
        block = results_df[results_df["dataset"] == dataset_name]
        winner = block.loc[block[metric_rank_col].idxmin(), "method"]
        rows.append({
            "dataset": dataset_name,
            "winner": winner,
        })
    return pd.DataFrame(rows)

def compute_average_ranks(results_df):
    """
    For each (dataset, method) row, compute two weighted-average ranks:
    one with equal weights, one with stability 2x
    """
    rows = []

    for dataset_name in results_df["dataset"].unique():
        block = results_df[results_df["dataset"] == dataset_name]

        for _, row in block.iterrows():
            equal_sum = 0.0
            for col in ALL_RANK_COLS:
                equal_sum += row[col]
            avg_rank_equal = equal_sum / len(ALL_RANK_COLS)

            stability_sum = 0.0
            for col in STABILITY_RANK_COLS:
                stability_sum += row[col]
            performance_sum = 0.0
            for col in PERFORMANCE_RANK_COLS:
                performance_sum += row[col]

            weighted_total = (
                STABILITY_WEIGHT * stability_sum
                + PERFORMANCE_WEIGHT * performance_sum
            )
            total_weight = (
                STABILITY_WEIGHT * len(STABILITY_RANK_COLS)
                + PERFORMANCE_WEIGHT * len(PERFORMANCE_RANK_COLS)
            )
            avg_rank_stability = weighted_total / total_weight

            rows.append({
                "dataset": dataset_name,
                "method": row["method"],
                "avg_rank_equal": avg_rank_equal,
                "avg_rank_stability": avg_rank_stability,
            })

    return pd.DataFrame(rows)

# get the labels for each dataset (depending on the scheme)
def pick_winners(avg_ranks_df):

    rows = []

    for dataset_name in avg_ranks_df["dataset"].unique():
        block = avg_ranks_df[avg_ranks_df["dataset"] == dataset_name]


        equal_winner = block.loc[block["avg_rank_equal"].idxmin(), "method"]
        stability_winner = block.loc[
            block["avg_rank_stability"].idxmin(), "method"
        ]

        rows.append({
            "dataset": dataset_name,
            "label_equal": equal_winner,
            "label_stability": stability_winner,
            # Binary labels: "balanced" if balanced won, else "other".
            "balanced_wins_equal":
                "balanced" if equal_winner == "balanced" else "other",
            "balanced_wins_stability":
                "balanced" if stability_winner == "balanced" else "other",
        })

    return pd.DataFrame(rows)



def build_tree_data(model):
    """
    model: 'lr' or 'svm'
    """
    results_path = Path(f"results/{model.upper()}_full_with_ranks.csv")
    features_path = Path(f"metadata/features_{model}.csv")
    out_path = Path(f"metadata/tree_data_{model}.csv")

    results_df = pd.read_csv(results_path)
    features_df = pd.read_csv(features_path)


    results_df = results_df[results_df["dataset"] != "MEAN"].copy()


    avg_ranks_df = compute_average_ranks(results_df)
    labels_df = pick_winners(avg_ranks_df)

        # Single-metric labels
    single_metrics = {
        "label_output_stab":   "output_stability_rank",
        "label_structural":    "structural_total_rank",
        "label_auc_worst":     "auc_worst_rank",
        "label_accuracy_worst":"accuracy_worst_rank",
    }
    for label_name, rank_col in single_metrics.items():
        single_df = pick_single_metric_winners(results_df, rank_col)
        single_df = single_df.rename(columns={"winner": label_name})
        single_df[f"{label_name}_balanced"] = single_df[label_name].apply(
            lambda m: "balanced" if m == "balanced" else "other"
        )
        labels_df = labels_df.merge(single_df, on="dataset", how="inner")


    tree_data = features_df.merge(labels_df, on="dataset", how="inner")

    # checking every feature dataset has a label
    if len(tree_data) != len(features_df):
        missing = set(features_df["dataset"]) - set(tree_data["dataset"])
        print(f"  WARNING: {len(missing)} datasets had no label: {missing}")

    tree_data.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(tree_data)} datasets)")

    # Quick summary of label distributions
    print(f"  label_equal distribution:")
    print(tree_data["label_equal"].value_counts().to_string())
    print(f"  label_stability distribution:")
    print(tree_data["label_stability"].value_counts().to_string())
    print(f"  balanced_wins_equal distribution:")
    print(tree_data["balanced_wins_equal"].value_counts().to_string())
    print(f"  balanced_wins_stability distribution:")
    print(tree_data["balanced_wins_stability"].value_counts().to_string())
    print()

    return tree_data


if __name__ == "__main__":
    for model in ["lr", "svm"]:
        print(f"=== {model.upper()} ===")
        build_tree_data(model)