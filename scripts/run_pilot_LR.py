

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loading import load_dataset
from src.models.LRstandard import StandardLR
from src.models.LRstable import StableLR
from src.models.LRbalanced import BalancedLR
from src.experiment.bootstrap import run_bootstrap_experiment
from src.experiment.metrics import compute_all_methods_metrics

PILOT_DATASETS = [

    "abalone_19_vs_10_11_12_13",
    "ecoli_0_1_3_7_vs_2_6",
    "kddcup_land_vs_portsweep",
    "kr_vs_k_zero_vs_eight",
    "poker_8_vs_6",
    "yeast6",
    "winequality_white_3_vs_7",
    "winequality_red_8_vs_6_7",
    "winequality_red_3_vs_5",
    "poker_8_9_vs_6",
    # done in differet runs since it takes too long

]

METHODS = {
    "standard": StandardLR,
    "stable":   StableLR,
    "balanced": BalancedLR,
}

PARAM_GRIDS = {
    "standard": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], #maybe change
    },
    "stable": {
        "k": [0.7, 0.75, 0.8, 0.85, 0.9], #maybe change
        "C": [0.001,0.01, 0.1, 1.0, 10.0, 100.0], #maybe change
    },
    "balanced": {
        "k": [0.7, 0.75, 0.8, 0.85, 0.9], #maybe change
        "C": [0.001,0.01, 0.1, 1.0, 10.0, 100.0], #maybe change
    },
}

N_BOOTSTRAPS = 100
TEST_SIZE = 0.1
SEED = 42
SCHEME = "subsample"

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def run_one_dataset(dataset_name: str) -> pd.DataFrame:

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")
    X, y, info = load_dataset(dataset_name)
    print(f"n_rows = {info['n_rows']}, n_features = {info['n_features']}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    # Scale features, fit scaler on training only and transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert pandas series to numpy arrays for the bootstrap func
    y_train_np = np.asarray(y_train)
    y_test_np = np.asarray(y_test)

    print(f"Running {N_BOOTSTRAPS} bootstrap rounds for {len(METHODS)} methods...")
    results = run_bootstrap_experiment(
        X_train_full=X_train_scaled,
        y_train=y_train_np,
        X_test=X_test_scaled,
        y_test=y_test_np,
        methods=METHODS,
        param_grids=PARAM_GRIDS,
        n_bootstraps=N_BOOTSTRAPS,
        seed=SEED,
        scheme=SCHEME
    )

    all_metrics = compute_all_methods_metrics(results)

    # dataset with all the information
    rows = []
    for method_name, metrics in all_metrics.items():
        row = {"dataset": dataset_name, "IR": info["imbalance_ratio"], "method": method_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)


    csv_path = RESULTS_DIR / f"{dataset_name}_{SCHEME}.csv"
    df.to_csv(csv_path, index=False)

    
    print_dataset_summary(df)

    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    
    cols_to_show = [
        "method",
        "auc_mean", "auc_std", "auc_worst",
        "accuracy_mean", "accuracy_std", "accuracy_worst",
        "output_stability",
        "structural_beta_std",
        "structural_beta0_std",
        "fit_time_mean_s", "fit_time_std_s", "fit_time_total_s",
    ]
    # (different methods have different hyperparameters)
    hyp_cols = [c for c in df.columns if c.startswith("hyp_")]
    cols_to_show += hyp_cols


    cols_to_show = [c for c in cols_to_show if c in df.columns]

  
    summary = df[cols_to_show].copy()
    for col in summary.select_dtypes(include="number").columns:
        summary[col] = summary[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")

    print("\nSummary:")
    print(summary.to_string(index=False))

def main():
    all_results = []
    for dataset_name in PILOT_DATASETS:
        df = run_one_dataset(dataset_name)
        all_results.append(df)

    # Combined CSV across all datasets (names 'pilot" since i started with a trial and then ended up using the same version)
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = RESULTS_DIR / f"pilot_{SCHEME}_combined.csv"
    combined.to_csv(combined_path, index=False)

    # mean per method across datasets
    numeric_cols = combined.select_dtypes(include="number").columns.tolist()
    mean_rows = combined.groupby("method")[numeric_cols].mean().reset_index()
    mean_rows["dataset"] = "MEAN"  # placeholder so the column lines up
    column_order = ["dataset"] + [c for c in combined.columns if c != "dataset"]
    mean_rows = mean_rows.reindex(columns=column_order)

    # original combined results, then mean rows at the bottom.
    final_combined = pd.concat([combined, mean_rows], ignore_index=True)
    final_combined.to_csv(combined_path, index=False)

    print(f"Combined results: {combined_path}")


if __name__ == "__main__":
    main()