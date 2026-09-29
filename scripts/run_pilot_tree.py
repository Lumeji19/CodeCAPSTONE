import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loading import load_dataset
from src.models.DecisionTreeStandard import StandardTree
from src.models.DecisionTreeStable import StableTree
from src.models.DecisionTreeBalanced import BalancedTree
from src.experiment.bootstrap_tree import run_bootstrap_experiment
from src.experiment.metrics_tree import compute_all_methods_metrics

PILOT_DATASETS = [
    # paste your full 66-dataset list here, same as run_pilot_LR.py
]

METHODS = {
    "standard": StandardTree,
    "stable":   StableTree,
    "balanced": BalancedTree,
}

PARAM_GRIDS = {
    "standard": {
        "depth": [3, 5, 7],
    },
    "stable": {
        "k": [0.7, 0.75, 0.8, 0.85, 0.9],
        "depth": [3, 5, 7],
    },
    "balanced": {
        "k": [0.7, 0.75, 0.8, 0.85, 0.9],
        "depth": [3, 5, 7],
    },
}

N_BOOTSTRAPS = 100
TEST_SIZE = 0.1
SEED = 42
SCHEME = "subsample"

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results_tree"
RESULTS_DIR.mkdir(exist_ok=True)


def run_one_dataset(dataset_name):
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    X, y, info = load_dataset(dataset_name)
    print(f"n_rows = {info['n_rows']}, n_features = {info['n_features']}")

    X_train_full, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

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
        scheme=SCHEME,
        seed=SEED,
    )

    metrics = compute_all_methods_metrics(results)

    rows = []
    for method_name, method_metrics in metrics.items():
        row = {"dataset": dataset_name, "method": method_name}
        row.update(method_metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / f"{dataset_name}_{SCHEME}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return df


def main():
    single_dataset = sys.argv[1] if len(sys.argv) > 1 else None
    datasets_to_run = [single_dataset] if single_dataset else PILOT_DATASETS

    all_results = []
    for dataset_name in datasets_to_run:
        df = run_one_dataset(dataset_name)
        all_results.append(df)

    if single_dataset is None:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = RESULTS_DIR / f"pilot_{SCHEME}_combined.csv"
        combined.to_csv(combined_path, index=False)

        numeric_cols = combined.select_dtypes(include="number").columns.tolist()
        mean_rows = combined.groupby("method")[numeric_cols].mean().reset_index()
        mean_rows["dataset"] = "MEAN"
        column_order = ["dataset"] + [c for c in combined.columns if c != "dataset"]
        mean_rows = mean_rows.reindex(columns=column_order)

        final_combined = pd.concat([combined, mean_rows], ignore_index=True)
        final_combined.to_csv(combined_path, index=False)
        print(f"Combined results: {combined_path}")


if __name__ == "__main__":
    main()