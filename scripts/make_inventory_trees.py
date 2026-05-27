from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.data_loading import load_dataset


def main():
    
    svm_results = pd.read_csv("results/pilot_svm_combined_40datasets.csv")
    svm_results = svm_results[svm_results["dataset"] != "MEAN"]
    dataset_names = sorted(svm_results["dataset"].unique())
    print(f"Found {len(dataset_names)} datasets")

    
    structural_rows = []
    for name in dataset_names:
        X, y, info = load_dataset(name)
        structural_rows.append({
            "dataset":    name,
            "source":     info["source"],
            "n_rows":     info["n_rows"],
            "p_encoded":  info["n_features"],
            "IR":         info["imbalance_ratio"],
            "n_minority": info["n_minority"],
        })
        print(f"  OK: {name}")
    structural_df = pd.DataFrame(structural_rows)

    
    lr_results = pd.read_csv("results/pilot_LR_combined_40datasets.csv")

    for method_label, results in [("lr", lr_results), ("svm", svm_results)]:
        
        standard = results[results["method"] == "standard"][[
            "dataset", "auc_mean", "accuracy_mean",
            "output_stability", "structural_total", "accuracy_worst",
        ]].rename(columns={
            "auc_mean":          "standard_auc",
            "accuracy_mean":     "standard_accuracy",
            "output_stability":  "standard_output_stab",
            "structural_total":  "standard_structural",
            "accuracy_worst":    "standard_accuracy_worst",
        })

        
        features = structural_df.merge(standard, on="dataset", how="left")

        out = Path(f"metadata/features_{method_label}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(out, index=False)
        print(f"Saved: {out}  ({len(features)} datasets)")

df = pd.read_csv("metadata/features_lr.csv")
df["IR"] = pd.to_numeric(df["IR"], errors="coerce")
df = df.sort_values(by="IR", ascending=True)
df.to_csv("metadata/features_lr_sorted_by_IR.csv", index=False) 

df2 = pd.read_csv("metadata/features_svm.csv")
df2["IR"] = pd.to_numeric(df2["IR"], errors="coerce")
df2 = df2.sort_values(by="IR", ascending=True)
df2.to_csv("metadata/features_svm_sorted_by_IR.csv", index=False)

if __name__ == "__main__":
    main()