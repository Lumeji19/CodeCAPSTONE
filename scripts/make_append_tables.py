from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

file_30 = RESULTS_DIR / "pilot_svm_combined_30datasets.csv"
file_10 = RESULTS_DIR / "pilot_svm_new_10datasets.csv"

output_file = RESULTS_DIR / "pilot_svm_combined_40datasets.csv"



dataset_col = "dataset"
method_col = "method"
ir_col = "IR"

method_order = ["standard", "stable", "balanced"]

method_order_map = {
    "standard": 0,
    "stable": 1,
    "balanced": 2,
}



def clean_file(path):
    df = pd.read_csv(path)
    df[method_col] = df[method_col].astype(str).str.lower().str.strip()
    df = df[df[dataset_col].astype(str).str.upper().str.strip() != "MEAN"].copy()
    df = df[df[method_col].isin(method_order)].copy()

    return df



df_30 = clean_file(file_30)
df_10 = clean_file(file_10)


combined = pd.concat([df_30, df_10], ignore_index=True)




combined[ir_col] = pd.to_numeric(combined[ir_col], errors="coerce")

combined["_method_order"] = combined[method_col].map(method_order_map)

combined = (
    combined
    .sort_values(by=[ir_col, dataset_col, "_method_order"])
    .drop(columns="_method_order")
    .reset_index(drop=True)
)




numeric_cols = combined.select_dtypes(include="number").columns

mean_rows = (
    combined
    .groupby(method_col, as_index=False)[numeric_cols]
    .mean()
)

mean_rows[dataset_col] = "MEAN"

# make sure columns match original combined dataframe
for col in combined.columns:
    if col not in mean_rows.columns:
        mean_rows[col] = ""

mean_rows = mean_rows[combined.columns]

# keep mean rows in standard/stable/balanced order
mean_rows["_method_order"] = mean_rows[method_col].map(method_order_map)

mean_rows = (
    mean_rows
    .sort_values("_method_order")
    .drop(columns="_method_order")
    .reset_index(drop=True)
)



final_df = pd.concat([combined, mean_rows], ignore_index=True)



final_df.to_csv(output_file, index=False)

print(f"Saved combined file as: {output_file}")
print(f"Rows without means: {len(combined)}")
print(f"Mean rows added: {len(mean_rows)}")