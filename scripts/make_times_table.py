from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = PROJECT_ROOT / "tables"

TABLES_DIR.mkdir(exist_ok=True)

input_files = {
    "LR": RESULTS_DIR / "LR_full_with_ranks.csv",
    "SVM": RESULTS_DIR / "SVM_full_with_ranks.csv",
}

output_csv = TABLES_DIR / "relative_computational_time_table.csv"
output_tex = TABLES_DIR / "relative_computational_time_table.tex"

dataset_col = "dataset"
method_col = "method"
time_col = "fit_time_mean_s"
rank_col = "fit_time_mean_s_rank"

method_order = ["standard", "stable", "balanced"]


def sigfig(x, sig=3):
    if pd.isna(x):
        return "—"
    return f"{x:#.{sig}g}"


def make_model_time_column(file_path):
    df = pd.read_csv(file_path)

    df[method_col] = df[method_col].astype(str).str.lower().str.strip()

    mean_rows = df[df[dataset_col].astype(str).str.upper().str.strip() == "MEAN"].copy()

    if mean_rows.empty:
        df = df[df[dataset_col].astype(str).str.upper().str.strip() != "MEAN"].copy()

        mean_rows = (
            df
            .groupby(method_col, as_index=False)
            [[time_col, rank_col]]
            .mean()
        )

    mean_rows = mean_rows[mean_rows[method_col].isin(method_order)].copy()
    mean_rows = mean_rows.set_index(method_col).reindex(method_order)

    standard_time = mean_rows.loc["standard", time_col]

    values = {}

    for method in method_order:
        mean_time = mean_rows.loc[method, time_col]
        mean_rank = mean_rows.loc[method, rank_col]

        relative_time = mean_time / standard_time

        values[method.capitalize()] = f"{sigfig(relative_time)} ({sigfig(mean_rank)})"

    return values


final_table = pd.DataFrame(index=[m.capitalize() for m in method_order])

for model_name, file_path in input_files.items():
    final_table[model_name] = pd.Series(make_model_time_column(file_path))

final_table.index.name = "Method"

final_table.to_csv(output_csv)

latex_table = final_table.to_latex(
    index=True,
    escape=False,
    column_format="lcc",
    caption="Relative computational time and average rank across datasets. Times are normalized by the standard method within each model class, so the standard method has relative time 1. Values are reported as relative mean time (average rank).",
    label="tab:relative_computational_time",
)

with open(output_tex, "w", encoding="utf-8") as f:
    f.write(latex_table)

print(final_table)
print()
print(f"Saved CSV: {output_csv}")
print(f"Saved LaTeX: {output_tex}")