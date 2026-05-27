from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


input_files = {
    "LR": RESULTS_DIR / "pilot_LR_combined_40datasets.csv",
    "SVM": RESULTS_DIR / "pilot_SVM_combined_40datasets.csv",
}

dataset_col = "dataset"
method_col = "method"

method_order = ["standard", "stable", "balanced"]

# Final rows in the table (and their corresponding columns in the CSV)
table_metrics = {
    "Output stability": "output_stability",
    "Structural stability": "structural_total",
    "Hyperparameter stability": "hyp_C_geom_std",
    "Hyperparameter frequency": "hyp_C_mode_frequency",
    "Accuracy": "accuracy_mean",
    "Accuracy worst": "accuracy_worst",
    "AUC": "auc_mean",
    "AUC worst": "auc_worst",
    "k stability" : "hyp_k_geom_std",
}

# Ranking direction 
rank_lowest_is_best = {
    "output_stability": True,

    "structural_beta_std": True,
    "structural_beta0_std": True,
    "structural_total": True,

    "hyp_C_geom_std": True,
    # hyp_C_mode is skipped
    "hyp_C_mode_frequency": False,

    "accuracy_mean": False,
    "accuracy_std": True,
    "accuracy_worst": False,

    "auc_mean": False,
    "auc_std": True,
    "auc_worst": False,

    "fit_time_mean_s": True,
    "fit_time_std_s": True,
    "fit_time_total_s": True,

    "hyp_k_geom_std": True,
    # hyp_k_mode is skipped
    "hyp_k_mode_frequency": False,
}


#round to 3 sig figs
def sigfig(x, sig=3):
    if pd.isna(x):
        return "—"
    return f"{x:#.{sig}g}"

# Format value and rank, bolding for best value/rank
def format_value_rank(mean_value, mean_rank, bold_value=False, bold_rank=False):
    value_text = sigfig(mean_value)
    rank_text = sigfig(mean_rank)

    if bold_value:
        value_text = f"\\textbf{{{value_text}}}"

    if bold_rank:
        rank_text = f"\\textbf{{{rank_text}}}"

    return f"{value_text} ({rank_text})"


for model_name, input_file in input_files.items():

    print()
    print(f"Processing {model_name}...")
    print(f"Reading: {input_file}")

  
    output_full_csv = RESULTS_DIR / f"{model_name}_full_with_ranks.csv"
    output_csv = RESULTS_DIR / f"{model_name}_main_table.csv"
    output_latex = RESULTS_DIR / f"{model_name}_main_table.tex"


    df = pd.read_csv(input_file)

    df[method_col] = df[method_col].astype(str).str.lower().str.strip()

    df = df[df[dataset_col].astype(str).str.upper().str.strip() != "MEAN"].copy()
    df = df[df[method_col].isin(method_order)].copy()


    for col, lowest_is_best in rank_lowest_is_best.items():
        if col not in df.columns:
            continue

        df[col + "_rank"] = (
            df
            .groupby(dataset_col)[col]
            .rank(
                ascending=lowest_is_best,
                method="average",
                na_option="keep"
            )
        )


    mean_values = (
        df
        .groupby(method_col)
        .mean(numeric_only=True)
        .reindex(method_order)
    )

    full_with_ranks = df.copy()

    mean_rows = mean_values.copy()
    mean_rows[dataset_col] = "MEAN"
    mean_rows[method_col] = mean_rows.index
    mean_rows = mean_rows.reset_index(drop=True)

    full_with_ranks = pd.concat([full_with_ranks, mean_rows], ignore_index=True)

    full_with_ranks.to_csv(output_full_csv, index=False)


    final_table = pd.DataFrame(index=table_metrics.keys(), columns=method_order)

    for row_name, value_col in table_metrics.items():
        rank_col = value_col + "_rank"

        metric_values = mean_values.loc[method_order, value_col]


        metric_ranks = mean_values.loc[method_order, rank_col]

        if rank_lowest_is_best[value_col]:
            best_value = metric_values.min()
        else:
            best_value = metric_values.max()

        best_rank = metric_ranks.min()

        for method in method_order:
            mean_value = mean_values.loc[method, value_col]
            mean_rank = mean_values.loc[method, rank_col]

            bold_value = np.isclose(mean_value, best_value, equal_nan=False)

            bold_rank = np.isclose(mean_rank, best_rank, equal_nan=False)

            final_table.loc[row_name, method] = format_value_rank(
                mean_value=mean_value,
                mean_rank=mean_rank,
                bold_value=bold_value,
                bold_rank=bold_rank
            )

    final_table.columns = ["Standard", "Stable", "Balanced"]
    final_table.to_csv(output_csv)

    latex_table = final_table.to_latex(
        index=True,
        escape=False,
        column_format="lccc",
        caption=f"Mean performance and average rank across datasets for {model_name}. Values are reported as mean (average rank).",
        label=f"tab:{model_name.lower()}_mean_performance_rank"
    )

    with open(output_latex, "w", encoding="utf-8") as f:
        f.write(latex_table)



    print(final_table)
    print()
    print(f"Saved full CSV: {output_full_csv}")
    print(f"Saved main table CSV: {output_csv}")
    print(f"Saved LaTeX table: {output_latex}")