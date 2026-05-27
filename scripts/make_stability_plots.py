from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

input_files = {
    "LR": RESULTS_DIR / "LR_full_with_ranks.csv",
    "SVM": RESULTS_DIR / "SVM_full_with_ranks.csv",
}

method_order = ["standard", "stable", "balanced"]

method_colors = {
    "standard": "tab:blue",
    "stable": "tab:orange",
    "balanced": "tab:green",
}

plots_to_make = {
    "output_stability_rank": "Output stability rank",
    "structural_total_rank": "Structural stability rank",
}

bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 90]
bin_labels = ["0–10", "10–20", "20–30", "30–40", "40–50", "50–60", "60–70", "70–90"]


def clean_data(df):
    df = df.copy()
    df["method"] = df["method"].astype(str).str.lower().str.strip()
    df = df[df["dataset"].astype(str).str.upper().str.strip() != "MEAN"].copy()
    df = df[df["method"].isin(method_order)].copy()
    df["IR"] = pd.to_numeric(df["IR"], errors="coerce")
    return df


def make_bucket_rank_barplot(df, model_name, rank_col, rank_label):
    df = df.copy()

    df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce")
    df = df.dropna(subset=["IR", rank_col])

    df["IR_bucket"] = pd.cut(
        df["IR"],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=False,
    )

    bucket_means = (
        df
        .groupby(["IR_bucket", "method"], observed=False)[rank_col]
        .mean()
        .reset_index()
    )

    plot_df = bucket_means.pivot(
        index="IR_bucket",
        columns="method",
        values=rank_col
    ).reindex(bin_labels)

    x = np.arange(len(plot_df.index))
    width = 0.25

    plt.figure(figsize=(10, 5))

    for i, method in enumerate(method_order):
        plt.bar(
            x + (i - 1) * width,
            plot_df[method],
            width=width,
            label=method.capitalize(),
            color=method_colors[method],
        )

    plt.axhline(1, color="black", linestyle="--", linewidth=1, label="Best possible rank")

    plt.xlabel("Imbalance ratio (IR) bucket")
    plt.ylabel("Mean rank")
    plt.title(f"{model_name}: Mean {rank_label.lower()} by IR bucket")

    plt.xticks(x, plot_df.index, rotation=45)
    plt.ylim(1, 3)
    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    output_path = FIGURES_DIR / f"{model_name}_{rank_col}_bucket_barplot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


for model_name, input_file in input_files.items():

    print()
    print(f"Processing {model_name}")
    print(f"Reading: {input_file}")

    df = pd.read_csv(input_file)
    df = clean_data(df)

    for rank_col, rank_label in plots_to_make.items():
        make_bucket_rank_barplot(
            df=df,
            model_name=model_name,
            rank_col=rank_col,
            rank_label=rank_label,
        )