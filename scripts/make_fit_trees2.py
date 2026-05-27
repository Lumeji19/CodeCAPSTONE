from xml.parsers.expat import model
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_COLS = [
    "n_rows",
    "p_encoded",
    "IR",
    "standard_auc",
    "standard_accuracy",
    "standard_accuracy_worst",
    "standard_output_stab",
    "standard_structural",
    "source_is_keel",
]

LABEL_SPECS = [
    ("label_equal",               "3class_equal",          "avg_rank_equal",     -1),
    ("label_stability",           "3class_stability",      "avg_rank_stability", -1),
    ("balanced_wins_equal",       "binary_equal",          "avg_rank_equal",     -1),
    ("balanced_wins_stability",   "binary_stability",      "avg_rank_stability", -1),
    ("label_output_stab",         "metric_output_stab",      "output_stability",  -1),
    ("label_structural",          "metric_structural",       "structural_total",  -1),
    ("label_auc_worst",           "metric_auc_worst",        "auc_worst",         +1),
    ("label_accuracy_worst",      "metric_accuracy_worst",   "accuracy_worst",    +1),
]
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
STABILITY_WEIGHT = 2.0
PERFORMANCE_WEIGHT = 1.0

DEPTH_GRID = [1, 2, 3, 4]
MIN_LEAF_GRID = [3, 4, 5, 6, 7, 8]
N_FOLDS = 4
RANDOM_STATE = 0


def add_rank_scores(results_df):

    df = results_df.copy()
    df["avg_rank_equal"] = df[ALL_RANK_COLS].sum(axis=1) / len(ALL_RANK_COLS)
    stab_sum = df[STABILITY_RANK_COLS].sum(axis=1)
    perf_sum = df[PERFORMANCE_RANK_COLS].sum(axis=1)
    total_w = STABILITY_WEIGHT * len(STABILITY_RANK_COLS) + PERFORMANCE_WEIGHT * len(PERFORMANCE_RANK_COLS)
    df["avg_rank_stability"] = (STABILITY_WEIGHT * stab_sum + PERFORMANCE_WEIGHT * perf_sum) / total_w
    return df


def select_hyperparameters(X, y):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best = (None, None, -1.0)
    for depth in DEPTH_GRID:
        for min_leaf in MIN_LEAF_GRID:
            clf = DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=min_leaf,
                random_state=RANDOM_STATE,
            )
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            mean_score = scores.mean()
            if mean_score > best[2]:
                best = (depth, min_leaf, mean_score)
    return best


def leave_one_out_accuracy(X, y, depth, min_leaf):
    n = len(y)
    correct = 0
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        clf = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=min_leaf,
            random_state=RANDOM_STATE,
        )
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[i:i+1])[0]
        if pred == y[i]:
            correct += 1
    return correct / n


def run_leaf_significance_tests(clf, tree_data, results_df, label_col,
                                tested_metric, direction):
  
    X = tree_data[FEATURE_COLS].values
    leaf_ids = clf.apply(X)
    unique_leaves = np.unique(leaf_ids)

    standard_rows = results_df[results_df["method"] == "standard"].set_index("dataset")
    stable_rows   = results_df[results_df["method"] == "stable"].set_index("dataset")
    balanced_rows = results_df[results_df["method"] == "balanced"].set_index("dataset")

    print(f"  Per-leaf significance tests on '{tested_metric}' "
          f"(Wilcoxon signed-rank, one-sided alt = balanced is better):")

    alternative = "greater" if direction == +1 else "less"
    leaf_results = []

    for leaf_id in unique_leaves:
        mask = (leaf_ids == leaf_id)
        leaf_datasets = tree_data.loc[mask, "dataset"].tolist()
        n_in_leaf = len(leaf_datasets)
        leaf_labels = tree_data.loc[mask, label_col]
        prescription = leaf_labels.value_counts().index[0]

        print(f"    Leaf {leaf_id}: n={n_in_leaf}, prescribes '{prescription}'")

        entry = {
            "leaf_id": int(leaf_id),
            "n_datasets": n_in_leaf,
            "prescription": prescription,
            "tested_metric": tested_metric,
            "pvalue_vs_standard": None,
            "pvalue_vs_stable":   None,
            "mean_diff_vs_standard": None,
            "mean_diff_vs_stable":   None,
        }

        if n_in_leaf < 3:
            print(f"      (too few datasets for a significance test)")
            leaf_results.append(entry)
            continue


        balanced_vals = balanced_rows.loc[leaf_datasets, tested_metric].values

        for opponent_name, opponent_rows in [("standard", standard_rows),
                                              ("stable",   stable_rows)]:
            opp_vals = opponent_rows.loc[leaf_datasets, tested_metric].values
            diffs = balanced_vals - opp_vals
            mean_diff = diffs.mean()
            if np.all(diffs == 0):
                print(f"      vs {opponent_name:8s}: all diffs zero, skipped")
                entry[f"mean_diff_vs_{opponent_name}"] = 0.0
                continue
            try:
                stat, p = wilcoxon(diffs, alternative=alternative)
                sign = "+" if mean_diff >= 0 else ""
                print(f"      vs {opponent_name:8s}: "
                      f"mean diff = {sign}{mean_diff:.4f}, p = {p:.3f}")
                entry[f"pvalue_vs_{opponent_name}"] = p
                entry[f"mean_diff_vs_{opponent_name}"] = mean_diff
            except ValueError as exc:
                print(f"      vs {opponent_name:8s}: wilcoxon failed ({exc})")

        leaf_results.append(entry)

    print()
    return leaf_results


def visualize_tree(clf, X, y, feature_names, out_path_stem,
                   cv_acc, loo_acc, majority_acc, depth,
                   model, label_name):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_axis_off()
    tree = clf.tree_
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value
    classes = clf.classes_

    subtree_width = np.zeros(n_nodes, dtype=float)
    node_depth = np.zeros(n_nodes, dtype=int)

    def compute_width(node, d):
        node_depth[node] = d
        if children_left[node] == -1:
            subtree_width[node] = 1.0
            return 1.0
        w = (compute_width(children_left[node], d + 1)
             + compute_width(children_right[node], d + 1))
        subtree_width[node] = w
        return w

    compute_width(0, 0)
    node_x = np.zeros(n_nodes, dtype=float)
    node_y = np.zeros(n_nodes, dtype=float)

    def assign_coords(node, x_left, y):
        w = subtree_width[node]
        node_x[node] = x_left + w / 2.0
        node_y[node] = y
        if children_left[node] != -1:
            assign_coords(children_left[node], x_left, y - 1)
            assign_coords(children_right[node],
                          x_left + subtree_width[children_left[node]],
                          y - 1)

    assign_coords(0, 0.0, 0.0)

    palette = plt.get_cmap("Set2")
    class_to_color = {
        cls: palette(i % palette.N) for i, cls in enumerate(classes)
    }

    for node in range(n_nodes):
        if children_left[node] == -1:
            continue
        for child, side in [(children_left[node], "yes"),
                            (children_right[node], "no")]:
            ax.plot(
                [node_x[node], node_x[child]],
                [node_y[node], node_y[child]],
                color="#888", linewidth=1.2, zorder=1,
            )
            mid_x = (node_x[node] + node_x[child]) / 2
            mid_y = (node_y[node] + node_y[child]) / 2
            ax.text(
                mid_x, mid_y, side,
                ha="center", va="center", fontsize=10,
                color="#444",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"),
                zorder=2,
            )

    for node in range(n_nodes):
        is_leaf = (children_left[node] == -1)
        counts = value[node][0]
        majority_idx = int(np.argmax(counts))
        majority_class = classes[majority_idx]
        n_samples = int(counts.sum())

        if is_leaf:
            face = class_to_color[majority_class]
            text = f"Prescribe: {majority_class}\nn = {n_samples}"
            ax.text(
                node_x[node], node_y[node], text,
                ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=face,
                          edgecolor="#333", linewidth=1.2),
                zorder=3,
            )
        else:
            feat_name = feature_names[feature[node]]
            thresh = threshold[node]
            text = f"{feat_name} <= {thresh:.3g}\nn = {n_samples}"
            ax.text(
                node_x[node], node_y[node], text,
                ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor="#333", linewidth=1.2),
                zorder=3,
            )

    total_w = subtree_width[0]
    max_d = node_depth.max()
    ax.set_xlim(-0.6, total_w + 0.6)
    ax.set_ylim(-max_d - 0.8, 0.8)

    pretty_titles = {
        "3class_equal":          f"{model.upper()} equal weight prescription tree for the 3 methods",
        "3class_stability":      f"{model.upper()} stability weighted prescription tree for the 3 methods",
        "binary_equal":          f"{model.upper()} equal weight prescription tree -- binary decision on Balanced method",
        "binary_stability":      f"{model.upper()} stability weighted prescription tree -- binary decision on Balanced method",
        "metric_output_stab":    f"{model.upper()} output stability prescription tree",
        "metric_structural":     f"{model.upper()} structural stability prescription tree",
        "metric_auc_worst":      f"{model.upper()} AUC worst prescription tree",
        "metric_accuracy_worst": f"{model.upper()} accuracy worst prescription tree",
    }
    title_top = pretty_titles.get(label_name, f"{model.upper()} -- {label_name}")
    ax.set_title(
        f"{title_top}\n"
        f"CV accuracy {cv_acc:.2f} | baseline accuracy {majority_acc:.2f}",
        fontsize=11,
    )

    fig.savefig(f"{out_path_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_path_stem}.png")


def export_pvalue_table(all_leaf_results, out_path):
    
    df = pd.DataFrame(all_leaf_results)

    def fmt_p(p):
        if pd.isna(p):
            return "--"
        if p < 0.001:
            return f"{p:.1e}"
        return f"{p:.3f}"

    table = df[[
        "model", "label_name", "leaf_id", "n_datasets",
        "prescription", "tested_metric",
        "pvalue_vs_standard", "pvalue_vs_stable",
    ]].copy()
    table["pvalue_vs_standard"] = table["pvalue_vs_standard"].apply(fmt_p)
    table["pvalue_vs_stable"]   = table["pvalue_vs_stable"].apply(fmt_p)

    table = table.rename(columns={
        "model": "Model",
        "label_name": "Tree",
        "leaf_id": "Leaf",
        "n_datasets": "n",
        "prescription": "Prescription",
        "tested_metric": "Tested metric",
        "pvalue_vs_standard": "p vs standard",
        "pvalue_vs_stable":   "p vs stable",
    })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path.with_suffix(".csv"), index=False)

    latex_str = table.to_latex(
        index=False,
        column_format="l" * len(table.columns),
        caption=("Per-leaf Wilcoxon signed-rank p-values for balanced "
                 "vs standard and balanced vs stable, across all trees. "
                 "Each leaf is tested on the metric the tree was built "
                 "around (column 'Tested metric'). One-sided alternative: "
                 "balanced is better. Small p indicates statistically "
                 "significant improvement."),
        label="tab:leaf-pvalues",
        escape=True,
    )
    with open(out_path.with_suffix(".tex"), "w") as f:
        f.write(latex_str)

    print(f"\n  Saved table: {out_path.with_suffix('.csv')}")
    print(f"  Saved LaTeX: {out_path.with_suffix('.tex')}")


def fit_one_tree(tree_data, results_df, label_col, model, label_name,
                 tested_metric, direction):
    X = tree_data[FEATURE_COLS].values
    y = tree_data[label_col].values

    depth, min_leaf, cv_acc = select_hyperparameters(X, y)
    clf = DecisionTreeClassifier(
        max_depth=depth, min_samples_leaf=min_leaf,
        random_state=RANDOM_STATE,
    )
    clf.fit(X, y)

    majority_label = pd.Series(y).value_counts().index[0]
    majority_acc = (y == majority_label).mean()
    loo_acc = leave_one_out_accuracy(X, y, depth, min_leaf)

    print(f"--- {model.upper()}  |  {label_name} ---")
    print(f"  chosen max_depth={depth}, min_samples_leaf={min_leaf}")
    print(f"  4-fold CV accuracy: {cv_acc:.3f}  (baseline {majority_acc:.3f})")
    print(f"  LOO CV accuracy:    {loo_acc:.3f}  "
          f"({int(round(loo_acc * len(y)))}/{len(y)} datasets)")
    print(f"  in-sample accuracy: {clf.score(X, y):.3f}")
    print()
    print("  Tree structure:")
    for line in export_text(clf, feature_names=FEATURE_COLS).splitlines():
        print(f"    {line}")
    print()

    leaf_results = run_leaf_significance_tests(
        clf, tree_data, results_df, label_col, tested_metric, direction
    )

    out_stem = f"figures/tree_{model}_{label_name}"
    visualize_tree(
        clf, X, y, FEATURE_COLS, out_stem,
        cv_acc=cv_acc, loo_acc=loo_acc, majority_acc=majority_acc,
        depth=depth, model=model, label_name=label_name,
    )
    print()

    for entry in leaf_results:
        entry["model"] = model
        entry["label_name"] = label_name

    return clf, leaf_results


def main():
    all_leaf_results = []

    for model in ["lr", "svm"]:
        tree_data = pd.read_csv(f"metadata/tree_data_{model}.csv")
        results_df = pd.read_csv(f"results/{model.upper()}_full_with_ranks.csv")
        results_df = results_df[results_df["dataset"] != "MEAN"].copy()
        results_df = add_rank_scores(results_df)

        tree_data["source_is_keel"] = (
            tree_data["source"].str.upper() == "KEEL"
        ).astype(int)

        for label_col, label_name, tested_metric, direction in LABEL_SPECS:
            clf, leaf_results = fit_one_tree(
                tree_data, results_df, label_col, model, label_name,
                tested_metric, direction,
            )
            all_leaf_results.extend(leaf_results)

    export_pvalue_table(all_leaf_results, "tables/leaf_pvalues")


if __name__ == "__main__":
    main()