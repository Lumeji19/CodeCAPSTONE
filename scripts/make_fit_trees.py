# NOT USED
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
    ("label_equal",              "3class_equal"),
    ("label_stability",          "3class_stability"),
    ("balanced_wins_equal",      "binary_equal"),
    ("balanced_wins_stability",  "binary_stability"),
]

DEPTH_GRID = [1, 2, 3, 4]
MIN_LEAF_GRID = [1,2,3,4,5,6,7,8]
N_FOLDS = 4
RANDOM_STATE = 0

# direction = +1: higher is better for balanced 
# direction = -1: lower is better for balanced 
LEAF_TEST_METRICS = [
    ("auc_mean",          +1),
    ("auc_worst",         +1),
    ("accuracy_mean",     +1),
    ("accuracy_worst",    +1),
    ("output_stability",  -1),
    ("structural_total",  -1),
]


def select_hyperparameters(X, y):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
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


# Leave-one-out CV accuracy

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


# Per-leaf Wilcoxon significance tests (not used in the final thesis in the end)

def run_leaf_significance_tests(clf, tree_data, results_df, label_col):
 
    X = tree_data[FEATURE_COLS].values
    leaf_ids = clf.apply(X)
    unique_leaves = np.unique(leaf_ids)

    standard_rows = results_df[results_df["method"] == "standard"].set_index("dataset")
    balanced_rows = results_df[results_df["method"] == "balanced"].set_index("dataset")

    print("  Per-leaf significance tests "
          "(balanced vs standard, Wilcoxon signed-rank):")

    leaf_results = []

    for leaf_id in unique_leaves:
        mask = (leaf_ids == leaf_id)
        leaf_datasets = tree_data.loc[mask, "dataset"].tolist()
        n_in_leaf = len(leaf_datasets)

        leaf_labels = tree_data.loc[mask, label_col]
        prescription = leaf_labels.value_counts().index[0]

        print(f"    Leaf {leaf_id}: n={n_in_leaf}, prescribes '{prescription}'")

        leaf_entry = {
            "leaf_id": int(leaf_id),
            "n_datasets": n_in_leaf,
            "prescription": prescription,
        }

        if n_in_leaf < 3:
            print(f"      (too few datasets for a significance test)")
            for metric, _ in LEAF_TEST_METRICS:
                leaf_entry[f"{metric}_pvalue"] = None
                leaf_entry[f"{metric}_mean_diff"] = None
            leaf_results.append(leaf_entry)
            continue

        for metric, direction in LEAF_TEST_METRICS:
            standard_vals = standard_rows.loc[leaf_datasets, metric].values
            balanced_vals = balanced_rows.loc[leaf_datasets, metric].values
            diffs = balanced_vals - standard_vals

            if np.all(diffs == 0):
                print(f"      {metric:20s}: all diffs zero, skipped")
                leaf_entry[f"{metric}_pvalue"] = None
                leaf_entry[f"{metric}_mean_diff"] = 0.0
                continue

            alternative = "greater" if direction == +1 else "less"
            try:
                stat, p = wilcoxon(diffs, alternative=alternative)
                mean_diff = diffs.mean()
                sign = "+" if mean_diff >= 0 else ""
                print(f"      {metric:20s}: "
                      f"mean diff = {sign}{mean_diff:.4f}, p = {p:.3f}")
                leaf_entry[f"{metric}_pvalue"] = p
                leaf_entry[f"{metric}_mean_diff"] = mean_diff
            except ValueError as exc:
                print(f"      {metric:20s}: wilcoxon failed ({exc})")
                leaf_entry[f"{metric}_pvalue"] = None
                leaf_entry[f"{metric}_mean_diff"] = None

        leaf_results.append(leaf_entry)

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
    value = tree.value  # shape: (n_nodes, 1, n_classes)
    classes = clf.classes_
    subtree_width = np.zeros(n_nodes, dtype=float)
    node_depth = np.zeros(n_nodes, dtype=int)

    def compute_width(node, d):
        node_depth[node] = d
        if children_left[node] == -1:  # leaf
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

    # Build a color map
    palette = plt.get_cmap("Set2")
    class_to_color = {
        cls: palette(i % palette.N) for i, cls in enumerate(classes)
    }

    # Draw edges (with yes/no labels) then nodes on top.
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
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white",
                          edgecolor="none"),
                zorder=2,
            )

    for node in range(n_nodes):
        is_leaf = (children_left[node] == -1)

        # Counts per class at this node, and the majority class.
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
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor=face,
                          edgecolor="#333",
                          linewidth=1.2),
                zorder=3,
            )
        else:
            # Internal node: split condition only with no color
            feat_name = feature_names[feature[node]]
            thresh = threshold[node]
            text = f"{feat_name} <= {thresh:.3g}\nn = {n_samples}"
            ax.text(
                node_x[node], node_y[node], text,
                ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor="white",
                          edgecolor="#333",
                          linewidth=1.2),
                zorder=3,
            )


    total_w = subtree_width[0]
    max_d = node_depth.max()
    ax.set_xlim(-0.6, total_w + 0.6)
    ax.set_ylim(-max_d - 0.8, 0.8)

    pretty_titles = {
        "3class_equal":      f"{model.upper()} equal weight prescription tree for the 3 methods",
        "3class_stability":  f"{model.upper()} stability weighted prescription tree for the 3 methods",
        "binary_equal":      f"{model.upper()} equal weight prescription tree -- binary decision on Balanced method",
        "binary_stability":  f"{model.upper()} stability weighted prescription tree -- binary decision on Balanced method",
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

    
    pvalue_cols = [f"{m}_pvalue" for m, _ in LEAF_TEST_METRICS]
    display_cols = ["model", "label_name", "leaf_id", "n_datasets",
                    "prescription"] + pvalue_cols

    table = df[display_cols].copy()


    for col in pvalue_cols:
        table[col] = table[col].apply(
            lambda p: ("—" if pd.isna(p)
                       else f"{p:.3f}" if p >= 0.001
                       else f"{p:.1e}")
        )

    # columns for the LaTeX table.
    rename = {
        "model": "Model",
        "label_name": "Tree",
        "leaf_id": "Leaf",
        "n_datasets": "n",
        "prescription": "Prescription",
    }
    for metric, _ in LEAF_TEST_METRICS:
        rename[f"{metric}_pvalue"] = metric.replace("_", " ")
    table = table.rename(columns=rename)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

  
    table.to_csv(out_path.with_suffix(".csv"), index=False)


    latex_str = table.to_latex(
        index=False,
        column_format="l" * len(table.columns),
        caption=("Per-leaf Wilcoxon signed-rank test p-values for "
                 "balanced vs standard, across all trees. "
                 "Lower p indicates stronger evidence that balanced "
                 "differs from standard on the corresponding metric."),
        label="tab:leaf-pvalues",
        escape=True,
    )
    with open(out_path.with_suffix(".tex"), "w") as f:
        f.write(latex_str)

    print(f"\n  Saved aggregate table: {out_path.with_suffix('.csv')}")
    print(f"  Saved aggregate LaTeX: {out_path.with_suffix('.tex')}")



def fit_one_tree(tree_data, results_df, label_col, model, label_name):
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
    print(f"  4-fold CV accuracy: {cv_acc:.3f}  "
          f"(baseline {majority_acc:.3f})")
    print(f"  LOO CV accuracy:    {loo_acc:.3f}  "
          f"({int(round(loo_acc * len(y)))}/{len(y)} datasets)")
    print(f"  in-sample accuracy: {clf.score(X, y):.3f}")
    print()
    print("  Tree structure:")
    for line in export_text(clf, feature_names=FEATURE_COLS).splitlines():
        print(f"    {line}")
    print()

    leaf_results = run_leaf_significance_tests(
        clf, tree_data, results_df, label_col
    )

    out_stem = f"figures/tree_{model}_{label_name}"
    visualize_tree(
        clf, X, y, FEATURE_COLS, out_stem,
        cv_acc=cv_acc,
        loo_acc=loo_acc,
        majority_acc=majority_acc,
        depth=depth,
        model=model,
        label_name=label_name,
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

        tree_data["source_is_keel"] = (
            tree_data["source"].str.upper() == "KEEL"
        ).astype(int)

        for label_col, label_name in LABEL_SPECS:
            clf, leaf_results = fit_one_tree(
                tree_data, results_df, label_col, model, label_name
            )
            all_leaf_results.extend(leaf_results)

    export_pvalue_table(all_leaf_results, "tables/leaf_pvalues")


if __name__ == "__main__":
    main()