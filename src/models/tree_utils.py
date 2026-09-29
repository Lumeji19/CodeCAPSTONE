import numpy as np
from sklearn.tree import DecisionTreeClassifier


def _pointwise_loss(tree, X, y):
    """
    Per-point log loss, f(m, x_i, y_i), the same loss family your
    LR/SVM robust formulations use. Returns one loss value per row.
    """
    probs = tree.predict_proba(X)[:, 1]
    eps = 1e-12  # avoid log(0)
    probs = np.clip(probs, eps, 1 - eps)
    return -(y * np.log(probs) + (1 - y) * np.log(1 - probs))


def _sample_subset(y, k, rng):
    """
    Sample one subset of size k from the uncertainty set Z (unconstrained
    class composition) — used for the stable (non-balanced) tree.
    """
    n = len(y)
    idx = rng.choice(n, size=k, replace=False)
    return idx


def _sample_balanced_subset(y, k, rng):
    """
    Sample one subset of size k from Z_balanced: exactly k+ positive rows
    and k - k+ negative rows, where k+ preserves the dataset's original
    class proportion (same k+ formula as BalancedLR).
    """
    n = len(y)
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    prop_pos = len(pos_indices) / n
    k_plus = int(round(k * prop_pos))
    k_minus = k - k_plus

    pos_sample = rng.choice(pos_indices, size=k_plus, replace=False)
    neg_sample = rng.choice(neg_indices, size=k_minus, replace=False)
    return np.concatenate([pos_sample, neg_sample])


def monte_carlo_select_tree(X, y, k, zeta, depth, min_samples_leaf, rng, balanced):
    """
    The Monte Carlo min-max tree selection:
      1. Sample zeta subsets of size k from the (balanced or unbalanced)
         uncertainty set.
      2. Fit one candidate tree per subset.
      3. Evaluate every candidate tree's total loss on every subset.
      4. Return the candidate whose worst-case (max) loss is smallest.
    """
    sample_fn = _sample_balanced_subset if balanced else _sample_subset

    subsets = [sample_fn(y, k, rng) for _ in range(zeta)]
    candidate_trees = []
    for subset_idx in subsets:
        tree = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=min_samples_leaf,
            random_state=int(rng.integers(0, 1_000_000)),
        )
        tree.fit(X[subset_idx], y[subset_idx])
        candidate_trees.append(tree)

    # zeta x zeta loss matrix: loss_matrix[j, l] = candidate j's total
    # loss evaluated on subset l
    loss_matrix = np.zeros((zeta, zeta))
    for j, tree in enumerate(candidate_trees):
        for l, subset_idx in enumerate(subsets):
            loss_matrix[j, l] = _pointwise_loss(tree, X[subset_idx], y[subset_idx]).sum()

    worst_case = loss_matrix.max(axis=1)
    best_j = int(np.argmin(worst_case))
    return candidate_trees[best_j], float(worst_case[best_j])