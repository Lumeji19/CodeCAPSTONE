import numpy as np
from src.models.tree_utils import monte_carlo_select_tree


class BalancedTree:
    def __init__(self, k=0.7, depth=5, zeta=20, min_samples_leaf=5):
        self.k = k
        self.depth = depth
        self.zeta = zeta
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = len(y)
        k_int = int(self.k * n)

        rng = np.random.default_rng(0)
        self.tree_, self.worst_case_loss_ = monte_carlo_select_tree(
            X, y,
            k=k_int,
            zeta=self.zeta,
            depth=self.depth,
            min_samples_leaf=self.min_samples_leaf,
            rng=rng,
            balanced=True,
        )
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self.tree_.predict_proba(X)[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_structure(self):
        return {
            "feature_importances": self.tree_.feature_importances_.copy(),
            "depth": int(self.tree_.get_depth()),
            "n_leaves": int(self.tree_.get_n_leaves()),
            "worst_case_loss": self.worst_case_loss_,
        }