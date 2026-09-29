import numpy as np
from sklearn.tree import DecisionTreeClassifier


class StandardTree:
    def __init__(self, depth=5, min_samples_leaf=5):
        self.depth = depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.tree_ = DecisionTreeClassifier(
            max_depth=self.depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=0,
        )
        self.tree_.fit(X, y)
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
        }