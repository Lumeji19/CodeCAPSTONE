

import numpy as np
from sklearn.svm import LinearSVC
from scipy.special import expit


class StandardSVM:


    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)


        n_samples, n_features = X.shape
        use_dual = n_samples <= n_features

        self._inner_model = LinearSVC(
            C=self.C,
            loss="squared_hinge",
            penalty="l2",
            dual="auto",          
            max_iter=10000,     
            random_state=0,
        )
        self._inner_model.fit(X, y)


        self.beta_ = self._inner_model.coef_.flatten()
        self.beta0_ = float(self._inner_model.intercept_[0])

        return self

    def predict_proba(self, X):

        X = np.asarray(X, dtype=float)
        scores = X @ self.beta_ + self.beta0_
        return expit(scores)

    def predict(self, X):

        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_structure(self):
        return {
            "beta": self.beta_.copy(),
            "beta0": self.beta0_,
        }