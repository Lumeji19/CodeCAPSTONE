

import numpy as np
from sklearn.linear_model import LogisticRegression


class StandardLR:


    def __init__(self, C=1.0, max_iter=1000):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)


        self._model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
        )
        self._model.fit(X, y)

        self.beta_ = self._model.coef_.ravel()
        self.beta0_ = float(self._model.intercept_[0])
        self.n_iterations_ = int(self._model.n_iter_[0])
     
        self.success_ = self.n_iterations_ < self.max_iter

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_structure(self):
        return {
            "beta": self.beta_.copy(),
            "beta0": self.beta0_,
        }