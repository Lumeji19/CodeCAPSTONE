

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _balanced_lr_objective(params, X, y_neg1, k, indic_pos, k_plus, C):
   
    n_features = X.shape[1]

    beta = params[:n_features]
    beta0 = params[n_features]
    theta = params[n_features + 1]
    lamb = params[n_features + 2]

    scores = X @ beta + beta0
    margins = y_neg1 * scores
    losses = np.logaddexp(0.0, -margins)

    maxlosses = np.maximum(losses - theta - lamb * indic_pos, 0.0)
    robust_term = theta + (1.0 / k) * np.sum(maxlosses) + (1.0 / k) * lamb * k_plus

    reg_term = (1.0 / (2.0 * C)) * np.dot(beta, beta)

    return float(robust_term + reg_term)


class BalancedLR:
 

    def __init__(self, k=0.7, C=1.0):
        self.k = k
        self.C = C

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        k_int = int(self.k * n_samples)

        y_neg1 = np.where(y == 1, 1.0, -1.0)

        indic_pos = (y_neg1 == 1).astype(float)
        prop_pos = np.mean(indic_pos)
        k_plus = int(round(k_int * prop_pos))

        rng = np.random.default_rng(0)
        initial_params = rng.normal(scale=0.01, size=n_features + 3)
        initial_params[-2] = np.log(2.0)
        initial_params[-1] = 0.0

        result = minimize(
            fun=_balanced_lr_objective,
            x0=initial_params,
            args=(X, y_neg1, k_int, indic_pos, k_plus, self.C),
            method="L-BFGS-B",
            options={"maxiter": 5000, "maxfun" : 50000},
        )

        for retry in range(3):
            if result.success:
                break
            rng = np.random.default_rng(retry + 1)
            initial_params = rng.normal(scale=0.01, size=n_features + 3)
            initial_params[-2] = np.log(2.0)
            initial_params[-1] = 0.0
            result = minimize(
                fun=_balanced_lr_objective,
                x0=initial_params,
                args=(X, y_neg1, k_int, indic_pos, k_plus, self.C),
                method="L-BFGS-B",
                options={"maxiter": 5000, "maxfun" : 50000},
            )

        if not result.success:
            raise RuntimeError(
                f" balanced lR failed."
                f"Message: {result.message}"
            )


        fitted = result.x
        self.beta_ = fitted[:n_features]
        self.beta0_ = float(fitted[n_features])
        self.theta_ = float(fitted[n_features + 1])
        self.lamb_ = float(fitted[n_features + 2])
        self.k_plus_ = k_plus

        self.objective_value_ = float(result.fun)
        self.n_iterations_ = int(result.nit)
        self.success_ = bool(result.success)
        self.message_ = str(result.message)

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