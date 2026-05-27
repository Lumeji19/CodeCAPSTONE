
import numpy as np
import cvxpy as cp
from scipy.special import expit


class BalancedSVM:

    def __init__(self, k=0.7, C=1.0, solver="CLARABEL"):
        self.k = k
        self.C = C
        self.solver = solver

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        k_int = int(self.k * n_samples)
        y_neg1 = np.where(y == 1, 1.0, -1.0)
        yX = y_neg1.reshape(-1, 1) * X


        indic_pos = (y_neg1 == 1).astype(float)
        prop_pos = np.mean(indic_pos)
        k_plus = int(round(k_int * prop_pos))

        beta = cp.Variable(n_features)
        beta0 = cp.Variable()
        theta = cp.Variable()
        lamb = cp.Variable()                       
        u = cp.Variable(n_samples, nonneg=True)


        margins = yX @ beta + y_neg1 * beta0
        hinge_squared = cp.pos(1.0 - margins) ** 2


        constraints = [theta + lamb * indic_pos + u >= hinge_squared]


        objective = cp.Minimize(
            k_int * theta
            + k_plus * lamb
            + cp.sum(u)
            + (1.0 / (2.0 * self.C)) * cp.sum_squares(beta)
        )

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"BalancedSVM failed: status={problem.status!r}"
            )

        self.beta_ = np.asarray(beta.value).flatten()
        self.beta0_ = float(beta0.value)
        self.theta_ = float(theta.value)
        self.lamb_ = float(lamb.value)
        self.k_plus_ = k_plus

        self.objective_value_ = float(problem.value)
        self.success_ = problem.status == "optimal"
        self.status_ = problem.status

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