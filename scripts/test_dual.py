
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

def make_tiny_dataset(n_samples=20, n_positives=5, n_features=3, seed=0):
    
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))

    y = np.zeros(n_samples, dtype=int)
    y[:n_positives] = 1

    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def per_sample_losses(X, y_neg1, beta, beta0):

    scores = X @ beta + beta0
    margins = y_neg1 * scores
    return np.logaddexp(0.0, -margins)


def primal_worst_case(losses, y_neg1, k, k_plus):

    n = len(losses)
    pos_indices = np.where(y_neg1 == 1)[0]
    neg_indices = np.where(y_neg1 == -1)[0]

    k_minus = k - k_plus  

    
    if k_plus > len(pos_indices) or k_minus > len(neg_indices):
        raise ValueError("Infeasible (k, k_plus) for this dataset.")

    best_sum = -np.inf
    
    for pos_subset in combinations(pos_indices, k_plus):
        for neg_subset in combinations(neg_indices, k_minus):
            subset = list(pos_subset) + list(neg_subset)
            s = losses[subset].sum()
            if s > best_sum:
                best_sum = s

    return best_sum

def dual_objective_over_theta_lamb(dual_params, losses, y_neg1, k, k_plus):
   
    theta, lamb = dual_params
    indic_pos = (y_neg1 == 1).astype(float)
    maxlosses = np.maximum(losses - theta - lamb * indic_pos, 0.0)
    return theta + (1.0 / k) * np.sum(maxlosses) + (1.0 / k) * lamb * k_plus


def dual_value(losses, y_neg1, k, k_plus):
    
    result = minimize(
        fun=dual_objective_over_theta_lamb,
        x0=np.array([np.log(2.0), 0.0]),  # reasonable initial guess
        args=(losses, y_neg1, k, k_plus),
        method="L-BFGS-B",
    )
    return float(result.fun)

def compare_primal_and_dual(X, y, beta, beta0, k, k_plus):

    y_neg1 = np.where(y == 1, 1.0, -1.0)

    
    losses = per_sample_losses(X, y_neg1, beta, beta0)

   
    primal_sum = primal_worst_case(losses, y_neg1, k, k_plus)
    primal = primal_sum / k

    dual = dual_value(losses, y_neg1, k, k_plus)

    return primal, dual, abs(primal - dual)

def main():
    
    n_samples = 20
    n_positives = 5
    n_features = 3

    X, y = make_tiny_dataset(
        n_samples=n_samples,
        n_positives=n_positives,
        n_features=n_features,
        seed=0,
    )

    
    k = int(0.8 * n_samples)                      # 16
    prop_pos = n_positives / n_samples             # 0.25
    k_plus = int(round(k * prop_pos))              # 4

    print(f"Dataset: n={n_samples}, positives={n_positives}, features={n_features}")
    print(f"k = {k}, k_plus = {k_plus}")
    print()

   
    rng = np.random.default_rng(42)
    tolerance = 1e-4
    all_passed = True

    for trial in range(5):
        beta = rng.normal(size=n_features)
        beta0 = rng.normal()

        primal, dual, diff = compare_primal_and_dual(
            X, y, beta, beta0, k, k_plus
        )

        passed = diff < tolerance
        all_passed = all_passed and passed

        status = "PASS" if passed else "FAIL"
        print(
            f"Trial {trial + 1}: "
            f"primal = {primal:.6f}, "
            f"dual = {dual:.6f}, "
            f"diff = {diff:.2e}  [{status}]"
        )

    print()
    if all_passed:
        print("All trials passed. Dual formulation verified.")
    else:
        print("FAILURE")


if __name__ == "__main__":
    main()