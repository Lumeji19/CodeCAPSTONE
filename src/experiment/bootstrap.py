
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time 

def _tune_and_fit(model_cls, param_grid, X_boot, y_boot, X_val, y_val):
    # Turn the param_grid dict into a list of all possible combinations.

    keys = list(param_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*param_grid.values())]

    best_params = None
    best_val_auc = -np.inf

    for params in combos: #try on all possible combos of k and C
        model = model_cls(**params) # stableLR(**{"k": 0.8, "C": 1.0}) = StableLR(k=0.8, C=1.0)
        model.fit(X_boot, y_boot)

        val_probs = model.predict_proba(X_val)

        # AUC needs both classes in y_val. If stratified sampling gave
        # a validation set with only one class (shouldn't happen with
        # stratified splitting but possible on tiny minority datasets),
        # give this combo a score of -inf so it loses.
        if len(np.unique(y_val)) < 2:
            val_auc = -np.inf
        else:
            val_auc = roc_auc_score(y_val, val_probs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_params = params

    # Refit on the full bootstrap with the chosen hyperparameters
    final_model = model_cls(**best_params)
    final_model.fit(X_boot, y_boot)

    return final_model, best_params, float(best_val_auc)

# SCHEMES: 
# plan A: stratified 80/20 split without replacement. Then tune on the 80%, validate on the 20%, refit on the 80% with best hyperparameters.
# plan B: True bootstrap, same size with replacement. Then 80/20 stratified split of training for validation.
#in the end used plan A
def _make_split_subsample(
    y_train: np.ndarray,
    seed: int,
    bootstrap_train_size: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plan A: stratified 80/20 split without replacement.
    """
    boot_idx, val_idx = train_test_split(
        np.arange(len(y_train)),
        train_size=bootstrap_train_size,
        stratify=y_train,
        random_state=seed,
    )
    return boot_idx, val_idx


def _make_split_true_bootstrap(
    y_train: np.ndarray,
    seed: int,
    internal_train_size: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plan B: stratified true bootstrap, then stratified
    internal 80/20 split for hyperparameter tuning.

    """
    rng = np.random.default_rng(seed)
    n = len(y_train)

    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    # Stratified bootstrap
    pos_boot = rng.choice(pos_indices, size=n_pos, replace=True)
    neg_boot = rng.choice(neg_indices, size=n_neg, replace=True)
    bootstrap_idx = np.concatenate([pos_boot, neg_boot])

    # Stratified internal 80/20 split.
    boot_positions, val_positions = train_test_split(
        np.arange(len(bootstrap_idx)),
        train_size=internal_train_size,
        stratify=y_train[bootstrap_idx],
        random_state=seed,
    )
    boot_idx = bootstrap_idx[boot_positions]
    val_idx = bootstrap_idx[val_positions]

    return boot_idx, val_idx


def run_bootstrap_experiment(
    X_train_full: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    methods: Dict[str, Any],
    param_grids: Dict[str, Dict[str, List]],
    n_bootstraps: int = 50,
    scheme: str = "subsample",
    bootstrap_train_size: float = 0.8,
    seed: int = 42,
) -> Dict[str, Dict]:
    
    if scheme == "subsample":
        split_fn = _make_split_subsample
    elif scheme == "true_bootstrap":
        split_fn = _make_split_true_bootstrap
    else:
        raise ValueError(
            f"Unkown scheme: {scheme!r}."
            f"Use 'subsample' or 'true_boostrap'."  
        )


    results = {
        name: {
            "probs": [],
            "preds": [],
            "betas": [],
            "beta0s": [],
            "params": [],
            "test_acc": [],
            "test_auc": [],
            "val_auc": [],
            "fit_times": [],
        }
        for name in methods
    }

    for b in range(n_bootstraps):
        # use the chosen split function
        boot_idx, val_idx = split_fn(y_train, seed=seed + b)

        X_boot, y_boot = X_train_full[boot_idx], y_train[boot_idx]
        X_val, y_val = X_train_full[val_idx], y_train[val_idx]

        for name, model_cls in methods.items():
            t_start = time.perf_counter()
            model, best_params, val_auc = _tune_and_fit(
                model_cls=model_cls,
                param_grid=param_grids[name],
                X_boot=X_boot, y_boot=y_boot,
                X_val=X_val, y_val=y_val,
            )
            fit_time = time.perf_counter() - t_start

            probs = model.predict_proba(X_test)
            preds = (probs >= 0.5).astype(int)
            test_acc = accuracy_score(y_test, preds)
            test_auc = roc_auc_score(y_test, probs)

            structure = model.get_structure()
            results[name]["probs"].append(probs)
            results[name]["preds"].append(preds)
            results[name]["betas"].append(structure["beta"])
            results[name]["beta0s"].append(structure["beta0"])
            results[name]["params"].append(best_params)
            results[name]["test_acc"].append(test_acc)
            results[name]["test_auc"].append(test_auc)
            results[name]["val_auc"].append(val_auc)
            results[name]["fit_times"].append(fit_time)

        if (b + 1) % 10 == 0:
            print(f"  bootstrap {b + 1}/{n_bootstraps} done")

    for name in methods:
        results[name]["probs"] = np.array(results[name]["probs"])
        results[name]["preds"] = np.array(results[name]["preds"])
        results[name]["test_acc"] = np.array(results[name]["test_acc"])
        results[name]["test_auc"] = np.array(results[name]["test_auc"])
        results[name]["val_auc"] = np.array(results[name]["val_auc"])
        results[name]["fit_times"] = np.array(results[name]["fit_times"])

    return results