
from typing import Any, Dict, List
from collections import Counter
import numpy as np

def output_stability(probs: np.ndarray) -> float:
    

    # For each test point, get the variance of its n_bootstraps predictions.
    per_point_variance = probs.var(axis=0, ddof=1)
    return float(per_point_variance.mean())


def structural_stability(structures: List[Dict[str, Any]]) -> Dict[str, float]:
    importance_matrix = np.array([s["feature_importances"] for s in structures])
    per_feature_std = importance_matrix.std(axis=0, ddof=1)
    importance_std_mean = float(per_feature_std.mean())
    depth_std = float(np.std([s["depth"] for s in structures], ddof=1))
    n_leaves_std = float(np.std([s["n_leaves"] for s in structures], ddof=1))
    return {
        "importance_std_mean": importance_std_mean,
        "depth_std": depth_std,
        "n_leaves_std": n_leaves_std,
        "total": importance_std_mean + depth_std + n_leaves_std,
    }


def hyperparameter_stability(params_list: List[Dict[str, Any]]) -> Dict[str, float]:
#geom std and mode freq
    hyperparam_names = list(params_list[0].keys())

    result = {}
    for name in hyperparam_names:  # for each dict p, get the value p[name] of each hyperparameter (name) across bootstraps into a 1D array
        values = np.array([p[name] for p in params_list], dtype=float)
        result[f"{name}_geom_std"] = float(np.exp(np.std(np.log(values), ddof=1)))
        # Geometric std: exp(std(log(values)))
        
        # Mode: most frequently chosen valuw
        counts = Counter(values.tolist())
        mode_value, mode_count = counts.most_common(1)[0]
        result[f"{name}_mode"] = float(mode_value)
        result[f"{name}_mode_frequency"] = mode_count / len(values)

    return result


def performance_summary(test_acc: np.ndarray, test_auc: np.ndarray, test_auprc: np.ndarray) -> Dict[str, float]:
    """
    Mean and std of test accuracy and AUC.
    """
    return {
        "accuracy_mean": float(test_acc.mean()),
        "accuracy_std": float(test_acc.std(ddof=1)),
        "accuracy_worst": float(test_acc.min()),
        "auc_mean": float(test_auc.mean()),
        "auc_std": float(test_auc.std(ddof=1)),
        "auc_worst": float(test_auc.min()),
        "auprc_mean": float(test_auprc.mean()),
        "auprc_std": float(test_auprc.std(ddof=1)),
        "auprc_worst": float(test_auprc.min()),

    }

def timing_summary(fit_times: np.ndarray) -> Dict[str, float]:
    """
    computational times
    """
    return {
        "fit_time_mean_s": float(fit_times.mean()),
        "fit_time_std_s": float(fit_times.std(ddof=1)),
        "fit_time_total_s": float(fit_times.sum()),
    }

def compute_all_metrics(method_results: Dict[str, Any]) -> Dict[str, Any]:
   
    metrics = {}

    metrics["output_stability"] = output_stability(method_results["probs"])

  
    struct = structural_stability(structures=method_results["structures"])
    for key, val in struct.items():
        metrics[f"structural_{key}"] = val



    hyper = hyperparameter_stability(method_results["params"])
    for key, val in hyper.items():
        metrics[f"hyp_{key}"] = val # "hyp_k_std" = 0.05

    
    perf = performance_summary(
        test_acc=method_results["test_acc"],
        test_auc=method_results["test_auc"],
        test_auprc=method_results["test_auprc"],
    )
    metrics.update(perf)

    fit_time_summary = timing_summary(np.array(method_results["fit_times"]))
    metrics.update(fit_time_summary)

    return metrics


def compute_all_methods_metrics(results):
    output = {}
    for name, method_results in results.items():
        output[name] = compute_all_metrics(method_results)
    return output