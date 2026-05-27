
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.data_loading import load_dataset
from src.models.SVMstandard import StandardSVM
from src.models.SVMstable_cvx import StableSVM
from src.models.SVMbalanced_cvx import BalancedSVM


def main():
    DATASET = "car_good"  

    X, y, info = load_dataset(DATASET)
    print(f"Dataset: {DATASET}, IR={info['imbalance_ratio']:.2f}")
    print(f"  n_rows = {info['n_rows']}, n_features = {info['n_features']}")
    print(f"  class counts: {dict(zip(*np.unique(y, return_counts=True)))}")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)

    methods = {
        "standard":         StandardSVM(C=1.0),
        "stable (k=0.99)":  StableSVM(k=0.99, C=1.0),
        "stable (k=0.8)":   StableSVM(k=0.8,  C=1.0),
        "balanced (k=0.8)": BalancedSVM(k=0.8, C=1.0),
    }

    print(f"{'method':<22} {'||beta||':>10} {'beta0':>10} "
          f"{'theta':>10} {'lamb':>10} {'AUC':>8}")
    print("-" * 80)

    for name, model in methods.items():
        model.fit(X_train_s, y_train)
        scores = X_test_s @ model.beta_ + model.beta0_
        auc = roc_auc_score(y_test, scores)  # use raw scores; same AUC

        norm = np.linalg.norm(model.beta_)
        b0 = model.beta0_
        theta = getattr(model, "theta_", float("nan"))
        lamb = getattr(model, "lamb_", float("nan"))

        print(f"{name:<22} {norm:>10.4f} {b0:>10.4f} "
              f"{theta:>10.4f} {lamb:>10.4f} {auc:>8.4f}")


if __name__ == "__main__":
    main()