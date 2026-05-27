import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


from src.data_loading import load_dataset
from src.models.LRstandard import StandardLR
from src.models.LRstable import StableLR
from src.models.LRbalanced import BalancedLR


def main():
    
    X, y, info = load_dataset("yeast5")
    print(f"Dataset: {info['dataset_name']}")
    print(f"n_rows = {info['n_rows']}, n_features = {info['n_features']}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    methods = {
        "standard": StandardLR(C=1.0),
        "stable":   StableLR(k=0.8, C=1.0),
        "balanced": BalancedLR(k=0.8, C=1.0),
    }

    for name, model in methods.items():
        model.fit(X_train_s, y_train)

        probs = model.predict_proba(X_test_s)
        preds = model.predict(X_test_s)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        print(f"--- {name} ---")
        print(f"  accuracy = {acc:.4f}")
        print(f"  auc      = {auc:.4f}")
        print(f"  ||beta|| = {np.linalg.norm(model.beta_):.4f}")
        print(f"  beta0    = {model.beta0_:.4f}")

        if hasattr(model, "theta_"):
            print(f"  theta    = {model.theta_:.4f}")
        if hasattr(model, "lamb_"):
            print(f"  lamb     = {model.lamb_:.4f}  (k_plus = {model.k_plus_})")
        print()


if __name__ == "__main__":
    main()