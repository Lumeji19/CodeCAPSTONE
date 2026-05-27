

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data_loading import load_dataset
from src.models.LRstable import StableLR
from src.models.LRbalanced import BalancedLR


def main():
    X, y, info = load_dataset("abalone_21_vs_8")
    print(f"IR: {info['imbalance_ratio']}, positives: {info['n_minority']}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    print(f"Training: {len(y_train)} rows, {(y_train == 1).sum()} positives")
    print(f"Test:     {len(y_test)} rows, {(y_test == 1).sum()} positives")
    print()

    for b in range(50):
        boot_idx, val_idx = train_test_split(
            np.arange(len(y_train)),
            train_size=0.8,
            stratify=y_train,
            random_state=42 + b,
        )
        X_boot = X_train_s[boot_idx]
        y_boot = y_train[boot_idx]
        X_val = X_train_s[val_idx]
        y_val = y_train[val_idx]

        n_pos_boot = int((y_boot == 1).sum())
        n_pos_val = int((y_val == 1).sum())


        model = StableLR(k=0.8, C=1.0)
        model.fit(X_boot, y_boot)

        probs = model.predict_proba(X_test_s)
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        
        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = float("nan")

        status = "BAD " if acc < 0.5 else "    "
        
        print(
            f"{status}b={b:3d}  "
            f"pos(boot)={n_pos_boot:2d}  pos(val)={n_pos_val}  "
            f"||beta||={np.linalg.norm(model.beta_):7.3f}  "
            f"beta0={model.beta0_:+7.3f}  "
            f"theta={model.theta_:+7.3f}  "
            f"success={model.success_}  "
            f"acc={acc:.4f}  auc={auc:.4f}"
        )


if __name__ == "__main__":
    main()