
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import time

from src.data_loading import load_dataset
from src.models.SVMbalanced_cvx import BalancedSVM


X, y, info = load_dataset("car_good")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
y_train, y_test = np.asarray(y_train), np.asarray(y_test)

print("Fitting BalancedSVM (k=0.7, C=100.0, Clarabel)...")
t = time.perf_counter()
m = BalancedSVM(k=0.7, C=100.0).fit(X_train_s, y_train)
elapsed = time.perf_counter() - t
print(f"  fit time: {elapsed:.2f}s")
print(f"  ||beta||: {np.linalg.norm(m.beta_):.4f}, beta0: {m.beta0_:+.4f}")
print(f"  theta:    {m.theta_:.4f}")
print(f"  lamb:     {m.lamb_:.4f}      (the new dual variable)")
print(f"  k_plus:   {m.k_plus_}        (positives required in worst-case)")
print(f"  status:   {m.status_}")

probs = m.predict_proba(X_test_s)
print(f"  AUC:      {roc_auc_score(y_test, probs):.4f}")