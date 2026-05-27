
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.data_loading import load_dataset
from src.models.SVMbalanced_cvx import BalancedSVM


DATASET = "car_good"  

X, y, info = load_dataset(DATASET)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
y_train, y_test = np.asarray(y_train), np.asarray(y_test)

m = BalancedSVM(k=0.8, C=100).fit(X_train_s, y_train)

print(f"beta_ values:")
print(m.beta_)
print(f"\n||beta||_1: {np.sum(np.abs(m.beta_)):.10f}")
print(f"||beta||_2: {np.linalg.norm(m.beta_):.10f}")
print(f"max |beta|: {np.max(np.abs(m.beta_)):.10f}")
print(f"min |beta|: {np.min(np.abs(m.beta_)):.10f}")
print(f"beta0: {m.beta0_:.10f}")


scores = X_test_s @ m.beta_ + m.beta0_
print(f"\nTest scores:")
print(f"  min: {scores.min():.6f}")
print(f"  max: {scores.max():.6f}")
print(f"  range: {scores.max() - scores.min():.6f}")
print(f"  std: {scores.std():.6f}")
print(f"  number of unique values: {len(np.unique(scores))}")

probs = m.predict_proba(X_test_s)
print(f"\nProbabilities:")
print(f"  min: {probs.min():.6f}")
print(f"  max: {probs.max():.6f}")
print(f"  unique: {len(np.unique(probs))}")


print(f"\nTest set: {(y_test == 1).sum()} positives, {(y_test == 0).sum()} negatives")
print(f"AUC: {roc_auc_score(y_test, probs):.4f}")