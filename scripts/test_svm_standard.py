
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.data_loading import load_dataset
from src.models.SVMstandard import StandardSVM


X, y, info = load_dataset("car_good")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
y_train, y_test = np.asarray(y_train), np.asarray(y_test)

print("Fitting StandardSVM (C=1.0)...")
t = time.perf_counter()
m = StandardSVM(C=1.0).fit(X_train_s, y_train)
print(f"  fit time: {time.perf_counter() - t:.3f}s")
print(f"  ||beta||: {np.linalg.norm(m.beta_):.4f}, beta0: {m.beta0_:+.4f}")
print(f"  AUC:      {roc_auc_score(y_test, m.predict_proba(X_test_s)):.4f}")