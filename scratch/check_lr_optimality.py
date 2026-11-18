"""Check gradient norm and loss optimality of the fitted Logistic Regression model."""

import numpy as np
from scipy.special import softmax
import pickle
import sys
from pathlib import Path

repo_root = Path("D:/Network IDS").resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data_loader import DataLoader
from src.openset.partitioner import partition_openset_data, extract_known_multiclass_labels
from src.preprocessing import NIDSPreprocessor

base_dir = Path("results/openset/LogisticRegression/features_48")
with open(base_dir / "model.pkl", "rb") as f:
    model = pickle.load(f)

preprocessor = NIDSPreprocessor.load(base_dir / "preprocessor.pkl")

loader = DataLoader()
raw_train, raw_test = loader.load_data()
splits = partition_openset_data(raw_train, raw_test, random_state=42)

X_tr, _, _ = preprocessor.transform(splits.df_train_known)
y_tr = extract_known_multiclass_labels(splits.df_train_known)

# Recompute probabilities on training set
W = model._underlying_model.coef_  # (6, 48)
b = model._underlying_model.intercept_  # (6,)

logits = np.dot(X_tr, W.T) + b  # (113288, 6)
probs = softmax(logits, axis=1)

# One-hot encode y_tr
N, K = probs.shape
Y_onehot = np.zeros((N, K))
Y_onehot[np.arange(N), y_tr] = 1.0

# Cross-entropy loss
log_probs = np.log(np.clip(probs, 1e-15, 1.0))
loss = -np.mean(np.sum(Y_onehot * log_probs, axis=1))

# Unregularized gradient w.r.t W: (1/N) * (P - Y)^T X + (1/C) * W
C = model._underlying_model.C
diff = probs - Y_onehot
grad_W = (1.0 / N) * np.dot(diff.T, X_tr) + (1.0 / C) * W
grad_b = np.mean(diff, axis=0)

grad_norm_W = np.linalg.norm(grad_W)
grad_norm_b = np.linalg.norm(grad_b)

print(f"Training Cross-Entropy Loss: {loss:.6f}")
print(f"Gradient Norm (W): {grad_norm_W:.6e}")
print(f"Gradient Norm (b): {grad_norm_b:.6e}")
print(f"Maximum absolute gradient element: {np.max(np.abs(grad_W)):.6e}")
