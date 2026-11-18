"""Detailed audit of Analysis (Class 1) and DoS (Class 5) Weibull tail fits."""

import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import weibull_min

repo_root = Path("D:/Network IDS").resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data_loader import DataLoader
from src.models.base_model import BaseNIDSModel
from src.openset.partitioner import partition_openset_data, extract_known_multiclass_labels
from src.preprocessing import NIDSPreprocessor

base_dir = Path("results/openset/LogisticRegression/features_48")
model = BaseNIDSModel.load(base_dir / "model.pkl")
preprocessor = NIDSPreprocessor.load(base_dir / "preprocessor.pkl")

loader = DataLoader()
raw_train, raw_test = loader.load_data()
splits = partition_openset_data(raw_train, raw_test, random_state=42)

X_evt, _, _ = preprocessor.transform(splits.df_evt_fit)
y_evt = extract_known_multiclass_labels(splits.df_evt_fit)

logits_evt = model.predict_logits(X_evt)
preds_evt = model.predict(X_evt)

# 1. Analysis (Class 1)
class_1_correct = (y_evt == 1) & (preds_evt == 1)
n_c1 = int(class_1_correct.sum())
print(f"Analysis total in D_evt_fit: {(y_evt == 1).sum()}")
print(f"Analysis correctly classified by LR (n_c1): {n_c1}")

logits_c1 = logits_evt[class_1_correct]
mav_1 = np.mean(logits_c1, axis=0)
dists_c1 = np.linalg.norm(logits_c1 - mav_1, axis=1)
sorted_dists_c1 = np.sort(dists_c1)
print("All 9 sorted distances for Analysis:", [round(x, 4) for x in sorted_dists_c1])

eta_c1 = 5
tail_c1 = sorted_dists_c1[-eta_c1:]
floc_c1 = float(np.min(tail_c1))
print("Tail distances (eta=5):", [round(x, 4) for x in tail_c1])
print("Tail min (floc):", floc_c1)
print("Tail max:", float(np.max(tail_c1)))
print("Tail span:", float(np.max(tail_c1) - floc_c1))

shape_c1, loc_c1, scale_c1 = weibull_min.fit(tail_c1, floc=floc_c1)
print(f"Fitted Weibull: shape={shape_c1:.6f}, loc={loc_c1:.6f}, scale={scale_c1:.6f}")

# Bootstrap sensitivity analysis on the 5 tail distances
print("\nBootstrap Sensitivity Check (B=1000) on Analysis tail:")
boot_shapes, boot_scales = [], []
rng = np.random.default_rng(42)
for _ in range(1000):
    b_tail = rng.choice(tail_c1, size=len(tail_c1), replace=True)
    b_floc = float(np.min(b_tail))
    if np.max(b_tail) - b_floc > 1e-4:
        try:
            b_sh, b_loc, b_sc = weibull_min.fit(b_tail, floc=b_floc)
            if np.isfinite(b_sh) and np.isfinite(b_sc) and b_sh > 0 and b_sc > 0:
                boot_shapes.append(b_sh)
                boot_scales.append(b_sc)
        except Exception:
            pass

print(f"Bootstrap Shapes: Median={np.median(boot_shapes):.4f}, 5th={np.percentile(boot_shapes, 5):.4f}, 95th={np.percentile(boot_shapes, 95):.4f}")
print(f"Bootstrap Scales: Median={np.median(boot_scales):.4f}, 5th={np.percentile(boot_scales, 5):.4f}, 95th={np.percentile(boot_scales, 95):.4f}")

# 2. DoS (Class 5)
class_5_correct = (y_evt == 5) & (preds_evt == 5)
n_c5 = int(class_5_correct.sum())
print(f"\nDoS total in D_evt_fit: {(y_evt == 5).sum()}")
print(f"DoS correctly classified by LR (n_c5): {n_c5}")
eta_c5 = min(20, max(5, int(np.floor(0.2 * n_c5))))
print(f"DoS eta rule: min(20, max(5, floor(0.2 * {n_c5}))) = {eta_c5}")

logits_c5 = logits_evt[class_5_correct]
mav_5 = np.mean(logits_c5, axis=0)
dists_c5 = np.linalg.norm(logits_c5 - mav_5, axis=1)
tail_c5 = np.sort(dists_c5)[-eta_c5:]
floc_c5 = float(np.min(tail_c5))
shape_c5, loc_c5, scale_c5 = weibull_min.fit(tail_c5, floc=floc_c5)
print(f"DoS Fitted Weibull: shape={shape_c5:.6f}, loc={loc_c5:.6f}, scale={scale_c5:.6f}")
