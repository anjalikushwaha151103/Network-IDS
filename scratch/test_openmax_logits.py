import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from pathlib import Path
from src.data_loader import DataLoader
from src.models.lightgbm_model import LightGBMModel
from src.preprocessing import NIDSPreprocessor
from src.openset.partitioner import partition_openset_data, extract_known_multiclass_labels

model = LightGBMModel.load("results/openset/LightGBM/features_48/model.pkl")
preprocessor = NIDSPreprocessor.load("results/openset/LightGBM/features_48/preprocessor.pkl")

loader = DataLoader()
train_df, test_df = loader.load_data()
splits = partition_openset_data(train_df, test_df, random_state=42)

df_evt = splits.df_evt_fit
X_evt, _, _ = preprocessor.transform(df_evt, feature_subset=preprocessor.feature_names_)
y_evt = extract_known_multiclass_labels(df_evt)

logits = model.predict_logits(X_evt)
preds = model.predict(X_evt)

print(f"Logits shape: {logits.shape}")
print(f"Logits min: {logits.min():.4f}, max: {logits.max():.4f}, mean: {logits.mean():.4f}")

correct = (preds == y_evt)
print(f"Correctly classified samples in D_evt_fit: {correct.sum()} / {len(y_evt)} ({correct.mean()*100:.2f}%)")

for c in range(6):
    mask = (y_evt == c) & correct
    n_c = mask.sum()
    eta_c = min(20, max(5, int(np.floor(0.2 * n_c))))
    c_logits = logits[mask]
    mav_c = np.mean(c_logits, axis=0)
    dists = np.linalg.norm(c_logits - mav_c, axis=1)
    print(f"Class {c}: total={sum(y_evt==c)}, correct={n_c}, eta={eta_c}, dist min={dists.min():.4f}, max={dists.max():.4f}, mean={dists.mean():.4f}")
