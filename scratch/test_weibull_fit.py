import sys
sys.path.insert(0, '.')
import numpy as np
from scipy.stats import weibull_min
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
correct = (preds == y_evt)

weibull_params = {}
for c in range(6):
    mask = (y_evt == c) & correct
    n_c = mask.sum()
    eta_c = min(20, max(5, int(np.floor(0.2 * n_c))))
    c_logits = logits[mask]
    mav_c = np.mean(c_logits, axis=0)
    dists = np.linalg.norm(c_logits - mav_c, axis=1)
    sorted_dists = np.sort(dists)
    tail_c = sorted_dists[-eta_c:]

    # Fit Weibull with floc = min(tail_c)
    floc_val = float(np.min(tail_c))
    try:
        shape, loc, scale = weibull_min.fit(tail_c, floc=floc_val)
        weibull_params[c] = {"shape": float(shape), "loc": float(loc), "scale": float(scale), "eta": eta_c}
        print(f"Class {c}: shape={shape:.4f}, loc={loc:.4f}, scale={scale:.4f}, eta={eta_c}")
    except Exception as e:
        print(f"Class {c} fitting failed: {e}")
