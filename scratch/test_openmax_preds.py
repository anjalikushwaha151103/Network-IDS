import sys
sys.path.insert(0, '.')
import numpy as np
from scipy.stats import weibull_min
from src.data_loader import DataLoader
from src.models.lightgbm_model import LightGBMModel
from src.preprocessing import NIDSPreprocessor
from src.openset.partitioner import partition_openset_data, extract_known_multiclass_labels, extract_is_unknown

model = LightGBMModel.load("results/openset/LightGBM/features_48/model.pkl")
preprocessor = NIDSPreprocessor.load("results/openset/LightGBM/features_48/preprocessor.pkl")

loader = DataLoader()
train_df, test_df = loader.load_data()
splits = partition_openset_data(train_df, test_df, random_state=42)

df_evt = splits.df_evt_fit
X_evt, _, _ = preprocessor.transform(df_evt, feature_subset=preprocessor.feature_names_)
y_evt = extract_known_multiclass_labels(df_evt)

logits_evt = model.predict_logits(X_evt)
preds_evt = model.predict(X_evt)
correct = (preds_evt == y_evt)

mavs = {}
weibulls = {}
for c in range(6):
    mask = (y_evt == c) & correct
    n_c = mask.sum()
    eta_c = min(20, max(5, int(np.floor(0.2 * n_c))))
    c_logits = logits_evt[mask]
    mav_c = np.mean(c_logits, axis=0)
    mavs[c] = mav_c
    dists = np.linalg.norm(c_logits - mav_c, axis=1)
    sorted_dists = np.sort(dists)
    tail_c = sorted_dists[-eta_c:]
    shape, loc, scale = weibull_min.fit(tail_c, floc=float(np.min(tail_c)))
    weibulls[c] = (shape, loc, scale)

df_test = splits.df_test
X_test, _, _ = preprocessor.transform(df_test.iloc[:5000], feature_subset=preprocessor.feature_names_)
y_unk = extract_is_unknown(df_test.iloc[:5000])
logits_test = model.predict_logits(X_test)
base_preds = model.predict(X_test)

alpha = 3
N = len(logits_test)
openmax_probs = np.zeros((N, 7))

for i in range(N):
    z = logits_test[i]
    ranked = np.argsort(z)[::-1]
    w = np.ones(6)
    for r in range(alpha):
        c = ranked[r]
        dist = np.linalg.norm(z - mavs[c])
        shape, loc, scale = weibulls[c]
        omega = 0.0 if dist <= loc else weibull_min.cdf(dist, shape, loc=loc, scale=scale)
        w[c] = 1.0 - ((alpha - (r + 1) + 1) / alpha) * omega
    
    z_hat_known = z * w
    z_hat_unk = np.sum(z[ranked[:alpha]] * (1.0 - w[ranked[:alpha]]))
    z_7 = np.append(z_hat_known, z_hat_unk)
    exp_z = np.exp(z_7 - np.max(z_7))
    openmax_probs[i] = exp_z / np.sum(exp_z)

openmax_known_preds = np.argmax(openmax_probs[:, :6], axis=1)
diff_count = (openmax_known_preds != base_preds).sum()
print(f"Differences between openmax_known_preds and base_preds on 5000 test samples: {diff_count} / 5000 ({diff_count/50:.2f}%)")
