"""MSP Results Consistency Audit Script."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

cm_fpr05_path = Path("results/openset/LightGBM/features_48/MSP/confusion_matrix_7x7_fpr05.csv")
cm_cov95_path = Path("results/openset/LightGBM/features_48/MSP/confusion_matrix_7x7_cov95.csv")
metrics_json_path = Path("results/openset/LightGBM/features_48/MSP/metrics_openset_msp.json")
metrics_csv_path = Path("results/openset/LightGBM/features_48/MSP/metrics_openset_msp.csv")
per_cat_csv_path = Path("results/openset/LightGBM/features_48/MSP/per_category_unknown_detection.csv")

cm_fpr05 = pd.read_csv(cm_fpr05_path, index_col=0)
cm_cov95 = pd.read_csv(cm_cov95_path, index_col=0)

with open(metrics_json_path, "r", encoding="utf-8") as f:
    mj = json.load(f)

csv_df = pd.read_csv(metrics_csv_path)
csv_dict = dict(zip(csv_df["metric"], csv_df["value"]))

per_cat_df = pd.read_csv(per_cat_csv_path)

def audit_matrix(cm_df, tag):
    cm = cm_df.values
    classes = cm_df.index.tolist()
    total = int(cm.sum())
    diag = np.diag(cm)
    correct = int(diag.sum())

    # 1. Accuracy
    acc = correct / total

    # 2. Per-class metrics
    tp = diag.astype(float)
    fn = (cm.sum(axis=1) - tp).astype(float)
    fp = (cm.sum(axis=0) - tp).astype(float)
    support = cm.sum(axis=1).astype(float)

    prec = np.divide(tp, (tp + fp), out=np.zeros_like(tp), where=(tp + fp) != 0)
    rec = np.divide(tp, (tp + fn), out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * prec * rec, (prec + rec), out=np.zeros_like(tp), where=(prec + rec) != 0)

    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.sum(f1 * support) / total)

    # 3. Known coverage & rejection rate
    # Rows 0..5 are known classes; column 6 is 'Unknown'
    known_support = int(support[:6].sum())
    known_rejected = int(cm[:6, 6].sum())
    known_rejection_rate = known_rejected / known_support
    known_coverage = 1.0 - known_rejection_rate

    # Normal FPR
    normal_support = int(support[0])
    normal_rejected = int(cm[0, 6])
    normal_fpr = normal_rejected / normal_support

    # Unknown recall
    unknown_support = int(support[6])
    unknown_detected = int(cm[6, 6])
    unknown_recall = unknown_detected / unknown_support

    json_op = mj["operating_points"][tag]

    print(f"=== AUDIT FOR {tag} ===")
    print(f"Matrix Total Sum:        {total} (Expected: 82332) -> Match: {total == 82332}")
    print(f"Accuracy:                Matrix={acc:.8f}, JSON={json_op['open_set_accuracy']:.8f} -> Match: {np.isclose(acc, json_op['open_set_accuracy'])}")
    print(f"Macro F1:                Matrix={macro_f1:.8f}, JSON={json_op['open_set_macro_f1']:.8f} -> Match: {np.isclose(macro_f1, json_op['open_set_macro_f1'])}")
    print(f"Weighted F1:             Matrix={weighted_f1:.8f}, JSON={json_op['open_set_weighted_f1']:.8f} -> Match: {np.isclose(weighted_f1, json_op['open_set_weighted_f1'])}")
    print(f"Normal FPR:              Matrix={normal_fpr:.8f}, JSON={json_op['realized_test_fpr_normal']:.8f} -> Match: {np.isclose(normal_fpr, json_op['realized_test_fpr_normal'])}")
    print(f"Known Rejection Rate:    Matrix={known_rejection_rate:.8f}, JSON={json_op['realized_test_fpr_known']:.8f} -> Match: {np.isclose(known_rejection_rate, json_op['realized_test_fpr_known'])}")
    print(f"Known Coverage:          Matrix={known_coverage:.8f}, JSON={json_op['known_coverage']:.8f} -> Match: {np.isclose(known_coverage, json_op['known_coverage'])}")
    print(f"Unknown Recall:          Matrix={unknown_recall:.8f}, JSON={json_op['unknown_detection_rate']:.8f} -> Match: {np.isclose(unknown_recall, json_op['unknown_detection_rate'])}")

    print("\n--- Cross-check with CSV metrics ---")
    print(f"CSV Threshold:           {csv_dict.get(f'{tag}_threshold')}")
    print(f"CSV Normal FPR:          {csv_dict.get(f'{tag}_realized_test_fpr_normal'):.8f} -> Match: {np.isclose(normal_fpr, csv_dict.get(f'{tag}_realized_test_fpr_normal'))}")
    print(f"CSV Known Coverage:      {csv_dict.get(f'{tag}_known_coverage'):.8f} -> Match: {np.isclose(known_coverage, csv_dict.get(f'{tag}_known_coverage'))}")
    print(f"CSV Unknown Recall:      {csv_dict.get(f'{tag}_unknown_detection_rate'):.8f} -> Match: {np.isclose(unknown_recall, csv_dict.get(f'{tag}_unknown_detection_rate'))}")
    print(f"CSV Macro F1:            {csv_dict.get(f'{tag}_open_set_macro_f1'):.8f} -> Match: {np.isclose(macro_f1, csv_dict.get(f'{tag}_open_set_macro_f1'))}")

    # Check per-class classification report in JSON
    rep = json_op["classification_report_7class"]
    print("\n--- Per-class report consistency ---")
    for i, cname in enumerate(classes):
        rep_p = rep[cname]["precision"]
        rep_r = rep[cname]["recall"]
        rep_f = rep[cname]["f1-score"]
        rep_s = rep[cname]["support"]
        p_match = np.isclose(prec[i], rep_p)
        r_match = np.isclose(rec[i], rep_r)
        f_match = np.isclose(f1[i], rep_f)
        s_match = (support[i] == rep_s)
        assert p_match and r_match and f_match and s_match, f"Discrepancy in class {cname}!"
        print(f"  {cname:12s} Prec: {prec[i]:.4f} (rep={rep_p:.4f}), Rec: {rec[i]:.4f} (rep={rep_r:.4f}), F1: {f1[i]:.4f} (rep={rep_f:.4f}), Supp: {int(support[i])} (rep={int(rep_s)}) [OK]")
    print()

audit_matrix(cm_fpr05, "tau_FPR05")
audit_matrix(cm_cov95, "tau_Cov95")
