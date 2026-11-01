"""Comprehensive Read-Only Consistency Audit for MSP and Energy Open-Set Detectors."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

MSP_DIR = Path("results/openset/LightGBM/features_48/MSP")
ENERGY_DIR = Path("results/openset/LightGBM/features_48/Energy")
BASE_DIR = Path("results/openset/LightGBM/features_48")

def get_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def analyze_matrix(cm_path: Path):
    cm_df = pd.read_csv(cm_path, index_col=0)
    cm = cm_df.values
    classes = cm_df.index.tolist()
    total = int(cm.sum())
    diag = np.diag(cm)
    correct = int(diag.sum())

    acc = correct / total

    tp = diag.astype(float)
    fn = (cm.sum(axis=1) - tp).astype(float)
    fp = (cm.sum(axis=0) - tp).astype(float)
    support = cm.sum(axis=1).astype(float)

    prec = np.divide(tp, (tp + fp), out=np.zeros_like(tp), where=(tp + fp) != 0)
    rec = np.divide(tp, (tp + fn), out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * prec * rec, (prec + rec), out=np.zeros_like(tp), where=(prec + rec) != 0)

    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.sum(f1 * support) / total)

    known_support = int(support[:6].sum())
    known_rejected = int(cm[:6, 6].sum())
    known_rejection_rate = known_rejected / known_support
    known_coverage = 1.0 - known_rejection_rate

    normal_support = int(support[0])
    normal_rejected = int(cm[0, 6])
    normal_fpr = normal_rejected / normal_support

    unknown_support = int(support[6])
    unknown_detected = int(cm[6, 6])
    unknown_recall = unknown_detected / unknown_support

    per_class = {}
    for i, c in enumerate(classes):
        per_class[c] = {
            "support": int(support[i]),
            "tp": int(tp[i]),
            "fn": int(fn[i]),
            "fp": int(fp[i]),
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
        }

    return {
        "classes": classes,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "normal_support": normal_support,
        "normal_rejected": normal_rejected,
        "normal_fpr": normal_fpr,
        "known_support": known_support,
        "known_rejected": known_rejected,
        "known_rejection_rate": known_rejection_rate,
        "known_coverage": known_coverage,
        "unknown_support": unknown_support,
        "unknown_detected": unknown_detected,
        "unknown_recall": unknown_recall,
        "per_class": per_class,
        "row_supports": {c: int(support[i]) for i, c in enumerate(classes)},
    }

def run_audit():
    print("=" * 80)
    print("COMPREHENSIVE READ-ONLY AUDIT: MSP & ENERGY ARTIFACTS")
    print("=" * 80)

    # 1. File inventory & SHA256
    print("\n--- 1. FILE INVENTORY AND INTEGRITY ---")
    for name, directory in [("MSP", MSP_DIR), ("ENERGY", ENERGY_DIR)]:
        print(f"\nDirectory: {directory}")
        for p in sorted(directory.iterdir()):
            if p.is_file():
                h = get_file_hash(p)
                size = p.stat().st_size
                mtime = pd.to_datetime(p.stat().st_mtime, unit='s')
                print(f"  {p.name:36s} | {size:8d} B | {mtime} | SHA256: {h[:12]}...")

    # 2. Base model check
    print("\n--- 2. BASE CLASSIFIER ARTIFACTS ---")
    model_file = BASE_DIR / "model.pkl"
    preproc_file = BASE_DIR / "preprocessor.pkl"
    print(f"Base Model:        {model_file} (SHA256: {get_file_hash(model_file)[:12]}...)")
    print(f"Base Preprocessor: {preproc_file} (SHA256: {get_file_hash(preproc_file)[:12]}...)")

    # 3. Experiment configs
    print("\n--- 3. EXPERIMENT CONFIGURATIONS ---")
    with open(MSP_DIR / "experiment_config.json") as f:
        msp_cfg = json.load(f)
    with open(ENERGY_DIR / "experiment_config.json") as f:
        energy_cfg = json.load(f)

    print("MSP Config:")
    for k, v in msp_cfg.items():
        print(f"  {k}: {v}")
    print("\nEnergy Config:")
    for k, v in energy_cfg.items():
        print(f"  {k}: {v}")

    # 4. Calibration statistics
    print("\n--- 4. CALIBRATION STATISTICS ---")
    with open(MSP_DIR / "thresholds.json") as f:
        msp_thresh = json.load(f)
    with open(ENERGY_DIR / "thresholds.json") as f:
        energy_thresh = json.load(f)
    print("MSP Thresholds:")
    for k, v in msp_thresh.items():
        print(f"  {k}: {v}")
    print("Energy Thresholds:")
    for k, v in energy_thresh.items():
        print(f"  {k}: {v}")

    # 5. Confusion Matrix Recalculations vs JSON & CSV
    print("\n--- 5. CONFUSION MATRIX RECALCULATIONS VS JSON/CSV ---")
    results = {}
    for det, det_dir, jname, cname in [
        ("MSP", MSP_DIR, "metrics_openset_msp.json", "metrics_openset_msp.csv"),
        ("Energy", ENERGY_DIR, "metrics_openset_energy.json", "metrics_openset_energy.csv")
    ]:
        with open(det_dir / jname) as f:
            jdata = json.load(f)
        cdf = pd.read_csv(det_dir / cname)
        cdict = dict(zip(cdf["metric"], cdf["value"]))

        for op, cm_fname in [("tau_FPR05", "confusion_matrix_7x7_fpr05.csv"), ("tau_Cov95", "confusion_matrix_7x7_cov95.csv")]:
            cm_path = det_dir / cm_fname
            m_res = analyze_matrix(cm_path)
            j_op = jdata["operating_points"][op]

            key = f"{det}_{op}"
            results[key] = {
                "matrix": m_res,
                "json": j_op,
                "csv": cdict,
                "json_full": jdata,
            }

            print(f"\n--- Detector: {det} | Operating Point: {op} ---")
            print(f"Source Matrix: {cm_path}")
            print(f"Total Flows:               {m_res['total']} (JSON: {int(sum(m_res['row_supports'].values()))})")
            print(f"Accuracy:                  Matrix={m_res['accuracy']:.8f} | JSON={j_op['open_set_accuracy']:.8f} | Diff={abs(m_res['accuracy'] - j_op['open_set_accuracy']):.2e}")
            print(f"Macro F1:                  Matrix={m_res['macro_f1']:.8f} | JSON={j_op['open_set_macro_f1']:.8f} | Diff={abs(m_res['macro_f1'] - j_op['open_set_macro_f1']):.2e}")
            print(f"Weighted F1:               Matrix={m_res['weighted_f1']:.8f} | JSON={j_op['open_set_weighted_f1']:.8f} | Diff={abs(m_res['weighted_f1'] - j_op['open_set_weighted_f1']):.2e}")
            print(f"Normal FPR:                Matrix={m_res['normal_fpr']:.8f} | JSON={j_op['realized_test_fpr_normal']:.8f} ({m_res['normal_rejected']}/{m_res['normal_support']})")
            print(f"Known Coverage:            Matrix={m_res['known_coverage']:.8f} | JSON={j_op['known_coverage']:.8f} (Rejected: {m_res['known_rejected']}/{m_res['known_support']})")
            print(f"Unknown Recall:            Matrix={m_res['unknown_recall']:.8f} | JSON={j_op['unknown_detection_rate']:.8f} ({m_res['unknown_detected']}/{m_res['unknown_support']})")

            # Check CSV agreement
            csv_thresh = cdict.get(f"{op}_threshold")
            csv_acc = cdict.get(f"{op}_open_set_accuracy")
            csv_mf1 = cdict.get(f"{op}_open_set_macro_f1")
            csv_wf1 = cdict.get(f"{op}_open_set_weighted_f1")
            csv_norm_fpr = cdict.get(f"{op}_realized_test_fpr_normal")
            csv_cov = cdict.get(f"{op}_known_coverage")
            csv_unk_rec = cdict.get(f"{op}_unknown_detection_rate")

            print(f"CSV Cross-Check:")
            print(f"  Threshold in CSV:        {csv_thresh}")
            if csv_acc is not None:
                print(f"  Accuracy in CSV:         {csv_acc:.8f} (Match: {np.isclose(m_res['accuracy'], csv_acc)})")
            if csv_mf1 is not None:
                print(f"  Macro F1 in CSV:         {csv_mf1:.8f} (Match: {np.isclose(m_res['macro_f1'], csv_mf1)})")
            if csv_wf1 is not None:
                print(f"  Weighted F1 in CSV:      {csv_wf1:.8f} (Match: {np.isclose(m_res['weighted_f1'], csv_wf1)})")
            if csv_norm_fpr is not None:
                print(f"  Normal FPR in CSV:       {csv_norm_fpr:.8f} (Match: {np.isclose(m_res['normal_fpr'], csv_norm_fpr)})")
            if csv_cov is not None:
                print(f"  Known Coverage in CSV:   {csv_cov:.8f} (Match: {np.isclose(m_res['known_coverage'], csv_cov)})")
            if csv_unk_rec is not None:
                print(f"  Unknown Recall in CSV:   {csv_unk_rec:.8f} (Match: {np.isclose(m_res['unknown_recall'], csv_unk_rec)})")

    # 6. Per-Category Unknown Detection
    print("\n--- 6. PER-CATEGORY UNKNOWN DETECTION COMPARISON ---")
    msp_cat_df = pd.read_csv(MSP_DIR / "per_category_unknown_detection.csv")
    energy_cat_df = pd.read_csv(ENERGY_DIR / "per_category_unknown_detection.csv")
    print("\nMSP Per-Category (CSV):")
    print(msp_cat_df.to_string(index=False))
    print("\nEnergy Per-Category (CSV):")
    print(energy_cat_df.to_string(index=False))

    # Cross-check with JSON per_category_unknown_detection
    print("\nVerifying JSON per-category dictionaries match CSV:")
    for det, det_dir, jname in [("MSP", MSP_DIR, "metrics_openset_msp.json"), ("Energy", ENERGY_DIR, "metrics_openset_energy.json")]:
        with open(det_dir / jname) as f:
            jd = json.load(f)
        cat_csv = pd.read_csv(det_dir / "per_category_unknown_detection.csv")
        for _, row in cat_csv.iterrows():
            cname = row["attack_category"]
            # Check FPR05
            j_fpr05 = jd["operating_points"]["tau_FPR05"]["per_category_unknown_detection"][cname]
            assert j_fpr05["support"] == row["support"]
            assert j_fpr05["detected"] == row["detected_tau_FPR05"]
            assert np.isclose(j_fpr05["recall"], row["recall_tau_FPR05"])
            # Check Cov95
            j_cov95 = jd["operating_points"]["tau_Cov95"]["per_category_unknown_detection"][cname]
            assert j_cov95["support"] == row["support"]
            assert j_cov95["detected"] == row["detected_tau_Cov95"]
            assert np.isclose(j_cov95["recall"], row["recall_tau_Cov95"])
        print(f"  {det} JSON and CSV per-category match EXACTLY.")

    # 7. Test set class supports check across all 4 confusion matrices
    print("\n--- 7. TEST SET SUPPORTS ACROSS ALL MATRICES ---")
    for k, v in results.items():
        print(f"{k:18s} Row Supports: {v['matrix']['row_supports']}")

    return results

if __name__ == '__main__':
    run_audit()
