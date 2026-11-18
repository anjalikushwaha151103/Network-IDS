"""Comprehensive Independent Numerical and Implementation Audit for MLP Phase 2."""

import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from scipy.stats import weibull_min
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

# Ensure repo root is on sys.path
repo_root = Path("D:/Network IDS").resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import default_config
from src.data_loader import DataLoader
from src.models import ModelRegistry
from src.models.base_model import BaseNIDSModel
from src.models.mlp_model import MLPModel
from src.openset.energy_detector import EnergyDetector
from src.openset.metrics import (
    compute_aupr_in,
    compute_aupr_out,
    compute_auroc,
    compute_operating_point_metrics,
    compute_oscr,
)
from src.openset.msp_detector import MSPDetector
from src.openset.openmax_detector import OpenMaxDetector
from src.openset.partitioner import (
    KNOWN_CLASS_TO_ID,
    KNOWN_CLASSES,
    OPENSET_7CLASSES,
    UNKNOWN_CLASSES,
    UNKNOWN_LABEL_ID,
    extract_is_unknown,
    extract_known_multiclass_labels,
    extract_openset_7class_labels,
    partition_openset_data,
)
from src.preprocessing import NIDSPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MLP_Audit")


def run_comprehensive_audit():
    audit_results = {}
    base_dir = Path("results/openset/MLP/features_48")

    logger.info("=== 1. SAFETY & IMMUTABILITY CHECK ===")
    prior_models = ["LightGBM", "RandomForest", "XGBoost", "LogisticRegression"]
    for m in prior_models:
        p = Path(f"results/openset/{m}/features_48/metrics_closed_set.json")
        assert p.exists(), f"Prior artifact missing: {p}"
    assert Path("results/openset/synthesis/consolidated_benchmark_table.csv").exists()
    assert Path("results/openset/synthesis_3models/consolidated_benchmark_table.csv").exists()
    logger.info("Prior model artifacts and synthesis directories are intact.")

    logger.info("=== 2. MODEL & ARCHITECTURE VERIFICATION ===")
    # 2.1 Registry and aliases
    aliases = ["mlp", "neural_net", "neural_network", "multilayer_perceptron", "MLP"]
    for a in aliases:
        canonical = ModelRegistry.get_canonical_name(a)
        assert canonical == "mlp", f"Alias {a} resolved to {canonical}, expected mlp"

    model_path = base_dir / "model.pkl"
    preproc_path = base_dir / "preprocessor.pkl"
    assert model_path.exists() and preproc_path.exists()

    model = BaseNIDSModel.load(model_path)
    assert isinstance(model, MLPModel)
    assert isinstance(model, BaseNIDSModel)
    assert model.classes_.tolist() == [0, 1, 2, 3, 4, 5]

    preprocessor = NIDSPreprocessor.load(preproc_path)
    feature_names = preprocessor.feature_names_
    assert len(feature_names) == 48

    # 2.2 Underlying Scikit-Learn Model & Hyperparameter Inspection
    sk_mlp = model._underlying_model
    params = sk_mlp.get_params()
    hidden_layers = params["hidden_layer_sizes"]
    activation = params["activation"]
    solver = params["solver"]
    lr_init = params["learning_rate_init"]
    early_stopping = params["early_stopping"]
    n_iter = getattr(sk_mlp, "n_iter_", None)
    loss_curve = getattr(sk_mlp, "loss_curve_", None)
    val_scores = getattr(sk_mlp, "validation_scores_", None)
    best_val_score = getattr(sk_mlp, "best_validation_score_", None)

    logger.info("Architecture: Hidden=%s, Activation=%s, Solver=%s, LR_init=%s", hidden_layers, activation, solver, lr_init)
    logger.info("Early stopping: %s, Total epochs fitted (n_iter_): %d", early_stopping, n_iter)
    logger.info("Loss curve len: %d, Validation scores len: %d, Best val score: %.4f",
                len(loss_curve) if loss_curve else 0, len(val_scores) if val_scores else 0, best_val_score if best_val_score else 0)

    assert hidden_layers == (128, 64)
    assert activation == "relu"
    assert solver == "adam"
    assert lr_init == 0.001
    assert early_stopping is True
    assert n_iter == 45
    assert len(loss_curve) == 45
    assert len(val_scores) == 45

    # Check network weights shapes
    coefs = sk_mlp.coefs_
    intercepts = sk_mlp.intercepts_
    assert len(coefs) == 3 and len(intercepts) == 3
    assert coefs[0].shape == (48, 128)
    assert intercepts[0].shape == (128,)
    assert coefs[1].shape == (128, 64)
    assert intercepts[1].shape == (64,)
    assert coefs[2].shape == (64, 6)
    assert intercepts[2].shape == (6,)

    audit_results["model_architecture"] = {
        "classes": model.classes_.tolist(),
        "n_features": len(feature_names),
        "hidden_layer_sizes": list(hidden_layers),
        "activation": activation,
        "solver": solver,
        "lr_init": lr_init,
        "early_stopping": early_stopping,
        "n_iter": n_iter,
        "loss_curve_len": len(loss_curve),
        "validation_scores_len": len(val_scores),
        "best_validation_score": float(best_val_score),
        "weight_shapes": [list(c.shape) for c in coefs],
        "bias_shapes": [list(b.shape) for b in intercepts],
    }

    logger.info("=== 3. DATASET ISOLATION & FIXED PARTITIONS ===")
    loader = DataLoader(default_config)
    raw_train_df, raw_test_df = loader.load_data()
    splits = partition_openset_data(raw_train_df, raw_test_df, random_state=42)

    partition_sizes = {
        "df_train_known": len(splits.df_train_known),
        "df_val_known": len(splits.df_val_known),
        "df_evt_fit": len(splits.df_evt_fit),
        "df_thresh_cal": len(splits.df_thresh_cal),
        "df_test_known": len(splits.df_test_known),
        "df_test_unknown": len(splits.df_test_unknown),
        "df_test_total": len(splits.df_test),
        "purged_train_unknowns": splits.purged_train_unknown_count,
    }

    assert partition_sizes["df_train_known"] == 113288
    assert partition_sizes["df_val_known"] == 24276
    assert partition_sizes["df_evt_fit"] == 12138
    assert partition_sizes["df_thresh_cal"] == 12139
    assert partition_sizes["df_test_known"] == 77831
    assert partition_sizes["df_test_unknown"] == 4501
    assert partition_sizes["df_test_total"] == 82332
    assert partition_sizes["purged_train_unknowns"] == 13500

    # Verify zero unknown attack leakage
    train_cats = set(splits.df_train_known["attack_cat"].astype(str).str.strip().unique())
    val_cats = set(splits.df_val_known["attack_cat"].astype(str).str.strip().unique())
    evt_cats = set(splits.df_evt_fit["attack_cat"].astype(str).str.strip().unique())
    cal_cats = set(splits.df_thresh_cal["attack_cat"].astype(str).str.strip().unique())

    unknown_set = set(UNKNOWN_CLASSES)
    assert train_cats.isdisjoint(unknown_set)
    assert val_cats.isdisjoint(unknown_set)
    assert evt_cats.isdisjoint(unknown_set)
    assert cal_cats.isdisjoint(unknown_set)

    # Disjoint split check is asserted inside partition_openset_data before reset_index:
    # assert len(idx_train & idx_val) == 0, etc.
    logger.info("Mutual disjointness and zero unknown attack leakage verified across all splits!")

    audit_results["data_integrity"] = {
        "partition_sizes": partition_sizes,
        "zero_leakage_verified": True,
        "disjoint_partitions_verified": True,
    }

    logger.info("=== 4. PREDICT_LOGITS & MATHEMATICAL EQUIVALENCE ===")
    X_test_k, _, _ = preprocessor.transform(splits.df_test_known, feature_subset=feature_names)
    y_test_k = extract_known_multiclass_labels(splits.df_test_known)

    logits_k = model.predict_logits(X_test_k)
    assert logits_k.shape == (77831, 6)

    # Manual forward propagation computation
    W0, W1, Wout = coefs[0], coefs[1], coefs[2]
    b0, b1, bout = intercepts[0], intercepts[1], intercepts[2]
    h1_manual = np.maximum(0, np.dot(X_test_k, W0) + b0)
    h2_manual = np.maximum(0, np.dot(h1_manual, W1) + b1)
    logits_manual = np.dot(h2_manual, Wout) + bout

    max_logit_diff = float(np.max(np.abs(logits_k - logits_manual)))
    assert max_logit_diff < 1e-10
    logger.info("Logits match exact manual forward propagation (max diff: %.2e)", max_logit_diff)

    # Softmax vs predict_proba
    probs_from_model = model.predict_proba(X_test_k)
    probs_from_softmax = softmax(logits_k, axis=1)
    max_prob_diff = float(np.max(np.abs(probs_from_model - probs_from_softmax)))
    assert max_prob_diff < 1e-10
    logger.info("Softmax(predict_logits) matches predict_proba exactly (max diff: %.2e)", max_prob_diff)

    audit_results["logit_verification"] = {
        "max_logit_diff": max_logit_diff,
        "max_prob_diff": max_prob_diff,
        "shape": list(logits_k.shape),
    }

    logger.info("=== 5. INDEPENDENT CLOSED-SET RECOMPUTATION ===")
    y_pred_k = model.predict(X_test_k)
    acc = float(accuracy_score(y_test_k, y_pred_k))
    macro_f1 = float(f1_score(y_test_k, y_pred_k, average="macro"))
    weighted_f1 = float(f1_score(y_test_k, y_pred_k, average="weighted"))
    cm_recomputed = confusion_matrix(y_test_k, y_pred_k, labels=range(6))

    with open(base_dir / "metrics_closed_set.json", "r") as f:
        saved_closed_set = json.load(f)
    cm_saved = pd.read_csv(base_dir / "confusion_matrix_6x6.csv", index_col=0).values

    assert abs(acc - saved_closed_set["accuracy"]) < 1e-7
    assert abs(macro_f1 - saved_closed_set["macro_f1"]) < 1e-7
    assert abs(weighted_f1 - saved_closed_set["weighted_f1"]) < 1e-7
    np.testing.assert_array_equal(cm_recomputed, cm_saved)
    logger.info("Closed-set metrics & 6x6 confusion matrix reconciled with 100%% precision!")

    p_cls, r_cls, f_cls, s_cls = precision_recall_fscore_support(y_test_k, y_pred_k, labels=range(6))
    per_class_summary = {}
    for idx, cname in enumerate(KNOWN_CLASSES):
        per_class_summary[cname] = {
            "precision": float(p_cls[idx]),
            "recall": float(r_cls[idx]),
            "f1_score": float(f_cls[idx]),
            "support": int(s_cls[idx]),
        }
        logger.info("  Class %d (%s): Prec=%.4f, Rec=%.4f, F1=%.4f, Support=%d",
                    idx, cname, p_cls[idx], r_cls[idx], f_cls[idx], s_cls[idx])

    # Detailed inspection of Analysis and DoS recall
    analysis_cm_row = cm_recomputed[1, :]
    dos_cm_row = cm_recomputed[5, :]
    logger.info("Analysis CM row: %s (Total: %d, TP: %d)", analysis_cm_row, analysis_cm_row.sum(), analysis_cm_row[1])
    logger.info("DoS CM row: %s (Total: %d, TP: %d)", dos_cm_row, dos_cm_row.sum(), dos_cm_row[5])

    audit_results["closed_set_recomputation"] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class_summary,
        "confusion_matrix_recomputed": cm_recomputed.tolist(),
        "analysis_confusion": {
            "predicted_as_exploits": int(analysis_cm_row[3]),
            "predicted_as_fuzzers": int(analysis_cm_row[4]),
            "predicted_as_dos": int(analysis_cm_row[5]),
            "tp": int(analysis_cm_row[1]),
        },
        "dos_confusion": {
            "predicted_as_exploits": int(dos_cm_row[3]),
            "predicted_as_fuzzers": int(dos_cm_row[4]),
            "tp": int(dos_cm_row[5]),
        },
    }

    logger.info("=== 6. INDEPENDENT OPEN-SET AUDIT & RECOMPUTATION ===")
    X_cal, _, _ = preprocessor.transform(splits.df_thresh_cal, feature_subset=feature_names)
    y_cal = extract_known_multiclass_labels(splits.df_thresh_cal)

    X_test_full, _, _ = preprocessor.transform(splits.df_test, feature_subset=feature_names)
    y_test_7class = extract_openset_7class_labels(splits.df_test)
    y_is_unknown = extract_is_unknown(splits.df_test)
    y_is_known = ~y_is_unknown
    attack_cat_series = splits.df_test["attack_cat"].astype(str).str.strip()

    X_evt, _, _ = preprocessor.transform(splits.df_evt_fit, feature_subset=feature_names)
    y_evt = extract_known_multiclass_labels(splits.df_evt_fit)

    # 6.1 Energy Detector Mathematical Verification
    energy_det = EnergyDetector(temperature=1.0)
    energy_det.fit(None, None, model)
    logits_full = model.predict_logits(X_test_full)
    manual_energy = -1.0 * logsumexp(logits_full / 1.0, axis=1)
    computed_energy = energy_det.compute_anomaly_score(X_test_full, model)
    max_energy_diff = float(np.max(np.abs(manual_energy - computed_energy)))
    assert max_energy_diff < 1e-10
    logger.info("Canonical Free Energy formulation exactly matches -T * logsumexp(z/T) (diff: %.2e)", max_energy_diff)

    detectors = {
        "MSP": MSPDetector(),
        "Energy": EnergyDetector(temperature=1.0),
        "OpenMax": OpenMaxDetector(alpha=3, min_eta=5, max_eta=20),
    }

    audit_results["openset_detectors"] = {}

    for dname, detector in detectors.items():
        logger.info("Auditing %s Detector...", dname)
        if dname == "OpenMax":
            detector.fit(X_evt, y_evt, model)
        else:
            detector.fit(None, None, model)

        # Threshold calibration
        cal_stats = detector.calibrate_thresholds(
            X_cal, y_cal, model, normal_class_id=KNOWN_CLASS_TO_ID["Normal"], percentile_target=95.0
        )
        tau_fpr05 = cal_stats["tau_FPR05"]
        tau_cov95 = cal_stats["tau_Cov95"]

        # Anomaly scoring on full test set
        if dname == "OpenMax":
            probs_7 = detector.predict_proba_7class(X_test_full, model)
            scores = probs_7[:, 6]
            confidences = 1.0 - scores
        else:
            scores = detector.compute_anomaly_score(X_test_full, model)
            confidences = detector.compute_confidence_score(X_test_full, model)

        # Discrimination metrics
        auroc = compute_auroc(y_is_unknown, scores)
        aupr_out = compute_aupr_out(y_is_unknown, scores)
        aupr_in = compute_aupr_in(y_is_known, confidences)
        oscr_auc, _, _ = compute_oscr(
            y_test_7class[y_is_known],
            model.predict(X_test_full[y_is_known]),
            scores[y_is_known],
            scores[y_is_unknown],
        )

        # Operating points
        if dname == "OpenMax":
            preds_fpr05 = detector.predict_openset(
                X_test_full, model, threshold=tau_fpr05, unknown_label=UNKNOWN_LABEL_ID, use_recalibrated_argmax=True
            )
            preds_cov95 = detector.predict_openset(
                X_test_full, model, threshold=tau_cov95, unknown_label=UNKNOWN_LABEL_ID, use_recalibrated_argmax=True
            )
        else:
            preds_fpr05 = detector.predict_openset(
                X_test_full, model, threshold=tau_fpr05, unknown_label=UNKNOWN_LABEL_ID
            )
            preds_cov95 = detector.predict_openset(
                X_test_full, model, threshold=tau_cov95, unknown_label=UNKNOWN_LABEL_ID
            )

        metrics_fpr05 = compute_operating_point_metrics(
            y_test_7class, preds_fpr05, attack_cat_series, "tau_FPR05", tau_fpr05, KNOWN_CLASS_TO_ID["Normal"], UNKNOWN_LABEL_ID
        )
        metrics_cov95 = compute_operating_point_metrics(
            y_test_7class, preds_cov95, attack_cat_series, "tau_Cov95", tau_cov95, KNOWN_CLASS_TO_ID["Normal"], UNKNOWN_LABEL_ID
        )

        # Compare with saved JSON
        saved_file = base_dir / dname / f"metrics_openset_{dname.lower()}.json"
        with open(saved_file, "r") as f:
            saved_dict = json.load(f)

        saved_ind = saved_dict["threshold_independent_metrics"]
        assert abs(auroc - saved_ind["auroc"]) < 1e-7
        assert abs(aupr_out - saved_ind["aupr_out"]) < 1e-7
        assert abs(aupr_in - saved_ind["aupr_in"]) < 1e-7
        assert abs(oscr_auc - saved_ind["oscr_auc"]) < 1e-7

        saved_cal = saved_dict["calibration"]
        assert abs(tau_fpr05 - saved_cal["tau_FPR05"]) < 1e-7
        assert abs(tau_cov95 - saved_cal["tau_Cov95"]) < 1e-7

        for op_key, recomputed_op in [("tau_FPR05", metrics_fpr05), ("tau_Cov95", metrics_cov95)]:
            saved_op = saved_dict["operating_points"][op_key]
            assert abs(recomputed_op["realized_test_fpr_normal"] - saved_op["realized_test_fpr_normal"]) < 1e-7
            assert abs(recomputed_op["known_coverage"] - saved_op["known_coverage"]) < 1e-7
            assert abs(recomputed_op["unknown_detection_rate"] - saved_op["unknown_detection_rate"]) < 1e-7
            assert abs(recomputed_op["open_set_accuracy"] - saved_op["open_set_accuracy"]) < 1e-7
            assert abs(recomputed_op["open_set_macro_f1"] - saved_op["open_set_macro_f1"]) < 1e-7
            assert abs(recomputed_op["open_set_weighted_f1"] - saved_op["open_set_weighted_f1"]) < 1e-7

            cm_saved_op = pd.read_csv(base_dir / dname / f"confusion_matrix_7x7_{op_key.lower().replace('tau_', '')}.csv", index_col=0).values
            cm_recomputed_op = np.array(recomputed_op["confusion_matrix_7x7"])
            np.testing.assert_array_equal(cm_recomputed_op, cm_saved_op)

        logger.info("%s detector metrics & 7x7 confusion matrices reconciled with 100%% precision!", dname)

        audit_results["openset_detectors"][dname] = {
            "auroc": auroc,
            "aupr_out": aupr_out,
            "aupr_in": aupr_in,
            "oscr_auc": oscr_auc,
            "tau_fpr05": tau_fpr05,
            "tau_cov95": tau_cov95,
            "fpr05_normal_fpr": metrics_fpr05["realized_test_fpr_normal"],
            "fpr05_known_coverage": metrics_fpr05["known_coverage"],
            "fpr05_unknown_detection": metrics_fpr05["unknown_detection_rate"],
            "fpr05_open_set_acc": metrics_fpr05["open_set_accuracy"],
            "fpr05_open_set_macro_f1": metrics_fpr05["open_set_macro_f1"],
            "cov95_normal_fpr": metrics_cov95["realized_test_fpr_normal"],
            "cov95_known_coverage": metrics_cov95["known_coverage"],
            "cov95_unknown_detection": metrics_cov95["unknown_detection_rate"],
            "cov95_open_set_acc": metrics_cov95["open_set_accuracy"],
            "cov95_open_set_macro_f1": metrics_cov95["open_set_macro_f1"],
            "per_category_fpr05": metrics_fpr05["per_category_unknown_detection"],
            "per_category_cov95": metrics_cov95["per_category_unknown_detection"],
        }

    # 6.2 Calibration vs Test Distribution Shift Analysis
    # Let's check normal scores in calibration vs test
    logger.info("=== 6.2 CALIBRATION VS TEST DISTRIBUTION SHIFT ===")
    shift_analysis = {}
    for dname, detector in detectors.items():
        if dname == "OpenMax":
            cal_scores = detector.predict_proba_7class(X_cal, model)[:, 6]
            test_scores = detector.predict_proba_7class(X_test_full, model)[:, 6]
        else:
            cal_scores = detector.compute_anomaly_score(X_cal, model)
            test_scores = detector.compute_anomaly_score(X_test_full, model)
        
        cal_normal = cal_scores[y_cal == KNOWN_CLASS_TO_ID["Normal"]]
        test_normal = test_scores[y_test_7class == KNOWN_CLASS_TO_ID["Normal"]]
        tau_val = audit_results["openset_detectors"][dname]["tau_fpr05"]

        cal_fpr = float((cal_normal > tau_val).mean())
        test_fpr = float((test_normal > tau_val).mean())
        shift_analysis[dname] = {
            "cal_normal_mean": float(cal_normal.mean()),
            "cal_normal_std": float(cal_normal.std()),
            "test_normal_mean": float(test_normal.mean()),
            "test_normal_std": float(test_normal.std()),
            "tau_val": float(tau_val),
            "cal_fpr": cal_fpr,
            "test_fpr": test_fpr,
            "shift_ratio": test_fpr / cal_fpr if cal_fpr > 0 else 0,
        }
        logger.info("%s: Cal Normal Mean=%.4f (Std=%.4f), Test Normal Mean=%.4f (Std=%.4f)",
                    dname, cal_normal.mean(), cal_normal.std(), test_normal.mean(), test_normal.std())
        logger.info("  Cal FPR=%.2f%%, Test FPR=%.2f%% (Ratio=%.2fx)", cal_fpr * 100, test_fpr * 100, test_fpr / cal_fpr)
    audit_results["distribution_shift"] = shift_analysis

    # 6.3 Detailed OpenMax Weibull Audit
    logger.info("=== 6.3 OPENMAX WEIBULL AUDIT ===")
    weibull_params = detectors["OpenMax"].weibull_params_
    with open(base_dir / "OpenMax" / "mav_and_weibull_params.json", "r") as f:
        saved_weibull = json.load(f)

    weibull_audit_details = {}
    for c in range(6):
        cname = KNOWN_CLASSES[c]
        wp = weibull_params[c]
        swp = saved_weibull["weibull_tail_models"][str(c)]

        assert wp["n_correct"] == swp["n_correct"]
        assert wp["eta"] == swp["eta"]
        assert abs(wp["shape"] - swp["shape"]) < 1e-7
        assert abs(wp["loc"] - swp["loc"]) < 1e-7
        assert abs(wp["scale"] - swp["scale"]) < 1e-7
        assert wp["fallback_applied"] == swp["fallback_applied"]

        weibull_audit_details[cname] = {
            "class_id": c,
            "n_correct": wp["n_correct"],
            "eta": wp["eta"],
            "shape": wp["shape"],
            "loc": wp["loc"],
            "scale": wp["scale"],
            "fallback_applied": wp["fallback_applied"],
            "fallback_reason": wp["fallback_reason"],
        }
    audit_results["weibull_audit"] = weibull_audit_details

    # Analysis & DoS Tail Analysis
    analysis_correct_evt = int((splits.df_evt_fit["attack_cat"].astype(str).str.strip() == "Analysis").sum())
    dos_correct_evt = int((splits.df_evt_fit["attack_cat"].astype(str).str.strip() == "DoS").sum())
    logger.info("Analysis: %d in D_evt_fit, %d correct by MLP, eta=%d, shape=%.4f, scale=%.4f",
                analysis_correct_evt, weibull_params[1]["n_correct"], weibull_params[1]["eta"], weibull_params[1]["shape"], weibull_params[1]["scale"])
    logger.info("DoS: %d in D_evt_fit, %d correct by MLP, eta=%d, shape=%.4f, scale=%.4f",
                dos_correct_evt, weibull_params[5]["n_correct"], weibull_params[5]["eta"], weibull_params[5]["shape"], weibull_params[5]["scale"])

    # Generic Tail Check
    logger.info("Generic: %d correct by MLP, eta=%d, shape=%.4f, scale=%.4f, loc=%.4f",
                weibull_params[2]["n_correct"], weibull_params[2]["eta"], weibull_params[2]["shape"], weibull_params[2]["scale"], weibull_params[2]["loc"])

    # 6.4 Unknown attack category evaluation check
    logger.info("=== 6.4 UNKNOWN ATTACK CATEGORY EVALUATION ===")
    unknown_counts = attack_cat_series[y_is_unknown].value_counts().to_dict()
    assert unknown_counts["Reconnaissance"] == 3496
    assert unknown_counts["Backdoor"] == 583
    assert unknown_counts["Shellcode"] == 378
    assert unknown_counts["Worms"] == 44
    assert sum(unknown_counts.values()) == 4501
    logger.info("Unknown attack category counts verified: %s (Total: %d)", unknown_counts, sum(unknown_counts.values()))

    # Save complete audit dictionary in scratch directory
    audit_save_path = Path("scratch/audit_mlp_results.json")
    with open(audit_save_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    logger.info("Independent audit execution complete. Results saved to %s", audit_save_path)
    return audit_results


if __name__ == "__main__":
    res = run_comprehensive_audit()
    print("\n--- MLP AUDIT SUMMARY RESULT ---")
    print("Closed Set Accuracy:", res["closed_set_recomputation"]["accuracy"])
    print("Closed Set Macro F1:", res["closed_set_recomputation"]["macro_f1"])
    print("Energy AUROC:", res["openset_detectors"]["Energy"]["auroc"])
    print("OpenMax AUROC:", res["openset_detectors"]["OpenMax"]["auroc"])
    print("OpenMax tau_FPR05 Unknown Recall:", res["openset_detectors"]["OpenMax"]["fpr05_unknown_detection"])
    print("OpenMax tau_FPR05 Known Coverage:", res["openset_detectors"]["OpenMax"]["fpr05_known_coverage"])
