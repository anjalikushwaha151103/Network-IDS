"""Comprehensive Independent Numerical and Implementation Audit for Logistic Regression Phase 2."""

import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
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
from src.models.logistic_regression import LogisticRegressionModel
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
logger = logging.getLogger("LR_Audit")


def run_comprehensive_audit():
    audit_results = {}
    base_dir = Path("results/openset/LogisticRegression/features_48")

    logger.info("=== 1. MODEL & IMPLEMENTATION VERIFICATION ===")
    # 1.1 Registry and Class hierarchy
    canonical = ModelRegistry.get_canonical_name("logistic_regression")
    aliases = ["lr", "logistic", "logisticregression", "logistic_regression"]
    resolved_aliases = [ModelRegistry.get_canonical_name(a) for a in aliases]
    assert all(r == "logistic_regression" for r in resolved_aliases)

    model_path = base_dir / "model.pkl"
    preproc_path = base_dir / "preprocessor.pkl"
    assert model_path.exists() and preproc_path.exists()

    model = BaseNIDSModel.load(model_path)
    assert isinstance(model, LogisticRegressionModel)
    assert isinstance(model, BaseNIDSModel)
    assert model.classes_.tolist() == [0, 1, 2, 3, 4, 5]

    preprocessor = NIDSPreprocessor.load(preproc_path)
    feature_names = preprocessor.feature_names_
    assert len(feature_names) == 48

    # 1.2 Underlying Scikit-Learn Model & Convergence Inspection
    sk_lr = model._underlying_model
    lr_params = sk_lr.get_params()
    n_iter = int(sk_lr.n_iter_[0])
    max_iter = int(lr_params["max_iter"])
    converged = n_iter < max_iter

    logger.info("Underlying Solver: %s, C: %s, max_iter: %d, tol: %s", lr_params["solver"], lr_params["C"], max_iter, lr_params["tol"])
    logger.info("Optimizer Iterations: %d / %d (Converged: %s)", n_iter, max_iter, converged)
    logger.info("Weight Matrix Shape: %s, Intercept Shape: %s", sk_lr.coef_.shape, sk_lr.intercept_.shape)

    audit_results["model_verification"] = {
        "classes": model.classes_.tolist(),
        "n_features": len(feature_names),
        "solver": lr_params["solver"],
        "C": lr_params["C"],
        "max_iter": max_iter,
        "n_iter": n_iter,
        "converged": converged,
        "random_state": lr_params["random_state"],
    }

    logger.info("=== 2. DATA INTEGRITY & PARTITION VERIFICATION ===")
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
    logger.info("Zero unknown attack leakage verified across all 4 training/calibration splits!")

    audit_results["data_integrity"] = {
        "partition_sizes": partition_sizes,
        "zero_leakage_verified": True,
    }

    logger.info("=== 3. CLOSED-SET NUMERICAL RECOMPUTATION ===")
    X_test_k, _, _ = preprocessor.transform(splits.df_test_known, feature_subset=feature_names)
    y_test_k = extract_known_multiclass_labels(splits.df_test_known)

    # 3.1 Verify logit mathematical equivalence on test set
    logits_k = model.predict_logits(X_test_k)
    exact_linear_values = np.dot(X_test_k, sk_lr.coef_.T) + sk_lr.intercept_
    np.testing.assert_allclose(logits_k, exact_linear_values, rtol=1e-7, atol=1e-7)

    probs_from_model = model.predict_proba(X_test_k)
    probs_from_softmax = softmax(logits_k, axis=1)
    max_prob_diff = float(np.max(np.abs(probs_from_model - probs_from_softmax)))
    assert max_prob_diff < 1e-6
    logger.info("Linear logits exactly match X @ W^T + b. Softmax difference vs predict_proba: %.2e", max_prob_diff)

    # 3.2 Predictions and Metrics
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

    # Per-class metrics
    p_cls, r_cls, f_cls, s_cls = precision_recall_fscore_support(y_test_k, y_pred_k, labels=range(6))
    per_class_summary = {}
    for idx, cname in enumerate(KNOWN_CLASSES):
        per_class_summary[cname] = {
            "precision": float(p_cls[idx]),
            "recall": float(r_cls[idx]),
            "f1_score": float(f_cls[idx]),
            "support": int(s_cls[idx]),
        }

    audit_results["closed_set_recomputation"] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class_summary,
        "confusion_matrix_reconciled": True,
    }

    logger.info("=== 4. OPEN-SET EVALUATION RECOMPUTATION (MSP, ENERGY, OPENMAX) ===")
    # Precompute transformed calibration and full test sets
    X_cal, _, _ = preprocessor.transform(splits.df_thresh_cal, feature_subset=feature_names)
    y_cal = extract_known_multiclass_labels(splits.df_thresh_cal)

    X_test_full, _, _ = preprocessor.transform(splits.df_test, feature_subset=feature_names)
    y_test_7class = extract_openset_7class_labels(splits.df_test)
    y_is_unknown = extract_is_unknown(splits.df_test)
    y_is_known = ~y_is_unknown
    attack_cat_series = splits.df_test["attack_cat"].astype(str).str.strip()

    X_evt, _, _ = preprocessor.transform(splits.df_evt_fit, feature_subset=feature_names)
    y_evt = extract_known_multiclass_labels(splits.df_evt_fit)

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

        # Load saved metrics
        saved_file = base_dir / dname / f"metrics_openset_{dname.lower()}.json"
        with open(saved_file, "r") as f:
            saved_dict = json.load(f)

        saved_ind = saved_dict["threshold_independent_metrics"]
        assert abs(auroc - saved_ind["auroc"]) < 1e-7
        assert abs(aupr_out - saved_ind["aupr_out"]) < 1e-7
        assert abs(aupr_in - saved_ind["aupr_in"]) < 1e-7
        assert abs(oscr_auc - saved_ind["oscr_auc"]) < 1e-7

        # Compare thresholds
        saved_cal = saved_dict["calibration"]
        assert abs(tau_fpr05 - saved_cal["tau_FPR05"]) < 1e-7
        assert abs(tau_cov95 - saved_cal["tau_Cov95"]) < 1e-7

        # Compare operating point metrics
        for op_key, recomputed_op in [("tau_FPR05", metrics_fpr05), ("tau_Cov95", metrics_cov95)]:
            saved_op = saved_dict["operating_points"][op_key]
            assert abs(recomputed_op["realized_test_fpr_normal"] - saved_op["realized_test_fpr_normal"]) < 1e-7
            assert abs(recomputed_op["known_coverage"] - saved_op["known_coverage"]) < 1e-7
            assert abs(recomputed_op["unknown_detection_rate"] - saved_op["unknown_detection_rate"]) < 1e-7
            assert abs(recomputed_op["open_set_accuracy"] - saved_op["open_set_accuracy"]) < 1e-7
            assert abs(recomputed_op["open_set_macro_f1"] - saved_op["open_set_macro_f1"]) < 1e-7
            assert abs(recomputed_op["open_set_weighted_f1"] - saved_op["open_set_weighted_f1"]) < 1e-7

            # Compare 7x7 confusion matrices
            cm_saved_op = pd.read_csv(base_dir / dname / f"confusion_matrix_7x7_{op_key.lower().replace('tau_', '')}.csv", index_col=0).values
            cm_recomputed_op = np.array(recomputed_op["confusion_matrix_7x7"])
            np.testing.assert_array_equal(cm_recomputed_op, cm_saved_op)

        logger.info("%s detector completely verified and reconciled!", dname)

        audit_results["openset_detectors"][dname] = {
            "auroc": auroc,
            "aupr_out": aupr_out,
            "aupr_in": aupr_in,
            "oscr_auc": oscr_auc,
            "tau_fpr05": tau_fpr05,
            "tau_cov95": tau_cov95,
            "fpr05_normal_fpr": metrics_fpr05["realized_test_fpr_normal"],
            "fpr05_unknown_detection": metrics_fpr05["unknown_detection_rate"],
            "cov95_normal_fpr": metrics_cov95["realized_test_fpr_normal"],
            "cov95_unknown_detection": metrics_cov95["unknown_detection_rate"],
            "reconciled": True,
        }

    # 4.1 Detailed OpenMax Weibull Audit
    logger.info("=== 4.1 DETAILED OPENMAX WEIBULL AUDIT ===")
    weibull_params = detectors["OpenMax"].weibull_params_
    mavs = detectors["OpenMax"].mavs_
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

    # Analysis Tail investigation
    analysis_correct_evt = int((splits.df_evt_fit["attack_cat"].astype(str).str.strip() == "Analysis").sum())
    logger.info("Total Analysis samples in D_evt_fit: %d", analysis_correct_evt)
    logger.info("Analysis correctly classified by LR in D_evt_fit: %d", weibull_params[1]["n_correct"])
    logger.info("Analysis eta: %d, shape: %.4f, scale: %.4f", weibull_params[1]["eta"], weibull_params[1]["shape"], weibull_params[1]["scale"])

    # Generic Tail investigation
    logger.info("Generic correctly classified in D_evt_fit: %d", weibull_params[2]["n_correct"])
    logger.info("Generic eta: %d, shape: %.4f, scale: %.4f", weibull_params[2]["eta"], weibull_params[2]["shape"], weibull_params[2]["scale"])

    logger.info("=== 5. ARTIFACT & MUTATION AUDIT ===")
    # Verify prior models and synthesis
    priors = [
        Path("results/openset/LightGBM/features_48/metrics_closed_set.json"),
        Path("results/openset/RandomForest/features_48/metrics_closed_set.json"),
        Path("results/openset/XGBoost/features_48/metrics_closed_set.json"),
        Path("results/openset/synthesis/consolidated_benchmark_table.csv"),
        Path("results/openset/synthesis_3models/consolidated_benchmark_table.csv"),
    ]
    for p in priors:
        assert p.exists(), f"Missing prior artifact {p}"

    # Verify actual metrics in LightGBM and RandomForest
    with open("results/openset/LightGBM/features_48/metrics_closed_set.json") as f:
        lgb_ref = json.load(f)
    with open("results/openset/RandomForest/features_48/metrics_closed_set.json") as f:
        rf_ref = json.load(f)

    audit_results["prior_model_references"] = {
        "LightGBM_actual": {
            "accuracy": lgb_ref["accuracy"],
            "macro_f1": lgb_ref["macro_f1"],
            "weighted_f1": lgb_ref["weighted_f1"],
        },
        "RandomForest_actual": {
            "accuracy": rf_ref["accuracy"],
            "macro_f1": rf_ref["macro_f1"],
            "weighted_f1": rf_ref["weighted_f1"],
        },
    }

    # Save complete audit dictionary in scratch directory
    audit_save_path = Path("scratch/audit_lr_results.json")
    with open(audit_save_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    logger.info("Independent audit completed successfully. Saved results to %s", audit_save_path)
    return audit_results


if __name__ == "__main__":
    res = run_comprehensive_audit()
    print("\n--- AUDIT SUMMARY RESULT ---")
    print("Model Converged:", res["model_verification"]["converged"])
    print("Optimizer Iterations:", res["model_verification"]["n_iter"])
    print("Closed Set Accuracy:", res["closed_set_recomputation"]["accuracy"])
    print("Closed Set Macro F1:", res["closed_set_recomputation"]["macro_f1"])
    print("Energy AUROC:", res["openset_detectors"]["Energy"]["auroc"])
    print("OpenMax AUROC:", res["openset_detectors"]["OpenMax"]["auroc"])
