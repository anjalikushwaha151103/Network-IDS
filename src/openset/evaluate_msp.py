"""MSP-Based Open-Set Detection Evaluation Pipeline (Phase 2 - Step 4).

Reuses the trained 6-class LightGBM classifier and fitted preprocessor from
results/openset/LightGBM/features_48/.
Calibrates thresholds exclusively on D_threshold_cal (12,139 samples).
Evaluates open-set performance on the untouched official test set (82,332 samples).
Persists all results to results/openset/LightGBM/features_48/MSP/.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src.config import Config, default_config
from src.data_loader import DataLoader
from src.models.lightgbm_model import LightGBMModel
from src.openset.metrics import (
    compute_aupr_in,
    compute_aupr_out,
    compute_auroc,
    compute_operating_point_metrics,
    compute_oscr,
)
from src.openset.msp_detector import MSPDetector
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

logger = logging.getLogger(__name__)


def run_msp_evaluation(
    classifier_dir: str = "results/openset/LightGBM/features_48",
    output_dir: Optional[str] = None,
    random_state: int = 42,
    config: Optional[Config] = None,
) -> Dict[str, Any]:
    """Runs complete calibration and evaluation pipeline for MSP Open-Set Detection."""
    cfg = config or default_config
    cls_path = Path(classifier_dir)
    out_path = Path(output_dir) if output_dir else cls_path / "MSP"
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Pre-trained Classifier and Fitted Preprocessor (Zero Retraining)
    model_file = cls_path / "model.pkl"
    preproc_file = cls_path / "preprocessor.pkl"

    if not model_file.exists() or not preproc_file.exists():
        raise FileNotFoundError(
            f"Missing base classifier artifacts in {cls_path}. Ensure Step 3 has been run."
        )

    logger.info("Loading pre-trained 6-class model from %s", model_file)
    model = LightGBMModel.load(model_file)

    logger.info("Loading fitted preprocessor from %s", preproc_file)
    preprocessor = NIDSPreprocessor.load(preproc_file)

    # Load closed-set metrics for reference
    with open(cls_path / "metrics_closed_set.json", "r", encoding="utf-8") as f:
        closed_set_reference = json.load(f)

    # 2. Load Official Dataset and Partition into Identical 4-Way Splits
    loader = DataLoader(cfg)
    raw_train_df, raw_test_df = loader.load_data()
    splits = partition_openset_data(raw_train_df, raw_test_df, random_state=random_state)

    df_cal = splits.df_thresh_cal  # 12,139 samples for threshold calibration ONLY
    df_test = splits.df_test        # 82,332 untouched official test flows

    # 3. Transform D_threshold_cal
    feature_names = preprocessor.feature_names_
    X_cal, _, _ = preprocessor.transform(df_cal, feature_subset=feature_names)
    y_cal = extract_known_multiclass_labels(df_cal)

    # 4. Instantiate MSP Detector and Calibrate on D_threshold_cal ONLY
    detector = MSPDetector()
    detector.fit(None, None, model)

    logger.info("Calibrating MSP operating thresholds on D_threshold_cal (%d samples)...", len(X_cal))
    calibrated_thresholds = detector.calibrate_thresholds(
        X_cal,
        y_cal,
        model,
        normal_class_id=KNOWN_CLASS_TO_ID["Normal"],
        percentile_target=95.0,
    )
    tau_fpr05 = calibrated_thresholds["tau_FPR05"]
    tau_cov95 = calibrated_thresholds["tau_Cov95"]

    logger.info("Calibrated tau_FPR05 = %.6f (target ~5%% Normal FPR)", tau_fpr05)
    logger.info("Calibrated tau_Cov95 = %.6f (target ~95%% Known Coverage)", tau_cov95)

    # 5. Transform Untouched Official Test Set (82,332 samples)
    logger.info("Transforming full untouched official test split (%d flows)...", len(df_test))
    X_test, _, _ = preprocessor.transform(df_test, feature_subset=feature_names)

    y_test_7class = extract_openset_7class_labels(df_test)
    y_is_unknown = extract_is_unknown(df_test)
    y_is_known = ~y_is_unknown
    attack_cat_series = df_test["attack_cat"].astype(str).str.strip()

    # 6. Compute MSP Anomaly Scores and Known Confidences on Test Set
    anomaly_scores = detector.compute_anomaly_score(X_test, model)
    confidence_scores = detector.compute_confidence_score(X_test, model)

    # Split scores for OSCR calculation
    scores_known = anomaly_scores[y_is_known]
    scores_unknown = anomaly_scores[y_is_unknown]
    y_true_known = y_test_7class[y_is_known]

    # Closed-set predictions on known test flows
    y_pred_closed = model.predict(X_test[y_is_known])

    # 7. Calculate Threshold-Independent Metrics
    logger.info("Computing threshold-independent discrimination metrics...")
    auroc = compute_auroc(y_is_unknown, anomaly_scores)
    aupr_out = compute_aupr_out(y_is_unknown, anomaly_scores)
    aupr_in = compute_aupr_in(y_is_known, confidence_scores)
    oscr_auc, oscr_fpr, oscr_ccr = compute_oscr(
        y_true_known,
        y_pred_closed,
        scores_known,
        scores_unknown,
    )

    logger.info("MSP AUROC:    %.4f", auroc)
    logger.info("MSP AUPR-Out: %.4f", aupr_out)
    logger.info("MSP AUPR-In:  %.4f", aupr_in)
    logger.info("MSP OSCR AUC: %.4f", oscr_auc)

    # 8. Evaluate Operating Points: tau_FPR05 and tau_Cov95
    logger.info("Evaluating Operating Point 1: tau_FPR05 = %.6f ...", tau_fpr05)
    preds_fpr05 = detector.predict_openset(X_test, model, threshold=tau_fpr05, unknown_label=UNKNOWN_LABEL_ID)
    metrics_fpr05 = compute_operating_point_metrics(
        y_test_7class,
        preds_fpr05,
        attack_cat_series,
        threshold_name="tau_FPR05",
        threshold_value=tau_fpr05,
        normal_class_id=KNOWN_CLASS_TO_ID["Normal"],
        unknown_label=UNKNOWN_LABEL_ID,
    )

    logger.info("Evaluating Operating Point 2: tau_Cov95 = %.6f ...", tau_cov95)
    preds_cov95 = detector.predict_openset(X_test, model, threshold=tau_cov95, unknown_label=UNKNOWN_LABEL_ID)
    metrics_cov95 = compute_operating_point_metrics(
        y_test_7class,
        preds_cov95,
        attack_cat_series,
        threshold_name="tau_Cov95",
        threshold_value=tau_cov95,
        normal_class_id=KNOWN_CLASS_TO_ID["Normal"],
        unknown_label=UNKNOWN_LABEL_ID,
    )

    # 9. Structure Final Output Metrics
    final_results = {
        "method": "Maximum Softmax Probability (MSP)",
        "model": "LightGBM",
        "feature_count": len(feature_names),
        "random_state": random_state,
        "calibration": detector.calibration_stats_,
        "threshold_independent_metrics": {
            "auroc": auroc,
            "aupr_out": aupr_out,
            "aupr_in": aupr_in,
            "oscr_auc": oscr_auc,
        },
        "operating_points": {
            "tau_FPR05": metrics_fpr05,
            "tau_Cov95": metrics_cov95,
        },
        "closed_set_reference": {
            "test_accuracy": closed_set_reference["accuracy"],
            "test_macro_f1": closed_set_reference["macro_f1"],
            "test_weighted_f1": closed_set_reference["weighted_f1"],
        },
    }

    # 10. Persist Artifacts under results/openset/LightGBM/features_48/MSP/
    # A. Save Thresholds JSON
    with open(out_path / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(detector.calibration_stats_, f, indent=2)

    # B. Save Metrics JSON
    with open(out_path / "metrics_openset_msp.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    # C. Save Summary Metrics CSV
    summary_rows = [
        {"metric": "AUROC", "value": auroc, "type": "threshold_independent"},
        {"metric": "AUPR-Out", "value": aupr_out, "type": "threshold_independent"},
        {"metric": "AUPR-In", "value": aupr_in, "type": "threshold_independent"},
        {"metric": "OSCR_AUC", "value": oscr_auc, "type": "threshold_independent"},
        {"metric": "tau_FPR05_threshold", "value": tau_fpr05, "type": "operating_point"},
        {"metric": "tau_FPR05_realized_test_fpr_normal", "value": metrics_fpr05["realized_test_fpr_normal"], "type": "operating_point"},
        {"metric": "tau_FPR05_known_coverage", "value": metrics_fpr05["known_coverage"], "type": "operating_point"},
        {"metric": "tau_FPR05_unknown_detection_rate", "value": metrics_fpr05["unknown_detection_rate"], "type": "operating_point"},
        {"metric": "tau_FPR05_open_set_macro_f1", "value": metrics_fpr05["open_set_macro_f1"], "type": "operating_point"},
        {"metric": "tau_Cov95_threshold", "value": tau_cov95, "type": "operating_point"},
        {"metric": "tau_Cov95_realized_test_fpr_normal", "value": metrics_cov95["realized_test_fpr_normal"], "type": "operating_point"},
        {"metric": "tau_Cov95_known_coverage", "value": metrics_cov95["known_coverage"], "type": "operating_point"},
        {"metric": "tau_Cov95_unknown_detection_rate", "value": metrics_cov95["unknown_detection_rate"], "type": "operating_point"},
        {"metric": "tau_Cov95_open_set_macro_f1", "value": metrics_cov95["open_set_macro_f1"], "type": "operating_point"},
    ]
    pd.DataFrame(summary_rows).to_csv(out_path / "metrics_openset_msp.csv", index=False)

    # D. Save Per-Category Unknown Detection CSV
    per_cat_df = pd.DataFrame([
        {
            "attack_category": cat,
            "support": metrics_fpr05["per_category_unknown_detection"][cat]["support"],
            "detected_tau_FPR05": metrics_fpr05["per_category_unknown_detection"][cat]["detected"],
            "recall_tau_FPR05": metrics_fpr05["per_category_unknown_detection"][cat]["recall"],
            "detected_tau_Cov95": metrics_cov95["per_category_unknown_detection"][cat]["detected"],
            "recall_tau_Cov95": metrics_cov95["per_category_unknown_detection"][cat]["recall"],
        }
        for cat in UNKNOWN_CLASSES
    ])
    per_cat_df.to_csv(out_path / "per_category_unknown_detection.csv", index=False)

    # E. Save 7x7 Confusion Matrices (CSV and PNG)
    cm_fpr05_df = pd.DataFrame(metrics_fpr05["confusion_matrix_7x7"], index=OPENSET_7CLASSES, columns=OPENSET_7CLASSES)
    cm_fpr05_df.to_csv(out_path / "confusion_matrix_7x7_fpr05.csv")
    _plot_7x7_confusion_matrix(
        np.array(metrics_fpr05["confusion_matrix_7x7"]),
        OPENSET_7CLASSES,
        "MSP 7x7 Open-Set Confusion Matrix (tau_FPR05)",
        out_path / "confusion_matrix_7x7_fpr05.png",
    )

    cm_cov95_df = pd.DataFrame(metrics_cov95["confusion_matrix_7x7"], index=OPENSET_7CLASSES, columns=OPENSET_7CLASSES)
    cm_cov95_df.to_csv(out_path / "confusion_matrix_7x7_cov95.csv")
    _plot_7x7_confusion_matrix(
        np.array(metrics_cov95["confusion_matrix_7x7"]),
        OPENSET_7CLASSES,
        "MSP 7x7 Open-Set Confusion Matrix (tau_Cov95)",
        out_path / "confusion_matrix_7x7_cov95.png",
    )

    # F. Save ROC & PR Curves Plot
    _plot_roc_and_pr_curves(y_is_unknown, anomaly_scores, y_is_known, confidence_scores, out_path / "roc_pr_curves.png")

    # G. Save OSCR Curve Plot
    _plot_oscr_curve(oscr_fpr, oscr_ccr, oscr_auc, out_path / "oscr_curve.png")

    # H. Save Experiment Config
    exp_config = {
        "detector": "MSPDetector",
        "model_file": str(model_file),
        "preprocessor_file": str(preproc_file),
        "n_features": len(feature_names),
        "calibration_set": "D_threshold_cal (12,139 samples)",
        "test_set": "Official untouched UNSW-NB15 test set (82,332 samples)",
        "thresholds": detector.calibrated_thresholds_,
        "reproducibility_seed": random_state,
    }
    with open(out_path / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    logger.info("MSP Open-Set evaluation completed. Artifacts saved in %s", out_path)
    return final_results


def _plot_7x7_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, save_path: Path):
    """Plots normalized 7x7 confusion matrix."""
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    plt.figure(figsize=(9, 7.5), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Ratio"},
    )
    plt.title(title, fontsize=12, pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _plot_roc_and_pr_curves(
    y_is_unknown: np.ndarray,
    anomaly_scores: np.ndarray,
    y_is_known: np.ndarray,
    confidence_scores: np.ndarray,
    save_path: Path,
):
    """Plots dual-panel ROC and PR curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: ROC Curve (Unknown as positive)
    fpr, tpr, _ = roc_curve(y_is_unknown.astype(int), anomaly_scores)
    roc_auc_val = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"MSP ROC (AUC = {roc_auc_val:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (Known flagged as Unknown)", fontsize=10)
    axes[0].set_ylabel("True Positive Rate (Unknown Recall)", fontsize=10)
    axes[0].set_title("OOD Discrimination ROC Curve (MSP)", fontsize=11)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: PR Curves (AUPR-Out & AUPR-In)
    prec_out, rec_out, _ = precision_recall_curve(y_is_unknown.astype(int), anomaly_scores)
    aupr_out_val = auc(rec_out, prec_out)

    prec_in, rec_in, _ = precision_recall_curve(y_is_known.astype(int), confidence_scores)
    aupr_in_val = auc(rec_in, prec_in)

    axes[1].plot(rec_out, prec_out, color="#d62728", lw=2, label=f"AUPR-Out (Unknowns: {aupr_out_val:.4f})")
    axes[1].plot(rec_in, prec_in, color="#2ca02c", lw=2, label=f"AUPR-In (Knowns: {aupr_in_val:.4f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].set_title("Precision-Recall Curves (MSP)", fontsize=11)
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _plot_oscr_curve(oscr_fpr: np.ndarray, oscr_ccr: np.ndarray, oscr_auc: float, save_path: Path):
    """Plots OSCR curve (CCR vs FPR_unknown)."""
    plt.figure(figsize=(8, 6), dpi=300)
    sort_idx = np.argsort(oscr_fpr)
    plt.plot(oscr_fpr[sort_idx], oscr_ccr[sort_idx], color="#9467bd", lw=2, label=f"MSP OSCR (Area = {oscr_auc:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate on Unknowns (FPR_U)", fontsize=11)
    plt.ylabel("Correct Classification Rate on Knowns (CCR_K)", fontsize=11)
    plt.title("Open-Set Classification Rate (OSCR) Curve - MSP", fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    results = run_msp_evaluation()
    print("MSP evaluation completed successfully.")
