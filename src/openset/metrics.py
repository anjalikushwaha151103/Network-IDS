"""Evaluation metrics for Open-Set / Zero-Day Attack Detection.

Provides:
- AUROC (Unknown as positive class)
- AUPR-Out (Unknown as positive class, anomaly score)
- AUPR-In (Known as positive class, known-confidence score)
- OSCR (Open-Set Classification Rate curve and Area Under Curve)
- Operating point metrics (FPR on Normal, Known Coverage, Unknown Recall, Per-Category Recall)
- 7x7 Open-Set Confusion Matrix
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.openset.partitioner import KNOWN_CLASSES, OPENSET_7CLASSES, UNKNOWN_CLASSES


def compute_auroc(y_is_unknown: np.ndarray, anomaly_scores: np.ndarray) -> float:
    """Computes AUROC treating Unknown attack flows as the positive class (1) and Known traffic as (0)."""
    return float(roc_auc_score(y_is_unknown.astype(int), anomaly_scores))


def compute_aupr_out(y_is_unknown: np.ndarray, anomaly_scores: np.ndarray) -> float:
    """Computes AUPR-Out (Precision-Recall AUC for Unknown attack detection).

    Target class: Unknowns (y=1). Score: Anomaly Score S(x).
    """
    prec, rec, _ = precision_recall_curve(y_is_unknown.astype(int), anomaly_scores)
    return float(auc(rec, prec))


def compute_aupr_in(y_is_known: np.ndarray, confidence_scores: np.ndarray) -> float:
    """Computes AUPR-In (Precision-Recall AUC for Known traffic retention).

    Target class: Knowns (y=1). Score: Known-Confidence Score C_known(x).
    """
    prec, rec, _ = precision_recall_curve(y_is_known.astype(int), confidence_scores)
    return float(auc(rec, prec))


def compute_oscr(
    y_true_known: np.ndarray,
    y_pred_closed: np.ndarray,
    anomaly_scores_known: np.ndarray,
    anomaly_scores_unknown: np.ndarray,
    num_thresholds: int = 2000,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Computes the Open-Set Classification Rate (OSCR) curve and Area Under OSCR.

    Definitions:
        - Query x is accepted as Known if S(x) <= theta.
        - Correct Classification Rate on Known:
            CCR(theta) = sum_{x in K} I(y_pred(x) == y_true(x) and S(x) <= theta) / |K|
        - False Positive Rate on Unknown (Unknown accepted as Known):
            FPR(theta) = sum_{x in U} I(S(x) <= theta) / |U|

    Args:
        y_true_known: True class labels of known test samples (0..5).
        y_pred_closed: Closed-set predictions on known test samples (0..5).
        anomaly_scores_known: Anomaly scores for known test samples.
        anomaly_scores_unknown: Anomaly scores for unknown test samples.
        num_thresholds: Number of evaluation thresholds for curve interpolation.

    Returns:
        (oscr_auc, fpr_curve, ccr_curve)
    """
    all_scores = np.concatenate([anomaly_scores_known, anomaly_scores_unknown])
    min_score = float(np.min(all_scores))
    max_score = float(np.max(all_scores))

    thresholds = np.linspace(min_score - 1e-6, max_score + 1e-6, num_thresholds)

    n_known = len(y_true_known)
    n_unknown = len(anomaly_scores_unknown)

    is_correct_known = (y_true_known == y_pred_closed)

    ccr_list = []
    fpr_list = []

    for theta in thresholds:
        # Known samples accepted and correctly classified
        ccr = np.sum(is_correct_known & (anomaly_scores_known <= theta)) / n_known
        # Unknown samples accepted as known (false positives)
        fpr = np.sum(anomaly_scores_unknown <= theta) / n_unknown

        ccr_list.append(ccr)
        fpr_list.append(fpr)

    fpr_curve = np.array(fpr_list)
    ccr_curve = np.array(ccr_list)

    # Sort by FPR for monotonic integration
    sort_idx = np.argsort(fpr_curve)
    fpr_sorted = fpr_curve[sort_idx]
    ccr_sorted = ccr_curve[sort_idx]

    # Prepend (0, 0) and append (1, max_ccr) to bound the curve properly
    fpr_bounded = np.concatenate([[0.0], fpr_sorted, [1.0]])
    ccr_bounded = np.concatenate([[0.0], ccr_sorted, [ccr_sorted[-1]]])

    oscr_auc = float(trapezoid(ccr_bounded, fpr_bounded))
    return oscr_auc, fpr_curve, ccr_curve


def compute_operating_point_metrics(
    y_true_7class: np.ndarray,
    y_pred_7class: np.ndarray,
    attack_cat_series: pd.Series,
    threshold_name: str,
    threshold_value: float,
    normal_class_id: int = 0,
    unknown_label: int = 6,
) -> Dict[str, Any]:
    """Calculates all operating point metrics for a specific calibrated threshold."""
    is_normal = (y_true_7class == normal_class_id)
    is_known = (y_true_7class != unknown_label)
    is_unknown = (y_true_7class == unknown_label)

    # 1. False Positive Rate on Normal traffic
    normal_total = int(is_normal.sum())
    normal_false_positives = int((is_normal & (y_pred_7class == unknown_label)).sum())
    fpr_normal = normal_false_positives / normal_total if normal_total > 0 else 0.0

    # 2. False Positive Rate on all Known traffic & Known Coverage
    known_total = int(is_known.sum())
    known_false_positives = int((is_known & (y_pred_7class == unknown_label)).sum())
    fpr_known = known_false_positives / known_total if known_total > 0 else 0.0
    known_coverage = 1.0 - fpr_known

    # 3. Overall Unknown Attack Detection Rate
    unknown_total = int(is_unknown.sum())
    unknown_true_positives = int((is_unknown & (y_pred_7class == unknown_label)).sum())
    unknown_detection_rate = unknown_true_positives / unknown_total if unknown_total > 0 else 0.0

    # 4. Per-Unknown-Category Detection Rates
    per_cat_recall = {}
    for cat_name in UNKNOWN_CLASSES:
        cat_mask = (attack_cat_series == cat_name)
        cat_count = int(cat_mask.sum())
        if cat_count > 0:
            detected = int((cat_mask & (y_pred_7class == unknown_label)).sum())
            per_cat_recall[cat_name] = {
                "support": cat_count,
                "detected": detected,
                "recall": detected / cat_count,
            }

    # 5. Full 7x7 Confusion Matrix
    cm_7x7 = confusion_matrix(y_true_7class, y_pred_7class, labels=range(7))

    # 6. Open-Set 7-class performance metrics
    acc_7class = float(accuracy_score(y_true_7class, y_pred_7class))
    macro_f1_7class = float(f1_score(y_true_7class, y_pred_7class, average="macro"))
    weighted_f1_7class = float(f1_score(y_true_7class, y_pred_7class, average="weighted"))

    # Classification report over all 7 categories
    report_dict = classification_report(
        y_true_7class,
        y_pred_7class,
        target_names=OPENSET_7CLASSES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "threshold_name": threshold_name,
        "threshold_value": threshold_value,
        "realized_test_fpr_normal": fpr_normal,
        "normal_support": normal_total,
        "normal_false_alarms": normal_false_positives,
        "realized_test_fpr_known": fpr_known,
        "known_support": known_total,
        "known_coverage": known_coverage,
        "unknown_detection_rate": unknown_detection_rate,
        "unknown_support": unknown_total,
        "unknown_detected_count": unknown_true_positives,
        "per_category_unknown_detection": per_cat_recall,
        "open_set_accuracy": acc_7class,
        "open_set_macro_f1": macro_f1_7class,
        "open_set_weighted_f1": weighted_f1_7class,
        "confusion_matrix_7x7": cm_7x7.tolist(),
        "classification_report_7class": report_dict,
    }
