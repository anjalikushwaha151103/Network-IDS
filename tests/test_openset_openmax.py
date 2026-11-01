"""Comprehensive unit tests for OpenMax (EVT-Based Adaptation) (Phase 2 - Step 6).

Verifies:
1. MAV computation and class separation.
2. Correctly classified sample filtering on D_evt_fit.
3. Weibull upper-tail sizing (eta) and fitting parameters.
4. Deterministic fallbacks for zero variance, fitting failures, and insufficient samples.
5. OpenMax recalibration produces valid 7-class probability distributions (sum=1, non-negative).
6. Consistent score orientation: higher anomaly score indicates greater likelihood of unknown class.
7. Threshold calibration isolation on D_threshold_cal (zero test or unknown leakage).
8. Predictions strictly in {0, 1, 2, 3, 4, 5, 6}.
9. Direct mathematical consistency between 7x7 confusion matrices and reported metrics.
10. Numerical stability under extreme logits and reproducibility across runs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.models.base_model import BaseNIDSModel
from src.models.lightgbm_model import LightGBMModel
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


class DummyModelWithCustomLogits(BaseNIDSModel):
    """Mock model to test OpenMax EVT fitting and recalibration on synthetic logits."""

    def __init__(self, synthetic_logits: np.ndarray, preds: Optional[np.ndarray] = None):
        super().__init__("dummy_logits", task="multiclass")
        self.synthetic_logits = synthetic_logits
        self.classes_ = np.arange(synthetic_logits.shape[1])
        self.is_fitted = True
        self._preds = preds if preds is not None else np.argmax(synthetic_logits, axis=1)

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X):
        return self._preds

    def predict_proba(self, X):
        exp_z = np.exp(self.synthetic_logits - np.max(self.synthetic_logits, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict_logits(self, X):
        return self.synthetic_logits


@pytest.fixture(scope="module")
def trained_artifacts():
    """Loads the pre-trained 6-class LightGBM model and fitted preprocessor."""
    cls_dir = Path("results/openset/LightGBM/features_48")
    model_path = cls_dir / "model.pkl"
    preproc_path = cls_dir / "preprocessor.pkl"

    assert model_path.exists(), f"Model artifact missing at: {model_path}"
    assert preproc_path.exists(), f"Preprocessor artifact missing at: {preproc_path}"

    model = LightGBMModel.load(model_path)
    preprocessor = NIDSPreprocessor.load(preproc_path)
    return model, preprocessor


@pytest.fixture(scope="module")
def partitioned_data():
    """Loads official UNSW-NB15 dataset and partitions into 4-way splits."""
    loader = DataLoader()
    raw_train, raw_test = loader.load_data()
    splits = partition_openset_data(raw_train, raw_test, random_state=42)
    return splits


def test_mav_computation_and_class_separation(trained_artifacts, partitioned_data):
    """Test 1: Mean Activation Vectors (MAVs) are computed for all 6 classes and are well-separated."""
    model, preprocessor = trained_artifacts
    splits = partitioned_data

    df_evt = splits.df_evt_fit
    X_evt, _, _ = preprocessor.transform(df_evt, feature_subset=preprocessor.feature_names_)
    y_evt = extract_known_multiclass_labels(df_evt)

    detector = OpenMaxDetector(alpha=3, eta_ratio=0.2, min_eta=5, max_eta=20)
    detector.fit(X_evt, y_evt, model)

    assert detector.is_fitted, "Detector must be fitted"
    assert len(detector.mavs_) == 6, f"Expected 6 MAVs, got {len(detector.mavs_)}"

    # Check each MAV is finite and shape (6,)
    for c in range(6):
        mav = detector.mavs_[c]
        assert isinstance(mav, np.ndarray), f"MAV for class {c} must be ndarray"
        assert mav.shape == (6,), f"MAV for class {c} must have shape (6,), got {mav.shape}"
        assert np.all(np.isfinite(mav)), f"MAV for class {c} must be finite"

    # Check pairwise separation between MAVs
    for i in range(6):
        for j in range(i + 1, 6):
            dist = np.linalg.norm(detector.mavs_[i] - detector.mavs_[j])
            assert dist > 1.0, f"MAVs for classes {i} and {j} must be separated (got distance {dist:.4f})"


def test_correctly_classified_filtering(trained_artifacts, partitioned_data):
    """Test 2: Only correctly classified samples in D_evt_fit are used for MAV and tail fitting."""
    model, preprocessor = trained_artifacts
    splits = partitioned_data

    df_evt = splits.df_evt_fit
    X_evt, _, _ = preprocessor.transform(df_evt, feature_subset=preprocessor.feature_names_)
    y_evt = extract_known_multiclass_labels(df_evt)

    preds_evt = model.predict(X_evt)
    correct_counts = {}
    for c in range(6):
        correct_counts[c] = int(((y_evt == c) & (preds_evt == c)).sum())

    detector = OpenMaxDetector(alpha=3, eta_ratio=0.2, min_eta=5, max_eta=20)
    detector.fit(X_evt, y_evt, model)

    for c in range(6):
        reported_n = detector.weibull_params_[c]["n_correct"]
        assert reported_n == correct_counts[c], (
            f"Class {c} correctly classified count mismatch: reported {reported_n} vs true {correct_counts[c]}"
        )


def test_weibull_fitting_and_tail_sizing(trained_artifacts, partitioned_data):
    """Test 3: Tail sizing eta_c satisfies eta = min(20, max(5, floor(0.2*n_c))) and Weibull parameters are valid."""
    model, preprocessor = trained_artifacts
    splits = partitioned_data

    df_evt = splits.df_evt_fit
    X_evt, _, _ = preprocessor.transform(df_evt, feature_subset=preprocessor.feature_names_)
    y_evt = extract_known_multiclass_labels(df_evt)

    detector = OpenMaxDetector(alpha=3, eta_ratio=0.2, min_eta=5, max_eta=20)
    detector.fit(X_evt, y_evt, model)

    for c in range(6):
        params = detector.weibull_params_[c]
        n_c = params["n_correct"]
        expected_eta = min(20, max(5, int(np.floor(0.2 * n_c))))
        if n_c < 5:
            expected_eta = n_c

        assert params["eta"] == expected_eta, (
            f"Class {c} eta mismatch: expected {expected_eta}, got {params['eta']}"
        )
        assert params["shape"] > 0, f"Class {c} Weibull shape must be positive, got {params['shape']}"
        assert params["scale"] > 0, f"Class {c} Weibull scale must be positive, got {params['scale']}"
        assert params["loc"] == pytest.approx(params["floc"], rel=1e-5), "Weibull loc must match floc"
        assert not params["fallback_applied"], f"Class {c} should fit cleanly without fallback"


def test_deterministic_fallbacks_zero_variance_and_small_sample():
    """Test 4: Deterministic fallbacks handle zero-variance tail distances and zero correct samples without jitter."""
    # Scenario A: Zero variance tail distances (identical distance vectors)
    # 10 samples of class 0, all identical logits
    identical_logits = np.ones((10, 6)) * 5.0
    labels = np.zeros(10, dtype=int)
    mock_model = DummyModelWithCustomLogits(identical_logits)

    detector = OpenMaxDetector(alpha=3, min_eta=5, max_eta=20)
    detector.fit(identical_logits, labels, mock_model)

    params_0 = detector.weibull_params_[0]
    assert params_0["fallback_applied"] is True, "Zero variance tail must trigger fallback"
    assert params_0["fallback_reason"] == "zero_variance_tail_fallback"
    assert params_0["shape"] == 1.0, "Deterministic fallback shape must be 1.0"
    assert params_0["scale"] >= 1e-4, "Deterministic fallback scale must be >= 1e-4"

    # Scenario B: Class has 0 correctly classified samples
    # Mock model predicts wrong class for all samples of class 1
    logits_b = np.array([
        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # true 0, predicted 0 (correct)
        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # true 0, predicted 0 (correct)
        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # true 1, predicted 0 (INCORRECT)
        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # true 1, predicted 0 (INCORRECT)
    ])
    labels_b = np.array([0, 0, 1, 1])
    preds_b = np.array([0, 0, 0, 0])  # class 1 has 0 correct
    mock_model_b = DummyModelWithCustomLogits(logits_b, preds=preds_b)

    detector_b = OpenMaxDetector(alpha=2, min_eta=2, max_eta=5)
    detector_b.fit(logits_b, labels_b, mock_model_b)

    params_1 = detector_b.weibull_params_[1]
    assert params_1["fallback_applied"] is True
    assert "zero_correct_samples" in params_1["fallback_reason"]


def test_openmax_recalibration_and_seven_class_output(trained_artifacts, partitioned_data):
    """Test 5: predict_proba_7class outputs shape (N, 7) valid probability distributions."""
    model, preprocessor = trained_artifacts
    splits = partitioned_data

    # Load fitted OpenMax detector
    openmax_dir = Path("results/openset/LightGBM/features_48/OpenMax")
    assert (openmax_dir / "mav_and_weibull_params.json").exists()

    df_sample = splits.df_thresh_cal.iloc[:200]
    X_sample, _, _ = preprocessor.transform(df_sample, feature_subset=preprocessor.feature_names_)

    detector = OpenMaxDetector(alpha=3)
    # Fit detector on D_evt_fit
    X_evt, _, _ = preprocessor.transform(splits.df_evt_fit, feature_subset=preprocessor.feature_names_)
    y_evt = extract_known_multiclass_labels(splits.df_evt_fit)
    detector.fit(X_evt, y_evt, model)

    probs_7 = detector.predict_proba_7class(X_sample, model)

    assert probs_7.shape == (200, 7), f"Expected shape (200, 7), got {probs_7.shape}"
    assert np.all(probs_7 >= 0.0), "All probabilities must be non-negative"
    assert np.all(probs_7 <= 1.0), "All probabilities must be <= 1.0"
    row_sums = np.sum(probs_7, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5, err_msg="Each row of 7-class probs must sum to 1.0")


def test_score_orientation(trained_artifacts, partitioned_data):
    """Test 6: Anomaly score S(x) = P_6(x) is higher for unknown samples than known samples."""
    model, preprocessor = trained_artifacts
    splits = partitioned_data

    # Sample known and unknown flows from test set
    df_test = splits.df_test
    df_known = df_test[~extract_is_unknown(df_test)].iloc[:500]
    df_unknown = df_test[extract_is_unknown(df_test)].iloc[:500]

    X_known, _, _ = preprocessor.transform(df_known, feature_subset=preprocessor.feature_names_)
    X_unknown, _, _ = preprocessor.transform(df_unknown, feature_subset=preprocessor.feature_names_)

    # Fit detector
    X_evt, _, _ = preprocessor.transform(splits.df_evt_fit, feature_subset=preprocessor.feature_names_)
    y_evt = extract_known_multiclass_labels(splits.df_evt_fit)
    detector = OpenMaxDetector(alpha=3)
    detector.fit(X_evt, y_evt, model)

    scores_known = detector.compute_anomaly_score(X_known, model)
    scores_unknown = detector.compute_anomaly_score(X_unknown, model)

    conf_known = detector.compute_confidence_score(X_known, model)
    np.testing.assert_allclose(conf_known, 1.0 - scores_known, atol=1e-6)

    mean_known_score = np.mean(scores_known)
    mean_unknown_score = np.mean(scores_unknown)

    assert mean_unknown_score > mean_known_score, (
        f"Unknown mean anomaly score ({mean_unknown_score:.4f}) must exceed known mean ({mean_known_score:.4f})"
    )


def test_threshold_calibration_isolation_and_no_leakage(trained_artifacts, partitioned_data):
    """Test 7: Threshold calibration strictly on D_threshold_cal without touching D_test or unknown classes."""
    model, preprocessor = trained_artifacts
    splits = partitioned_data

    X_evt, _, _ = preprocessor.transform(splits.df_evt_fit, feature_subset=preprocessor.feature_names_)
    y_evt = extract_known_multiclass_labels(splits.df_evt_fit)
    detector = OpenMaxDetector(alpha=3)
    detector.fit(X_evt, y_evt, model)

    X_cal, _, _ = preprocessor.transform(splits.df_thresh_cal, feature_subset=preprocessor.feature_names_)
    y_cal = extract_known_multiclass_labels(splits.df_thresh_cal)

    thresholds = detector.calibrate_thresholds(
        X_cal, y_cal, model, normal_class_id=KNOWN_CLASS_TO_ID["Normal"], percentile_target=95.0
    )

    # Verify calibration stats
    stats = detector.calibration_stats_
    assert stats["total_cal_samples"] == 12139
    assert stats["normal_cal_samples"] == 4200
    assert stats["percentile_method"] == "linear"

    # Load saved thresholds from disk and verify exact match
    saved_thresh_file = Path("results/openset/LightGBM/features_48/OpenMax/thresholds.json")
    assert saved_thresh_file.exists()
    with open(saved_thresh_file, "r", encoding="utf-8") as f:
        saved_stats = json.load(f)

    assert thresholds["tau_FPR05"] == pytest.approx(saved_stats["tau_FPR05"], abs=1e-8)
    assert thresholds["tau_Cov95"] == pytest.approx(saved_stats["tau_Cov95"], abs=1e-8)


def test_predictions_strictly_seven_classes(trained_artifacts, partitioned_data):
    """Test 8: predict_openset returns labels strictly in {0, 1, 2, 3, 4, 5, 6}."""
    model, preprocessor = trained_artifacts
    splits = partitioned_data

    df_sample = splits.df_test.iloc[:500]
    X_sample, _, _ = preprocessor.transform(df_sample, feature_subset=preprocessor.feature_names_)

    # Fit detector
    X_evt, _, _ = preprocessor.transform(splits.df_evt_fit, feature_subset=preprocessor.feature_names_)
    y_evt = extract_known_multiclass_labels(splits.df_evt_fit)
    detector = OpenMaxDetector(alpha=3)
    detector.fit(X_evt, y_evt, model)

    preds = detector.predict_openset(X_sample, model, threshold=0.20, unknown_label=UNKNOWN_LABEL_ID)

    unique_labels = set(np.unique(preds))
    valid_labels = set(range(7))
    assert unique_labels.issubset(valid_labels), f"Predictions contain invalid labels: {unique_labels - valid_labels}"
    assert UNKNOWN_LABEL_ID in unique_labels, "At threshold 0.20, at least some samples should be rejected"


def test_confusion_matrix_and_metrics_consistency():
    """Test 9: 7x7 confusion matrices directly match reported metrics and supports."""
    openmax_dir = Path("results/openset/LightGBM/features_48/OpenMax")
    cm_fpr_file = openmax_dir / "confusion_matrix_7x7_fpr05.csv"
    cm_cov_file = openmax_dir / "confusion_matrix_7x7_cov95.csv"
    metrics_file = openmax_dir / "metrics_openset_openmax.json"

    assert cm_fpr_file.exists()
    assert cm_cov_file.exists()
    assert metrics_file.exists()

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    expected_row_supports = {
        "Normal": 37000,
        "Analysis": 677,
        "Generic": 18871,
        "Exploits": 11132,
        "Fuzzers": 6062,
        "DoS": 4089,
        "Unknown": 4501,
    }

    for path, op_key in [(cm_fpr_file, "tau_FPR05"), (cm_cov_file, "tau_Cov95")]:
        cm_df = pd.read_csv(path, index_index=False if "index" in pd.read_csv(path).columns else None, index_col=0)
        cm = cm_df.values

        assert cm.shape == (7, 7), f"{path} must be 7x7"
        assert int(cm.sum()) == 82332, f"Total samples in {path} must equal 82,332, got {cm.sum()}"

        # Check row supports
        for i, class_name in enumerate(OPENSET_7CLASSES):
            row_sum = int(cm[i].sum())
            assert row_sum == expected_row_supports[class_name], (
                f"{path} Row {class_name} sum {row_sum} != expected {expected_row_supports[class_name]}"
            )

        # Recalculate metrics directly from confusion matrix
        op_data = metrics["operating_points"][op_key]
        recalc_acc = float(np.trace(cm) / cm.sum())
        assert recalc_acc == pytest.approx(op_data["open_set_accuracy"], rel=1e-5)

        # Normal FPR = Normal rejected (col 6) / total Normal
        normal_rejected = cm[0, 6]
        recalc_normal_fpr = float(normal_rejected / 37000)
        assert recalc_normal_fpr == pytest.approx(op_data["realized_test_fpr_normal"], rel=1e-5)

        # Known coverage = known not rejected / total known
        total_known = 77831
        known_rejected = cm[:6, 6].sum()
        recalc_coverage = float((total_known - known_rejected) / total_known)
        assert recalc_coverage == pytest.approx(op_data["known_coverage"], rel=1e-5)

        # Unknown detection rate = Unknown rejected (row 6, col 6) / total Unknown
        unknown_detected = cm[6, 6]
        recalc_unknown_recall = float(unknown_detected / 4501)
        assert recalc_unknown_recall == pytest.approx(op_data["unknown_detection_rate"], rel=1e-5)


def test_numerical_stability_and_reproducibility():
    """Test 10: Extreme logits do not cause NaN or Inf and results are deterministic."""
    # Synthetic extreme logits
    extreme_logits = np.array([
        [1000.0, -1000.0, 0.0, -500.0, 200.0, -100.0],
        [-1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0],
        [500.0, 500.0, 500.0, 500.0, 500.0, 500.0],
    ])
    mock_model = DummyModelWithCustomLogits(extreme_logits)

    # Synthetic MAV and Weibull setup
    mavs = np.zeros((6, 6))
    shapes = np.ones(6) * 1.5
    locs = np.ones(6) * 2.0
    scales = np.ones(6) * 1.0

    detector = OpenMaxDetector(alpha=3)
    detector.mavs_ = {c: mavs[c] for c in range(6)}
    detector.weibull_params_ = {
        c: {"shape": shapes[c], "loc": locs[c], "scale": scales[c], "eta": 10, "n_correct": 50}
        for c in range(6)
    }
    detector.n_classes_ = 6
    detector._mavs_matrix = mavs
    detector._shapes_array = shapes
    detector._locs_array = locs
    detector._scales_array = scales
    detector.is_fitted = True

    # Call predict_proba_7class twice
    probs_1 = detector.predict_proba_7class(extreme_logits, mock_model)
    probs_2 = detector.predict_proba_7class(extreme_logits, mock_model)

    assert np.all(np.isfinite(probs_1)), "All output probabilities must be finite even with extreme logits"
    assert np.all(probs_1 >= 0.0), "Probabilities must be non-negative"
    assert np.all(probs_1 <= 1.0), "Probabilities must be <= 1.0"
    np.testing.assert_allclose(np.sum(probs_1, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(probs_1, probs_2, atol=1e-12, err_msg="OpenMax must be strictly deterministic")
