"""Comprehensive unit tests for Energy-Based Open-Set Detector (Phase 2 - Step 5).

Verifies:
1. Raw logits from LightGBM have expected shape (N, 6) and finite values.
2. Energy values are finite and numerically stable for extreme logits (+1000 and -1000).
3. Higher energy corresponds to greater unknown likelihood (and negative energy to known confidence).
4. Thresholds (tau_FPR05, tau_Cov95) use exclusively D_threshold_cal.
5. Unknown test data is never used for calibration.
6. Predictions remain strictly within labels 0 through 6.
7. Both thresholds produce separate and distinguishable evaluation results.
8. Metrics match the saved 7x7 confusion matrices directly.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.models.base_model import BaseNIDSModel
from src.models.lightgbm_model import LightGBMModel
from src.openset.energy_detector import EnergyDetector
from src.openset.partitioner import (
    KNOWN_CLASS_TO_ID,
    KNOWN_CLASSES,
    UNKNOWN_CLASSES,
    UNKNOWN_LABEL_ID,
    extract_known_multiclass_labels,
    partition_openset_data,
)
from src.preprocessing import NIDSPreprocessor


class DummyModelWithCustomLogits(BaseNIDSModel):
    """Mock model to test energy numerical stability on arbitrary synthetic logits."""

    def __init__(self, synthetic_logits: np.ndarray):
        super().__init__("dummy_logits", task="multiclass")
        self.synthetic_logits = synthetic_logits
        self.classes_ = np.arange(synthetic_logits.shape[1])
        self.is_fitted = True

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X):
        return np.argmax(self.synthetic_logits, axis=1)

    def predict_proba(self, X):
        # Softmax of synthetic logits
        exp_z = np.exp(self.synthetic_logits - np.max(self.synthetic_logits, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict_logits(self, X):
        return self.synthetic_logits


@pytest.fixture(scope="module")
def trained_artifacts():
    """Loads the pre-trained 6-class LightGBM model and preprocessor."""
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


def test_raw_logits_shape_and_finiteness(trained_artifacts, partitioned_data):
    """Test 1: LightGBM predict_logits returns shape (N, 6) with finite unnormalized margin values."""
    model, preprocessor = trained_artifacts
    splits = partitioned_data

    # Sample from D_thresh_cal
    df_sample = splits.df_thresh_cal.iloc[:250]
    X_sample, _, _ = preprocessor.transform(df_sample, feature_subset=preprocessor.feature_names_)

    logits = model.predict_logits(X_sample)

    assert isinstance(logits, np.ndarray), "Logits must be a numpy array"
    assert logits.shape == (250, 6), f"Expected shape (250, 6), got {logits.shape}"
    assert np.all(np.isfinite(logits)), "Logits must contain only finite numbers (no NaN or Inf)"

    # Verify these are genuine unnormalized logits, not softmax probabilities
    # 1) Softmax sum to 1.0; logits almost never sum to 1.0
    row_sums = np.sum(logits, axis=1)
    assert not np.allclose(row_sums, 1.0, atol=1e-2), "Logits appear to be softmax probabilities!"
    # 2) Logits typically contain negative numbers
    assert np.any(logits < 0.0), "Logits must contain negative values"


def test_energy_numerical_stability_extreme_logits():
    """Test 2: Numerical stability for large positive (+1000) and large negative (-1000) logits."""
    # Synthetic logits designed to cause overflow/underflow if not stably implemented
    extreme_logits = np.array([
        [1000.0, 0.0, -500.0, -1000.0, 200.0, 0.0],          # High confidence positive (+1000)
        [-1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0], # Extreme negative logits
        [500.0, 500.0, 500.0, 500.0, 500.0, 500.0],          # Flat high logits
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],                      # Flat zero logits
        [-50.0, -50.0, -50.0, -50.0, -50.0, -50.0],          # Flat negative logits
    ])

    dummy_model = DummyModelWithCustomLogits(extreme_logits)
    detector = EnergyDetector(temperature=1.0)
    detector.fit(None, None, dummy_model)

    dummy_X = np.zeros((len(extreme_logits), 10))
    energy_scores = detector.compute_anomaly_score(dummy_X, dummy_model)
    confidence_scores = detector.compute_confidence_score(dummy_X, dummy_model)

    assert len(energy_scores) == 5
    assert np.all(np.isfinite(energy_scores)), f"Energy scores must be finite, got: {energy_scores}"
    assert np.all(np.isfinite(confidence_scores)), f"Confidence scores must be finite, got: {confidence_scores}"

    # Verify exact analytical values for flat logits:
    # E = -z_max - T * log(sum(exp((z_k - z_max)/T)))
    # When all 6 logits are equal to c:
    # z_max = c, (z_k - z_max) = 0, exp(0) = 1, sum = 6
    # E = -c - 1.0 * log(6)
    expected_flat_neg1000 = 1000.0 - np.log(6.0)
    expected_flat_pos500 = -500.0 - np.log(6.0)
    expected_flat_zero = -np.log(6.0)
    expected_flat_neg50 = 50.0 - np.log(6.0)

    assert np.isclose(energy_scores[1], expected_flat_neg1000, rtol=1e-5)
    assert np.isclose(energy_scores[2], expected_flat_pos500, rtol=1e-5)
    assert np.isclose(energy_scores[3], expected_flat_zero, rtol=1e-5)
    assert np.isclose(energy_scores[4], expected_flat_neg50, rtol=1e-5)

    # For row 0 (z_max = 1000, other logits negligible):
    # exp((1000-1000)/1) = 1, exp(-800) ~ 0, exp(-1000) ~ 0...
    # E ≈ -1000.0 - log(1.0) = -1000.0
    assert np.isclose(energy_scores[0], -1000.0, atol=1e-2)

    # Confidence must be exact negative of energy
    np.testing.assert_allclose(confidence_scores, -energy_scores, rtol=1e-6)


def test_higher_energy_corresponds_to_greater_unknown_likelihood():
    """Test 3: Higher energy corresponds to greater unknown likelihood; negative energy to known confidence."""
    # Synthetic samples representing:
    # Sample A: High-confidence known class 0
    # Sample B: Ambiguous among known classes
    # Sample C: Out-of-distribution (all classes low activation)
    sample_logits = np.array([
        [15.0, -5.0, -5.0, -5.0, -5.0, -5.0],  # Sample A: Sharp peak -> Very Known
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],        # Sample B: Uncertain
        [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # Sample C: Low overall density -> Unknown
    ])

    dummy_model = DummyModelWithCustomLogits(sample_logits)
    detector = EnergyDetector(temperature=1.0)
    dummy_X = np.zeros((3, 5))

    energy = detector.compute_anomaly_score(dummy_X, dummy_model)
    confidence = detector.compute_confidence_score(dummy_X, dummy_model)

    # Sample A (sharp peak) should have lowest energy (most in-distribution)
    # Sample C (low density) should have highest energy (most out-of-distribution / unknown)
    assert energy[0] < energy[1] < energy[2], (
        f"Expected E(A) < E(B) < E(C), got {energy[0]} < {energy[1]} < {energy[2]}"
    )

    # Confidence must have reverse order
    assert confidence[0] > confidence[1] > confidence[2], (
        f"Expected Conf(A) > Conf(B) > Conf(C), got {confidence[0]} > {confidence[1]} > {confidence[2]}"
    )


def test_thresholds_use_only_d_threshold_cal(trained_artifacts, partitioned_data):
    """Test 4 & 5: Calibration uses ONLY D_threshold_cal; unknown test data is never used."""
    model, preprocessor = trained_artifacts
    splits = partitioned_data

    # Calibrate on D_threshold_cal
    df_cal = splits.df_thresh_cal
    assert len(df_cal) == 12139, f"Expected 12,139 samples in D_thresh_cal, got {len(df_cal)}"

    # Check zero unknown attacks in calibration set
    cal_cats = set(df_cal["attack_cat"].unique())
    assert cal_cats.issubset(set(KNOWN_CLASSES)), f"Unknown classes found in D_threshold_cal: {cal_cats - set(KNOWN_CLASSES)}"
    for unk in UNKNOWN_CLASSES:
        assert (df_cal["attack_cat"] == unk).sum() == 0, f"Found {unk} in D_thresh_cal!"

    X_cal, _, _ = preprocessor.transform(df_cal, feature_subset=preprocessor.feature_names_)
    y_cal = extract_known_multiclass_labels(df_cal)

    detector = EnergyDetector(temperature=1.0)
    thresholds = detector.calibrate_thresholds(
        X_cal,
        y_cal,
        model,
        normal_class_id=KNOWN_CLASS_TO_ID["Normal"],
        percentile_target=95.0,
    )

    tau_fpr05 = thresholds["tau_FPR05"]
    tau_cov95 = thresholds["tau_Cov95"]

    assert isinstance(tau_fpr05, float)
    assert isinstance(tau_cov95, float)
    assert np.isfinite(tau_fpr05)
    assert np.isfinite(tau_cov95)

    assert detector.calibration_stats_["total_cal_samples"] == 12139
    assert detector.calibration_stats_["normal_cal_samples"] == (df_cal["attack_cat"] == "Normal").sum()
    assert detector.calibration_stats_["percentile_method"] == "linear"


def test_predictions_remain_within_labels_0_through_6(trained_artifacts, partitioned_data):
    """Test 6: Predictions remain strictly within labels 0 through 6."""
    model, preprocessor = trained_artifacts
    splits = partitioned_data

    X_sample, _, _ = preprocessor.transform(splits.df_test.iloc[:500], feature_subset=preprocessor.feature_names_)

    detector = EnergyDetector(temperature=1.0)
    # Test with varying thresholds (including very low and very high)
    for thresh in [-50.0, -10.0, -2.0, 0.0, 5.0, 50.0]:
        preds = detector.predict_openset(X_sample, model, threshold=thresh, unknown_label=UNKNOWN_LABEL_ID)
        unique_labels = set(np.unique(preds))
        assert unique_labels.issubset({0, 1, 2, 3, 4, 5, 6}), (
            f"Invalid predicted labels found: {unique_labels - {0, 1, 2, 3, 4, 5, 6}}"
        )


def test_both_thresholds_produce_separate_evaluation_results(trained_artifacts, partitioned_data):
    """Test 7: Both thresholds (tau_FPR05 and tau_Cov95) yield separate, distinguishable predictions."""
    model, preprocessor = trained_artifacts
    splits = partitioned_data

    X_cal, _, _ = preprocessor.transform(splits.df_thresh_cal, feature_subset=preprocessor.feature_names_)
    y_cal = extract_known_multiclass_labels(splits.df_thresh_cal)

    detector = EnergyDetector(temperature=1.0)
    thresholds = detector.calibrate_thresholds(X_cal, y_cal, model)

    tau_fpr05 = thresholds["tau_FPR05"]
    tau_cov95 = thresholds["tau_Cov95"]

    # Thresholds should not be identical because Normal traffic distribution differs from complex known attacks
    assert tau_fpr05 != tau_cov95, f"tau_FPR05 ({tau_fpr05}) and tau_Cov95 ({tau_cov95}) must be distinct"

    X_test_sample, _, _ = preprocessor.transform(splits.df_test.iloc[:2000], feature_subset=preprocessor.feature_names_)
    preds_fpr05 = detector.predict_openset(X_test_sample, model, threshold=tau_fpr05)
    preds_cov95 = detector.predict_openset(X_test_sample, model, threshold=tau_cov95)

    # Rejection counts should differ
    rejections_fpr05 = (preds_fpr05 == UNKNOWN_LABEL_ID).sum()
    rejections_cov95 = (preds_cov95 == UNKNOWN_LABEL_ID).sum()
    assert rejections_fpr05 != rejections_cov95, (
        f"Rejection counts should differ: FPR05 had {rejections_fpr05}, Cov95 had {rejections_cov95}"
    )


def test_energy_metrics_match_confusion_matrices():
    """Test 8: Verifies reported accuracy and operating-point metrics directly against saved confusion matrices."""
    energy_dir = Path("results/openset/LightGBM/features_48/Energy")
    cm_fpr05_path = energy_dir / "confusion_matrix_7x7_fpr05.csv"
    cm_cov95_path = energy_dir / "confusion_matrix_7x7_cov95.csv"
    metrics_json_path = energy_dir / "metrics_openset_energy.json"

    if not metrics_json_path.exists():
        pytest.skip("Energy evaluation artifacts not yet generated. Run evaluate_energy.py first.")

    assert cm_fpr05_path.exists(), f"Missing {cm_fpr05_path}"
    assert cm_cov95_path.exists(), f"Missing {cm_cov95_path}"

    cm_fpr05 = pd.read_csv(cm_fpr05_path, index_col=0)
    cm_cov95 = pd.read_csv(cm_cov95_path, index_col=0)

    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics_json = json.load(f)

    expected_supports = {
        "Normal": 37000,
        "Analysis": 677,
        "Generic": 18871,
        "Exploits": 11132,
        "Fuzzers": 6062,
        "DoS": 4089,
        "Unknown": 4501,
    }

    for cm_df, tag in [(cm_fpr05, "tau_FPR05"), (cm_cov95, "tau_Cov95")]:
        cm = cm_df.values

        # 1. Total sum and row support checks
        assert cm.sum() == 82332, f"{tag}: Total sum must be exactly 82,332, got {cm.sum()}"
        for cname, expected_supp in expected_supports.items():
            actual_supp = int(cm_df.loc[cname].sum())
            assert actual_supp == expected_supp, (
                f"{tag}: Row support mismatch for {cname}: expected {expected_supp}, got {actual_supp}"
            )

        # 2. Arithmetic metric calculations directly from matrix counts
        diag = np.diag(cm)
        acc = float(diag.sum() / cm.sum())

        tp = diag.astype(float)
        fn = (cm.sum(axis=1) - tp).astype(float)
        fp = (cm.sum(axis=0) - tp).astype(float)
        support = cm.sum(axis=1).astype(float)

        prec = np.divide(tp, (tp + fp), out=np.zeros_like(tp), where=(tp + fp) != 0)
        rec = np.divide(tp, (tp + fn), out=np.zeros_like(tp), where=(tp + fn) != 0)
        f1 = np.divide(2 * prec * rec, (prec + rec), out=np.zeros_like(tp), where=(prec + rec) != 0)

        macro_f1 = float(np.mean(f1))
        weighted_f1 = float(np.sum(f1 * support) / cm.sum())

        normal_fpr = float(cm[0, 6] / support[0])
        known_rejected = int(cm[:6, 6].sum())
        known_support = int(support[:6].sum())
        known_fpr = float(known_rejected / known_support)
        known_coverage = float(1.0 - known_fpr)
        unknown_recall = float(cm[6, 6] / support[6])

        # 3. Verify exact match with JSON metrics
        op_json = metrics_json["operating_points"][tag]
        assert np.isclose(acc, op_json["open_set_accuracy"], atol=1e-7), f"{tag}: Accuracy mismatch!"
        assert np.isclose(macro_f1, op_json["open_set_macro_f1"], atol=1e-7), f"{tag}: Macro F1 mismatch!"
        assert np.isclose(weighted_f1, op_json["open_set_weighted_f1"], atol=1e-7), f"{tag}: Weighted F1 mismatch!"
        assert np.isclose(normal_fpr, op_json["realized_test_fpr_normal"], atol=1e-7), f"{tag}: Normal FPR mismatch!"
        assert np.isclose(known_fpr, op_json["realized_test_fpr_known"], atol=1e-7), f"{tag}: Known FPR mismatch!"
        assert np.isclose(known_coverage, op_json["known_coverage"], atol=1e-7), f"{tag}: Known coverage mismatch!"
        assert np.isclose(unknown_recall, op_json["unknown_detection_rate"], atol=1e-7), f"{tag}: Unknown recall mismatch!"
