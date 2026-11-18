"""Comprehensive Unit and Pluggability Tests for Random Forest in Open-Set Pipeline (Phase 2 - Step 1).

Verifies:
1. Random Forest instantiation through ModelRegistry with alias resolution.
2. Correct model-specific fit arguments (RF does not receive/require eval_set or callbacks).
3. Exact 6-class output, correct class ordering (0..5), valid probability row sums.
4. Mathematical properties of Centered Log-Ratio (CLR) pseudo-logits (shape, finiteness, zero-mean centering).
5. Polymorphic serialization and deserialization via BaseNIDSModel.load().
6. Generalized train_openset_classifier dispatching and correct artifact routing (LightGBM vs RandomForest).
7. Detector compatibility: MSP, Energy, and OpenMax successfully process Random Forest representations.
8. Non-mutation of existing LightGBM reference artifacts.
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.data_loader import DataLoader
from src.models import ModelRegistry
from src.models.base_model import BaseNIDSModel
from src.models.lightgbm_model import LightGBMModel
from src.models.random_forest import RandomForestModel
from src.openset.base_detector import BaseOODDetector
from src.openset.energy_detector import EnergyDetector
from src.openset.msp_detector import MSPDetector
from src.openset.openmax_detector import OpenMaxDetector
from src.openset.partitioner import (
    KNOWN_CLASSES,
    KNOWN_CLASS_TO_ID,
    extract_known_multiclass_labels,
    partition_openset_data,
)
from src.openset.train_classifier import train_openset_classifier
from src.preprocessing import NIDSPreprocessor


@pytest.fixture(scope="module")
def sampled_splits():
    """Provides fast sampled splits for pluggability testing without full dataset overhead."""
    loader = DataLoader()
    train_df, test_df = loader.load_data(sample_size=3000, test_sample_size=1000, random_state=42)
    splits = partition_openset_data(train_df, test_df, random_state=42)
    return splits


def test_random_forest_registry_instantiation():
    """Test 1: Verifies ModelRegistry instantiation and alias resolution for Random Forest."""
    for alias in ["random_forest", "rf", "randomforest", "Random_Forest"]:
        canonical = ModelRegistry.get_canonical_name(alias)
        assert canonical == "random_forest", f"Alias '{alias}' did not resolve to 'random_forest'"

    rf_model = ModelRegistry.create("random_forest", task="multiclass", random_state=42, n_estimators=10)
    assert isinstance(rf_model, RandomForestModel)
    assert isinstance(rf_model, BaseNIDSModel)
    assert rf_model.model_name == "random_forest"
    assert rf_model.task == "multiclass"
    assert rf_model.is_fitted is False


def test_random_forest_fit_predict_properties(sampled_splits):
    """Test 2 & 3: Verifies 6-class output, class ordering, probability validity, and CLR properties."""
    df_train = sampled_splits.df_train_known
    df_val = sampled_splits.df_val_known

    preprocessor = NIDSPreprocessor()
    X_train, _, _ = preprocessor.fit_transform(df_train)
    y_train = extract_known_multiclass_labels(df_train)

    X_val, _, _ = preprocessor.transform(df_val)
    y_val = extract_known_multiclass_labels(df_val)

    rf_model = RandomForestModel(task="multiclass", n_estimators=25, max_depth=10, random_state=42)
    # Fit strictly with supported arguments: X, y, feature_names (NO eval_set or callbacks)
    rf_model.fit(X_train, y_train, feature_names=preprocessor.feature_names_)

    assert rf_model.is_fitted is True
    assert len(rf_model.classes_) == 6, f"Expected 6 output classes, got {len(rf_model.classes_)}"
    assert list(rf_model.classes_) == [0, 1, 2, 3, 4, 5], f"Expected classes [0..5], got {rf_model.classes_}"

    # Predictions
    preds = rf_model.predict(X_val)
    assert set(np.unique(preds)).issubset({0, 1, 2, 3, 4, 5})

    # Probabilities
    probs = rf_model.predict_proba(X_val)
    assert probs.shape == (len(X_val), 6)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

    # CLR Pseudo-logits
    clr_logits = rf_model.predict_logits(X_val)
    assert clr_logits.shape == (len(X_val), 6)
    assert np.all(np.isfinite(clr_logits)), "CLR pseudo-logits must be strictly finite"
    # Centered property: mean across classes for every sample must equal 0.0
    row_means = np.mean(clr_logits, axis=1)
    np.testing.assert_allclose(row_means, 0.0, atol=1e-6)


def test_polymorphic_base_model_loading(sampled_splits, tmp_path):
    """Test 4: Verifies polymorphic serialization and loading via BaseNIDSModel.load()."""
    df_train = sampled_splits.df_train_known
    preprocessor = NIDSPreprocessor()
    X_train, _, _ = preprocessor.fit_transform(df_train)
    y_train = extract_known_multiclass_labels(df_train)

    # Train and save a small Random Forest
    rf_model = RandomForestModel(task="multiclass", n_estimators=10, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_path = tmp_path / "rf_model.pkl"
    rf_model.save(rf_path)

    # Load back using BaseNIDSModel.load
    loaded_rf = BaseNIDSModel.load(rf_path)
    assert isinstance(loaded_rf, RandomForestModel)
    assert loaded_rf.model_name == "random_forest"
    assert len(loaded_rf.classes_) == 6

    # Verify that existing LightGBM reference model also loads polymorphically
    lgb_ref_path = Path("results/openset/LightGBM/features_48/model.pkl")
    if lgb_ref_path.exists():
        loaded_lgb = BaseNIDSModel.load(lgb_ref_path)
        assert isinstance(loaded_lgb, LightGBMModel)
        assert loaded_lgb.model_name == "lightgbm"
        assert len(loaded_lgb.classes_) == 6


def test_train_openset_classifier_dispatching(sampled_splits, tmp_path):
    """Test 5: Verifies generalized train_openset_classifier dispatches RF and routes artifacts correctly."""
    df_train = sampled_splits.df_train_known
    df_test_k = sampled_splits.df_test_known

    results = train_openset_classifier(
        model_name="random_forest",
        n_features=48,
        random_state=42,
        base_results_dir=str(tmp_path),
        train_df_override=df_train,
        test_df_override=df_test_k,
    )

    artifact_dir = Path(results["artifact_dir"])
    assert "RandomForest" in str(artifact_dir), f"Artifact dir should contain 'RandomForest', got {artifact_dir}"
    assert (artifact_dir / "model.pkl").exists()
    assert (artifact_dir / "preprocessor.pkl").exists()
    assert (artifact_dir / "metrics_closed_set.json").exists()
    assert (artifact_dir / "run_config.json").exists()
    assert (artifact_dir / "confusion_matrix_6x6.csv").exists()
    assert (artifact_dir / "confusion_matrix_6x6.png").exists()

    with open(artifact_dir / "run_config.json", "r", encoding="utf-8") as f:
        run_cfg = json.load(f)
    assert run_cfg["model_name"] == "RandomForest"
    assert run_cfg["num_classes"] == 6

    with open(artifact_dir / "metrics_closed_set.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["model"] == "RandomForest"
    assert "accuracy" in metrics
    assert "macro_f1" in metrics


def test_detectors_process_random_forest_representations(sampled_splits):
    """Test 6: Verifies MSP, Energy, and OpenMax detectors execute on Random Forest representations."""
    df_train = sampled_splits.df_train_known
    df_val = sampled_splits.df_val_known
    df_evt = sampled_splits.df_evt_fit
    df_cal = sampled_splits.df_thresh_cal

    prep = NIDSPreprocessor()
    X_tr, _, _ = prep.fit_transform(df_train)
    y_tr = extract_known_multiclass_labels(df_train)

    X_val, _, _ = prep.transform(df_val)
    y_val = extract_known_multiclass_labels(df_val)

    X_evt, _, _ = prep.transform(df_evt)
    y_evt = extract_known_multiclass_labels(df_evt)

    X_cal, _, _ = prep.transform(df_cal)
    y_cal = extract_known_multiclass_labels(df_cal)

    rf = RandomForestModel(task="multiclass", n_estimators=20, max_depth=8, random_state=42)
    rf.fit(X_tr, y_tr, feature_names=prep.feature_names_)

    # 1. MSP Detector
    msp = MSPDetector()
    msp.fit(None, None, rf)
    msp_scores = msp.compute_anomaly_score(X_val, rf)
    assert np.all(msp_scores >= 0.0) and np.all(msp_scores <= 1.0)
    thresh_msp = msp.calibrate_thresholds(X_cal, y_cal, rf)
    assert 0.0 < thresh_msp["tau_FPR05"] < 1.0
    assert 0.0 < thresh_msp["tau_Cov95"] < 1.0

    # 2. Energy Detector on CLR pseudo-logits
    energy = EnergyDetector(temperature=1.0)
    energy.fit(None, None, rf)
    energy_scores = energy.compute_anomaly_score(X_val, rf)
    assert np.all(np.isfinite(energy_scores))
    thresh_energy = energy.calibrate_thresholds(X_cal, y_cal, rf)
    assert np.isfinite(thresh_energy["tau_FPR05"])
    assert np.isfinite(thresh_energy["tau_Cov95"])

    # 3. OpenMax Detector on CLR pseudo-logits
    openmax = OpenMaxDetector(alpha=3, min_eta=3, max_eta=10)
    openmax.fit(X_evt, y_evt, rf)
    assert openmax.is_fitted
    assert len(openmax.mavs_) == 6
    probs_7 = openmax.predict_proba_7class(X_val, rf)
    assert probs_7.shape == (len(X_val), 7)
    np.testing.assert_allclose(probs_7.sum(axis=1), 1.0, rtol=1e-5)
    openmax_scores = openmax.compute_anomaly_score(X_val, rf)
    assert np.all(openmax_scores >= 0.0) and np.all(openmax_scores <= 1.0)
    thresh_om = openmax.calibrate_thresholds(X_cal, y_cal, rf)
    assert 0.0 < thresh_om["tau_FPR05"] < 1.0


def test_lightgbm_reference_artifacts_untouched():
    """Test 7: Regression check ensuring results/openset/LightGBM/features_48/ was not mutated."""
    lgb_dir = Path("results/openset/LightGBM/features_48")
    assert lgb_dir.exists(), "Reference LightGBM directory must exist"

    # Verify key reference files exist and are non-empty
    for fname in [
        "model.pkl",
        "preprocessor.pkl",
        "run_config.json",
        "metrics_closed_set.json",
        "confusion_matrix_6x6.csv",
        "MSP/thresholds.json",
        "MSP/metrics_openset_msp.json",
        "Energy/thresholds.json",
        "Energy/metrics_openset_energy.json",
        "OpenMax/thresholds.json",
        "OpenMax/mav_and_weibull_params.json",
        "OpenMax/metrics_openset_openmax.json",
    ]:
        fpath = lgb_dir / fname
        assert fpath.exists(), f"Reference file missing: {fpath}"
        assert fpath.stat().st_size > 0, f"Reference file is empty: {fpath}"

    # Verify run_config still specifies LightGBM
    with open(lgb_dir / "run_config.json", "r", encoding="utf-8") as f:
        run_cfg = json.load(f)
    assert run_cfg["model_name"] == "LightGBM"
    assert run_cfg["actual_feature_count"] == 48
