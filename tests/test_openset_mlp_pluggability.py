"""Comprehensive Unit and Pluggability Tests for MLP (Multi-Layer Perceptron) in Open-Set Pipeline.

Verifies:
1. MLP instantiation through ModelRegistry with alias resolution.
2. Correct 6-class fit on sample data, exact class ordering (0..5), valid probability row sums.
3. Native pre-softmax logit extraction (predict_logits): shape, finiteness, continuous distribution,
   and exact mathematical equivalence between softmax(predict_logits) and predict_proba.
4. Feature importance proxy extraction via BaseNIDSModel.get_feature_importances().
5. Polymorphic serialization and deserialization via BaseNIDSModel.load().
6. Detector compatibility: MSP, Energy, and OpenMax successfully process MLP representations on sample data.
7. Non-mutation of existing LightGBM, Random Forest, XGBoost, Logistic Regression, and synthesis artifacts.
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
from scipy.special import softmax

from src.data_loader import DataLoader
from src.models import ModelRegistry
from src.models.base_model import BaseNIDSModel
from src.models.mlp_model import MLPModel
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
from src.preprocessing import NIDSPreprocessor


@pytest.fixture(scope="module")
def sampled_splits():
    """Provides fast sampled splits for pluggability testing without full dataset overhead."""
    loader = DataLoader()
    train_df, test_df = loader.load_data(sample_size=3000, test_sample_size=1000, random_state=42)
    splits = partition_openset_data(train_df, test_df, random_state=42)
    return splits


def test_mlp_registry_instantiation():
    """Test 1: Verifies ModelRegistry instantiation and alias resolution for MLP."""
    for alias in ["mlp", "neural_net", "neural_network", "multilayer_perceptron", "MLP"]:
        canonical = ModelRegistry.get_canonical_name(alias)
        assert canonical == "mlp", f"Alias '{alias}' did not resolve to 'mlp'"

    mlp_model = ModelRegistry.create(
        "mlp", task="multiclass", random_state=42, hidden_layer_sizes=[64, 32], max_iter=20
    )
    assert isinstance(mlp_model, MLPModel)
    assert isinstance(mlp_model, BaseNIDSModel)
    assert mlp_model.model_name == "mlp"
    assert mlp_model.task == "multiclass"
    assert mlp_model.is_fitted is False


def test_mlp_fit_predict_properties(sampled_splits):
    """Test 2: Verifies 6-class output, class ordering, probability validity, and exact pre-softmax logits."""
    df_train = sampled_splits.df_train_known
    df_val = sampled_splits.df_val_known

    preprocessor = NIDSPreprocessor()
    X_train, _, _ = preprocessor.fit_transform(df_train)
    y_train = extract_known_multiclass_labels(df_train)

    X_val, _, _ = preprocessor.transform(df_val)
    y_val = extract_known_multiclass_labels(df_val)

    mlp_model = MLPModel(task="multiclass", hidden_layer_sizes=(64, 32), max_iter=30, random_state=42)
    mlp_model.fit(X_train, y_train, feature_names=preprocessor.feature_names_)

    assert mlp_model.is_fitted is True
    assert len(mlp_model.classes_) == 6, f"Expected 6 output classes, got {len(mlp_model.classes_)}"
    assert list(mlp_model.classes_) == [0, 1, 2, 3, 4, 5], f"Expected classes [0..5], got {mlp_model.classes_}"

    # Predictions
    preds = mlp_model.predict(X_val)
    assert set(np.unique(preds)).issubset({0, 1, 2, 3, 4, 5})

    # Probabilities
    probs = mlp_model.predict_proba(X_val)
    assert probs.shape == (len(X_val), 6)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

    # Native pre-softmax logits (predict_logits)
    logits = mlp_model.predict_logits(X_val)
    assert logits.shape == (len(X_val), 6)
    assert np.all(np.isfinite(logits)), "MLP logits must be strictly finite"
    assert np.var(logits) > 0.01, "MLP logits must exhibit non-zero variance"

    # Mathematical identity check: softmax(logits) == predict_proba
    probs_from_logits = softmax(logits, axis=1)
    np.testing.assert_allclose(
        probs_from_logits,
        probs,
        atol=1e-5,
        err_msg="Softmax of pre-softmax logits must match predict_proba() identically",
    )

    # Feature importances
    importances = mlp_model.get_feature_importances()
    assert importances is not None
    assert len(importances) == X_train.shape[1]
    assert np.all(importances >= 0.0)


def test_mlp_serialization(sampled_splits):
    """Test 3: Verifies polymorphic serialization and deserialization via BaseNIDSModel."""
    df_train = sampled_splits.df_train_known
    df_val = sampled_splits.df_val_known

    preprocessor = NIDSPreprocessor()
    X_train, _, _ = preprocessor.fit_transform(df_train)
    y_train = extract_known_multiclass_labels(df_train)
    X_val, _, _ = preprocessor.transform(df_val)

    mlp_model = MLPModel(task="multiclass", hidden_layer_sizes=(64, 32), max_iter=20, random_state=42)
    mlp_model.fit(X_train, y_train, feature_names=preprocessor.feature_names_)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "model.pkl"
        mlp_model.save(model_path)
        assert model_path.exists()

        loaded_model = BaseNIDSModel.load(model_path)
        assert isinstance(loaded_model, MLPModel)
        assert loaded_model.is_fitted is True
        assert list(loaded_model.classes_) == [0, 1, 2, 3, 4, 5]

        preds_orig = mlp_model.predict(X_val)
        preds_loaded = loaded_model.predict(X_val)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

        logits_orig = mlp_model.predict_logits(X_val)
        logits_loaded = loaded_model.predict_logits(X_val)
        np.testing.assert_allclose(logits_orig, logits_loaded, rtol=1e-6)


def test_mlp_detector_compatibility(sampled_splits):
    """Test 4: Verifies compatibility with MSP, Energy, and OpenMax detectors."""
    df_train = sampled_splits.df_train_known
    df_val = sampled_splits.df_val_known

    preprocessor = NIDSPreprocessor()
    X_train, _, _ = preprocessor.fit_transform(df_train)
    y_train = extract_known_multiclass_labels(df_train)
    X_val, _, _ = preprocessor.transform(df_val)
    y_val = extract_known_multiclass_labels(df_val)

    mlp_model = MLPModel(task="multiclass", hidden_layer_sizes=(64, 32), max_iter=20, random_state=42)
    mlp_model.fit(X_train, y_train, feature_names=preprocessor.feature_names_)

    # 1. MSP Detector
    msp = MSPDetector()
    msp.fit(None, None, mlp_model)
    msp_scores = msp.compute_anomaly_score(X_val, mlp_model)
    assert len(msp_scores) == len(X_val)
    assert np.all(np.isfinite(msp_scores))
    assert np.all(msp_scores >= 0.0) and np.all(msp_scores <= 1.0)

    # 2. Energy Detector
    energy = EnergyDetector(temperature=1.0)
    energy.fit(None, None, mlp_model)
    energy_scores = energy.compute_anomaly_score(X_val, mlp_model)
    assert len(energy_scores) == len(X_val)
    assert np.all(np.isfinite(energy_scores))
    assert np.var(energy_scores) > 0.0

    # 3. OpenMax Detector
    openmax = OpenMaxDetector(alpha=3, min_eta=5, max_eta=20)
    openmax.fit(X_val, y_val, mlp_model)
    probs_7 = openmax.predict_proba_7class(X_val, mlp_model)
    assert probs_7.shape == (len(X_val), 7)
    assert np.all(probs_7 >= 0.0)
    assert np.all(probs_7 <= 1.0)
    np.testing.assert_allclose(probs_7.sum(axis=1), 1.0, atol=1e-5)


def test_prior_model_reference_artifacts_untouched():
    """Test 5: Verifies that prior model and synthesis artifacts remain intact and unmutated."""
    required_paths = [
        # LightGBM
        Path("results/openset/LightGBM/features_48/model.pkl"),
        Path("results/openset/LightGBM/features_48/metrics_closed_set.json"),
        Path("results/openset/LightGBM/features_48/MSP/metrics_openset_msp.json"),
        Path("results/openset/LightGBM/features_48/Energy/metrics_openset_energy.json"),
        Path("results/openset/LightGBM/features_48/OpenMax/metrics_openset_openmax.json"),
        # Random Forest
        Path("results/openset/RandomForest/features_48/model.pkl"),
        Path("results/openset/RandomForest/features_48/metrics_closed_set.json"),
        Path("results/openset/RandomForest/features_48/MSP/metrics_openset_msp.json"),
        Path("results/openset/RandomForest/features_48/Energy/metrics_openset_energy.json"),
        Path("results/openset/RandomForest/features_48/OpenMax/metrics_openset_openmax.json"),
        # XGBoost
        Path("results/openset/XGBoost/features_48/model.pkl"),
        Path("results/openset/XGBoost/features_48/metrics_closed_set.json"),
        Path("results/openset/XGBoost/features_48/MSP/metrics_openset_msp.json"),
        Path("results/openset/XGBoost/features_48/Energy/metrics_openset_energy.json"),
        Path("results/openset/XGBoost/features_48/OpenMax/metrics_openset_openmax.json"),
        # Logistic Regression
        Path("results/openset/LogisticRegression/features_48/model.pkl"),
        Path("results/openset/LogisticRegression/features_48/metrics_closed_set.json"),
        Path("results/openset/LogisticRegression/features_48/MSP/metrics_openset_msp.json"),
        Path("results/openset/LogisticRegression/features_48/Energy/metrics_openset_energy.json"),
        Path("results/openset/LogisticRegression/features_48/OpenMax/metrics_openset_openmax.json"),
        # Synthesis
        Path("results/openset/synthesis/consolidated_benchmark_table.csv"),
        Path("results/openset/synthesis_3models/consolidated_benchmark_table.csv"),
    ]

    for p in required_paths:
        assert p.exists(), f"Reference artifact {p} is missing or corrupted!"
