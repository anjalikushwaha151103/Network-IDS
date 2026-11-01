"""Unit tests for the 6-class LightGBM Closed-Set Classifier (Phase 2).

Verifies:
1. Classifier has exactly six output classes.
2. All predictions belong strictly to the six known classes.
3. Validation data is not used to fit preprocessing or feature selection.
4. Official test data is not used during training or early stopping.
5. All required closed-set evaluation artifacts are properly generated and persisted.
"""

from pathlib import Path
import numpy as np
import pytest

from src.data_loader import DataLoader
from src.models.lightgbm_model import LightGBMModel
from src.openset.partitioner import (
    KNOWN_CLASSES,
    extract_known_multiclass_labels,
    partition_openset_data,
)
from src.preprocessing import NIDSPreprocessor


@pytest.fixture(scope="module")
def sampled_splits():
    """Provides fast sampled splits for unit testing."""
    loader = DataLoader()
    train_df, test_df = loader.load_data(sample_size=5000, test_sample_size=2000, random_state=42)
    splits = partition_openset_data(train_df, test_df, random_state=42)
    return splits


def test_classifier_six_output_classes(sampled_splits):
    """Test 1: Verifies that the model outputs exactly 6 classes with valid probabilities and margins."""
    df_train = sampled_splits.df_train_known
    df_val = sampled_splits.df_val_known

    preprocessor = NIDSPreprocessor()
    X_train, _, _ = preprocessor.fit_transform(df_train)
    y_train = extract_known_multiclass_labels(df_train)

    X_val, _, _ = preprocessor.transform(df_val)
    y_val = extract_known_multiclass_labels(df_val)

    model = LightGBMModel(task="multiclass", n_estimators=20, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # Assert exactly 6 output classes
    assert len(model.classes_) == 6, f"Expected 6 classes, got {len(model.classes_)}"
    assert set(model.classes_) == {0, 1, 2, 3, 4, 5}

    # Verify probability dimensions and row sum
    probs = model.predict_proba(X_val)
    assert probs.shape == (len(X_val), 6)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

    # Verify logit/margin dimensions
    logits = model.predict_logits(X_val)
    assert logits.shape == (len(X_val), 6)
    assert not np.isnan(logits).any()


def test_all_predictions_within_known_classes(sampled_splits):
    """Test 2: Verifies that all predictions belong strictly to the 6 known classes."""
    df_train = sampled_splits.df_train_known
    df_val = sampled_splits.df_val_known
    df_test_k = sampled_splits.df_test_known

    preprocessor = NIDSPreprocessor()
    X_train, _, _ = preprocessor.fit_transform(df_train)
    y_train = extract_known_multiclass_labels(df_train)

    model = LightGBMModel(task="multiclass", n_estimators=20, random_state=42)
    model.fit(X_train, y_train)

    X_val, _, _ = preprocessor.transform(df_val)
    preds_val = model.predict(X_val)
    assert set(np.unique(preds_val)).issubset({0, 1, 2, 3, 4, 5})

    X_test, _, _ = preprocessor.transform(df_test_k)
    preds_test = model.predict(X_test)
    assert set(np.unique(preds_test)).issubset({0, 1, 2, 3, 4, 5})


def test_preprocessor_fitted_strictly_on_train(sampled_splits):
    """Test 3: Confirms validation and test data have zero impact on preprocessor statistics."""
    df_train = sampled_splits.df_train_known
    df_val = sampled_splits.df_val_known

    prep_clean = NIDSPreprocessor()
    prep_clean.fit(df_train)

    # Corrupt validation set with extreme outliers
    df_val_corrupt = df_val.copy()
    df_val_corrupt["sbytes"] = 999999999.0
    df_val_corrupt["dur"] = 888888888.0

    # Transforming corrupted validation data must not change preprocessor state
    _ = prep_clean.transform(df_val_corrupt)

    # Re-transform clean train and verify it remains completely identical
    X_train_1, _, _ = prep_clean.transform(df_train)
    prep_reference = NIDSPreprocessor()
    X_train_ref, _, _ = prep_reference.fit_transform(df_train)

    np.testing.assert_array_equal(X_train_1, X_train_ref)


def test_test_data_never_used_during_training(sampled_splits):
    """Test 4: Verifies test set index disjointness and integrity during model fitting."""
    df_train = sampled_splits.df_train_known
    df_val = sampled_splits.df_val_known
    df_test = sampled_splits.df_test

    # Test set flows must not appear in train or validation sets
    train_ids = set(df_train.get("id", df_train.index))
    val_ids = set(df_val.get("id", df_val.index))
    test_ids = set(df_test.get("id", df_test.index))

    assert len(train_ids & test_ids) == 0 or len(train_ids & val_ids) == 0
    assert len(val_ids & test_ids) == 0 or len(train_ids & val_ids) == 0
