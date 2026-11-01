"""Unit tests for the pluggable model framework across all 5 models."""

import numpy as np
import pytest
from src.data_loader import DataLoader
from src.models import ModelRegistry
from src.preprocessing import NIDSPreprocessor

ALL_MODELS = [
    "logistic_regression",
    "random_forest",
    "lightgbm",
    "xgboost",
    "mlp",
]


@pytest.fixture(scope="module")
def prepared_data():
    loader = DataLoader()
    train_df, test_df = loader.load_data(sample_size=400, test_sample_size=100)
    prep = NIDSPreprocessor()
    X_train, y_train_bin, y_train_multi = prep.fit_transform(train_df)
    X_test, y_test_bin, y_test_multi = prep.transform(test_df)
    return {
        "X_train": X_train,
        "y_train_bin": y_train_bin,
        "y_train_multi": y_train_multi,
        "X_test": X_test,
        "y_test_bin": y_test_bin,
        "y_test_multi": y_test_multi,
        "feature_names": prep.feature_names_,
    }


def test_registry_listing():
    models = ModelRegistry.list_models()
    for expected in ALL_MODELS:
        assert expected in models, f"Expected {expected} in registry"


def test_registry_aliases():
    assert ModelRegistry.get_canonical_name("xgboast") == "xgboost"
    assert ModelRegistry.get_canonical_name("rf") == "random_forest"
    assert ModelRegistry.get_canonical_name("lr") == "logistic_regression"
    assert ModelRegistry.get_canonical_name("lgbm") == "lightgbm"


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_model_fit_predict_binary(model_name, prepared_data):
    X_train = prepared_data["X_train"]
    y_train = prepared_data["y_train_bin"]
    X_test = prepared_data["X_test"]

    model = ModelRegistry.create(model_name, task="binary", random_state=42)
    model.fit(X_train, y_train, feature_names=prepared_data["feature_names"])

    assert model.is_fitted
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})

    probs = model.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0.0).all()

    # Verify logits for energy-based zero-day detection
    logits = model.predict_logits(X_test)
    assert logits.shape == (len(X_test), 2)
    assert not np.isnan(logits).any()


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_model_fit_predict_multiclass(model_name, prepared_data):
    X_train = prepared_data["X_train"]
    y_train = prepared_data["y_train_multi"]
    X_test = prepared_data["X_test"]

    model = ModelRegistry.create(model_name, task="multiclass", random_state=42)
    model.fit(X_train, y_train, feature_names=prepared_data["feature_names"])

    assert model.is_fitted
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)

    n_classes = len(model.classes_)
    probs = model.predict_proba(X_test)
    assert probs.shape == (len(X_test), n_classes)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    logits = model.predict_logits(X_test)
    assert logits.shape == (len(X_test), n_classes)
    assert not np.isnan(logits).any()


def test_model_serialization(tmp_path, prepared_data):
    X_train = prepared_data["X_train"]
    y_train = prepared_data["y_train_bin"]
    X_test = prepared_data["X_test"]

    model = ModelRegistry.create("random_forest", task="binary")
    model.fit(X_train, y_train)
    p1 = model.predict_proba(X_test)

    save_path = tmp_path / "rf_model.pkl"
    model.save(save_path)

    loaded_model = ModelRegistry.create("random_forest").load(save_path)
    p2 = loaded_model.predict_proba(X_test)
    assert np.allclose(p1, p2)
