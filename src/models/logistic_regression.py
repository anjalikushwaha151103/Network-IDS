"""Logistic Regression wrapper for XAI-NIDS."""

from typing import List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.base_model import BaseNIDSModel


class LogisticRegressionModel(BaseNIDSModel):
    """Logistic Regression classifier with exact logit computation."""

    def __init__(self, task: str = "multiclass", random_state: int = 42, **hyperparameters):
        super().__init__("logistic_regression", task=task, random_state=random_state, **hyperparameters)
        default_params = {
            "max_iter": 1000,
            "C": 1.0,
            "solver": "lbfgs",
            "random_state": random_state,
        }
        default_params.update(hyperparameters)
        self._underlying_model = LogisticRegression(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "LogisticRegressionModel":
        self._underlying_model.fit(X, y)
        self.classes_ = self._underlying_model.classes_
        self.is_fitted = True
        if feature_names is not None:
            self.feature_names_ = feature_names
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._underlying_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._underlying_model.predict_proba(X)

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Computes exact linear decision logits (X @ W^T + b)."""
        coef = self._underlying_model.coef_
        intercept = self._underlying_model.intercept_
        logits = np.dot(X, coef.T) + intercept
        if len(self.classes_) == 2 and logits.shape[1] == 1:
            # Binary classification: expand to [ -logit, logit ]
            logits = np.hstack([-logits, logits])
        return logits
