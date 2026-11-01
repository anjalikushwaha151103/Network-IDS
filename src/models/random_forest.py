"""Random Forest wrapper for XAI-NIDS."""

from typing import List, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base_model import BaseNIDSModel


class RandomForestModel(BaseNIDSModel):
    """Random Forest classifier with ensemble probability and Centered Log-Ratio (CLR) logit extraction."""

    def __init__(self, task: str = "multiclass", random_state: int = 42, **hyperparameters):
        super().__init__("random_forest", task=task, random_state=random_state, **hyperparameters)
        default_params = {
            "n_estimators": 100,
            "max_depth": 16,
            "random_state": random_state,
            "n_jobs": -1,
        }
        default_params.update(hyperparameters)
        self._underlying_model = RandomForestClassifier(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "RandomForestModel":
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
        """Computes Centered Log-Ratio (CLR) transformed logits from ensemble probabilities.

        Since Random Forest outputs leaf-vote posterior probabilities rather than explicit additive margins,
        raw log(p) yields a constant sum of 1. The Centered Log-Ratio:
            f_c(x) = log(p_c) - (1/K) * sum_j log(p_j)
        preserves relative class likelihoods while providing the dynamic range and variance required
        for mathematically valid Energy-Based Out-of-Distribution (OOD) / Zero-Day detection.
        """
        probs = self.predict_proba(X)
        eps = 1e-7
        clipped = np.clip(probs, eps, 1.0)
        log_p = np.log(clipped)
        # Center across classes for each sample
        clr_logits = log_p - np.mean(log_p, axis=1, keepdims=True)
        return clr_logits
