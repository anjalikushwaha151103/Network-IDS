"""LightGBM wrapper for XAI-NIDS."""

from typing import List, Optional
import lightgbm as lgb
import numpy as np

from src.models.base_model import BaseNIDSModel


class LightGBMModel(BaseNIDSModel):
    """LightGBM gradient booster with raw margin logit extraction."""

    def __init__(self, task: str = "multiclass", random_state: int = 42, **hyperparameters):
        super().__init__("lightgbm", task=task, random_state=random_state, **hyperparameters)
        default_params = {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 10,
            "num_leaves": 31,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        default_params.update(hyperparameters)
        self._underlying_model = lgb.LGBMClassifier(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None, **fit_kwargs) -> "LightGBMModel":
        self._underlying_model.fit(X, y, **fit_kwargs)
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
        """Returns raw additive margin scores prior to sigmoid/softmax."""
        raw_scores = self._underlying_model.predict(X, raw_score=True)
        if len(self.classes_) == 2 and (raw_scores.ndim == 1 or raw_scores.shape[1] == 1):
            raw_scores = raw_scores.reshape(-1, 1)
            return np.hstack([-raw_scores, raw_scores])
        return raw_scores
