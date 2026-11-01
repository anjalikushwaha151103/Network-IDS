"""XGBoost wrapper for XAI-NIDS."""

from typing import List, Optional
import numpy as np
import xgboost as xgb

from src.models.base_model import BaseNIDSModel


class XGBoostModel(BaseNIDSModel):
    """XGBoost gradient booster with output margin extraction."""

    def __init__(self, task: str = "multiclass", random_state: int = 42, **hyperparameters):
        super().__init__("xgboost", task=task, random_state=random_state, **hyperparameters)
        default_params = {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 8,
            "random_state": random_state,
            "n_jobs": -1,
            "eval_metric": "mlogloss" if task == "multiclass" else "logloss",
        }
        default_params.update(hyperparameters)
        self._underlying_model = xgb.XGBClassifier(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "XGBoostModel":
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
        """Returns raw additive margin scores before logistic/softmax transform."""
        margins = self._underlying_model.predict(X, output_margin=True)
        if len(self.classes_) == 2 and (margins.ndim == 1 or margins.shape[1] == 1):
            margins = margins.reshape(-1, 1)
            return np.hstack([-margins, margins])
        return margins
