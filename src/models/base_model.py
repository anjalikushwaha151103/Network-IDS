"""Abstract Base Model interface for XAI-NIDS.

Ensures pluggable interoperability across Logistic Regression, Random Forest,
LightGBM, XGBoost, and MLP.
"""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class BaseNIDSModel(ABC):
    """Abstract base class for all pluggable NIDS classification models."""

    def __init__(self, model_name: str, task: str = "multiclass", random_state: int = 42, **hyperparameters):
        self.model_name = model_name
        self.task = task
        self.random_state = random_state
        self.hyperparameters = hyperparameters

        self.is_fitted: bool = False
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []
        self._underlying_model: Any = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "BaseNIDSModel":
        """Fits the underlying model on the training feature matrix X and labels y."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for input matrix X."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for input matrix X of shape (n_samples, n_classes)."""
        pass

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Returns unnormalized logits or log-odds for input matrix X.

        Essential for Energy-based Out-of-Distribution / Zero-Day Detection (Novelty 1).
        Default implementation converts clipped probabilities to log-odds.
        Subclasses with native logits (e.g. Logistic Regression, MLP, XGBoost margins) override this.
        """
        probs = self.predict_proba(X)
        clipped_probs = np.clip(probs, 1e-12, 1.0)
        # Log-odds representation: log(p_i)
        return np.log(clipped_probs)

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Returns feature importance scores if available from the underlying model."""
        if hasattr(self._underlying_model, "feature_importances_"):
            return self._underlying_model.feature_importances_
        if hasattr(self._underlying_model, "coef_"):
            # For linear models, use mean absolute coefficient across classes
            return np.mean(np.abs(self._underlying_model.coef_), axis=0)
        return None

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes the fitted model wrapper to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved model '%s' to %s", self.model_name, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseNIDSModel":
        """Deserializes a fitted model wrapper from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded model '%s' from %s", model.model_name, filepath)
        return model
