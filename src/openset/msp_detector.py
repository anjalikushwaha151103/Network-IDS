"""Maximum Softmax Probability (MSP) Baseline OOD Detector (Hendrycks & Gimpel, 2017)."""

import numpy as np

from src.models.base_model import BaseNIDSModel
from src.openset.base_detector import BaseOODDetector


class MSPDetector(BaseOODDetector):
    """Maximum Softmax Probability baseline open-set detector.

    Anomaly score:
        S(x) = 1.0 - max_{k in {1..K}} p_k(x)
        Higher score indicates greater prediction uncertainty and higher likelihood of being Unknown.

    Known-confidence:
        C_known(x) = max_{k in {1..K}} p_k(x) = 1.0 - S(x)
        Higher confidence indicates greater certainty in a known-class classification.
    """

    def __init__(self):
        super().__init__("Maximum Softmax Probability (MSP)")

    def fit(self, X_evt: np.ndarray, y_evt: np.ndarray, model: BaseNIDSModel) -> "MSPDetector":
        """MSP requires no parameter fitting."""
        self.is_fitted = True
        return self

    def compute_anomaly_score(self, X: np.ndarray, model: BaseNIDSModel) -> np.ndarray:
        """Computes anomaly score: 1.0 - max class probability."""
        probs = model.predict_proba(X)
        max_p = np.max(probs, axis=1)
        return 1.0 - max_p

    def compute_confidence_score(self, X: np.ndarray, model: BaseNIDSModel) -> np.ndarray:
        """Computes known-confidence score: max class probability."""
        probs = model.predict_proba(X)
        return np.max(probs, axis=1)
