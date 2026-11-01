"""Abstract base class for Open-Set / Out-of-Distribution Detectors."""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.models.base_model import BaseNIDSModel

logger = logging.getLogger(__name__)


class BaseOODDetector(ABC):
    """Abstract interface for all open-set / OOD detection mechanisms.

    Subclasses implement specific anomaly scoring mechanisms (MSP, Energy, OpenMax).
    """

    def __init__(self, name: str):
        self.name = name
        self.is_fitted: bool = False
        self.calibrated_thresholds_: Dict[str, float] = {}
        self.calibration_stats_: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, X_evt: np.ndarray, y_evt: np.ndarray, model: BaseNIDSModel) -> "BaseOODDetector":
        """Fits detector parameters (e.g., EVT Weibull tails, class centroids).

        MSP and Energy do not require parameter fitting and may implement this as a no-op.
        """
        pass

    @abstractmethod
    def compute_anomaly_score(self, X: np.ndarray, model: BaseNIDSModel) -> np.ndarray:
        """Computes OOD anomaly score for input matrix X.

        Convention: Higher score indicates greater anomaly / greater likelihood of being an Unknown sample.

        Returns:
            1D array of shape (n_samples,) with float anomaly scores.
        """
        pass

    @abstractmethod
    def compute_confidence_score(self, X: np.ndarray, model: BaseNIDSModel) -> np.ndarray:
        """Computes Known-confidence score for input matrix X.

        Convention: Higher score indicates higher confidence that the sample belongs to a Known class.

        Returns:
            1D array of shape (n_samples,) with float confidence scores.
        """
        pass

    def calibrate_thresholds(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        model: BaseNIDSModel,
        normal_class_id: int = 0,
        percentile_target: float = 95.0,
    ) -> Dict[str, float]:
        """Calibrates operating thresholds strictly on D_threshold_cal without observing test or unknown samples.

        Args:
            X_cal: Feature matrix of D_threshold_cal.
            y_cal: Known-class integer labels of D_threshold_cal (0..5).
            model: Fitted 6-class classifier.
            normal_class_id: Integer class ID for Normal traffic (default=0).
            percentile_target: Percentile used for calibration (default=95.0).

        Returns:
            Dictionary containing calibrated threshold values:
            - 'tau_FPR05': 95th percentile on Normal calibration samples (~5% FPR target on Normal).
            - 'tau_Cov95': 95th percentile on all Known calibration samples (~95% coverage target on Known).
        """
        scores = self.compute_anomaly_score(X_cal, model)

        # 1. Calibrate tau_FPR05 on Normal calibration samples only
        is_normal = (y_cal == normal_class_id)
        if not is_normal.any():
            raise ValueError(f"No Normal samples (class ID {normal_class_id}) found in calibration data!")

        normal_scores = scores[is_normal]
        tau_fpr05 = float(np.percentile(normal_scores, percentile_target, method="linear"))

        # 2. Calibrate tau_Cov95 on all known calibration samples
        tau_cov95 = float(np.percentile(scores, percentile_target, method="linear"))

        self.calibrated_thresholds_ = {
            "tau_FPR05": tau_fpr05,
            "tau_Cov95": tau_cov95,
        }

        self.calibration_stats_ = {
            "percentile_target": percentile_target,
            "percentile_method": "linear",
            "normal_cal_samples": int(is_normal.sum()),
            "total_cal_samples": len(scores),
            "tau_FPR05": tau_fpr05,
            "tau_Cov95": tau_cov95,
            "normal_score_mean": float(np.mean(normal_scores)),
            "normal_score_std": float(np.std(normal_scores)),
            "all_known_score_mean": float(np.mean(scores)),
            "all_known_score_std": float(np.std(scores)),
        }

        logger.info(
            "Calibrated [%s] on %d samples: tau_FPR05=%.6f, tau_Cov95=%.6f",
            self.name,
            len(scores),
            tau_fpr05,
            tau_cov95,
        )
        return self.calibrated_thresholds_

    def predict_openset(
        self,
        X: np.ndarray,
        model: BaseNIDSModel,
        threshold: float,
        unknown_label: int = 6,
    ) -> np.ndarray:
        """Predicts 7-class open-set labels using the base classifier and calibrated rejection threshold.

        Decision Rule:
            If S(x) > threshold:
                predict unknown_label (6)
            Else:
                predict base classifier's argmax class in {0, 1, ..., 5}

        Args:
            X: Input feature matrix.
            model: Fitted 6-class classifier.
            threshold: Rejection threshold.
            unknown_label: Integer label representing 'Unknown' (default=6).

        Returns:
            1D array of predicted open-set class labels in {0, 1, 2, 3, 4, 5, 6}.
        """
        base_preds = model.predict(X)
        scores = self.compute_anomaly_score(X, model)

        openset_preds = np.copy(base_preds)
        is_rejected = (scores > threshold)
        openset_preds[is_rejected] = unknown_label
        return openset_preds
