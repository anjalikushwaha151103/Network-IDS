"""Multi-Layer Perceptron (MLP) wrapper for XAI-NIDS."""

from typing import List, Optional
import numpy as np
from sklearn.neural_network import MLPClassifier

from src.models.base_model import BaseNIDSModel


class MLPModel(BaseNIDSModel):
    """Multi-Layer Perceptron neural network with pre-softmax logit extraction."""

    def __init__(self, task: str = "multiclass", random_state: int = 42, **hyperparameters):
        super().__init__("mlp", task=task, random_state=random_state, **hyperparameters)
        default_params = {
            "hidden_layer_sizes": (128, 64),
            "max_iter": 150,
            "learning_rate_init": 0.001,
            "random_state": random_state,
            "early_stopping": True,
            "n_iter_no_change": 10,
        }
        default_params.update(hyperparameters)
        self._underlying_model = MLPClassifier(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "MLPModel":
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
        """Computes exact forward pass activations up to the output layer prior to softmax/sigmoid."""
        activation = X
        # Propagate through hidden layers
        for i in range(len(self._underlying_model.coefs_) - 1):
            w = self._underlying_model.coefs_[i]
            b = self._underlying_model.intercepts_[i]
            z = np.dot(activation, w) + b
            # Default activation in MLPClassifier is relu
            activation = np.maximum(0, z)

        # Output layer logits
        w_out = self._underlying_model.coefs_[-1]
        b_out = self._underlying_model.intercepts_[-1]
        logits = np.dot(activation, w_out) + b_out

        if len(self.classes_) == 2 and logits.shape[1] == 1:
            logits = np.hstack([-logits, logits])
        return logits
