"""Models module for XAI-NIDS: base interface, concrete models, and factory registry."""

from src.models.base_model import BaseNIDSModel
from src.models.lightgbm_model import LightGBMModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.mlp_model import MLPModel
from src.models.random_forest import RandomForestModel
from src.models.registry import ModelRegistry
from src.models.xgboost_model import XGBoostModel

# Auto-register all 5 supported models
ModelRegistry.register("logistic_regression", LogisticRegressionModel)
ModelRegistry.register("random_forest", RandomForestModel)
ModelRegistry.register("lightgbm", LightGBMModel)
ModelRegistry.register("xgboost", XGBoostModel)
ModelRegistry.register("mlp", MLPModel)

__all__ = [
    "BaseNIDSModel",
    "ModelRegistry",
    "LogisticRegressionModel",
    "RandomForestModel",
    "LightGBMModel",
    "XGBoostModel",
    "MLPModel",
]
