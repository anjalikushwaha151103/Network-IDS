"""Master training pipeline for XAI-NIDS.

Trains models, computes feature subsets (49, 30, 20, 15), and stores all evaluation metrics,
confusion matrices, ROC/PR curves, and comparison reports systematically in results/.
"""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Optional
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from src.config import default_config
from src.data_loader import DataLoader
from src.feature_selection import FeatureSelector
from src.models import ModelRegistry
from src.preprocessing import NIDSPreprocessor
from src.results_manager import ResultsManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_nids(
    model_name: str = "lightgbm",
    n_features: int = 30,
    task: str = "multiclass",
    train_sample_size: Optional[int] = None,
    test_sample_size: Optional[int] = None,
    save_dir: str = "models_saved",
    results_dir: str = "results",
) -> dict:
    """End-to-end training, evaluation, and visualization pipeline for any registered NIDS model."""
    start_time = time.time()
    logger.info("=================================================================")
    logger.info("Starting XAI-NIDS Training Pipeline")
    logger.info("Model: %s | Features: %d | Task: %s", model_name, n_features, task)
    logger.info("=================================================================")

    # 1. Ingestion
    loader = DataLoader(default_config)
    train_df, test_df = loader.load_data(
        sample_size=train_sample_size,
        test_sample_size=test_sample_size,
        random_state=42,
    )

    # 2. Preprocessing
    preprocessor = NIDSPreprocessor(default_config)
    X_train_full, y_train_bin, y_train_multi = preprocessor.fit_transform(train_df)
    X_test_full, y_test_bin, y_test_multi = preprocessor.transform(test_df)

    y_train = y_train_bin if task == "binary" else y_train_multi
    y_test = y_test_bin if task == "binary" else y_test_multi

    # 3. Feature Selection
    selector = FeatureSelector(default_config)
    selector.compute_ranking(X_train_full, y_train, preprocessor.feature_names_)
    active_features = selector.get_tier_features(n_features, all_features=preprocessor.feature_names_)

    logger.info("Active feature tier (%d features): %s", len(active_features), active_features[:10])
    feature_indices = [preprocessor.feature_names_.index(f) for f in active_features]
    X_train = X_train_full[:, feature_indices]
    X_test = X_test_full[:, feature_indices]

    # 4. Model Instantiation & Training
    model = ModelRegistry.create(model_name, task=task, random_state=42)
    fit_start = time.time()
    model.fit(X_train, y_train, feature_names=active_features)
    fit_duration = time.time() - fit_start
    logger.info("Model fitting completed in %.2f seconds", fit_duration)

    # 5. Evaluation
    eval_start = time.time()
    preds = model.predict(X_test)
    eval_duration = time.time() - eval_start

    y_probs = None
    try:
        y_probs = model.predict_proba(X_test)
    except Exception as e:
        logger.warning("Could not compute probabilities for ROC/PR curves: %s", e)

    accuracy = float(np.mean(preds == y_test))
    f1_macro = float(f1_score(y_test, preds, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, preds, average="weighted", zero_division=0))
    precision_macro = float(precision_score(y_test, preds, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_test, preds, average="macro", zero_division=0))

    if task == "multiclass":
        all_unique_ids = sorted(list(set(y_test) | set(preds)))
        class_names = [preprocessor.id_to_class_.get(c, str(c)) for c in all_unique_ids]
    else:
        class_names = ["Normal", "Attack"]

    report_dict = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)

    logger.info("---------------- Evaluation Results ----------------")
    logger.info("Accuracy:           %.4f", accuracy)
    logger.info("Macro F1:           %.4f", f1_macro)
    logger.info("Weighted F1:        %.4f", f1_weighted)
    logger.info("Macro Precision:    %.4f", precision_macro)
    logger.info("Macro Recall:       %.4f", recall_macro)
    logger.info("Inference Latency:  %.4f ms/sample", (eval_duration / len(X_test)) * 1000)
    logger.info("---------------------------------------------------")

    # 6. Artifact Persistence & Results Management
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model_file = save_path / f"{model.model_name}_{n_features}_{task}.pkl"
    prep_file = save_path / f"preprocessor_{model.model_name}_{n_features}_{task}.pkl"
    model.save(model_file)
    preprocessor.save(prep_file)

    metrics = {
        "model_name": model.model_name,
        "n_features": len(active_features),
        "requested_tier": n_features,
        "task": task,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "active_features": active_features,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "fit_time_seconds": fit_duration,
        "eval_time_seconds": eval_duration,
        "total_time_seconds": time.time() - start_time,
        "hyperparameters": model.hyperparameters,
        "classification_report": report_dict,
    }

    # Save to structured results directory (results/<Model>/features_<N>/)
    results_mgr = ResultsManager(results_dir)
    feature_importances = model.get_feature_importances()
    results_mgr.save_experiment_results(
        model_name=model.model_name,
        n_features=n_features,
        metrics=metrics,
        y_true=y_test,
        y_pred=preds,
        y_probs=y_probs,
        class_names=class_names,
        feature_names=active_features,
        feature_importances=feature_importances,
    )

    # Generate / update consolidated comparison report
    results_mgr.generate_consolidated_report()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an XAI-NIDS Model")
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        help="Model: logistic_regression, random_forest, lightgbm, xgboost, mlp",
    )
    parser.add_argument("--features", type=int, default=30, help="Feature tier: 49, 30, 20, 15")
    parser.add_argument("--task", type=str, default="multiclass", choices=["binary", "multiclass"])
    parser.add_argument("--train-samples", type=int, default=None, help="Optional sample size (omit for full dataset)")
    parser.add_argument("--test-samples", type=int, default=None, help="Optional sample size (omit for full dataset)")
    args = parser.parse_args()

    train_nids(
        model_name=args.model,
        n_features=args.features,
        task=args.task,
        train_sample_size=args.train_samples,
        test_sample_size=args.test_samples,
    )
