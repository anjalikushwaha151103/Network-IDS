"""Closed-Set Classifier Training and Evaluation for Open-Set NIDS (Phase 2).

Trains a 6-class classifier strictly on D_train_known (113,288 samples), validates
on D_val_known (24,276 samples) for early stopping and model selection, and evaluates
closed-set performance on the 77,831 known flows in the official test split.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

from src.config import Config, default_config
from src.data_loader import DataLoader
from src.models.lightgbm_model import LightGBMModel
from src.openset.partitioner import (
    KNOWN_CLASS_TO_ID,
    KNOWN_CLASSES,
    OpenSetSplits,
    extract_known_multiclass_labels,
    partition_openset_data,
)
from src.preprocessing import NIDSPreprocessor

logger = logging.getLogger(__name__)


def train_openset_classifier(
    model_name: str = "lightgbm",
    n_features: int = 48,
    random_state: int = 42,
    base_results_dir: str = "results/openset",
    config: Optional[Config] = None,
) -> Dict[str, Any]:
    """Trains and evaluates the 6-class closed-set classifier.

    Args:
        model_name: Name of model to train (currently "lightgbm").
        n_features: Number of features to use (default 48 for all actual available features).
        random_state: Fixed random seed (default 42).
        base_results_dir: Directory where results will be saved.
        config: System configuration object.

    Returns:
        Dictionary containing closed-set metrics, timing, and artifact paths.
    """
    cfg = config or default_config

    # 1. Load official UNSW-NB15 dataset
    logger.info("Loading official UNSW-NB15 dataset...")
    loader = DataLoader(cfg)
    raw_train_df, raw_test_df = loader.load_data()

    # 2. Partition dataset using strict 4-way stratified partitioner
    logger.info("Partitioning data into disjoint 4-way open-set splits (seed=%d)...", random_state)
    splits = partition_openset_data(raw_train_df, raw_test_df, random_state=random_state)

    df_train = splits.df_train_known  # 113,288 flows
    df_val = splits.df_val_known      # 24,276 flows
    df_test_known = splits.df_test_known  # 77,831 flows

    logger.info(
        "Splits prepared: Train Known=%d, Val Known=%d, Test Known=%d",
        len(df_train),
        len(df_val),
        len(df_test_known),
    )

    # 3. Fit Preprocessor exclusively on D_train_known
    logger.info("Fitting preprocessor exclusively on D_train_known (%d samples)...", len(df_train))
    preprocessor = NIDSPreprocessor(cfg)
    preprocessor.fit(df_train)

    feature_pool = preprocessor.feature_names_
    logger.info("Extracted %d input features from preprocessor", len(feature_pool))

    # Determine feature subset if n_features < 48
    if n_features < len(feature_pool):
        from src.feature_selection import FeatureSelector
        selector = FeatureSelector(cfg)
        X_tr_full, _, _ = preprocessor.transform(df_train)
        y_tr_full = extract_known_multiclass_labels(df_train)
        selector.compute_ranking(X_tr_full, y_tr_full, feature_pool)
        selected_features = selector.get_tier_features(n_features, all_features=feature_pool)
    else:
        selected_features = feature_pool

    actual_feature_count = len(selected_features)
    logger.info("Active feature count: %d features", actual_feature_count)

    # Transform all splits using preprocessor fitted on train ONLY
    X_train, _, _ = preprocessor.transform(df_train, feature_subset=selected_features)
    y_train = extract_known_multiclass_labels(df_train)

    X_val, _, _ = preprocessor.transform(df_val, feature_subset=selected_features)
    y_val = extract_known_multiclass_labels(df_val)

    X_test_k, _, _ = preprocessor.transform(df_test_known, feature_subset=selected_features)
    y_test_k = extract_known_multiclass_labels(df_test_known)

    # 4. Instantiate LightGBM Model with configuration hyperparameters
    lgb_params = dict(cfg.model.get("hyperparameters", {}).get("lightgbm", {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 10,
        "num_leaves": 31,
        "n_jobs": -1,
        "verbose": -1,
    }))
    lgb_params.pop("random_state", None)

    logger.info("Instantiating LightGBMModel with parameters: %s (seed=%d)", lgb_params, random_state)
    model = LightGBMModel(task="multiclass", random_state=random_state, **lgb_params)

    # 5. Train Model on D_train_known with D_val_known for early stopping
    logger.info("Training 6-class LightGBM classifier on %d samples...", len(X_train))
    t0_fit = time.perf_counter()
    callbacks = [lgb.early_stopping(stopping_rounds=15, verbose=False)]
    model.fit(
        X_train,
        y_train,
        feature_names=selected_features,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    fit_time = time.perf_counter() - t0_fit
    logger.info("Model training completed in %.2f seconds", fit_time)

    # Verify model classes
    assert len(model.classes_) == 6, f"Expected 6 output classes, got {len(model.classes_)}"
    assert set(model.classes_) == set(range(6)), f"Expected classes 0..5, got {model.classes_}"

    # 6. Evaluate Closed-Set Performance on D_test_known (77,831 flows)
    logger.info("Evaluating closed-set performance on D_test_known (%d flows)...", len(X_test_k))
    t0_inf = time.perf_counter()
    y_pred = model.predict(X_test_k)
    y_probs = model.predict_proba(X_test_k)
    inf_time = time.perf_counter() - t0_inf
    latency_ms_per_sample = (inf_time / len(X_test_k)) * 1000.0

    acc = float(accuracy_score(y_test_k, y_pred))
    macro_f1 = float(f1_score(y_test_k, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test_k, y_pred, average="weighted"))

    cls_report_dict = classification_report(
        y_test_k,
        y_pred,
        target_names=KNOWN_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test_k, y_pred, labels=range(6))

    # Also record validation metrics
    y_val_pred = model.predict(X_val)
    val_acc = float(accuracy_score(y_val, y_val_pred))
    val_macro_f1 = float(f1_score(y_val, y_val_pred, average="macro"))

    logger.info("=== CLOSED-SET EVALUATION RESULTS ===")
    logger.info("Test Accuracy:     %.4f", acc)
    logger.info("Test Macro F1:     %.4f", macro_f1)
    logger.info("Test Weighted F1:  %.4f", weighted_f1)
    logger.info("Validation Acc:    %.4f", val_acc)
    logger.info("Validation F1:     %.4f", val_macro_f1)
    logger.info("Fit Time:          %.2fs", fit_time)
    logger.info("Inference Latency: %.4f ms/sample", latency_ms_per_sample)

    # 7. Persist Artifacts under results/openset/LightGBM/features_<N>/
    out_dir = Path(base_results_dir) / "LightGBM" / f"features_{actual_feature_count}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # A. Save Model & Preprocessor
    model_path = out_dir / "model.pkl"
    model.save(model_path)

    preproc_path = out_dir / "preprocessor.pkl"
    preprocessor.save(preproc_path)

    # B. Save Run Configuration
    run_config = {
        "model_name": "LightGBM",
        "task": "closed_set_multiclass",
        "random_state": random_state,
        "num_classes": 6,
        "known_classes": KNOWN_CLASSES,
        "known_class_to_id": KNOWN_CLASS_TO_ID,
        "actual_feature_count": actual_feature_count,
        "feature_names": selected_features,
        "dataset_split_counts": {
            "train_known": len(df_train),
            "val_known": len(df_val),
            "evt_fit": len(splits.df_evt_fit),
            "thresh_cal": len(splits.df_thresh_cal),
            "test_known": len(df_test_known),
            "test_unknown": len(splits.df_test_unknown),
            "purged_train_unknowns": splits.purged_train_unknown_count,
        },
        "hyperparameters": lgb_params,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    # C. Save Feature Names & Class Mapping
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(selected_features, f, indent=2)

    with open(out_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(KNOWN_CLASS_TO_ID, f, indent=2)

    # D. Save Metrics JSON
    metrics_data = {
        "model": "LightGBM",
        "feature_count": actual_feature_count,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "validation_accuracy": val_acc,
        "validation_macro_f1": val_macro_f1,
        "training_time_seconds": fit_time,
        "inference_latency_ms_per_sample": latency_ms_per_sample,
        "total_test_samples": len(X_test_k),
        "classification_report": cls_report_dict,
    }
    with open(out_dir / "metrics_closed_set.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # E. Save Tabular Classification Report CSV
    report_df = pd.DataFrame(cls_report_dict).transpose()
    report_df.to_csv(out_dir / "classification_report.csv")

    # F. Save Confusion Matrix CSV
    cm_df = pd.DataFrame(cm, index=KNOWN_CLASSES, columns=KNOWN_CLASSES)
    cm_df.to_csv(out_dir / "confusion_matrix_6x6.csv")

    # G. Generate and Save Confusion Matrix Plot
    _plot_confusion_matrix(cm, KNOWN_CLASSES, out_dir / "confusion_matrix_6x6.png")

    # H. Generate ROC and PR Curves
    _plot_multiclass_roc(y_test_k, y_probs, KNOWN_CLASSES, out_dir / "roc_curves_closed_set.png")
    _plot_multiclass_pr(y_test_k, y_probs, KNOWN_CLASSES, out_dir / "pr_curves_closed_set.png")

    # I. Generate Feature Importance Plot
    importances = model.get_feature_importances()
    if importances is not None:
        _plot_feature_importance(selected_features, importances, out_dir / "feature_importance.png")

    logger.info("All closed-set artifacts successfully persisted to: %s", out_dir)

    return {
        "metrics": metrics_data,
        "artifact_dir": str(out_dir),
        "confusion_matrix": cm_df,
        "model": model,
        "preprocessor": preprocessor,
        "splits": splits,
    }


def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    """Plots normalized confusion matrix heatmap."""
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    plt.figure(figsize=(8, 6.5), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Ratio"},
    )
    plt.title("6-Class Closed-Set Confusion Matrix (LightGBM)", fontsize=13, pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _plot_multiclass_roc(y_true: np.ndarray, y_probs: np.ndarray, class_names: List[str], save_path: Path):
    """Plots One-vs-Rest ROC curves for each of the 6 known classes."""
    plt.figure(figsize=(8, 6), dpi=300)
    for idx, cname in enumerate(class_names):
        y_binary = (y_true == idx).astype(int)
        if len(np.unique(y_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_binary, y_probs[:, idx])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.5, label=f"{cname} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Closed-Set Multiclass ROC Curves (LightGBM)", fontsize=13)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _plot_multiclass_pr(y_true: np.ndarray, y_probs: np.ndarray, class_names: List[str], save_path: Path):
    """Plots One-vs-Rest Precision-Recall curves for each of the 6 known classes."""
    plt.figure(figsize=(8, 6), dpi=300)
    for idx, cname in enumerate(class_names):
        y_binary = (y_true == idx).astype(int)
        if len(np.unique(y_binary)) > 1:
            prec, rec, _ = precision_recall_curve(y_binary, y_probs[:, idx])
            pr_auc = auc(rec, prec)
            plt.plot(rec, prec, lw=1.5, label=f"{cname} (AUC = {pr_auc:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Closed-Set Precision-Recall Curves (LightGBM)", fontsize=13)
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _plot_feature_importance(feature_names: List[str], importances: np.ndarray, save_path: Path, top_n: int = 25):
    """Plots top N feature importances."""
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    plt.figure(figsize=(10, 7), dpi=300)
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_scores, align="center", color="#2b5c8f", alpha=0.85)
    plt.yticks(y_pos, top_features)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance Score (Gain)", fontsize=11)
    plt.title(f"Top {top_n} Feature Importances (LightGBM)", fontsize=13)
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    results = train_openset_classifier()
    print("Closed-set training and evaluation complete.")
