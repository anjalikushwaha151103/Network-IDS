"""Results management, plotting, and reporting module for XAI-NIDS.

Manages experiment storage under results/<Model>/features_<N>/ and produces
reproducible plots, CSV/JSON summaries, and consolidated comparison reports.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plot generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

logger = logging.getLogger(__name__)

# Canonical directory names for models
MODEL_DIR_NAMES = {
    "logistic_regression": "LogisticRegression",
    "random_forest": "RandomForest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}


class ResultsManager:
    """Manages structured persistence of experiment metrics, plots, and comparison reports."""

    def __init__(self, base_results_dir: str = "results"):
        self.base_dir = Path(base_results_dir)
        self.comparisons_dir = self.base_dir / "comparisons"
        self.logs_dir = self.base_dir / "experiment_logs"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.comparisons_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def get_experiment_dir(self, model_name: str, n_features: int) -> Path:
        """Returns the canonical output directory for a model and feature tier."""
        folder_name = MODEL_DIR_NAMES.get(model_name.lower().strip(), model_name)
        exp_dir = self.base_dir / folder_name / f"features_{n_features}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def save_experiment_results(
        self,
        model_name: str,
        n_features: int,
        metrics: Dict[str, Any],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray],
        class_names: List[str],
        feature_names: List[str],
        feature_importances: Optional[np.ndarray] = None,
    ) -> Path:
        """Saves all evaluation metrics, classification reports, and high-resolution plots."""
        exp_dir = self.get_experiment_dir(model_name, n_features)

        # 1. Save Metrics JSON
        metrics_file = exp_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # 2. Save Classification Report CSV
        if "classification_report" in metrics:
            report_df = pd.DataFrame(metrics["classification_report"]).transpose()
            report_df.to_csv(exp_dir / "classification_report.csv")

        # 3. Generate Confusion Matrix Plot
        self._plot_confusion_matrix(y_true, y_pred, class_names, exp_dir / "confusion_matrix.png")

        # 4. Generate ROC & PR Curves (if probabilities available)
        if y_probs is not None and y_probs.ndim == 2:
            self._plot_roc_curves(y_true, y_probs, class_names, exp_dir / "roc_curves.png")
            self._plot_pr_curves(y_true, y_probs, class_names, exp_dir / "pr_curves.png")

        # 5. Generate Feature Importance Plot
        if feature_importances is not None and len(feature_importances) == len(feature_names):
            self._plot_feature_importance(feature_names, feature_importances, exp_dir / "feature_importance.png")

        logger.info("Saved complete experiment results to: %s", exp_dir)
        return exp_dir

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], save_path: Path):
        """Plots and saves a high-resolution normalized confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

        plt.figure(figsize=(9, 7), dpi=300)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
        )
        plt.title("Normalized Confusion Matrix", fontsize=14, pad=12, fontweight="bold")
        plt.ylabel("True Class", fontsize=11)
        plt.xlabel("Predicted Class", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_roc_curves(self, y_true: np.ndarray, y_probs: np.ndarray, class_names: List[str], save_path: Path):
        """Plots multi-class or binary ROC curves."""
        plt.figure(figsize=(8, 6), dpi=300)
        n_classes = len(class_names)

        if n_classes == 2:
            # Binary ROC
            pos_probs = y_probs[:, 1] if y_probs.shape[1] > 1 else y_probs.ravel()
            fpr, tpr, _ = roc_curve(y_true, pos_probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        else:
            # Multiclass One-vs-Rest ROC
            for i in range(min(n_classes, y_probs.shape[1])):
                y_bin = (y_true == i).astype(int)
                if len(np.unique(y_bin)) > 1:
                    fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=1.5, label=f"{class_names[i]} ({roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=11)
        plt.ylabel("True Positive Rate", fontsize=11)
        plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_pr_curves(self, y_true: np.ndarray, y_probs: np.ndarray, class_names: List[str], save_path: Path):
        """Plots Precision-Recall curves."""
        plt.figure(figsize=(8, 6), dpi=300)
        n_classes = len(class_names)

        if n_classes == 2:
            pos_probs = y_probs[:, 1] if y_probs.shape[1] > 1 else y_probs.ravel()
            precision, recall, _ = precision_recall_curve(y_true, pos_probs)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, color="#1b7837", lw=2, label=f"PR (AUC = {pr_auc:.3f})")
        else:
            for i in range(min(n_classes, y_probs.shape[1])):
                y_bin = (y_true == i).astype(int)
                if len(np.unique(y_bin)) > 1:
                    precision, recall, _ = precision_recall_curve(y_bin, y_probs[:, i])
                    pr_auc = auc(recall, precision)
                    plt.plot(recall, precision, lw=1.5, label=f"{class_names[i]} ({pr_auc:.2f})")

        plt.xlabel("Recall", fontsize=11)
        plt.ylabel("Precision", fontsize=11)
        plt.title("Precision-Recall Curve", fontsize=13, fontweight="bold")
        plt.legend(loc="lower left", fontsize=8)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_feature_importance(self, feature_names: List[str], importances: np.ndarray, save_path: Path):
        """Plots horizontal bar chart of top features."""
        top_k = min(20, len(feature_names))
        indices = np.argsort(importances)[-top_k:]

        plt.figure(figsize=(9, 6), dpi=300)
        y_pos = np.arange(top_k)
        plt.barh(y_pos, importances[indices], color="#2c7fb8", edgecolor="none", alpha=0.85)
        plt.yticks(y_pos, [feature_names[i] for i in indices], fontsize=9)
        plt.xlabel("Relative Importance / Weight Magnitude", fontsize=10)
        plt.title(f"Top {top_k} Features", fontsize=12, fontweight="bold")
        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def generate_consolidated_report(self) -> Path:
        """Scans all experiment subdirectories and compiles a master comparison CSV and markdown report."""
        records = []
        for model_folder in self.base_dir.iterdir():
            if not model_folder.is_dir() or model_folder.name in ["comparisons", "experiment_logs"]:
                continue
            for feat_dir in model_folder.iterdir():
                if not feat_dir.is_dir() or not feat_dir.name.startswith("features_"):
                    continue
                metrics_file = feat_dir / "metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        records.append({
                            "Model": data.get("model_name"),
                            "Feature Tier": data.get("n_features"),
                            "Task": data.get("task"),
                            "Accuracy": data.get("accuracy"),
                            "Macro F1": data.get("f1_macro"),
                            "Weighted F1": data.get("f1_weighted"),
                            "Precision": data.get("precision_macro"),
                            "Recall": data.get("recall_macro"),
                            "Fit Time (s)": round(data.get("fit_time_seconds", 0.0), 2),
                            "Inference Latency (ms/sample)": round((data.get("eval_time_seconds", 0.0) / max(1, data.get("n_test_samples", 1))) * 1000, 4),
                        })
                    except Exception as e:
                        logger.warning("Failed to parse %s: %s", metrics_file, e)

        if not records:
            logger.warning("No experiment records found to build comparison report.")
            return self.comparisons_dir

        df = pd.DataFrame(records).sort_values(by=["Model", "Feature Tier"], ascending=[True, False])
        csv_path = self.comparisons_dir / "model_comparison.csv"
        df.to_csv(csv_path, index=False)

        # Generate Grouped Performance Bar Plot
        self._plot_model_comparison_bars(df, self.comparisons_dir / "performance_summary.png")

        # Generate Markdown Final Report
        report_md_path = self.comparisons_dir / "final_report.md"
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write("# XAI-NIDS Model & Feature Subset Comparison Report\n\n")
            f.write(f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("### Consolidated Performance Matrix\n\n")
            try:
                table_str = df.to_markdown(index=False)
            except Exception:
                headers = list(df.columns)
                lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                for _, row in df.iterrows():
                    lines.append("| " + " | ".join(str(val) for val in row.values) + " |")
                table_str = "\n".join(lines)
            f.write(table_str)
            f.write("\n\n### Key Findings\n")
            f.write("- **Best Performing Model**: " + str(df.loc[df["Weighted F1"].idxmax()]["Model"]) + " (Weighted F1: " + f"{df['Weighted F1'].max():.4f})\n")
            f.write("- **Fastest Model**: " + str(df.loc[df["Inference Latency (ms/sample)"].idxmin()]["Model"]) + " (" + f"{df['Inference Latency (ms/sample)'].min():.4f} ms/sample)\n")
            f.write("- **Feature Degradation**: Performance comparison across 49, 30, 20, and 15 feature subsets.\n")

        logger.info("Generated consolidated report at: %s", report_md_path)
        return report_md_path

    def _plot_model_comparison_bars(self, df: pd.DataFrame, save_path: Path):
        """Generates a comparison bar chart across models and feature subsets."""
        plt.figure(figsize=(11, 6), dpi=300)
        sns.barplot(
            data=df,
            x="Model",
            y="Weighted F1",
            hue="Feature Tier",
            palette="mako",
        )
        plt.title("Weighted F1-Score Across Models & Feature Tiers", fontsize=13, fontweight="bold", pad=12)
        plt.ylabel("Weighted F1-Score", fontsize=11)
        plt.xlabel("Model Architecture", fontsize=11)
        plt.ylim([0.0, 1.05])
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.legend(title="Feature Tier", loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
