"""End-to-End Execution Script for Logistic Regression Open-Set Experiment (Phase 2).

Executes:
1. Stage 1: Closed-Set 6-class Training and Evaluation (48 features, D_train_known=113,288).
2. Stage 2: MSP Open-Set Evaluation (D_thresh_cal=12,139, D_test=82,332).
3. Stage 3: Energy-Based Open-Set Evaluation with exact linear decision logits (T=1.0).
4. Stage 4: OpenMax EVT-Based Open-Set Evaluation (alpha=3, D_evt_fit=12,138).
5. Stage 5: Verification of all generated artifacts.
"""

import json
import logging
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
repo_root = Path("D:/Network IDS").resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.openset.train_classifier import train_openset_classifier
from src.openset.evaluate_msp import run_msp_evaluation
from src.openset.evaluate_energy import run_energy_evaluation
from src.openset.evaluate_openmax import run_openmax_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("LR_Experiment_Runner")


def run_full_lr_pipeline():
    logger.info("=================================================================")
    logger.info("STARTING COMPLETE LOGISTIC REGRESSION PHASE 2 EXPERIMENT")
    logger.info("=================================================================")
    t_start = time.perf_counter()

    base_dir = Path("results/openset/LogisticRegression/features_48")

    # -------------------------------------------------------------
    # STAGE 1: Closed-Set Training & Evaluation
    # -------------------------------------------------------------
    logger.info("\n--- [STAGE 1/4] Closed-Set 6-Class Training & Evaluation ---")
    t0 = time.perf_counter()
    train_res = train_openset_classifier(
        model_name="logistic_regression",
        n_features=48,
        random_state=42,
    )
    t_stage1 = time.perf_counter() - t0
    logger.info(
        "Stage 1 Complete in %.2fs. Test Acc: %.4f, Macro F1: %.4f, Weighted F1: %.4f",
        t_stage1,
        train_res["metrics"]["accuracy"],
        train_res["metrics"]["macro_f1"],
        train_res["metrics"]["weighted_f1"],
    )

    # -------------------------------------------------------------
    # STAGE 2: Open-Set MSP Evaluation
    # -------------------------------------------------------------
    logger.info("\n--- [STAGE 2/4] MSP Open-Set Detection Evaluation ---")
    t0 = time.perf_counter()
    msp_res = run_msp_evaluation(
        classifier_dir=str(base_dir),
        random_state=42,
    )
    t_stage2 = time.perf_counter() - t0
    logger.info(
        "Stage 2 Complete in %.2fs. MSP AUROC: %.4f, AUPR-Out: %.4f, OSCR: %.4f",
        t_stage2,
        msp_res["threshold_independent_metrics"]["auroc"],
        msp_res["threshold_independent_metrics"]["aupr_out"],
        msp_res["threshold_independent_metrics"]["oscr_auc"],
    )

    # -------------------------------------------------------------
    # STAGE 3: Open-Set Energy Evaluation
    # -------------------------------------------------------------
    logger.info("\n--- [STAGE 3/4] Energy-Based Open-Set Detection Evaluation ---")
    t0 = time.perf_counter()
    energy_res = run_energy_evaluation(
        classifier_dir=str(base_dir),
        temperature=1.0,
        random_state=42,
    )
    t_stage3 = time.perf_counter() - t0
    logger.info(
        "Stage 3 Complete in %.2fs. Energy AUROC: %.4f, AUPR-Out: %.4f, OSCR: %.4f",
        t_stage3,
        energy_res["threshold_independent_metrics"]["auroc"],
        energy_res["threshold_independent_metrics"]["aupr_out"],
        energy_res["threshold_independent_metrics"]["oscr_auc"],
    )

    # -------------------------------------------------------------
    # STAGE 4: Open-Set OpenMax Evaluation
    # -------------------------------------------------------------
    logger.info("\n--- [STAGE 4/4] OpenMax EVT-Based Open-Set Detection Evaluation ---")
    t0 = time.perf_counter()
    openmax_res = run_openmax_evaluation(
        classifier_dir=str(base_dir),
        alpha=3,
        eta_ratio=0.2,
        min_eta=5,
        max_eta=20,
        random_state=42,
    )
    t_stage4 = time.perf_counter() - t0
    logger.info(
        "Stage 4 Complete in %.2fs. OpenMax AUROC: %.4f, AUPR-Out: %.4f, OSCR: %.4f",
        t_stage4,
        openmax_res["threshold_independent_metrics"]["auroc"],
        openmax_res["threshold_independent_metrics"]["aupr_out"],
        openmax_res["threshold_independent_metrics"]["oscr_auc"],
    )

    # -------------------------------------------------------------
    # STAGE 5: Artifact Verification
    # -------------------------------------------------------------
    logger.info("\n--- [STAGE 5/5] Comprehensive Artifact Verification ---")
    expected_artifacts = [
        # Closed-set
        base_dir / "model.pkl",
        base_dir / "preprocessor.pkl",
        base_dir / "run_config.json",
        base_dir / "feature_names.json",
        base_dir / "class_mapping.json",
        base_dir / "metrics_closed_set.json",
        base_dir / "classification_report.csv",
        base_dir / "confusion_matrix_6x6.csv",
        base_dir / "confusion_matrix_6x6.png",
        base_dir / "roc_curves_closed_set.png",
        base_dir / "pr_curves_closed_set.png",
        base_dir / "feature_importance.png",
        # MSP
        base_dir / "MSP" / "thresholds.json",
        base_dir / "MSP" / "experiment_config.json",
        base_dir / "MSP" / "metrics_openset_msp.json",
        base_dir / "MSP" / "metrics_openset_msp.csv",
        base_dir / "MSP" / "confusion_matrix_7x7_fpr05.csv",
        base_dir / "MSP" / "confusion_matrix_7x7_fpr05.png",
        base_dir / "MSP" / "confusion_matrix_7x7_cov95.csv",
        base_dir / "MSP" / "confusion_matrix_7x7_cov95.png",
        base_dir / "MSP" / "per_category_unknown_detection.csv",
        base_dir / "MSP" / "roc_pr_curves.png",
        base_dir / "MSP" / "oscr_curve.png",
        # Energy
        base_dir / "Energy" / "thresholds.json",
        base_dir / "Energy" / "experiment_config.json",
        base_dir / "Energy" / "metrics_openset_energy.json",
        base_dir / "Energy" / "metrics_openset_energy.csv",
        base_dir / "Energy" / "confusion_matrix_7x7_fpr05.csv",
        base_dir / "Energy" / "confusion_matrix_7x7_fpr05.png",
        base_dir / "Energy" / "confusion_matrix_7x7_cov95.csv",
        base_dir / "Energy" / "confusion_matrix_7x7_cov95.png",
        base_dir / "Energy" / "per_category_unknown_detection.csv",
        base_dir / "Energy" / "roc_pr_curves.png",
        base_dir / "Energy" / "oscr_curve.png",
        # OpenMax
        base_dir / "OpenMax" / "thresholds.json",
        base_dir / "OpenMax" / "experiment_config.json",
        base_dir / "OpenMax" / "mav_and_weibull_params.json",
        base_dir / "OpenMax" / "metrics_openset_openmax.json",
        base_dir / "OpenMax" / "metrics_openset_openmax.csv",
        base_dir / "OpenMax" / "confusion_matrix_7x7_fpr05.csv",
        base_dir / "OpenMax" / "confusion_matrix_7x7_fpr05.png",
        base_dir / "OpenMax" / "confusion_matrix_7x7_cov95.csv",
        base_dir / "OpenMax" / "confusion_matrix_7x7_cov95.png",
        base_dir / "OpenMax" / "per_category_unknown_detection.csv",
        base_dir / "OpenMax" / "roc_pr_curves.png",
        base_dir / "OpenMax" / "oscr_curve.png",
    ]

    missing = []
    for art in expected_artifacts:
        if not art.exists():
            missing.append(str(art))

    if missing:
        logger.error("Verification failed! Missing %d artifacts:\n%s", len(missing), "\n".join(missing))
        raise FileNotFoundError(f"Missing {len(missing)} required artifacts after pipeline completion.")

    logger.info("All %d expected artifacts successfully verified on disk!", len(expected_artifacts))

    total_time = time.perf_counter() - t_start
    logger.info("=================================================================")
    logger.info("LOGISTIC REGRESSION EXPERIMENT COMPLETED SUCCESSFULLY IN %.2fs", total_time)
    logger.info("=================================================================")


if __name__ == "__main__":
    run_full_lr_pipeline()
