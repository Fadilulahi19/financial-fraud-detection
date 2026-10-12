"""
evaluate.py
===========
Metrics, threshold analysis, and comparison plots for the Moniepoint
fraud detection feasibility study — the programmatic version of Phases
7-8 of the analysis (business-KPI evaluation and threshold optimization).

Accuracy is deliberately not used as a headline metric here given class
imbalance — see FRAUD CAPTURE RATE (recall) and FALSE POSITIVE RATE
below, which are the business- and customer-impact KPIs this study is
built around.

Usage:
    python evaluate.py [path/to/data.csv] [--output-dir ../reports/images]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score, f1_score,
)

from train import run_training_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LEGIT_COLOR = "#4C72B0"
FRAUD_COLOR = "#C44E52"
DEFAULT_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


# ---------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------
def compute_metrics_at_threshold(
    proba: np.ndarray, y_true: np.ndarray, amounts: np.ndarray, threshold: float
) -> dict:
    """
    Full metric set at a single threshold, including the two KPIs this
    study prioritizes over raw accuracy:
      - Fraud Capture Rate (business KPI) = recall
      - False Positive Rate (customer-impact KPI)
    plus the naira value of fraud captured and legitimate value blocked.
    """
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = Fraud Capture Rate
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    fraud_total = amounts[y_true == 1].sum()
    fraud_captured = amounts[(y_true == 1) & (pred == 1)].sum()
    legit_blocked = amounts[(y_true == 0) & (pred == 1)].sum()

    return {
        "threshold": threshold, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": precision, "recall": recall, "fraud_capture_rate": recall,
        "f1": f1, "false_positive_rate": fpr,
        "fraud_value_total": float(fraud_total), "fraud_value_captured": float(fraud_captured),
        "fraud_pct_captured": float(fraud_captured / fraud_total * 100) if fraud_total > 0 else 0.0,
        "legit_value_blocked": float(legit_blocked),
    }


def evaluate_at_recall_targets(
    proba: np.ndarray, y_true: np.ndarray, amounts: np.ndarray,
    targets: list[float] = (0.5, 0.8, 1.0),
) -> pd.DataFrame:
    """
    For each target recall level, find the threshold that achieves at
    least that recall at the highest available precision — this is how
    Phase 7's three operating points (50%/80%/100% recall) were chosen.
    """
    prec, rec, thresh = precision_recall_curve(y_true, proba)
    rows = []
    for target in targets:
        idx = np.where(rec >= target)[0]
        if len(idx) == 0:
            continue
        best_i = idx[np.argmax(prec[idx])]
        th = thresh[best_i] if best_i < len(thresh) else 0.0
        rows.append({"target_recall": target, **compute_metrics_at_threshold(proba, y_true, amounts, th)})
    return pd.DataFrame(rows)


def threshold_sweep(
    proba: np.ndarray, y_true: np.ndarray, amounts: np.ndarray,
    thresholds: list[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """
    Full sweep across explicit thresholds rather than assuming 0.5 —
    the central practice from Phase 8. Always run this before picking
    an operating point; a model's probabilities may never reach 0.5
    (true for Logistic Regression in this study), which silently makes
    the conventional default predict zero positives.
    """
    rows = [compute_metrics_at_threshold(proba, y_true, amounts, t) for t in thresholds]
    return pd.DataFrame(rows)


def summary_metrics(proba: np.ndarray, y_true: np.ndarray) -> dict:
    """Threshold-independent metrics: PR-AUC and ROC-AUC."""
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "max_proba": float(proba.max()),
    }


def recommend_threshold(sweep_df: pd.DataFrame, min_recall: float = 0.5) -> dict:
    """
    Picks the threshold with the best fraud-value-captured-to-legit-value-blocked
    ratio among thresholds meeting a minimum recall floor. This reproduces
    the reasoning behind Phase 8's threshold=0.30 recommendation — it is
    a suggestion to sanity-check against business risk appetite, not a
    substitute for that judgment call.
    """
    candidates = sweep_df[sweep_df["recall"] >= min_recall].copy()
    if candidates.empty:
        candidates = sweep_df.copy()
    candidates["value_ratio"] = candidates["fraud_value_captured"] / candidates["legit_value_blocked"].replace(0, np.nan)
    best = candidates.sort_values("value_ratio", ascending=False).iloc[0]
    return best.to_dict()


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def plot_pr_curves(y_true: np.ndarray, proba_dict: dict[str, np.ndarray], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [LEGIT_COLOR, FRAUD_COLOR]
    for (name, proba), color in zip(proba_dict.items(), colors):
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(rec, prec, label=f"{name} (PR-AUC={ap:.3f})", color=color, linewidth=2)
    base_rate = y_true.mean()
    ax.axhline(base_rate, color="gray", linestyle="--", linewidth=1, label=f"Random baseline ({base_rate*100:.0f}%)")
    ax.axhline(0.80, color="green", linestyle=":", linewidth=1.5, label="Target precision (80%)")
    ax.axvline(0.80, color="green", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves"); ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    logger.info(f"Saved {save_path}")


def plot_fcr_vs_fpr(y_true: np.ndarray, proba_dict: dict[str, np.ndarray], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [LEGIT_COLOR, FRAUD_COLOR]
    for (name, proba), color in zip(proba_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(tpr, fpr, label=name, color=color, linewidth=2)
    ax.set_xlabel("Recall (Fraud Capture Rate)"); ax.set_ylabel("False Positive Rate")
    ax.set_title("Fraud Capture Rate vs. Customer False Positive Rate")
    ax.legend(); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    logger.info(f"Saved {save_path}")


def plot_threshold_tradeoff(sweep_dict: dict[str, pd.DataFrame], save_path: Path) -> None:
    fig, axes = plt.subplots(1, len(sweep_dict), figsize=(6.5 * len(sweep_dict), 5))
    if len(sweep_dict) == 1:
        axes = [axes]
    for ax, (name, sweep_df) in zip(axes, sweep_dict.items()):
        ax.plot(sweep_df["threshold"], sweep_df["recall"] * 100, marker="o", color=LEGIT_COLOR, label="Fraud Capture Rate")
        ax.plot(sweep_df["threshold"], sweep_df["false_positive_rate"] * 100, marker="s", color=FRAUD_COLOR, label="False Positive Rate")
        ax.set_xlabel("Threshold"); ax.set_ylabel("%"); ax.set_title(name)
        ax.legend(fontsize=8); ax.set_ylim(0, 105)
    plt.suptitle("Threshold Trade-off: Fraud Capture Rate vs. False Positive Rate")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    logger.info(f"Saved {save_path}")


def plot_confusion_matrix(y_true: np.ndarray, proba: np.ndarray, threshold: float,
                           model_name: str, save_path: Path) -> None:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Legit", "Pred Fraud"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Legit", "Actual Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"{model_name} — Confusion Matrix @ threshold={threshold:.2f}")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    logger.info(f"Saved {save_path}")


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def run_evaluation(data_path: str, output_dir: str = "../reports/images",
                    recommended_threshold: float = 0.30) -> dict:
    results = run_training_pipeline(data_path)
    y_test = results["y_test"].values
    amt_test = results["amt_test"].values
    proba_dict = {"Logistic Regression": results["logreg_proba"], "XGBoost": results["xgb_proba"]}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {name: summary_metrics(proba, y_test) for name, proba in proba_dict.items()}
    for name, m in summary.items():
        logger.info(f"{name}: PR-AUC={m['pr_auc']:.3f}  ROC-AUC={m['roc_auc']:.3f}")

    sweeps = {name: threshold_sweep(proba, y_test, amt_test) for name, proba in proba_dict.items()}
    recall_targets = {
        name: evaluate_at_recall_targets(proba, y_test, amt_test)
        for name, proba in proba_dict.items()
    }

    plot_pr_curves(y_test, proba_dict, out / "eval_pr_curves.png")
    plot_fcr_vs_fpr(y_test, proba_dict, out / "eval_fcr_vs_fpr.png")
    plot_threshold_tradeoff(sweeps, out / "eval_threshold_tradeoff.png")
    plot_confusion_matrix(y_test, proba_dict["XGBoost"], recommended_threshold, "XGBoost",
                           out / "eval_confusion_matrix.png")

    recommendation = recommend_threshold(sweeps["XGBoost"])

    return {
        "summary_metrics": summary,
        "threshold_sweeps": sweeps,
        "recall_target_tables": recall_targets,
        "recommended_threshold_analysis": recommendation,
        "models": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fraud detection models")
    parser.add_argument("data_path", nargs="?", default="../data/raw/balanced_fraud_dataset.csv")
    parser.add_argument("--output-dir", default="../reports/images")
    parser.add_argument("--threshold", type=float, default=0.30,
                         help="Threshold to use for the confusion matrix plot")
    args = parser.parse_args()

    results = run_evaluation(args.data_path, args.output_dir, args.threshold)

    print("\n=== Summary Metrics ===")
    for name, m in results["summary_metrics"].items():
        print(f"{name}: PR-AUC={m['pr_auc']:.3f}  ROC-AUC={m['roc_auc']:.3f}  max_proba={m['max_proba']:.3f}")

    print(f"\n=== XGBoost — Metrics at Recall Targets ===")
    cols = ["target_recall", "threshold", "precision", "recall", "false_positive_rate", "fraud_pct_captured"]
    print(results["recall_target_tables"]["XGBoost"][cols].round(3).to_string(index=False))

    print(f"\n=== Recommended Operating Point (XGBoost) ===")
    r = results["recommended_threshold_analysis"]
    print(f"Threshold: {r['threshold']:.2f}  Precision: {r['precision']:.3f}  "
          f"Recall: {r['recall']:.3f}  FPR: {r['false_positive_rate']:.3f}  "
          f"Fraud value captured: {r['fraud_pct_captured']:.1f}%")


if __name__ == "__main__":
    main()
