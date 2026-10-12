"""
train.py
========
Trains the two models compared in the Moniepoint fraud detection
feasibility study:
  - Logistic Regression (baseline, recommended for the pilot)
  - XGBoost (comparison model)

Both are trained on the same train/test split (stratified, 25% test,
random_state=42 — fixed for reproducibility with the rest of this
project's analysis) so their results are directly comparable.

Usage:
    python train.py [path/to/data.csv] [--output-dir models/]

Saves:
    models/logreg_model.pkl      — trained LogisticRegression
    models/logreg_scaler.pkl     — StandardScaler fit on the training split
    models/xgb_model.json        — trained XGBoost (native format)
    models/feature_cols.json     — exact feature order both models expect
    models/training_metadata.json — split sizes, fraud rates, headline metrics
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from preprocessing import load_raw_data, preprocess
from features import FEATURE_COLS, get_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.25


def split_data(X: pd.DataFrame, y: pd.Series, amounts: pd.Series):
    """Stratified split — fixed random_state to stay reproducible with
    every other phase of this analysis (Phases 6-9 all use the same split)."""
    X_train, X_test, y_train, y_test, amt_train, amt_test = train_test_split(
        X, y, amounts, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return (X_train.reset_index(drop=True), X_test.reset_index(drop=True),
            y_train.reset_index(drop=True), y_test.reset_index(drop=True),
            amt_train.reset_index(drop=True), amt_test.reset_index(drop=True))


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Baseline model. Requires scaling (unlike XGBoost) since it's a
    linear model sensitive to feature magnitude — the scaler is returned
    alongside the model and must travel with it for any future scoring.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    """Comparison model. No scaling needed — tree splits are scale-invariant."""
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric="aucpr", random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_quick(model_name: str, proba: np.ndarray, y_test: pd.Series) -> dict:
    """Threshold-independent metrics only — full threshold sweep and
    business-KPI evaluation live in evaluate.py, not here."""
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    max_proba = float(proba.max())
    logger.info(f"{model_name}: PR-AUC={pr_auc:.3f}  ROC-AUC={roc_auc:.3f}  max_proba={max_proba:.3f}")
    if max_proba < 0.5:
        logger.warning(
            f"{model_name}'s max predicted probability ({max_proba:.3f}) never reaches 0.5 — "
            "the conventional 0.5 decision threshold will silently predict zero positives. "
            "Use evaluate.py's threshold sweep, not a hardcoded 0.5 cutoff, to choose an operating point."
        )
    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "max_proba": max_proba}


def get_logreg_coefficients(model: LogisticRegression) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": FEATURE_COLS,
        "coefficient": model.coef_[0],
    }).sort_values("coefficient", ascending=False)


def get_xgb_importances(model: xgb.XGBClassifier) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)


def save_artifacts(output_dir: Path, logreg, scaler, xgb_model, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(logreg, output_dir / "logreg_model.pkl")
    joblib.dump(scaler, output_dir / "logreg_scaler.pkl")
    xgb_model.save_model(str(output_dir / "xgb_model.json"))

    with open(output_dir / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved model artifacts to {output_dir}/")


def run_training_pipeline(data_path: str, output_dir: str = "models") -> dict:
    """Full pipeline: load -> preprocess -> engineer -> split -> train -> evaluate -> save."""
    raw = load_raw_data(data_path)
    clean, prep_report = preprocess(raw)
    X, y, amounts = get_feature_matrix(clean)

    X_train, X_test, y_train, y_test, amt_train, amt_test = split_data(X, y, amounts)
    logger.info(f"Train: {X_train.shape}  Test: {X_test.shape}")
    logger.info(f"Train fraud rate: {y_train.mean():.3f}  Test fraud rate: {y_test.mean():.3f}")

    logreg, scaler = train_logistic_regression(X_train, y_train)
    logreg_proba = logreg.predict_proba(scaler.transform(X_test))[:, 1]
    logreg_metrics = evaluate_quick("Logistic Regression", logreg_proba, y_test)

    xgb_model = train_xgboost(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate_quick("XGBoost", xgb_proba, y_test)

    logger.info("\nLogistic Regression coefficients (top 5):")
    logger.info("\n" + get_logreg_coefficients(logreg).head().to_string(index=False))
    logger.info("\nXGBoost feature importances (top 5):")
    logger.info("\n" + get_xgb_importances(xgb_model).head().to_string(index=False))

    metadata = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_fraud_rate": round(float(y_train.mean()), 4),
        "test_fraud_rate": round(float(y_test.mean()), 4),
        "feature_cols": FEATURE_COLS,
        "logistic_regression": logreg_metrics,
        "xgboost": xgb_metrics,
        "pr_auc_relative_improvement_xgb_vs_logreg": round(
            (xgb_metrics["pr_auc"] - logreg_metrics["pr_auc"]) / logreg_metrics["pr_auc"], 4
        ),
    }

    save_artifacts(Path(output_dir), logreg, scaler, xgb_model, metadata)

    return {
        "logreg": logreg, "scaler": scaler, "xgb_model": xgb_model,
        "X_test": X_test, "y_test": y_test, "amt_test": amt_test,
        "logreg_proba": logreg_proba, "xgb_proba": xgb_proba,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument("data_path", nargs="?", default="../data/raw/balanced_fraud_dataset.csv",
                         help="Path to the raw transactions CSV")
    parser.add_argument("--output-dir", default="../models", help="Where to save trained model artifacts")
    args = parser.parse_args()

    results = run_training_pipeline(args.data_path, args.output_dir)

    print("\n=== Training Summary ===")
    print(f"Logistic Regression — PR-AUC: {results['metadata']['logistic_regression']['pr_auc']:.3f}")
    print(f"XGBoost             — PR-AUC: {results['metadata']['xgboost']['pr_auc']:.3f}")
    improvement = results['metadata']['pr_auc_relative_improvement_xgb_vs_logreg']
    print(f"XGBoost relative improvement over baseline: {improvement*100:+.1f}%")
    if abs(improvement) < 0.05:
        print("-> Not material. See Phase 6 of the analysis: Logistic Regression is recommended for the pilot.")


if __name__ == "__main__":
    main()
