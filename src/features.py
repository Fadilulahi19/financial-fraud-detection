"""
features.py
============
Fraud-specific feature engineering for the Moniepoint fraud detection
feasibility study: velocity, customer-deviation, and device/geo/merchant
risk features.

This module assumes the input has already been through preprocessing.py
(dead/redundant columns dropped, time_since_last_transaction sign bug
fixed, missing values handled). It does not re-clean the data — see
preprocessing.py for that.

Every feature here maps to a specific finding from Phase 4/5 of the
analysis (see reports/fraud_case_study.md). In particular:
  - new_device_transaction carries the overwhelming majority of model
    signal (Phase 4: 1.15x lift, 100% recall; Phase 6: 73% of XGBoost
    feature importance).
  - The four "risk score" columns (geo_anomaly_score, merchant_fraud_rate,
    channel_risk_score) were explicitly checked for target leakage in
    Phase 2 — see check_leakage() below to reproduce that check on new
    data before trusting these features in a new context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Final feature set used for modeling (Phase 5/6). Order matters for
# consistency with saved model coefficients/importances.
FEATURE_COLS = [
    "transaction_hour", "is_weekend", "is_night_transaction",
    "amount_deviation", "transaction_velocity",
    "customer_avg_amount", "customer_amount_deviation",
    "device_seen_count", "device_shared_flag", "ip_shared_flag",
    "geo_anomaly_score", "merchant_fraud_rate", "channel_risk_score",
    "new_device_transaction",
]

TARGET_COL = "is_fraud"
AMOUNT_COL = "amount_ngn"

# Risk-derived columns whose relationship to the label must be re-verified
# on any new dataset before being trusted (see check_leakage()).
RISK_DERIVED_COLS = {
    "merchant_fraud_rate": "merchant_category",
    "channel_risk_score": "payment_channel",
    "geo_anomaly_score": None,  # not category-derived; row-level by construction
}


@dataclass
class LeakageCheckResult:
    column: str
    group_col: str | None
    correlation_with_actual_group_rate: float | None
    is_constant_within_group: bool | None
    verdict: str


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """transaction_hour (passthrough) and is_night_transaction (derived fresh).

    is_night_transaction is rebuilt here rather than reused from the raw
    is_night_txn column, which Phase 2 found constant (all-zero) in the
    sample and preprocessing.py drops for that reason.
    """
    df = df.copy()
    df["transaction_hour"] = df["txn_hour"]
    df["is_night_transaction"] = df["transaction_hour"].apply(lambda h: 1 if (h >= 22 or h < 6) else 0)
    return df


def build_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """amount_deviation (population-level) and customer_amount_deviation (personal-level).

    customer_amount_deviation is a z-score of this transaction against the
    customer's OWN historical average/std — distinct from amount_deviation,
    which reflects deviation from the wider population. First-transaction
    customers (no std to divide by) default to 0 rather than NaN or inf.
    """
    df = df.copy()
    df["amount_deviation"] = df["spending_deviation_score"]
    df["customer_avg_amount"] = df["user_avg_txn_amt"]
    df["customer_amount_deviation"] = (
        (df[AMOUNT_COL] - df["user_avg_txn_amt"]) / df["user_std_txn_amt"].replace(0, np.nan)
    ).fillna(0)
    return df


def build_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """transaction_velocity from txn_count_last_24h.

    Phase 5 found txn_count_last_1h identical to txn_count_last_24h for
    every row in this sample; preprocessing.py drops the former, so only
    the 24h window is used here. If a future dataset has genuinely
    distinct 1h/24h counts, both could be reintroduced as separate
    features — re-check with the same equality test before assuming that.
    """
    df = df.copy()
    df["transaction_velocity"] = df["txn_count_last_24h"]
    return df


def build_device_features(df: pd.DataFrame) -> pd.DataFrame:
    """device_shared_flag, ip_shared_flag, device_seen_count, new_device_transaction.

    Note from Phase 5: ip_shared_flag was found near-constant (99.8% = 1)
    in the sample — kept for completeness but expect it to contribute
    little. new_device_transaction is the single strongest feature found
    anywhere in this analysis; do not drop it without re-running the
    typology quantification in Phase 4's logic.
    """
    df = df.copy()
    df["device_shared_flag"] = df["is_device_shared"]
    df["ip_shared_flag"] = df["is_ip_shared"]
    df["new_device_transaction"] = df["new_device_transaction"].astype(int)
    return df


def build_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """geo_anomaly_score, merchant_fraud_rate, channel_risk_score — passthrough.

    These are used as-is from the source data. Call check_leakage() before
    trusting them on a new dataset — see that function's docstring.
    """
    return df.copy()


def check_leakage(df: pd.DataFrame, target_col: str = TARGET_COL) -> list[LeakageCheckResult]:
    """
    Reproduce the Phase 2 leakage check: for each risk-derived column,
    verify it is NOT simply the target's own group-level rate restated.

    A column is flagged as likely leakage if its correlation with the
    dataset's own actual group fraud rate is suspiciously high (>0.95)
    AND it is constant within each group — i.e., it looks like it was
    computed directly from this dataset's own labels rather than an
    independent/historical source.

    Run this on any NEW dataset before reusing these features — the
    Phase 2 result ("no leakage") is specific to the sample it was
    computed on and is not guaranteed to hold elsewhere.
    """
    results = []
    for col, group_col in RISK_DERIVED_COLS.items():
        if col not in df.columns:
            continue

        if group_col is None or group_col not in df.columns:
            results.append(LeakageCheckResult(
                column=col, group_col=group_col,
                correlation_with_actual_group_rate=None,
                is_constant_within_group=None,
                verdict="Not category-derived — no group-level leakage test applicable. Verify row-level construction manually.",
            ))
            continue

        actual_rate = df.groupby(group_col)[target_col].transform("mean")
        corr = float(np.corrcoef(df[col], actual_rate)[0, 1])
        is_constant = bool((df.groupby(group_col)[col].nunique() == 1).all())

        if abs(corr) > 0.95 and is_constant:
            verdict = "LIKELY LEAKAGE — matches this dataset's own group fraud rate almost exactly. Do not use without investigation."
        elif is_constant:
            verdict = "No leakage — fixed per-category value, uncorrelated with actual outcomes here. Likely a static assigned weight."
        else:
            verdict = "No leakage — varies within category, consistent with a legitimate rolling/historical estimate."

        results.append(LeakageCheckResult(
            column=col, group_col=group_col,
            correlation_with_actual_group_rate=round(corr, 3),
            is_constant_within_group=is_constant,
            verdict=verdict,
        ))

    return results


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline and return the input
    dataframe with all engineered columns added (originals untouched)."""
    df = build_time_features(df)
    df = build_amount_features(df)
    df = build_velocity_features(df)
    df = build_device_features(df)
    df = build_risk_features(df)
    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Convenience wrapper: engineer features, then return (X, y, amounts)
    ready for train_test_split.

        X, y, amounts = get_feature_matrix(clean_df)
        X_train, X_test, y_train, y_test, amt_train, amt_test = train_test_split(
            X, y, amounts, test_size=0.25, stratify=y, random_state=42)
    """
    df = engineer_features(df)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Feature engineering did not produce expected columns: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int)
    amounts = df[AMOUNT_COL]
    return X, y, amounts


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from preprocessing import load_raw_data, preprocess

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/balanced_fraud_dataset.csv"
    raw = load_raw_data(path)
    clean, _ = preprocess(raw)

    print("\n=== Leakage Check ===")
    for r in check_leakage(clean):
        print(f"{r.column} (grouped by {r.group_col}): corr={r.correlation_with_actual_group_rate}, "
              f"constant_within_group={r.is_constant_within_group}\n  -> {r.verdict}")

    X, y, amounts = get_feature_matrix(clean)
    print(f"\n=== Feature Matrix ===")
    print(f"Shape: {X.shape}")
    print(f"Fraud rate: {y.mean():.4f}")
    print(f"\n{X.describe().T}")
