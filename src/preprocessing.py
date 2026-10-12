"""
preprocessing.py
=================
Cleaning, missing-value handling, and categorical encoding for the
Moniepoint fraud detection feasibility study.

This module handles general data cleaning only. Fraud-specific derived
features (velocity, customer deviation, etc.) live in features.py —
keeping the two separate means preprocessing can be reused for any
downstream task on this dataset, not just this project's feature set.

Every fix in here traces back to a specific finding from Phase 2 (Data
Validation) of the analysis — see reports/data_quality_report.md for
the full evidence behind each one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Columns confirmed broken or redundant during Phase 2/5 validation.
# is_night_txn: constant (all-zero) in the sample — dead signal.
# geospatial_velocity_anomaly: constant (all-False) in the sample — dead signal.
# txn_count_last_1h: identical to txn_count_last_24h for every row — redundant.
KNOWN_DEAD_COLUMNS = ["is_night_txn", "geospatial_velocity_anomaly"]
KNOWN_REDUNDANT_COLUMNS = ["txn_count_last_1h"]

# Categorical columns worth encoding for models that need numeric input
# (tree models can often take raw categories; linear models cannot).
CATEGORICAL_COLUMNS = [
    "transaction_type", "merchant_category", "location", "device_used",
    "payment_channel", "sender_persona", "ip_geo_region",
]


@dataclass
class PreprocessingReport:
    """Summary of what preprocess() actually did, for logging/audit."""
    rows_in: int = 0
    rows_out: int = 0
    dead_columns_dropped: list[str] = field(default_factory=list)
    redundant_columns_dropped: list[str] = field(default_factory=list)
    negative_time_deltas_fixed: int = 0
    missing_values_filled: dict[str, int] = field(default_factory=dict)


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV with timestamp parsing."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns from {path}")
    return df


def drop_dead_and_redundant_columns(df: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
    """
    Remove columns confirmed to carry no information in this dataset.

    A column is only dropped here if it was explicitly verified constant
    or exactly duplicated during validation — this is not a generic
    variance filter, since a legitimately rare-but-real signal (e.g. a
    flag that's 99% one value but occasionally meaningful) should not be
    silently discarded the same way.
    """
    df = df.copy()
    for col in KNOWN_DEAD_COLUMNS:
        if col in df.columns:
            n_unique = df[col].nunique()
            if n_unique <= 1:
                df = df.drop(columns=[col])
                report.dead_columns_dropped.append(col)
                logger.info(f"Dropped '{col}': confirmed constant ({n_unique} unique value)")
            else:
                logger.warning(f"'{col}' is no longer constant in this data (n_unique={n_unique}) — kept, re-check assumption")

    for col in KNOWN_REDUNDANT_COLUMNS:
        if col in df.columns and "txn_count_last_24h" in df.columns:
            if (df[col] == df["txn_count_last_24h"]).all():
                df = df.drop(columns=[col])
                report.redundant_columns_dropped.append(col)
                logger.info(f"Dropped '{col}': identical to txn_count_last_24h for every row")
            else:
                logger.warning(f"'{col}' no longer matches txn_count_last_24h — kept, re-check assumption")

    return df


def fix_time_since_last_transaction(df: pd.DataFrame, report: PreprocessingReport,
                                     strategy: str = "abs") -> pd.DataFrame:
    """
    Fix the sign-convention bug found in Phase 2: 43.5% of
    time_since_last_transaction values were negative, which is
    impossible for elapsed time.

    strategy:
        "abs"    — take the absolute value (default; assumes a simple
                   sign-flip bug, e.g. computed as previous - current
                   instead of current - previous)
        "drop"   — drop the column entirely (safest if the sign error's
                   root cause is not confirmed)
        "null"   — set negative values to NaN, leave everything else
    """
    df = df.copy()
    col = "time_since_last_transaction"
    if col not in df.columns:
        return df

    n_negative = (df[col] < 0).sum()
    report.negative_time_deltas_fixed = int(n_negative)

    if n_negative == 0:
        return df

    if strategy == "abs":
        df.loc[df[col] < 0, col] = df.loc[df[col] < 0, col].abs()
        logger.info(f"Fixed {n_negative:,} negative '{col}' values via abs() — confirm this matches the true root cause before production use")
    elif strategy == "drop":
        df = df.drop(columns=[col])
        logger.info(f"Dropped '{col}' entirely: {n_negative:,} negative values, root cause unconfirmed")
    elif strategy == "null":
        df.loc[df[col] < 0, col] = np.nan
        logger.info(f"Nulled {n_negative:,} negative '{col}' values")
    else:
        raise ValueError(f"Unknown strategy '{strategy}' — expected 'abs', 'drop', or 'null'")

    return df


def handle_missing_values(df: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
    """
    Fill or flag missing values based on what Phase 2 established about
    *why* each column has nulls — not a blanket fillna(0).

    - fraud_type: null is expected and meaningful (non-fraud transaction).
      Filled with 'not_fraud' so downstream encoders don't choke on NaN.
    - time_since_last_transaction: null means a customer's first observed
      transaction (no prior transaction to compare to). Filled with 0
      with a companion flag, since "first transaction" is itself
      informative and shouldn't be silently imputed away.
    """
    df = df.copy()
    before = df.isnull().sum()

    if "fraud_type" in df.columns:
        df["fraud_type"] = df["fraud_type"].fillna("not_fraud")

    if "time_since_last_transaction" in df.columns:
        df["is_first_transaction"] = df["time_since_last_transaction"].isnull().astype(int)
        df["time_since_last_transaction"] = df["time_since_last_transaction"].fillna(0)

    after = df.isnull().sum()
    filled = (before - after)[(before - after) > 0]
    report.missing_values_filled = filled.to_dict()

    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    if len(remaining) > 0:
        logger.warning(f"Missing values remain after handling: {remaining.to_dict()}")

    return df


def encode_categoricals(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    encoder: OneHotEncoder | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """
    One-hot encode categorical columns for models that require numeric
    input (e.g. Logistic Regression). Tree models like XGBoost can often
    take raw/label-encoded categories directly, so this step is optional
    depending on the downstream model.

    Fit the encoder on the TRAINING split only, then reuse it (fit=False)
    on the test split — fitting on the full dataset before splitting
    leaks test-set category information into training.

    Example:
        X_train_enc, enc = encode_categoricals(X_train, fit=True)
        X_test_enc, _    = encode_categoricals(X_test, encoder=enc, fit=False)
    """
    columns = columns or [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    if not columns:
        return df, encoder

    df = df.copy()

    if fit or encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(df[columns])
    else:
        encoded = encoder.transform(df[columns])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(columns),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=columns), encoded_df], axis=1)
    logger.info(f"Encoded {len(columns)} categorical columns into {encoded_df.shape[1]} dummy columns")
    return df, encoder


def preprocess(df: pd.DataFrame, time_delta_strategy: str = "abs") -> tuple[pd.DataFrame, PreprocessingReport]:
    """
    Full cleaning pipeline: drop dead/redundant columns, fix the known
    sign bug, handle missing values. Does NOT encode categoricals or
    engineer fraud-specific features — see encode_categoricals() above
    and features.py respectively for those, since not every downstream
    use needs them.
    """
    report = PreprocessingReport(rows_in=len(df))

    df = drop_dead_and_redundant_columns(df, report)
    df = fix_time_since_last_transaction(df, report, strategy=time_delta_strategy)
    df = handle_missing_values(df, report)

    report.rows_out = len(df)
    return df, report


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/balanced_fraud_dataset.csv"
    raw = load_raw_data(path)
    clean, rpt = preprocess(raw)

    print("\n=== Preprocessing Report ===")
    print(f"Rows in / out: {rpt.rows_in:,} / {rpt.rows_out:,}")
    print(f"Dead columns dropped: {rpt.dead_columns_dropped}")
    print(f"Redundant columns dropped: {rpt.redundant_columns_dropped}")
    print(f"Negative time deltas fixed: {rpt.negative_time_deltas_fixed:,}")
    print(f"Missing values filled: {rpt.missing_values_filled}")
    print(f"\nFinal shape: {clean.shape}")
