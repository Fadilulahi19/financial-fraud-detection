"""
validation.py
=============
Column-level checks and schema enforcement for the Moniepoint fraud
detection feasibility study — the programmatic version of Phase 2's
Data Quality Report (see reports/data_quality_report.md for the
narrative writeup this module's numbers were originally computed for).

This module only VALIDATES and REPORTS. It does not fix anything —
see preprocessing.py for the fixes each finding here leads to. Run this
first, on the raw data, before preprocessing touches it, so the report
reflects what was actually delivered rather than an already-cleaned view.

Usage:
    python validation.py [path/to/data.csv]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Columns expected to be null under specific, checkable conditions.
# Any other null is treated as a real data quality issue.
EXPECTED_NULL_COLUMNS = {
    "fraud_type": "is_fraud == False",
    "time_since_last_transaction": "customer's first observed transaction",
}

ID_COLUMNS = ["transaction_id", "sender_account", "receiver_account"]
TARGET_COL = "is_fraud"
AMOUNT_COL = "amount_ngn"


@dataclass
class ValidationReport:
    n_rows: int = 0
    n_cols: int = 0
    dtypes: dict = field(default_factory=dict)
    missing_pct: dict = field(default_factory=dict)
    unexpected_missing: dict = field(default_factory=dict)
    full_row_duplicates: int = 0
    duplicate_transaction_ids: int = 0
    unique_counts: dict = field(default_factory=dict)
    invalid_value_issues: dict = field(default_factory=dict)
    outlier_count: int = 0
    outlier_pct: float = 0.0
    target_integrity_mismatches: int = 0
    timestamp_issues: dict = field(default_factory=dict)
    identifier_issues: dict = field(default_factory=dict)
    class_balance: dict = field(default_factory=dict)
    passed: bool = True
    warnings: list[str] = field(default_factory=list)

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning(msg)


def check_data_types(df: pd.DataFrame, report: ValidationReport) -> None:
    report.dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}


def check_missing_values(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Flags missing values, distinguishing expected nulls (documented in
    EXPECTED_NULL_COLUMNS, e.g. fraud_type is null for every non-fraud
    row by design) from unexpected ones (a real data quality problem).
    """
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    report.missing_pct = missing_pct[missing_pct > 0].to_dict()

    for col, pct in report.missing_pct.items():
        if col not in EXPECTED_NULL_COLUMNS:
            report.unexpected_missing[col] = pct
            report.add_warning(f"Unexpected missing values in '{col}': {pct}% — not in the documented expected-null list")


def check_duplicates(df: pd.DataFrame, report: ValidationReport) -> None:
    report.full_row_duplicates = int(df.duplicated().sum())
    if "transaction_id" in df.columns:
        report.duplicate_transaction_ids = int(df["transaction_id"].duplicated().sum())

    if report.full_row_duplicates > 0:
        report.add_warning(f"{report.full_row_duplicates} full-row duplicates found")
        report.passed = False
    if report.duplicate_transaction_ids > 0:
        report.add_warning(f"{report.duplicate_transaction_ids} duplicate transaction_ids found")
        report.passed = False


def check_unique_values(df: pd.DataFrame, report: ValidationReport,
                         columns: list[str] | None = None) -> None:
    columns = columns or [
        "transaction_id", "sender_account", "receiver_account", "device_hash",
        "ip_address", "transaction_type", "merchant_category", "location",
        "device_used", "payment_channel", "sender_persona",
    ]
    report.unique_counts = {c: int(df[c].nunique()) for c in columns if c in df.columns}


def check_invalid_values(df: pd.DataFrame, report: ValidationReport) -> None:
    """Domain-specific sanity checks: amounts can't be non-positive, no
    self-transfers, and elapsed-time fields can't be negative."""
    issues = {}

    if AMOUNT_COL in df.columns:
        n_nonpositive = int((df[AMOUNT_COL] <= 0).sum())
        issues["nonpositive_amount"] = n_nonpositive
        if n_nonpositive > 0:
            report.add_warning(f"{n_nonpositive} rows with non-positive {AMOUNT_COL}")
            report.passed = False

    if "sender_account" in df.columns and "receiver_account" in df.columns:
        n_self_transfer = int((df["sender_account"] == df["receiver_account"]).sum())
        issues["self_transfers"] = n_self_transfer
        if n_self_transfer > 0:
            report.add_warning(f"{n_self_transfer} self-transfer transactions found")

    if "time_since_last_transaction" in df.columns:
        n_negative = int((df["time_since_last_transaction"] < 0).sum())
        pct_negative = round(n_negative / len(df) * 100, 1)
        issues["negative_time_since_last_transaction"] = n_negative
        if n_negative > 0:
            report.add_warning(
                f"{n_negative} ({pct_negative}%) negative values in time_since_last_transaction — "
                "elapsed time cannot be negative; likely a sign-convention bug (see preprocessing.py)"
            )
            report.passed = False

    report.invalid_value_issues = issues


def check_outliers(df: pd.DataFrame, report: ValidationReport, column: str = AMOUNT_COL) -> None:
    """IQR-based outlier count. Flagged for awareness, not treated as an
    error — in a fraud context, high-value outliers are plausible signal,
    not necessarily noise to be removed."""
    if column not in df.columns:
        return
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = int(((df[column] < lo) | (df[column] > hi)).sum())
    report.outlier_count = n_outliers
    report.outlier_pct = round(n_outliers / len(df) * 100, 2)


def check_target_integrity(df: pd.DataFrame, report: ValidationReport,
                            target_col: str = TARGET_COL, type_col: str = "fraud_type") -> None:
    """fraud_type must be populated if and only if is_fraud is True."""
    if target_col not in df.columns:
        return

    unique_targets = set(df[target_col].dropna().unique())
    if not unique_targets.issubset({True, False, 0, 1}):
        report.add_warning(f"Unexpected values in '{target_col}': {unique_targets}")
        report.passed = False

    if type_col in df.columns:
        mismatch = int(
            ((df[target_col] == True) & (df[type_col].isnull())).sum()
            + ((df[target_col] == False) & (df[type_col].notnull())).sum()
        )
        report.target_integrity_mismatches = mismatch
        if mismatch > 0:
            report.add_warning(f"{mismatch} rows where '{type_col}' is inconsistent with '{target_col}'")
            report.passed = False


def check_timestamp_validity(df: pd.DataFrame, report: ValidationReport,
                              ts_col: str = "timestamp") -> None:
    if ts_col not in df.columns:
        return
    ts = pd.to_datetime(df[ts_col])
    issues = {
        "min": str(ts.min()),
        "max": str(ts.max()),
        "future_dates": int((ts > pd.Timestamp.now()).sum()),
    }

    if "txn_hour" in df.columns:
        issues["txn_hour_mismatches"] = int((ts.dt.hour != df["txn_hour"]).sum())
    if "is_weekend" in df.columns:
        computed_weekend = ts.dt.dayofweek.isin([5, 6]).astype(int)
        issues["is_weekend_mismatches"] = int((computed_weekend != df["is_weekend"]).sum())

    report.timestamp_issues = issues
    if issues["future_dates"] > 0:
        report.add_warning(f"{issues['future_dates']} transactions with future-dated timestamps")
        report.passed = False


def check_identifier_consistency(df: pd.DataFrame, report: ValidationReport) -> None:
    """Confirms account numbers and transaction IDs follow a consistent format."""
    issues = {}
    if "sender_account" in df.columns:
        lengths = df["sender_account"].astype(str).str.len()
        issues["sender_account_length_range"] = [int(lengths.min()), int(lengths.max())]
        if lengths.min() != lengths.max():
            report.add_warning("sender_account has inconsistent digit lengths")

    if "transaction_id" in df.columns:
        matches = df["transaction_id"].astype(str).str.match(r"^T\d+$")
        issues["transaction_id_format_ok"] = bool(matches.all())
        if not matches.all():
            report.add_warning(f"{(~matches).sum()} transaction_ids don't match expected format 'T<digits>'")

    report.identifier_issues = issues


def check_class_imbalance(df: pd.DataFrame, report: ValidationReport,
                           target_col: str = TARGET_COL) -> None:
    if target_col not in df.columns:
        return
    balance = df[target_col].value_counts(normalize=True).to_dict()
    report.class_balance = {str(k): round(v, 4) for k, v in balance.items()}


def run_validation(df: pd.DataFrame) -> ValidationReport:
    """Run every check and return a single consolidated report."""
    report = ValidationReport(n_rows=len(df), n_cols=df.shape[1])

    check_data_types(df, report)
    check_missing_values(df, report)
    check_duplicates(df, report)
    check_unique_values(df, report)
    check_invalid_values(df, report)
    check_outliers(df, report)
    check_target_integrity(df, report)
    check_timestamp_validity(df, report)
    check_identifier_consistency(df, report)
    check_class_imbalance(df, report)

    return report


def print_report(report: ValidationReport) -> None:
    print(f"\n{'='*60}\nDATA VALIDATION REPORT\n{'='*60}")
    print(f"Rows: {report.n_rows:,}  |  Columns: {report.n_cols}")
    print(f"Overall passed: {report.passed}")

    print(f"\n--- Missing Values ---")
    for col, pct in report.missing_pct.items():
        flag = "" if col in EXPECTED_NULL_COLUMNS else "  <-- UNEXPECTED"
        print(f"  {col}: {pct}%{flag}")

    print(f"\n--- Duplicates ---")
    print(f"  Full-row: {report.full_row_duplicates}  |  Duplicate transaction_id: {report.duplicate_transaction_ids}")

    print(f"\n--- Invalid Values ---")
    for k, v in report.invalid_value_issues.items():
        print(f"  {k}: {v}")

    print(f"\n--- Outliers ({AMOUNT_COL}, IQR method) ---")
    print(f"  {report.outlier_count} rows ({report.outlier_pct}%)")

    print(f"\n--- Target Integrity ---")
    print(f"  fraud_type / is_fraud mismatches: {report.target_integrity_mismatches}")

    print(f"\n--- Identifier Consistency ---")
    for k, v in report.identifier_issues.items():
        print(f"  {k}: {v}")

    print(f"\n--- Timestamp ---")
    for k, v in report.timestamp_issues.items():
        print(f"  {k}: {v}")

    print(f"\n--- Class Balance ---")
    for k, v in report.class_balance.items():
        print(f"  {k}: {v*100:.2f}%")

    if report.warnings:
        print(f"\n--- Warnings ({len(report.warnings)}) ---")
        for w in report.warnings:
            print(f"  ! {w}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/balanced_fraud_dataset.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    report = run_validation(df)
    print_report(report)
