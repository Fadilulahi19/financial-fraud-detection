# Data Quality Report — Phase 2: Data Validation
**Dataset:** Nigerian Financial Transactions and Fraud Detection Dataset (5,000,000 rows × 45 columns)
**Source:** Nigerian Financial Transactions and Fraud Detection Dataset

---

## 1. Data Types
All 45 columns typed as expected: identifiers and categoricals as strings, `timestamp` as datetime, binary flags as bool/int, engineered scores as float64, counts as int64. No type coercion issues found.

## 2. Missing Values
| Column | Missing % | Assessment |
|---|---|---|
| `fraud_type` | 96.41% | Expected — null for all non-fraud rows (4,820,500/5,000,000) |
| `time_since_last_transaction` | 17.93% | Expected — null for a user's first observed transaction |

No missing values in any other column, including all engineered risk/behavioral features.

## 3. Duplicate Records
- Full-row duplicates: **0**
- Duplicate `transaction_id`: **0**

Clean on uniqueness.

## 4. Unique Values
| Column | Unique count | Notes |
|---|---|---|
| `transaction_id` | 5,000,000 | Fully unique, as expected |
| `sender_account` / `receiver_account` | ~19,775 each | Slight overlap expected (repeat users) |
| `device_hash` | 19,976 | Near-1:1 with transactions — most devices seen once |
| `ip_address` | 17,369 | Some IP reuse, consistent with shared NAT/mobile networks |
| `transaction_type` | 4 | transfer, deposit, withdrawal, payment |
| `merchant_category` | 21 | |
| `location` | 10 | Nigerian cities |
| `payment_channel` | 4 | |
| `sender_persona` | 3 | |

## 5. Invalid Values
- No negative or zero transaction amounts.
- All bounded scores fall within sensible ranges (`geo_anomaly_score` 0–1, `channel_risk_score` 0.3–0.8, `persona_fraud_risk` 0.4–0.7, `location_fraud_risk` ~0.02–0.10, `merchant_fraud_rate` 0–0.078).
- No self-transfers (`sender_account == receiver_account`): 6 cases.
- **Issue found:** `time_since_last_transaction` is **negative in 2,050,000 rows (41.0%)**. Elapsed time since a prior transaction cannot be negative — this points to a sign-convention bug in how the field was generated (likely computed as `previous − current` instead of `current − previous`) or clock/session misalignment. **Recommendation:** take `abs()` if the fix is confirmed to be a sign error, or exclude the feature from modeling until the generation logic is clarified.

## 6. Outliers
- `amount_ngn`: IQR method flags **2,487 rows (12.4%)** as outliers, range ₦24 – ₦15.46M. Given fraud detection contexts, high-value outliers are plausible signal rather than noise — no rows recommended for removal, but worth a log-transform for modeling.

## 7. Target Integrity
- `is_fraud`: only `{True, False}`, no invalid values.
- `fraud_type` populated for 100% of fraud rows and 0% of non-fraud rows — fully consistent, no mismatches.

## 8. Timestamp Validity
- Range: 2023-01-01 00:00:02.020872 to 2024-01-28 23:59:50.336777, no future-dated transactions.
- `txn_hour` matches `timestamp.hour` for all rows (0 mismatches).
- `is_weekend` matches computed weekday for all rows (0 mismatches).
- Derived time features are internally consistent and trustworthy.

## 9. Identifier Consistency
- `sender_account` / `receiver_account`: consistently 10-digit numeric, matching standard Nigerian account number format.
- `transaction_id`: consistently matches pattern `T<digits>`.
- No malformed identifiers found.

## 10. Potential Leakage
Checked whether the four precomputed "risk" features are simply the target's own group-level fraud rate restated (a common leakage pattern):
- `merchant_fraud_rate` vs. actual in-sample merchant fraud rate: r = 0.313 (not derived from this sample's labels)
- `persona_fraud_risk`: r = -0.11
- `location_fraud_risk`: r = 0.01
- `channel_risk_score`: r = -0.75 (**inversely** related to this sample's channel fraud rate)

**No direct leakage** — these features were not computed from this sample's own labels. However, the inverse relationship for `channel_risk_score` strongly suggests these are static features carrying the **true population's** base rates, and the artificial 30% balancing in this sample has **distorted their apparent relationship to fraud**. This is an important caveat for Phase 3 modeling: these features may carry more real signal in the true (rare-event) population than this balanced sample suggests, and their in-sample correlation should not be used to judge their production value.

## 11. Class Imbalance
- This sample: 3.5% fraud / 96.5% legitimate — deliberately balanced, **not representative of production prevalence**.
- True population fraud rate is not yet confirmed (flagged as open item since Phase 1).
- Modeling in Phase 3 should either re-derive a natural-rate holdout set for honest evaluation, or apply a documented correction factor when translating sample metrics to production expectations.

---

## Summary
Data is structurally clean: no duplicates, no malformed identifiers, no type errors, and target labels are internally consistent. Two issues require attention before modeling:
1. **`time_since_last_transaction` sign bug** (43.5% negative) — needs correction or exclusion.
2. **Balanced sampling distorts feature-target relationships** — risk scores likely more predictive in the true population than they appear here; do not discard them based on this sample's weak correlations alone.

No leakage detected. Class imbalance is by design in this sample and must be accounted for when generalizing results.
