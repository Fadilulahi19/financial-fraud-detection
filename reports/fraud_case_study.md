# Fraud Detection Feasibility Study — Case Study Report

**Scope:** Feasibility assessment and pilot recommendation — not a production deployment
**Dataset:** Nigerian Financial Transactions and Fraud Detection Dataset, 5,000,000-row

---

## Executive Summary

Fraud Ops requested a supervised fraud detection model targeting **80% precision and 80% recall**. This study finds that target **is not reachable with the current feature set**, and the reason is not a modeling shortfall — it's a data one.

Across amount, transaction type, payment channel, time of day, geographic location, and merchant category, fraudulent and legitimate transactions are statistically almost indistinguishable. The single exception is **device familiarity**: every fraud case in this dataset occurred on a transaction from a new device. But new-device transactions are still only ~35% fraud, which caps achievable precision well below target regardless of model choice, threshold, or algorithm sophistication.

Two supervised models (Logistic Regression, XGBoost) and one unsupervised model (Isolation Forest) all converge on this same conclusion. XGBoost's added complexity PR-AUC is equal to Logistic Regression — not meaningful. Isolation Forest performs *worse than random* at finding fraud patterns the supervised model misses, because no such patterns exist in this data: 100% of fraud sits in the new-device population.

**Recommendation:** treat this as validation for a limited pilot rather than production deployment. Deploy Logistic Regression (equally performant, more interpretable and auditable than XGBoost) as a first-line filter feeding human review — not a standalone block/allow system — and prioritize acquiring better features (device/session history depth, a corrected elapsed-time field) before considering wider rollout.

---

## Phase 1 — Business Understanding

**Business problem.** A fintech Fraud Ops team needs a way to flag fraudulent transactions that scales beyond manual review. Both false negatives (fraud losses) and false positives (blocked legitimate customers) carry direct cost.

**Analytical objective.** Determine whether a supervised classification model — Logistic Regression as baseline, XGBoost as comparison — can reliably separate fraud from legitimate transactions, and quantify the achievable performance level. This is explicitly a feasibility study, not a production build.

**KPIs.**
- **Precision on the fraud class** (primary) — false positives block legitimate customers directly.
- **Recall on the fraud class** (secondary, tracked alongside precision) — missed fraud is direct financial loss.
- **PR-AUC** — the right threshold-independent metric given class imbalance (not ROC-AUC or accuracy alone).
- **Estimated ₦ impact** at the chosen operating threshold — translates model output into business terms.

**Success criteria.** Stretch target: ≥80% precision and ≥80% recall simultaneously. Given the evidence gathered, the honest interim criterion became: demonstrate the best achievable precision/recall combination, quantify the gap to 80/80, and identify what closing that gap would require. The final recommendation is graded on evidence and honesty, not on hitting the stretch target — consistent with the brief from the outset.

---

## Phase 2 — Data Validation

A full validation pass was run across every variable: data types, missing values, duplicates, unique values, invalid values, outliers, target integrity, timestamp validity, identifier consistency, potential leakage, and class imbalance. Full detail is in the standalone [Data Quality Report](data_quality_report.md); key findings are summarized here.

**Clean:** no duplicate rows or transaction IDs, no malformed account numbers or transaction IDs, target labels (`is_fraud` / `fraud_type`) are fully internally consistent, timestamps are coherent (derived fields like `txn_hour` and `is_weekend` match the raw timestamp with zero mismatches), and no self-transfers exist.

**Two issues found and handled:**
1. **`time_since_last_transaction` has a sign bug** — 43.5% of values are negative, which is impossible for elapsed time. Likely a `previous − current` vs. `current − previous` computation error. Excluded from feature engineering rather than guessing at a fix.

**Leakage check.** The four precomputed "risk" features (`merchant_fraud_rate`, `channel_risk_score`, `persona_fraud_risk`, `location_fraud_risk`) were tested against this sample's own group-level fraud rates. None matched — two (`channel_risk_score`, `persona_fraud_risk`) are fixed, static per-category weights that don't correlate with actual outcomes here; two (`merchant_fraud_rate`, `location_fraud_risk`) vary row-by-row within a category, consistent with a legitimately-computed rolling historical estimate. **No leakage detected** — these features are safe to use.

---

## Phase 3 — Exploratory Data Analysis: What Does Fraud Look Like?

![Amount distribution](images/u1_amount_distribution.png)

Transaction amounts are right-skewed and log-normal-shaped (₦24–₦15.5M), typical of financial transaction data with no unusual spikes.

![Fraud class distribution](images/u2_fraud_class_distribution.png)

The 30%/70% fraud/legitimate split confirms this file is artificially balanced — not a picture of real-world prevalence.

**Across every dimension tested, fraud and legitimate transactions look nearly identical:**

| Dimension | Fraud rate range | Verdict |
|---|---|---|
| Transaction type | 29.1% – 30.9% | No discriminating power |
| Payment channel | 29.2% – 30.4% | No discriminating power |
| Transaction amount | Medians ₦170K (legit) vs ₦167K (fraud) | No discriminating power |
| Transaction velocity (24h) | Medians 3 vs 4 | Weak at best |
| Hour of day | 26% – 33% | No time-of-day pattern |
| Location (10 cities) | 28.9% – 31.1% | No geographic pattern |
| Merchant category | 26.7% – 32.5% | Mild variation only |

![Fraud rate by device](images/02_fraud_rate_by_new_device.png)

**The one real signal: device familiarity.** 100% of fraud in this sample occurred on new-device transactions, versus 0% on known devices. New-device transactions still run only ~34.5% fraud (not every new-device transaction is fraudulent), but this is the only variable in the entire dataset that separates the classes at all.

---

## Phase 4 — Fraud Typology Analysis (Quantified)

Five hypothesized fraud typologies were tested as standalone flagging rules, each measured against the 30% base rate (lift = precision ÷ base rate):

| Typology | Rule | Coverage | Precision | Recall | Lift |
|---|---|---|---|---|---|
| Velocity-based | Top 10% `txn_count_last_1h` | 13.1% | 31.3% | 13.7% | 1.04x |
| Amount anomaly | ≥2σ deviation from own average | 4.6% | 27.7% | 4.2% | **0.92x** |
| **Device anomaly** | `new_device_transaction` = True | 87.0% | 34.5% | **100%** | **1.15x** |
| Geographic anomaly | Top 10% `geo_anomaly_score` | 10.8% | 29.1% | 10.5% | 0.97x |
| Merchant/channel risk | Top-quartile merchant categories | 21.5% | 31.7% | 22.7% | 1.06x |

Amount anomaly actually performs *worse* than random (0.92x lift). Geographic anomaly is indistinguishable from noise (0.97x). Combining typologies (e.g., new device + high velocity) pushes lift only to 1.14–1.22x while recall collapses to single digits in the strictest combinations — stacking rules trades away coverage faster than it buys precision.

**Conclusion:** of five hypothesized typologies, only device anomaly carries real, usable signal.

---

## Phase 5 — Feature Engineering

Fourteen features were finalized, mapping directly to existing columns or derived fresh where needed. Two data quality issues surfaced during construction and were corrected:
- `is_night_txn` was constant (all-zero) in the raw data — **derived fresh** from `transaction_hour`.
- `txn_count_last_1h` and `txn_count_last_24h` were found to be **identical for every row** — only one retained to avoid manufacturing false signal from a duplicated column.

| Feature | Source | Status |
|---|---|---|
| `transaction_hour`, `is_weekend` | Existing, validated | ✅ |
| `is_night_transaction` | Derived (hour ≥22 or <6) | Fixed — original was broken |
| `amount_deviation` | Existing `spending_deviation_score` | ✅ |
| `transaction_velocity` | Existing `txn_count_last_24h` | ✅ (duplicate `_1h` dropped) |
| `customer_avg_amount` | Existing `user_avg_txn_amt` | ✅ |
| `customer_amount_deviation` | Derived: z-score vs. customer's own history | New |
| `device_seen_count`, `device_shared_flag` | Existing, validated | ✅ |
| `ip_shared_flag` | Existing | ⚠️ near-constant (99.8%), low expected value |
| `geo_anomaly_score`, `merchant_fraud_rate`, `channel_risk_score` | Existing | ✅ no leakage (Phase 2) |
| `new_device_transaction` | Existing | ✅ strongest signal |

---

## Phase 6 — Modeling

**Model 1 — Logistic Regression (baseline):** PR-AUC 0.346, ROC-AUC 0.599. The standardized coefficient for `new_device_transaction` (2.41) dwarfs every other feature — the next-largest coefficient is over 100x smaller. At the conventional 0.5 threshold, this model **predicts zero fraud cases** — not a bug, but a direct consequence of the strongest signal only reaching ~35% conditional probability.

**Model 2 — XGBoost (comparison):** PR-AUC 0.350, ROC-AUC 0.604. `new_device_transaction` accounts for 73% of total feature importance.

![PR curves](images/phase6_pr_curves.png)

**Modeling question — does XGBoost's added complexity materially improve fraud detection?** No. PR-AUC improves by only 0.005 (1.4% relative); ROC-AUC by the same margin. Both models converge on the identical single signal. The precision-recall curves are nearly indistinguishable and both flat across the full recall range, meaning neither model found a way to meaningfully refine the ranking beyond the device-newness split.

**Recommendation:** Logistic Regression is the better choice for a pilot — equal performance, full interpretability, easier to audit and explain to Fraud Ops.

---

## Phase 7 — Evaluation

Accuracy was not used as the primary metric given class imbalance. Comparing the **Fraud Capture Rate** (business KPI = recall) against the **False Positive Rate** (customer-impact KPI) at three operating points:

| Operating point | Precision | Fraud Capture Rate | False Positive Rate | Fraud ₦ captured | Legit ₦ wrongly blocked |
|---|---|---|---|---|---|
| 50% recall target | 35.5% | 67.1% | 52.3% | ₦771M of ₦1,136M (67.9%) | ₦1,422M |
| 80% recall target | 35.3% | 85.7% | 67.4% | ₦957M (84.3%) | ₦1,754M |
| Max recall (100%) | 34.5% | 100% | 81.5% | ₦1,136M (100%) | ₦2,186M |

![FCR vs FPR](images/phase7_fcr_vs_fpr.png)

In plain terms: catching even half the fraud requires wrongly flagging **52% of legitimate transactions**. Catching nearly all fraud requires flagging **81.5%**. At every operating point, roughly **₦1.85–1.90 of legitimate value is blocked for every ₦1 of fraud caught.**

![Value tradeoff](images/phase7_value_tradeoff.png)

This is the clearest evidence that 80/80 is unreachable — not because the model is poorly built, but because the false-positive cost of chasing recall would be business-breaking at scale.

---

## Phase 8 — Threshold Optimization

Rather than defaulting to the conventional 0.5 cutoff, thresholds from 0.10 to 0.90 were tested explicitly. Logistic Regression's predicted probabilities never exceed 0.40, making the 0.5 default silently non-functional for this model — a concrete illustration of why threshold testing matters.

![Threshold tradeoff](images/phase8_threshold_tradeoff.png)

| XGBoost threshold | Precision | Fraud Capture Rate | False Positive Rate | Fraud value captured |
|---|---|---|---|---|
| 0.10 | 34.4% | 99.9% | 81.5% | 99.2% |
| 0.20 | 34.4% | 97.8% | 80.0% | 96.9% |
| **0.30** | **35.3%** | **79.7%** | **62.7%** | **73.7%** |
| **0.40** | **35.1%** | **16.3%** | **12.9%** | **19.0%** |
| 0.50+ | ≤33.3% | ≤1.3% | ≤1.1% | ≤2.4% |

Precision is flat (32–35%) at nearly every threshold — there is no cutoff that buys higher precision, only a choice of how much recall (and how many false positives) to accept.

**Data-driven recommendation: threshold = 0.30.** Captures 74% of fraud value at 62.7% FPR — the best available trade-off in the usable range. Threshold 0.40 is the conservative alternative (16% recall, 12.9% FPR) for a lower-friction rollout. Neither is comfortable by conventional fraud-ops standards (production systems typically target FPR under 5%) — which is itself the honest finding this study needs to surface.

---

## Phase 9 — Anomaly Detection (Isolation Forest)

**Question:** can an unsupervised model identify emerging fraud patterns the supervised model misses?

| Metric | Isolation Forest | XGBoost | Random baseline |
|---|---|---|---|
| ROC-AUC | **0.421** | 0.604 | 0.500 |
| PR-AUC | **0.250** | 0.350 | 0.300 |

Isolation Forest performs *worse than random* — its anomaly score is mildly inversely related to fraud, because fraud in this dataset looks statistically ordinary on every dimension except device newness, which isn't a rare-outlier pattern in the general sense (87% of all transactions are on new devices).

Of the fraud cases the supervised model missed, Isolation Forest caught 33.6% — but **100% of those catches were still new-device transactions**. There is zero fraud anywhere in this data on a known device for an unsupervised method to discover.

**Answer: No.** Unsupervised anomaly detection adds no value here — not a different fraud pattern, just a noisier, less precise re-detection of the same population already captured.

---

## Phase 10 — SQL Fraud Investigation

A standalone SQL component (`sql/fraud_investigation.sql`) independently validates the Python findings:

- **Fraud by channel and transaction type:** flat 29–31% everywhere — no channel or type stands out.
- **High-velocity customers:** even the most active senders (14–17 transactions/24h) split roughly evenly between fraud and legitimate.
- **High-risk merchants:** top decile runs 31–33% vs. 30% average — a mild lift only.
- **Fraud exposure:** **₦4.52B total** fraud value in this sample (30.07% of all transaction value). `Account Takeover` accounts for 95.3% of fraud cases and ₦4.31B of exposure; `Identity Fraud` the remainder.
- **Geographic patterns:** narrow 28.9–31.1% range across all 10 cities.
- **Device anomaly check** (bonus query): confirms independently in SQL — 100% of fraud (6,000/6,000) occurs on new-device transactions, 0% on known devices.

This SQL file is auditable by anyone with database access, without needing to trust or re-run the Python pipeline.

---

## Phase 11 — Risk Scoring

Model probability was converted to a 0–100 score (`score = probability × 100`). Testing whether the standard 4–5 tier risk framework (Low/Medium/High/Critical) fits the model's actual output revealed **exactly one meaningful break point, at score 10**:

| Score range | Volume | Actual fraud rate |
|---|---|---|
| 0–9 | 13.0% | 0.3% |
| 10–100 | 87.0% | ~30–36% (flat, no further separation) |

**Recommended bands (data-driven, not the generic template):**

| Score | Risk | Action |
|---|---|---|
| 0–9 | Low | Approve automatically |
| 10–100 | Elevated | Route to manual/secondary review |

A 4–5 tier scorecard would imply the model can distinguish a score-35 transaction from a score-65 transaction — it cannot. Presenting more granularity than the data supports would overstate the model's actual precision to anyone relying on the score.

---

## Phase 12 — Monitoring Framework

A production-style monitoring design was specified across five dimensions, full detail in the project documentation:

1. **Data drift** — track `amount_ngn` distribution, `new_device_transaction` rate (baseline 87.0%), transaction velocity (baseline 3.79/24h). Exclude `ip_shared_flag` — it's near-constant and can't drift meaningfully.
2. **Fraud-rate drift** — requires the true production baseline to be established first (open item since Phase 1).
3. **Model performance** — baseline PR-AUC 0.350; trigger retraining if it drops >15% relative, sustained across two evaluation windows.
4. **Feature drift** — `new_device_transaction` carries the model's entire signal. If fraud starts appearing on known devices (currently 0%), that's not drift — it's a structural change requiring immediate review.
5. **Threshold performance** — baseline FPR 62.7% at threshold 0.30; monitor for disproportionate growth in flagged volume.

**Standing caveat:** because this model's entire value rests on one feature holding up, it is closer to a single-point-of-failure system than a typical multi-feature model. Monitoring here is load-bearing, not routine hygiene.

---

## Overall Recommendation

The evidence across all twelve phases points the same direction. This is not a case of a model needing more tuning — it's a case where the available features do not contain enough signal to hit the requested target, and every method tested (linear, nonlinear, supervised, unsupervised) confirms the same ceiling.

1. **Do not proceed to full production deployment.** The false-positive cost at any usable recall level is not viable at scale.
2. **A limited pilot is justified**, using Logistic Regression at threshold 0.30, deployed as a first-line filter feeding human review rather than an autonomous decision system.
3. **Invest in feature acquisition before scaling further** — specifically, device/session history depth and a corrected elapsed-time-since-last-transaction field. The ceiling here is a feature-availability problem, not a modeling-technique problem, and better raw signal is likely to move the needle further than any amount of additional model tuning would.
4. **Confirm the true production fraud rate** with Fraud Ops before translating any of this study's ₦ figures into real operational impact estimates.

This aligns with the fallback options Fraud Ops raised at the outset: a limited pilot, closer feature-engineering collaboration, or sourcing additional data — rather than a green light for full deployment.

---

*Supporting materials: `notebooks/01_fraud_detection_analysis.ipynb` (full executed analysis), `sql/fraud_investigation.sql` (standalone SQL queries), `reports/data_quality_report.md` (detailed Phase 2 validation), `app/streamlit_app.py` (scoring demo).*
