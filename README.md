# Fraud Detection — Feasibility Study

A feasibility assessment and pilot recommendation for fraud detection at a Fintech company. This is **not** a production deployment — it's an evidence-based evaluation of whether a supervised model can meet a target of 80% precision and 80% recall, using the tools and data currently available.

## Key finding

**The 80/80 target is not reachable with the current feature set.** Across amount, transaction type, payment channel, time of day, location, and merchant category, fraud and legitimate transactions are statistically indistinguishable. The only usable signal is **device familiarity** — 100% of fraud in the sample occurs on new-device transactions, but new-device transactions are still only ~35% fraud, capping precision well below target. Neither a more complex model (XGBoost) nor an unsupervised approach (Isolation Forest) overcomes this — both converge on the same single feature. Full details and evidence are in the notebook and written report.

**Recommendation:** deploy Logistic Regression (not XGBoost — added complexity buys no meaningful improvement) as a first-line filter feeding human review, at threshold 0.10, while prioritizing better features (device/session history, a fixed elapsed-time field) before considering a wider rollout.

## Repo structure

```
financial-fraud-detection/
├── data/
│   └── raw/
│       └── balanced_fraud_dataset.csv   # 5,000,000-row 
├── notebooks/
│   └── 01_fraud_detection_analysis.ipynb   # full 12-phase analysis, executed end-to-end
├── sql/
│   └── fraud_investigation.sql             # standalone SQL investigation queries
├── reports/
│   ├── data_quality_report.md              # Phase 2 formal data validation report
|   ├── fraud_case_study.md 
│   └── images/                             # all chart exports
├── app/
│   └── streamlit_app.py                    # scoring demo (not production)
├── requirements.txt
└── README.md
```

## Dataset

Sample of the [Nigerian Financial Transactions and Fraud Detection Dataset](https://huggingface.co/datasets/electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset) (full dataset: 5M rows, 45 columns, 2.84 GB on HuggingFace). This is *not* the real-world fraud prevalence, and any business-metric translation must correct for that (see `reports/data_quality_report.md`, Section 11).

**Known data quality issues** (documented in full in the data quality report):
- `time_since_last_transaction` has a sign bug (43.5% negative values) — excluded from modeling
- `txn_count_last_1h` and `txn_count_last_24h` are identical in this sample — only one used as a feature
- `ip_shared_flag` is near-constant (99.8%) — low value, kept but flagged
- True production fraud rate is unconfirmed — open item for Fraud Ops

## What's in the analysis

The notebook (`notebooks/01_fraud_detection_analysis.ipynb`) walks through 12 phases:

1. **Business Understanding** — problem, objective, KPIs, success criteria
2. **Data Validation** — types, nulls, duplicates, outliers, target integrity, leakage check
3. **Exploratory Data Analysis** — univariate and multivariate views of what fraud looks like
4. **Fraud Typology Analysis** — quantified lift/precision/recall for 5 hypothesized fraud patterns
5. **Feature Engineering** — mapped and derived features, with an explicit leakage check
6. **Modeling** — Logistic Regression baseline vs. XGBoost comparison
7. **Evaluation** — precision, recall, F1, ROC-AUC, PR-AUC, Fraud Capture Rate, False Positive Rate, ₦ value captured
8. **Threshold Optimization** — full sweep (0.10–0.90), data-driven operating point
9. **Anomaly Detection** — Isolation Forest, testing for fraud patterns the supervised model misses
10. **SQL Fraud Investigation** — standalone queries by channel, type, velocity, merchant, geography, exposure
11. **Risk Scoring** — 0–100 score converted from model probability, with data-driven risk bands
12. **Monitoring Framework** — drift, performance, and retraining triggers for production

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_fraud_detection_analysis.ipynb
```

To run the SQL investigation independently:
```bash
sqlite3 fraud.db < sql/fraud_investigation.sql
```

To run the Streamlit scoring demo:
```bash
streamlit run app/streamlit_app.py
```

## Scope note

Full production deployment with engineering integration was judged unrealistic within this project's timeframe and is explicitly out of scope. This deliverable positions the work as a feasibility assessment with a pilot recommendation, per the fallback options raised by Fraud Ops (limited pilot, feature engineering collaboration, or alternative data sources).
