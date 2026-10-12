"""
Fraud Detection — Scoring Demo (Streamlit)
======================================================
This is a DEMO app for the feasibility study, not a production system.
It trains a Logistic Regression model (the recommended baseline from
Phase 6) on the balanced sample and lets you explore:
  1. Scoring individual transactions or a batch CSV
  2. The precision / recall / false-positive trade-off across thresholds
  3. The key limitation of this feature set (see "About" tab)

Run with:  streamlit run app/streamlit_app.py
Expects the dataset at ../data/raw/balanced_fraud_dataset.csv relative
to this file (matches the repo scaffold).
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_curve, roc_curve, confusion_matrix,
                              average_precision_score, roc_auc_score)

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(page_title="Fraud Scoring Demo", page_icon="🔍", layout="wide")

FEATURE_COLS = [
    'transaction_hour', 'is_weekend', 'is_night_transaction', 'amount_deviation',
    'transaction_velocity', 'customer_avg_amount', 'customer_amount_deviation',
    'device_seen_count', 'device_shared_flag', 'ip_shared_flag',
    'geo_anomaly_score', 'merchant_fraud_rate', 'channel_risk_score',
    'new_device_transaction'
]

DATA_PATH_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), '..', 'data', 'V2-nigerian-financial-transactions-and-fraud-detection-dataset-for-model-training.csv'),
    '..//data/V2-nigerian-financial-transactions-and-fraud-detection-dataset-for-model-training.csv',
    'V2-nigerian-financial-transactions-and-fraud-detection-dataset-for-model-training.csv',
]


# ---------------------------------------------------------------------
# Data + model loading (cached so the app doesn't retrain on every click)
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    for path in DATA_PATH_CANDIDATES:
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['timestamp'])
            return df, path
    return None, None


def engineer_features(df):
    df = df.copy()
    df['transaction_hour'] = df['txn_hour']
    df['is_night_transaction'] = df['transaction_hour'].apply(lambda h: 1 if (h >= 22 or h < 6) else 0)
    df['amount_deviation'] = df['spending_deviation_score']
    df['transaction_velocity'] = df['txn_count_last_24h']
    df['customer_avg_amount'] = df['user_avg_txn_amt']
    df['customer_amount_deviation'] = (
        (df['amount_ngn'] - df['user_avg_txn_amt']) / df['user_std_txn_amt'].replace(0, np.nan)
    ).fillna(0)
    df['device_shared_flag'] = df['is_device_shared']
    df['ip_shared_flag'] = df['is_ip_shared']
    df['new_device_transaction'] = df['new_device_transaction'].astype(int)
    return df


@st.cache_resource
def train_model(_df):
    df = engineer_features(_df)
    X = df[FEATURE_COLS].copy()
    y = df['is_fraud'].astype(int)
    amounts = df['amount_ngn']

    X_train, X_test, y_train, y_test, amt_train, amt_test = train_test_split(
        X, y, amounts, test_size=0.25, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    test_proba = model.predict_proba(X_test_s)[:, 1]

    return {
        'model': model,
        'scaler': scaler,
        'X_test': X_test.reset_index(drop=True),
        'y_test': y_test.reset_index(drop=True),
        'amt_test': amt_test.reset_index(drop=True),
        'test_proba': test_proba,
        'pr_auc': average_precision_score(y_test, test_proba),
        'roc_auc': roc_auc_score(y_test, test_proba),
        'max_proba': test_proba.max(),
    }


def score_to_band(score):
    if score < 10:
        return "Low", "Approve automatically", "#2E7D32"
    else:
        return "Elevated", "Route to manual/secondary review", "#C44E52"


def compute_metrics_at_threshold(proba, y_true, amounts, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fraud_total = amounts[y_true == 1].sum()
    fraud_captured = amounts[(y_true == 1) & (pred == 1)].sum()
    legit_blocked = amounts[(y_true == 0) & (pred == 1)].sum()
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall, fpr=fpr,
                fraud_total=fraud_total, fraud_captured=fraud_captured, legit_blocked=legit_blocked)


# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------
st.title("🔍 Fraud Detection — Scoring Demo")
st.caption(
    "**Demo only, not a production system.** Built for the fraud detection feasibility study. "
    "Model: Logistic Regression, trained on a 5,000,000-row."
)

raw_df, found_path = load_data()

if raw_df is None:
    st.error(
        "Couldn't find `V2-nigerian-financial-transactions-and-fraud-detection-dataset-for-model-training.csv`. Expected it at "
        "`data/V2-nigerian-financial-transactions-and-fraud-detection-dataset-for-model-training.csv` relative to the repo root. "
        "Place the file there and restart the app."
    )
    st.stop()

with st.spinner("Training model..."):
    artifacts = train_model(raw_df)

tab_score, tab_threshold, tab_about = st.tabs(
    ["🎯 Score Transactions", "⚖️ Threshold Explorer", "ℹ️ About & Limitations"]
)

# =======================================================================
# TAB 1 — Score Transactions
# =======================================================================
with tab_score:
    st.subheader("Score a transaction")
    mode = st.radio("Input method", ["Manual entry", "Upload CSV"], horizontal=True)

    threshold = st.session_state.get("threshold", 0.30)

    if mode == "Manual entry":
        col1, col2, col3 = st.columns(3)
        with col1:
            transaction_hour = st.slider("Transaction hour", 0, 23, 14)
            is_weekend = st.checkbox("Weekend transaction")
            new_device = st.checkbox("New device", value=True)
            device_seen_count = st.number_input("Device seen count (times this device seen before)", 0, 500, 1)
        with col2:
            amount_deviation = st.slider("Amount deviation score", -3.0, 4.0, 0.0, 0.1)
            customer_avg_amount = st.number_input("Customer's average transaction amount (₦)", 0.0, 20_000_000.0, 200000.0)
            customer_amount_deviation = st.slider("This transaction's deviation from customer's own average (z-score)", -5.0, 10.0, 0.0, 0.1)
            transaction_velocity = st.number_input("Transactions in last 24h", 0, 50, 3)
        with col3:
            device_shared = st.checkbox("Device shared across accounts")
            ip_shared = st.checkbox("IP shared across accounts", value=True)
            geo_anomaly_score = st.slider("Geo anomaly score", 0.0, 1.0, 0.1, 0.01)
            merchant_fraud_rate = st.slider("Merchant fraud rate (historical)", 0.0, 0.1, 0.03, 0.001)
            channel_risk_score = st.select_slider("Channel risk score", options=[0.3, 0.4, 0.6, 0.8], value=0.4)

        is_night = 1 if (transaction_hour >= 22 or transaction_hour < 6) else 0

        input_row = pd.DataFrame([{
            'transaction_hour': transaction_hour,
            'is_weekend': int(is_weekend),
            'is_night_transaction': is_night,
            'amount_deviation': amount_deviation,
            'transaction_velocity': transaction_velocity,
            'customer_avg_amount': customer_avg_amount,
            'customer_amount_deviation': customer_amount_deviation,
            'device_seen_count': device_seen_count,
            'device_shared_flag': int(device_shared),
            'ip_shared_flag': int(ip_shared),
            'geo_anomaly_score': geo_anomaly_score,
            'merchant_fraud_rate': merchant_fraud_rate,
            'channel_risk_score': channel_risk_score,
            'new_device_transaction': int(new_device),
        }])

        if st.button("Score this transaction", type="primary"):
            X_scaled = artifacts['scaler'].transform(input_row[FEATURE_COLS])
            proba = artifacts['model'].predict_proba(X_scaled)[0, 1]
            score = round(proba * 100, 1)
            band, action, color = score_to_band(score)

            c1, c2, c3 = st.columns(3)
            c1.metric("Risk Score (0-100)", f"{score}")
            c2.metric("Risk Band", band)
            c3.metric("Recommended Action", action)
            st.markdown(
                f"<div style='padding:12px;border-radius:8px;background-color:{color}22;"
                f"border-left:5px solid {color}'><b>{band} risk</b> — {action}. "
                f"Model confidence this is fraud: {proba*100:.1f}%</div>",
                unsafe_allow_html=True,
            )
            if not new_device:
                st.info(
                    "Note: in the training data, **zero fraud cases occurred on known devices**. "
                    "A known-device transaction scoring above 'Low' here would be unusual — worth "
                    "treating with extra scrutiny if it happens on real data."
                )

    else:
        st.write("Upload a CSV with the same columns as the training data (or the 14 engineered feature columns directly).")
        uploaded = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                if not set(FEATURE_COLS).issubset(batch_df.columns):
                    batch_df = engineer_features(batch_df)
                X_batch = batch_df[FEATURE_COLS]
                X_batch_scaled = artifacts['scaler'].transform(X_batch)
                proba = artifacts['model'].predict_proba(X_batch_scaled)[:, 1]
                batch_df['risk_score'] = (proba * 100).round(1)
                batch_df['risk_band'] = np.where(batch_df['risk_score'] < 10, 'Low', 'Elevated')
                batch_df['recommended_action'] = np.where(
                    batch_df['risk_band'] == 'Low', 'Approve automatically', 'Route to manual review'
                )
                st.success(f"Scored {len(batch_df):,} transactions.")
                st.dataframe(
                    batch_df[['risk_score', 'risk_band', 'recommended_action'] + FEATURE_COLS].head(200),
                    use_container_width=True,
                )
                st.download_button(
                    "Download scored results (CSV)",
                    batch_df.to_csv(index=False).encode('utf-8'),
                    "scored_transactions.csv",
                    "text/csv",
                )
            except Exception as e:
                st.error(f"Couldn't process this file: {e}")

# =======================================================================
# TAB 2 — Threshold Explorer
# =======================================================================
with tab_threshold:
    st.subheader("Explore the precision / recall / false-positive trade-off")
    st.write(
        "This model's probabilities never exceed "
        f"**{artifacts['max_proba']:.2f}**, so the conventional 0.5 cutoff silently predicts zero fraud. "
        "Use the slider below to see what actually happens at different thresholds, using the held-out test set."
    )

    threshold = st.slider("Decision threshold", 0.0, min(0.5, float(np.ceil(artifacts['max_proba']*100)/100)), 0.30, 0.01)
    st.session_state["threshold"] = threshold

    m = compute_metrics_at_threshold(artifacts['test_proba'], artifacts['y_test'].values,
                                      artifacts['amt_test'].values, threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{m['precision']*100:.1f}%")
    c2.metric("Fraud Capture Rate (Recall)", f"{m['recall']*100:.1f}%")
    c3.metric("False Positive Rate", f"{m['fpr']*100:.1f}%")
    c4.metric("PR-AUC (overall)", f"{artifacts['pr_auc']:.3f}")

    fraud_pct = m['fraud_captured'] / m['fraud_total'] * 100 if m['fraud_total'] > 0 else 0
    st.markdown(
        f"At this threshold: **₦{m['fraud_captured']:,.0f}** of fraud value captured "
        f"({fraud_pct:.1f}% of ₦{m['fraud_total']:,.0f} total), while **₦{m['legit_blocked']:,.0f}** "
        f"of legitimate transaction value gets wrongly blocked."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred Legit', 'Pred Fraud'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Actual Legit', 'Actual Fraud'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center', fontsize=13,
                         color='white' if cm[i,j] > cm.max()/2 else 'black')
        ax.set_title(f"Confusion Matrix @ threshold={threshold:.2f}")
        st.pyplot(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(5, 4))
        prec, rec, _ = precision_recall_curve(artifacts['y_test'], artifacts['test_proba'])
        ax.plot(rec, prec, color='#4C72B0', linewidth=2)
        ax.axvline(m['recall'], color='#C44E52', linestyle='--', label=f'Current: recall={m["recall"]:.2f}')
        ax.axhline(0.80, color='green', linestyle=':', label='80% target')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        st.pyplot(fig)

    st.info(
        "**Reading this chart honestly:** precision stays flat around 30-36% across almost the "
        "entire recall range. There is no threshold that meaningfully improves precision — only a "
        "choice of how much recall (and how many false positives) to accept."
    )

# =======================================================================
# TAB 3 — About & Limitations
# =======================================================================
with tab_about:
    st.subheader("About this demo")
    st.markdown(f"""
This app trains a **Logistic Regression** model — the recommended baseline from the feasibility
study's Phase 6 modeling comparison — on a 5,000,000-row sample of the Nigerian Financial
Transactions and Fraud Detection Dataset.

**Model performance (held-out test set):**
- PR-AUC: {artifacts['pr_auc']:.3f}
- ROC-AUC: {artifacts['roc_auc']:.3f}
- Maximum predicted probability: {artifacts['max_proba']:.2f} (never reaches 0.5)

**The core limitation, in one sentence:** the only feature that meaningfully separates fraud from
legitimate transactions in this data is `new_device_transaction` — 100% of fraud in the training
data occurred on new devices, but new-device transactions are still only ~35% fraud. Amount,
transaction type, payment channel, time of day, location, and merchant category all showed
essentially no discriminating power (see the full analysis notebook, Phases 3-4).

**What this means practically:**
- No threshold gets this model to 80% precision *and* 80% recall simultaneously
- XGBoost (tested in Phase 6) performs almost identically — the added complexity isn't justified
- An unsupervised Isolation Forest (Phase 9) performed *worse than random* — no hidden fraud
  pattern was found that the supervised model missed

**This app is a demo, not a production system.** It doesn't handle real-time scoring
infrastructure, model versioning, drift monitoring, or the retraining triggers specified in
Phase 12 of the analysis. Treat any score shown here as illustrative of the model's behavior,
not as a decision to act on for real transactions.

**Data source:** [Nigerian Financial Transactions and Fraud Detection Dataset](https://huggingface.co/datasets/electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset)
(full dataset: {len(raw_df):,} rows).
""")
