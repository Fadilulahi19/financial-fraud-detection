-- =====================================================================
-- FRAUD INVESTIGATION QUERIES
-- Moniepoint Fraud Detection Feasibility Study — Phase 10
-- Table: transactions (loaded from balanced_fraud_dataset.csv)
-- NOTE: this table is an artificially balanced sample (30% fraud rate).
-- Fraud rate figures here do NOT reflect production prevalence; use
-- relative ranking / lift between categories, not absolute rates.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. FRAUD BY CHANNEL
-- Ranks payment channels by fraud rate to see if any channel is riskier.
-- ---------------------------------------------------------------------
SELECT
    payment_channel,
    COUNT(*) AS transactions,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN 1.0 ELSE 0 END), 4) AS fraud_rate,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount_ngn ELSE 0 END), 2) AS fraud_value_ngn
FROM transactions
GROUP BY payment_channel
ORDER BY fraud_rate DESC;


-- ---------------------------------------------------------------------
-- 2. FRAUD BY TRANSACTION TYPE
-- ---------------------------------------------------------------------
SELECT
    transaction_type,
    COUNT(*) AS transactions,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN 1.0 ELSE 0 END), 4) AS fraud_rate,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount_ngn ELSE 0 END), 2) AS fraud_value_ngn
FROM transactions
GROUP BY transaction_type
ORDER BY fraud_rate DESC;


-- ---------------------------------------------------------------------
-- 3. HIGH-VELOCITY CUSTOMERS
-- Senders with the most transactions in the trailing 24h window,
-- and whether high velocity co-occurs with fraud.
-- ---------------------------------------------------------------------
SELECT
    sender_account,
    COUNT(*) AS transactions_in_sample,
    MAX(txn_count_last_24h) AS max_velocity_24h,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN 1.0 ELSE 0 END), 4) AS fraud_rate,
    ROUND(SUM(amount_ngn), 2) AS total_value_ngn
FROM transactions
GROUP BY sender_account
HAVING MAX(txn_count_last_24h) >= 7          -- top-decile velocity threshold (Phase 4)
ORDER BY max_velocity_24h DESC, fraud_rate DESC
LIMIT 25;


-- ---------------------------------------------------------------------
-- 4. HIGH-RISK MERCHANTS
-- Merchant categories ranked by fraud rate, with minimum volume filter
-- to avoid small-sample noise.
-- ---------------------------------------------------------------------
SELECT
    merchant_category,
    COUNT(*) AS transactions,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN 1.0 ELSE 0 END), 4) AS fraud_rate,
    ROUND(AVG(merchant_fraud_rate), 4) AS avg_precomputed_risk_score,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount_ngn ELSE 0 END), 2) AS fraud_value_ngn
FROM transactions
GROUP BY merchant_category
HAVING COUNT(*) >= 300
ORDER BY fraud_rate DESC
LIMIT 10;


-- ---------------------------------------------------------------------
-- 5. FRAUD EXPOSURE
-- Overall monetary exposure: total fraud value, average fraud amount,
-- and exposure broken out by fraud_type.
-- ---------------------------------------------------------------------
SELECT
    fraud_type,
    COUNT(*) AS fraud_transactions,
    ROUND(SUM(amount_ngn), 2) AS total_exposure_ngn,
    ROUND(AVG(amount_ngn), 2) AS avg_fraud_amount_ngn,
    ROUND(MAX(amount_ngn), 2) AS max_single_fraud_ngn
FROM transactions
WHERE is_fraud = 1
GROUP BY fraud_type
ORDER BY total_exposure_ngn DESC;

-- Overall exposure summary (single row)
SELECT
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN is_fraud=1 THEN 1 ELSE 0 END) AS total_fraud_transactions,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amount_ngn ELSE 0 END), 2) AS total_fraud_exposure_ngn,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amount_ngn ELSE 0 END) * 1.0
          / SUM(amount_ngn), 4) AS fraud_share_of_total_value
FROM transactions;


-- ---------------------------------------------------------------------
-- 6. GEOGRAPHIC PATTERNS
-- Fraud rate and exposure by location (city).
-- ---------------------------------------------------------------------
SELECT
    location,
    COUNT(*) AS transactions,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN 1.0 ELSE 0 END), 4) AS fraud_rate,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount_ngn ELSE 0 END), 2) AS fraud_value_ngn
FROM transactions
GROUP BY location
ORDER BY fraud_rate DESC;


-- ---------------------------------------------------------------------
-- BONUS: DEVICE ANOMALY CHECK (Phase 4/9 finding, validated in SQL)
-- Confirms: fraud is concentrated entirely on new-device transactions.
-- ---------------------------------------------------------------------
SELECT
    new_device_transaction,
    COUNT(*) AS transactions,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN 1.0 ELSE 0 END), 4) AS fraud_rate
FROM transactions
GROUP BY new_device_transaction
ORDER BY fraud_rate DESC;
