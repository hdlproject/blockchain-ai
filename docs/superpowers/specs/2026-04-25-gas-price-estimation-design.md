# Gas Price Estimation — Design Spec

**Date:** 2026-04-25  
**Dataset:** `data/raw/ethereum-transactions.zip` (100k rows, 22 columns)  
**Target:** `gas_price` (Wei, heavily right-skewed)

---

## Goal

Estimate Ethereum `gas_price` using XGBoost regression trained on the existing transaction dataset. Predictions are made in log-space and exponentiated back to Wei for reporting.

---

## 1. Data Ingestion & Feature Engineering (`ingest.py`)

**Drop unusable columns at load time:**
- `hash`, `block_hash`, `from_address`, `to_address`, `input` — identifiers / free text
- `receipt_contract_address`, `receipt_root` — 99.9%–100% null

**Handle nulls selectively:**
- Fill `max_fee_per_gas` and `max_priority_fee_per_gas` with `0` (pre-EIP-1559 transactions legitimately lack these fields)
- No blanket `dropna()` — avoids discarding ~20% of rows unnecessarily

**Timestamp parsing:**
- Parse `block_timestamp` to Unix epoch integer

**Log-transform the target:**
- Add `log_gas_price = log1p(gas_price)`
- Drop raw `gas_price` from the feature set
- Predictions are inverted with `expm1()` at evaluation time

**Output:** `data/processed/ethereum-transactions.csv`

---

## 2. Model Training (`train.py`)

- Load `data/processed/ethereum-transactions.csv`
- Split `X` / `y` where `y = log_gas_price`
- 80/20 train/test split, stratified by `transaction_type`
- Replace `LinearRegression` with `XGBRegressor`:
  - `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`
  - `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`
- `model_type` parameter retained for future extensibility; default changed from `"linear"` to `"xgboost"`
- Save model to `models/model.joblib`
- Save test split to `data/processed/ethereum-transactions-test.csv`

---

## 3. Evaluation (`evaluate.py`)

- Load test split from `data/processed/ethereum-transactions-test.csv`
- Predict in log-space, exponentiate with `expm1()` before computing metrics
- Metrics: RMSE, MAE, R² — all in Wei (interpretable units)
- Report saved to `reports/report.json`

---

## 4. Pipeline & Dependencies

- `run_pipeline.py`: default `--model-type` changes from `"linear"` to `"xgboost"`; structure unchanged
- Add `xgboost` to `pyproject.toml` dependencies

---

## Data Flow

```
ethereum-transactions.zip
        ↓ ingest.py (drop cols, fill nulls, log-transform target)
data/processed/ethereum-transactions.csv  (full processed)
        ↓ train.py (80/20 split, XGBoost on log_gas_price)
models/model.joblib + data/processed/ethereum-transactions-test.csv (test split)
        ↓ evaluate.py (expm1 predictions, RMSE/MAE/R² in Wei)
reports/report.json
```

---

## Out of Scope

- Hyperparameter tuning / cross-validation
- Feature importance analysis
- Real-time inference API
