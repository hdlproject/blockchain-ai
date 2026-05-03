# Address Classification Model — Design Spec

## Goal

Train an XGBoost multi-class classifier that labels Ethereum addresses as `sanctioned`, `scammer`, or `phishing` using on-chain features derived from Etherscan. Expose predictions via an async REST API where the address itself is the idempotency key.

---

## Architecture Overview

Two independent workflows share the same config file and feature extraction logic:

**Offline (training):**
```
collect_labels.py → collect_address_features.py → run_pipeline.py → model.joblib
```

**Online (inference):**
```
GET /predict/address/{address}
  ├── known address (SQLite hit) → return cached result
  └── unknown address → enqueue background job → return 202 pending
        └── job: Etherscan fetch → compute features → model.predict → write to SQLite
```

---

## Config

Single config file `configs/address-classifier.yaml`. `PipelineConfig` gains three new optional fields (`goplus`, `ofac`, `forta`) populated only when `task = "classification"`. Existing gas price config (`configs/ethereum-gas-price.yaml`) is unchanged.

```yaml
task: classification

goplus:
  base_url: https://api.gopluslabs.io/api/v1
  chain_id: 1
  rate_limit_per_sec: 2
  timeout_sec: 30

ofac:
  alt_url: https://www.treasury.gov/ofac/downloads/alt.csv
  timeout_sec: 60

forta:
  graphql_url: https://api.forta.network/graphql
  timeout_sec: 30
  max_alerts: 500
  scam_bot_ids:
    - "0x6aa2012744a3eb210fc4e4b794d9df59684d36d502fd9ebb481d58f19b917d51"
    - "0x4c7e56a9a753e29ca92bd57dd593bdab0c03e762bdd04e2bc578cb82b842c1f3"

etherscan:
  base_url: https://api.etherscan.io/v2/api
  chain_id: 1
  rate_limit_per_sec: 5
  timeout_sec: 30

ingest:
  feature_cols:
    - tx_count
    - account_age_days
    - unique_counterparties
    - avg_tx_value_eth
    - failed_tx_ratio
    - contract_creation_count
    - erc20_token_count
    - incoming_to_outgoing_ratio
    - is_contract
    - hour_entropy
    - gas_price_avg_gwei
  fill_zero_cols:
    - failed_tx_ratio
    - contract_creation_count
    - erc20_token_count
  target_col: label

train:
  target_col: label
  model_type: xgboost
  test_size: 0.2
  hyperparameters:
    n_estimators: 300
    learning_rate: 0.05
    max_depth: 6
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42

serve:
  model_path: models/address_classifier.joblib
  confidence_threshold: 0.5
  db_path: data/jobs.db
```

### Config changes to `config.py`

- `PipelineConfig` gains: `task: str = "regression"`, `goplus: GoPlusConfig | None`, `ofac: OFACConfig | None`, `forta: FortaConfig | None`
- `ServeConfig` gains: `confidence_threshold: float | None`, `db_path: str | None`; existing regression fields (`title`, `fields`, `log_transform`, etc.) become optional
- `load_config()` validates: when `task = "classification"`, require `goplus`, `ofac`, `forta`, `serve.confidence_threshold`, `serve.db_path`; when `task = "regression"`, require `serve.title`, `serve.fields`

---

## Features

All 11 features are computed from two Etherscan calls per address.

| Feature | Source | Description |
|---------|--------|-------------|
| `tx_count` | `txlist` | Total number of transactions |
| `account_age_days` | `txlist` | Days from first tx to now |
| `unique_counterparties` | `txlist` | Count of unique `to`/`from` addresses |
| `avg_tx_value_eth` | `txlist` | Mean transaction value in ETH |
| `failed_tx_ratio` | `txlist` | Fraction of txs where `isError = 1` |
| `contract_creation_count` | `txlist` | Txs where `to` is empty (contract deploys) |
| `incoming_to_outgoing_ratio` | `txlist` | Count(received) / (count(sent) + 1) |
| `gas_price_avg_gwei` | `txlist` | Mean gas price in Gwei |
| `hour_entropy` | `txlist` | Shannon entropy of tx hour distribution (0 = all same hour, high = random) |
| `is_contract` | `txlist` | 1 if address appears as `contractAddress` in any tx, else 0 |
| `erc20_token_count` | `tokentx` | Count of unique ERC-20 token contracts interacted with |

### Etherscan client extensions

Two new methods added to the existing `EtherscanClient`:

```python
def get_tx_list(self, address: str) -> list[dict]
def get_token_transfers(self, address: str) -> list[dict]
```

Both use the existing `_get()` method with `module=account`.

---

## Training Pipeline (Offline)

Run in order:

### Step 1 — collect_labels.py
Fetches GoPlus + OFAC + Forta, writes `data/processed/labels/addresses.csv` with columns: `address, chain_id, label, confidence, sources, flags, fetched_at`. Labels are `sanctioned`, `scammer`, `phishing` only.

> **Note:** `collect_labels.py` (defined in the label collection plan) is updated to use `load_config()` instead of `load_label_config()`, reading `config.goplus`, `config.ofac`, `config.forta` from the unified config. `LabelPipelineConfig` and `load_label_config()` are removed.

### Step 2 — collect_address_features.py (new)
Reads each address from the labels CSV, calls `get_tx_list()` + `get_token_transfers()` via `EtherscanClient`, computes all 11 features via `AddressFeatureExtractor`, joins with label, writes `data/processed/features/address_features.csv`.

### Step 3 — run_pipeline.py (extended)
Reads config `task`. When `"classification"`:
- Reads `data/processed/features/address_features.csv`
- Encodes labels: `sanctioned=0`, `scammer=1`, `phishing=2`
- Trains XGBoost with `objective="multi:softprob"`, `num_class=3`
- Evaluates with accuracy + per-class F1
- Saves model to `serve.model_path`

---

## Inference Pipeline (Online)

### Job Store — `src/blockchain_ai/job_store.py`

SQLite wrapper around a single `jobs` table:

```sql
CREATE TABLE IF NOT EXISTS jobs (
    address    TEXT PRIMARY KEY,
    status     TEXT NOT NULL,   -- pending | done | failed
    result     TEXT,            -- JSON string when done
    error      TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
```

Methods:
- `get(address) -> dict | None`
- `create_pending(address) -> None`
- `mark_done(address, result: dict) -> None`
- `mark_failed(address, error: str) -> None`

### Feature Extractor — `src/blockchain_ai/address_features.py`

```python
class AddressFeatureExtractor:
    def __init__(self, client: EtherscanClient): ...
    def extract(self, address: str) -> dict[str, float]
```

Returns a dict of all 11 features. Used both in `collect_address_features.py` (training) and the inference background job (serving).

### Router — `src/blockchain_ai/router_address.py`

Single endpoint:

```
GET /predict/address/{address}
```

**Logic:**
1. Normalize address to lowercase
2. Look up in job store
3. If `None`: call `job_store.create_pending(address)`, enqueue `BackgroundTasks`, return 202
4. If `status = "pending"`: return 202
5. If `status = "done"`: return 200 with result
6. If `status = "failed"`: return 200 with error

**Background task:**
1. `AddressFeatureExtractor.extract(address)`
2. `predict_address(features, model, feature_cols, threshold)`
3. `job_store.mark_done(address, result)` or `mark_failed(address, error)`

### Prediction — `predict.py` extension

Label mapping is a hardcoded constant (classes are fixed):

```python
LABEL_ENCODER = {0: "sanctioned", 1: "scammer", 2: "phishing"}
```

New function:

```python
def predict_address(
    features: dict[str, float],
    model,
    feature_cols: list[str],
    threshold: float,
) -> dict:
```

Returns:
```json
{
  "label": "scammer",
  "probabilities": {
    "sanctioned": 0.03,
    "scammer": 0.81,
    "phishing": 0.12
  }
}
```

If `max(probabilities) < threshold`, `label` is overridden to `"unknown"`.

### app.py

At startup, checks `config.task`. When `"classification"`: instantiates `JobStore`, loads model, mounts `router_address` with dependencies injected. Regression and classification can run as separate server instances from their respective config files.

---

## API Response Shapes

**202 — pending:**
```json
{"address": "0xabc...", "status": "pending"}
```

**200 — done:**
```json
{
  "address": "0xabc...",
  "status": "done",
  "label": "scammer",
  "probabilities": {
    "sanctioned": 0.03,
    "scammer": 0.81,
    "phishing": 0.12
  }
}
```

**200 — unknown (low confidence):**
```json
{
  "address": "0xabc...",
  "status": "done",
  "label": "unknown",
  "probabilities": {
    "sanctioned": 0.20,
    "scammer": 0.35,
    "phishing": 0.45
  }
}
```

**200 — failed:**
```json
{"address": "0xabc...", "status": "failed", "error": "Etherscan timeout"}
```

---

## File Map

| Action | Path |
|--------|------|
| Create | `configs/address-classifier.yaml` |
| Modify | `src/blockchain_ai/config.py` |
| Modify | `src/blockchain_ai/etherscan.py` |
| Modify | `src/blockchain_ai/train.py` |
| Modify | `src/blockchain_ai/predict.py` |
| Modify | `src/blockchain_ai/evaluate.py` |
| Modify | `app.py` |
| Create | `src/blockchain_ai/address_features.py` |
| Create | `src/blockchain_ai/job_store.py` |
| Create | `src/blockchain_ai/router_address.py` |
| Create | `scripts/collect_address_features.py` |
| Create | `tests/test_address_features.py` |
| Create | `tests/test_job_store.py` |
| Create | `tests/test_router_address.py` |
