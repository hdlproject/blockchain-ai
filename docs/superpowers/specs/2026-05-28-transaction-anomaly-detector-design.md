# Transaction Anomaly Detector — Design Spec

**Date:** 2026-05-28
**Task:** `clustering`
**Algorithm:** DBSCAN
**Use case:** Prevention — flag unusual transactions before submission using raw feature input

---

## 1. Overview

An unsupervised anomaly detector that learns what normal Ethereum transactions look like by sweeping recent blocks, then scores new transactions at inference time. A transaction is flagged as anomalous if it falls outside all dense clusters in feature space. Designed as a prevention step: the user inputs transaction parameters before submitting and receives an anomaly label + score.

---

## 2. Config

New file: `configs/transaction-anomaly-detector.yaml`

Sections reused from existing configs: `etherscan`, `ingest`, `paths`, `serve`.

New `task: clustering` value alongside existing `regression` and `classification`.

`train.model_type: dbscan` with hyperparameters:
- `eps` — DBSCAN neighbourhood radius (default: 0.5)
- `min_samples` — minimum points to form a core (default: 5)

`collect.n_blocks` — number of recent blocks to sweep for training data (default: 500).

`serve.fields` — the 9 input features with types and descriptions (same pattern as `router_gas_price`).

`paths.test_path` — unused for clustering; omitted or left blank.

---

## 3. Data Collection

**Script:** `scripts/run_local_transaction_anomaly_detector.sh`
**Workflow:** `src/blockchain_ai/workflow/collect_transactions.py`
**Output:** `data/raw/transactions.csv`

Steps:
1. Fetch latest block number via `EtherscanClient.get_latest_block_number()`
2. For each of the last `collect.n_blocks` blocks: call `eth_getBlockByNumber` with `boolean: true` to get the full transaction list
3. Extract per-transaction rows
4. Compute sender/receiver window stats (address-level features) from the collected dataset in memory — no additional Etherscan calls
5. Write to `data/raw/transactions.csv`

No new connector needed — reuses `EtherscanClient`.

---

## 4. Features

Computed by `src/blockchain_ai/feature/transaction_features.py` → `TransactionFeatureExtractor`.

| Feature | Description |
|---|---|
| `value_eth` | Transaction value in ETH |
| `gas_price_gwei` | Gas price in Gwei |
| `gas_used` | Gas consumed |
| `input_data_len` | Calldata byte length (0 = simple ETH transfer) |
| `is_contract_call` | 1 if `to` is a contract, 0 otherwise |
| `hour_of_day` | UTC hour of block timestamp (0–23) |
| `sender_tx_count_window` | Sender's transaction count in the collected window |
| `sender_avg_value_eth` | Sender's average transaction value in the collected window |
| `receiver_tx_count_window` | Receiver's transaction count in the collected window |

Address window stats (`sender_*`, `receiver_*`) are built from a lookup table over the collected dataset — no per-address Etherscan lookups at fit or inference time.

---

## 5. Model

**File:** `src/blockchain_ai/model/dbscan_wrapper.py`
**Class:** `DBSCANWrapper`

`__init__` params mirror the config hyperparameters: `eps`, `min_samples`.

`fit(X)`:
1. Fit `StandardScaler` (required — DBSCAN is distance-based, sensitive to scale)
2. Fit `sklearn.cluster.DBSCAN` on scaled data
3. Store core sample vectors
4. Fit `NearestNeighbors` over core sample vectors for new-point inference

`predict(X) -> np.ndarray`:
- Scale X with fitted scaler
- For each point, find distance to nearest core sample
- Distance ≤ `eps` → `0` (normal); distance > `eps` → `-1` (anomaly)

`anomaly_score(X) -> np.ndarray`:
- Returns raw distance to nearest core sample as a continuous score
- Lower score = more normal; score > `eps` = anomaly territory

Saved and loaded via `joblib` (consistent with all other models).

---

## 6. Training & Evaluation

**train.py** — new `task: clustering` branch:
- No `train_test_split` (unsupervised; all collected data is used for fitting)
- Lazy-imports `DBSCANWrapper` (avoids loading sklearn DBSCAN alongside other backends)
- Fits and saves model via `joblib`
- Skips test CSV generation

**evaluate.py** — new clustering branch reports:
- `anomaly_ratio` — fraction of training points DBSCAN labeled as noise
- `n_clusters` — number of dense clusters found
- `n_noise` — raw count of noise points

Report saved to `paths.report_path` (e.g. `reports/transaction_anomaly.json`).

**run_pipeline.py** — clustering guard:
- `--input` flag not required (data comes from `collect_transactions.py`)
- Skips evaluate step gracefully when `test_path` is absent

---

## 7. API

**File:** `src/blockchain_ai/server/router_anomaly.py`

**Endpoint:** `POST /detect/transaction`

Request body: Pydantic model with the 9 feature fields, built dynamically from `serve.fields` (same pattern as `router_gas_price`).

Response:
```json
{
  "anomaly": true,
  "score": 1.34,
  "label": "anomaly"
}
```

`score` is the distance to the nearest core sample. `anomaly` is `true` when `score > eps`.

**app.py** — new `task: clustering` branch mounts `router_anomaly`. Lazy-imports `DBSCANWrapper`.

---

## 8. Streamlit UI

**New tab:** "Transaction Anomaly" (5th tab)

- API URL: `TXANOMALY_API_URL` env var, defaults to `http://localhost:8000`
- Input: number fields for all 9 features
- Button: "Check Transaction"
- Result:
  - `st.error("Anomaly detected")` + score metric when `anomaly: true`
  - `st.success("Normal transaction")` + score metric when `anomaly: false`
- Sidebar: shows `anomaly_ratio` and `n_clusters` from `reports/transaction_anomaly.json` when present

---

## 9. Run Script

**File:** `scripts/run_local_transaction_anomaly_detector.sh`

Three-step pattern consistent with existing scripts:
```
[1/3] Collecting transactions from Etherscan...
[2/3] Running training pipeline...
[3/3] Starting API + Streamlit UI...
```

---

## 10. File Checklist

New files:
- `configs/transaction-anomaly-detector.yaml`
- `src/blockchain_ai/model/dbscan_wrapper.py`
- `src/blockchain_ai/feature/transaction_features.py`
- `src/blockchain_ai/workflow/collect_transactions.py`
- `src/blockchain_ai/server/router_anomaly.py`
- `scripts/run_local_transaction_anomaly_detector.sh`

Modified files:
- `src/blockchain_ai/train.py` — clustering branch
- `src/blockchain_ai/evaluate.py` — clustering branch
- `src/blockchain_ai/workflow/run_pipeline.py` — clustering guard
- `app.py` — clustering router mount
- `ui/streamlit_app.py` — 5th tab + sidebar metrics + new API URL
