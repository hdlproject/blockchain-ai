# LSTM Gas Price Predictor v2 — Design

**Date:** 2026-05-17
**Status:** Approved

## Problem

The existing XGBoost gas price model (`/predict/gas-price`) treats each block row as independent, ignoring sequential dependencies. Ethereum base fees are a time series — each block's fee is mechanically derived from the previous one (EIP-1559). A sequence model captures this dependency and improves multi-step forecasting accuracy.

## Goal

Add an LSTM-based gas price predictor exposed under `/predict/gas-price/v2/latest`, keeping all existing v1 code and API untouched.

---

## Architecture

Three new files, two small modifications to existing files:

```
src/blockchain_ai/
  model/
    __init__.py
    lstm.py                              ← LSTMWrapper + _LSTMNet

configs/
  ethereum-gas-price-predictor-v2.yaml  ← model_type: lstm, LSTM hyperparams

src/blockchain_ai/server/
  router_gas_price_v2.py                ← GET /predict/gas-price/v2/latest
```

Modified:
- `src/blockchain_ai/train.py` — adds `elif model_type == "lstm"` branch
- `app.py` — registers v2 router alongside v1

---

## Component Design

### `model/lstm.py`

**`_LSTMNet(nn.Module)`**
- Input → LSTM layers → linear head → scalar output
- `input_size`: number of feature columns
- `hidden_size`, `num_layers`, `dropout`: configurable

**`LSTMWrapper`** (sklearn-compatible)

Constructor parameters (all become `hyperparameters` dict keys in config):
| Key | Default | Description |
|---|---|---|
| `sequence_length` | 30 | Lookback window in blocks (~6 min) |
| `hidden_size` | 64 | LSTM hidden state size |
| `num_layers` | 2 | Stacked LSTM layers |
| `dropout` | 0.2 | Dropout between LSTM layers |
| `epochs` | 50 | Max training epochs |
| `lr` | 0.001 | Adam learning rate |
| `batch_size` | 32 | Training batch size |

**`.fit(X, y)`**
1. Build sliding windows: `(n_samples - sequence_length)` sequences of shape `(sequence_length, n_features)`
2. Split last 10% as validation set for early stopping
3. Train with Adam optimizer + MSE loss
4. Stop early if validation loss does not improve for 5 consecutive epochs
5. Store `feature_cols` and `sequence_length` as instance attributes

**`.predict(X)`**
- Takes last `sequence_length` rows of X as context
- Returns numpy array of shape `(1,)` — one prediction per call
- Compatible with existing `evaluate.py` (`joblib.load(model_path).predict(X)`)

**Serialization:** `joblib.dump` / `joblib.load` — PyTorch modules are picklable.

---

### `configs/ethereum-gas-price-predictor-v2.yaml`

Same top-level structure as v1. Key differences:

```yaml
train:
  model_type: lstm
  hyperparameters:
    sequence_length: 30
    hidden_size: 64
    num_layers: 2
    dropout: 0.2
    epochs: 50
    lr: 0.001
    batch_size: 32

serve:
  model_path: models/gas_price_predictor_v2.joblib
```

All other sections (etherscan, collect, ingest, paths) identical to v1.

---

### `server/router_gas_price_v2.py`

Single endpoint:

```
GET /predict/gas-price/v2/latest?n_blocks=1
```

**Flow:**
1. Fetch `sequence_length + n_blocks` blocks from Etherscan
2. Feed last `sequence_length` rows as sequence to LSTM → step 1 prediction
3. For steps 2+: append predicted fee to rolling window, slide forward, re-run inference (auto-regression)
4. `sequence_length` is read from `model.sequence_length` at runtime — no config change needed

**Response:** Identical shape to v1 `/predict/gas-price/latest`:
```json
{
  "block_number": 21000000,
  "block_history": [{"block": ..., "base_fee_gwei": ...}],
  "predictions": [{"step": 1, "block_number": ..., "base_fee_gwei": ..., "base_fee_wei": ..., "method": "lstm"}]
}
```

Same response shape means the Streamlit UI works with v2 without modification.

---

### `train.py` modification

```python
elif train_config.model_type == "lstm":
    from blockchain_ai.model.lstm import LSTMWrapper
    model = LSTMWrapper(**train_config.hyperparameters)
```

No other changes to `train.py`, `evaluate.py`, or the pipeline script.

---

### `app.py` modification

`app.py` optionally loads a second model via a `CONFIG_V2` env var:

```
CONFIG=configs/ethereum-gas-price-predictor.yaml \
CONFIG_V2=configs/ethereum-gas-price-predictor-v2.yaml \
uvicorn app:app --reload
```

- If `CONFIG_V2` is set, load its model and register `router_gas_price_v2` (v2 paths)
- If `CONFIG_V2` is unset, only v1 router is registered (existing behaviour, zero change)
- Both routers can coexist on the same server instance sharing the same Etherscan client

---

## Data Flow

```
collect_blocks.py → data/raw/ethereum-blocks.csv
    ↓
ethereum_gas_price_pipeline.py --config configs/ethereum-gas-price-predictor-v2.yaml
    ↓ feature_engineering.py (unchanged)
data/processed/ethereum-blocks.csv
    ↓
train.py → LSTMWrapper.fit(X, y) → models/gas_price_predictor_v2.joblib
    ↓
evaluate.py → reports/gas_price_predictor_v2.json
    ↓
app.py → router_gas_price_v2 → GET /predict/gas-price/v2/latest
```

---

## Testing

- `tests/test_lstm.py` — unit tests for `LSTMWrapper`:
  - `fit` + `predict` round-trip on synthetic data
  - Sliding window shape correctness
  - Early stopping fires when validation loss plateaus
  - `joblib` serialization round-trip preserves weights
- `tests/test_router_gas_price_v2.py` — router tests with mocked model and Etherscan client:
  - Returns correct response shape
  - Handles missing Etherscan client (503)
  - Handles unloaded model (503)

---

## Out of Scope

- Manual single-row endpoint (`POST /predict/gas-price/v2`) — LSTM requires a sequence; not supported in v2
- Hyperparameter tuning (HPO) for LSTM — can be added later via the existing `tune.py` pattern
- Streamlit UI changes — v2 response is shape-compatible with v1, UI works as-is
