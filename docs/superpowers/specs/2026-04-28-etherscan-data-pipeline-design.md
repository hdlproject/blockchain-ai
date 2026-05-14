# Etherscan Data Pipeline Design

**Date:** 2026-04-28
**Status:** Approved

## Overview

Replace the existing zip-based transaction dataset with block-level data fetched from the Etherscan API. The model shifts from predicting per-transaction gas price to predicting the next block's `base_fee_per_gas` — the EIP-1559 floor price that any transaction must meet to be included.

## Problem & Motivation

The existing feature set (`value`, `gas`, `max_fee_per_gas`, etc.) consists of fields the sender sets *in response to* network conditions, making them circular predictors. Gas price is driven by network congestion (supply/demand of block space), so the correct features are block-level network state signals. The Etherscan free tier exposes exactly these via `eth_feeHistory`.

## Target

`base_fee_gwei[i+1]` — the next block's base fee in Gwei. Row `i` features predict row `i+1` target. The target shift happens in `collect_blocks.py`; the last row is dropped.

**Why `base_fee` not median gas price:** Deterministic (computed by protocol from previous block utilization), no per-transaction sampling needed, directly actionable for users setting their gas price.

## Features

| Column | Source | Description |
|---|---|---|
| `base_fee_gwei` | block header | Current block's base fee in Gwei |
| `gas_used_ratio` | block header | `gas_used / gas_limit` — congestion 0..1 |
| `base_fee_trend` | derived | `(base_fee[i] - base_fee[i-10]) / base_fee[i-10]` — momentum |
| `hour_of_day` | derived | 0–23 from block timestamp (UTC) |
| `day_of_week` | derived | 0–6 from block timestamp (UTC) |

## Config Changes

### `.env` (new, git-ignored)
```
ETHERSCAN_API_KEY=your_key_here
```

### `ethereum-gas-price-predictor.yaml` additions
```yaml
etherscan:
  base_url: https://api.etherscan.io/api
  rate_limit_per_sec: 5
  timeout_sec: 10

collect:
  n_blocks: 2000
  output_path: data/raw/blocks.csv

ingest:
  feature_cols:
    - base_fee_gwei
    - gas_used_ratio
    - base_fee_trend
    - hour_of_day
    - day_of_week
  fill_zero_cols: []
  target_col: base_fee_gwei   # next block's value after target shift
```

`api_key` is not in the YAML. `EtherscanClient` reads `ETHERSCAN_API_KEY` from the environment directly. `python-dotenv` loads `.env` at startup.

## Components

### `src/blockchain_ai/etherscan.py` (new)

Thin HTTP wrapper. No feature logic.

```
EtherscanClient(base_url, rate_limit_per_sec, timeout_sec)
  ├── get_latest_block_number() -> int
  ├── get_block(block_number) -> dict
  └── get_fee_history(block_count, newest_block) -> list[dict]
```

- Reads `ETHERSCAN_API_KEY` from `os.environ` at construction time
- `time.sleep(1 / rate_limit_per_sec)` between every call
- HTTP non-200 → `RuntimeError`
- Etherscan `"status": "0"` response → `RuntimeError` with `message`
- Pre-EIP-1559 blocks (no `baseFeePerGas`) → skipped silently

### `scripts/collect_blocks.py` (new)

One-shot collector. Reads config, fetches blocks, computes features, writes CSV.

- Uses `eth_feeHistory` (up to 1024 blocks/call → 2 calls for 2000 blocks)
- Computes all derived features
- Applies target shift: `target[i] = base_fee_gwei[i+1]`, drops last row
- Raises if resulting dataset has fewer than 2 rows
- Prints progress to stdout (mirrors `run_pipeline.py` style)

### `src/blockchain_ai/ingest.py` (modified)

- Drops zip-reading logic entirely
- Reads plain CSV from `input_path`
- Interface unchanged: `load_and_clean(input_path, output_path, config)`

### `scripts/run_pipeline.py` (modified)

- `--raw` argument renamed to `--input`

### `ethereum-gas-price-predictor.yaml` (modified)

- `train.stratify_col` removed — the new dataset has no categorical column suitable for stratification; `train_test_split` will use plain random split instead
- `TrainConfig.stratify_col` becomes optional (`str | None`, defaults to `None`)

### `app.py` (modified)

- `load_dotenv()` called at startup
- `serve.fields` updated to match new feature set

## Data Flow

```
.env  ──ETHERSCAN_API_KEY──►  EtherscanClient
                                    │ get_fee_history(n_blocks)  [2 API calls]
                                    ▼
                            collect_blocks.py
                                    │ derives features + target shift
                                    │ writes data/raw/blocks.csv
                                    ▼
                            ingest.load_and_clean()
                                    │ selects feature_cols, adds log_base_fee_gwei
                                    │ writes data/processed/blocks.csv
                                    ▼
                            train.py / tune.py
                                    │ writes models/model.joblib
                                    ▼
                            app.py (FastAPI)
                                    │ user sends current network state
                                    │ returns predicted next-block base_fee (Gwei + Wei)
```

## Error Handling

| Location | Scenario | Behaviour |
|---|---|---|
| `EtherscanClient` | HTTP non-200 | `RuntimeError` with status code |
| `EtherscanClient` | API `status=0` | `RuntimeError` with Etherscan message |
| `EtherscanClient` | Pre-EIP-1559 block | Skip silently |
| `collect_blocks.py` | Fewer blocks than requested | Warn, continue with available |
| `collect_blocks.py` | Dataset < 2 rows after shift | `RuntimeError` |

## Testing

| File | What it tests |
|---|---|
| `tests/test_etherscan.py` | Unit tests with `unittest.mock.patch` on `requests.get`: success, API error, rate limit cadence, pre-EIP-1559 skip |
| `tests/test_collect.py` | Feature derivation (trend, hour, day, target shift) on synthetic block list — no client needed |
| `tests/test_ingest.py` | Updated: plain CSV fixture replaces zip; zip-related assertions removed |

## Dependencies Added

- `python-dotenv` — load `.env` into environment at startup
- `requests` — HTTP calls in `EtherscanClient`
