# Airdrop Farmer Detection — Design Spec
**Date:** 2026-06-08

## Overview

End-to-end airdrop farmer / sybil detection system using Etherscan data and Gaussian Mixture Model (GMM) clustering. Produces a `farmer_score` and `priority_tier` per wallet address, served as an async REST API endpoint consistent with the existing address classifier pattern.

---

## Architecture

Two phases, mirroring the existing address classifier:

1. **Offline seed step** — run once via a workflow script to fetch all wallets that called the airdrop contract, engineer features, fit GMM + StandardScaler, and persist the model artifacts.
2. **Online serving** — a FastAPI async endpoint accepts a single wallet address, loads the pre-fitted model, extracts features, and scores the wallet. Same `JobStore` + `BackgroundTasks` pattern as `router_address.py`.

---

## File Structure

```
configs/
  airdrop-farmer.yaml              ← AIRDROP_CONTRACT_ADDRESS, AIRDROP_DATE, etherscan, GMM params

src/blockchain_ai/
  feature/
    airdrop_features.py            ← AirdropFeatureExtractor: fetches txlist + tokentx, computes 7 features
  model/
    gmm_wrapper.py                 ← GMMWrapper: fit, predict_proba, farmer_score, bic_scores
  server/
    router_airdrop_farmer.py       ← POST /airdrop-farmer/analyze/{address}
                                      GET  /airdrop-farmer/results/{address}
  workflow/
    run_airdrop_farmer_seed.py     ← offline: fetch seed wallets, fit model, save artifacts

models/
  airdrop_farmer_gmm.joblib        ← fitted GMMWrapper (includes scaler + funding_addresses set)

data/airdrop_farmer/
  wallet_scores.csv                ← output of seed step (for reference / BIC scores)
```

`app.py` mounts `router_airdrop_farmer` when `configs/airdrop-farmer.yaml` is present (same lazy-load pattern as existing routers).

No changes to `JobStore` — the existing per-address table and state machine (`pending → done/failed`) are reused as-is.

---

## Configuration (`configs/airdrop-farmer.yaml`)

```yaml
airdrop:
  contract_address: "0x..."       # AIRDROP_CONTRACT_ADDRESS
  date: "2024-01-01"              # AIRDROP_DATE (ISO format)

etherscan:
  base_url: https://api.etherscan.io/v2/api
  chain_id: 1
  rate_limit_per_sec: 5
  timeout_sec: 30

ingest:
  feature_cols:
    - wallet_age_days
    - tx_count_pre_airdrop
    - token_type_diversity
    - claim_to_withdraw_hours
    - gas_source_shared
    - inter_tx_time_variance
    - unique_counterparty_count
  fill_zero_cols:
    - claim_to_withdraw_hours
    - inter_tx_time_variance

train:
  model_type: gmm
  hyperparameters:
    n_components: 4
    covariance_type: full
    random_state: 42

serve:
  model_path: models/airdrop_farmer_gmm.joblib
  db_path: data/jobs.db
```

`ETHERSCAN_API_KEY` stays in `.env` per the existing project convention.

---

## Data Flow

### Offline Seed Step (`run_airdrop_farmer_seed.py`)

1. Call `EtherscanClient.get_tx_list(AIRDROP_CONTRACT_ADDRESS)` to get all transactions sent to the contract.
2. Extract unique `from` addresses — these are all wallets that called `claim()`.
3. For each wallet, call `AirdropFeatureExtractor.extract(address)` to compute the 7 features.
4. Collect all funding wallet addresses (first inbound tx `from` per wallet) into a set — stored in the model artifact for `gas_source_shared` cross-checks during online scoring.
5. Apply `StandardScaler`. Run BIC loop k=2..8 with `GaussianMixture`. Record BIC scores. Fit final model at k=4.
6. Label clusters by centroid feature means: genuine user, casual claimer, light farmer, heavy/sybil farmer.
7. Save `GMMWrapper` (includes fitted scaler + funding_address_set + cluster_labels + bic_scores) to `models/airdrop_farmer_gmm.joblib`.
8. Write `data/airdrop_farmer/wallet_scores.csv` with all seed wallet scores.

### Online Serving (`router_airdrop_farmer.py`)

```
POST /airdrop-farmer/analyze/{address}
  → JobStore.get(address)
  → if None: JobStore.create_pending(address)
             BackgroundTasks.add_task(_run_job, address, ...)
             return 202 {"address": ..., "status": "pending"}
  → if pending: return 202
  → if done:    return 200 {"address": ..., "status": "done", ...result}
  → if failed:  return 200 {"status": "failed", "error": ...}

_run_job(address):
  1. AirdropFeatureExtractor.extract(address)
  2. GMMWrapper.score(features)  ← uses pre-fitted scaler + GMM
     gas_source_shared: check funding wallet against model's funding_address_set
  3. farmer_score = sum of proba for the two farmer clusters
  4. priority_tier = "normal" | "watch" | "deprioritize"
  5. JobStore.mark_done(address, {farmer_score, priority_tier, features, bic_scores})
```

---

## Feature Engineering

| Feature | Source | Notes |
|---|---|---|
| `wallet_age_days` | `txlist` | days between first tx timestamp and today |
| `tx_count_pre_airdrop` | `txlist` | count of txs with timestamp < `AIRDROP_DATE` |
| `token_type_diversity` | `tokentx` | count of unique `contractAddress` values |
| `claim_to_withdraw_hours` | `txlist` + `tokentx` | hours from first tx to `AIRDROP_CONTRACT_ADDRESS` to first outbound `tokentx`; NaN → impute 0 |
| `gas_source_shared` | `txlist` (first inbound tx) | 1 if funding wallet appears in `funding_address_set` from seed step, else 0 |
| `inter_tx_time_variance` | `txlist` | variance of gaps between consecutive tx timestamps; single-tx wallets → 0 |
| `unique_counterparty_count` | `txlist` | count of unique `from` + `to` addresses |

Claim event definition: first transaction where `to == AIRDROP_CONTRACT_ADDRESS` (pull-style Merkle drop pattern).

---

## Modeling

- `GMMWrapper` wraps `sklearn.mixture.GaussianMixture(n_components=4, covariance_type='full', random_state=42)`.
- `StandardScaler` is fitted on the seed population and stored inside the wrapper (same approach as `DBSCANWrapper`'s `RobustScaler`).
- BIC loop: fit k=2..8, store `bic_scores` as `[{"k": 2, "bic": ...}, ..., {"k": 8, "bic": ...}]` — returned in online job results.
- Cluster labelling: inspect centroid feature means post-fit; assign human-readable labels (`genuine_user`, `casual_claimer`, `light_farmer`, `heavy_sybil`).
- Farmer clusters are identified post-fit by centroid inspection: low `claim_to_withdraw_hours` (quick dump), high `gas_source_shared`, low `tx_count_pre_airdrop`, low `wallet_age_days`.
- `farmer_score = sum(predict_proba(X)[farmer_cluster_indices])` — value in [0, 1].

### Priority Tiers

| Tier | Condition |
|---|---|
| `normal` | `farmer_score < 0.4` |
| `watch` | `0.4 ≤ farmer_score ≤ 0.8` |
| `deprioritize` | `farmer_score > 0.8` |

---

## Error Handling

- **Per-wallet fetch failure during seed step**: log warning, skip wallet. If >50% of wallets fail, abort and raise.
- **Missing claim event** (`claim_to_withdraw_hours`): wallet never called the contract → `NaN` → imputed to 0 (treated as instant withdrawal, a farmer signal).
- **Single-transaction wallet** (`inter_tx_time_variance`): variance undefined → imputed to 0.
- **Model not loaded** (seed step not run): API returns 503 `{"error": "Model not available — run seed step first"}`.
- **Online job failure**: `JobStore.mark_failed(address, error)`, returned via `GET /airdrop-farmer/results/{address}`.

---

## Testing

- **Unit — `AirdropFeatureExtractor`**: mocked Etherscan responses for a normal wallet and a sybil-pattern wallet; assert correct feature values.
- **Unit — `GMMWrapper`**: fit on synthetic data (≥50 samples), assert `farmer_score` in [0, 1], assert `bic_scores` has 7 entries.
- **Integration — `JobStore`**: create pending → mark done → retrieve; assert result round-trips correctly.
- **API — `router_airdrop_farmer`**: mock feature extractor + pre-loaded model fixture; assert 202 on first call, 200 with result on second call.

---

## Out of Scope

- Re-clustering on every new wallet (model is fixed after seed step).
- Multi-chain support (chain is configured in YAML).
- UI / dashboard for results.
