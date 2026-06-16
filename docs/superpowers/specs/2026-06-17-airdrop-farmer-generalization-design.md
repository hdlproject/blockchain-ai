# Airdrop Farmer Detection — Generalization Design Spec
**Date:** 2026-06-17
**Builds on:** `2026-06-08-airdrop-farmer-design.md`

## Overview

The current airdrop farmer detector only works against one airdrop at a time: `AirdropConfig` holds a single `contract_address` + `date`, and two of the seven model features (`tx_count_pre_airdrop`, `claim_to_withdraw_hours`) are defined relative to that one contract/date. Serving requires the same fixed contract/date the model was seeded with, so the system can only ever answer "is this wallet farming *this specific* airdrop."

This spec generalizes the pipeline so that:
1. **Training** can pull from any number of known airdrop contracts to build a richer, farmer-dense training population (still config-driven for now; automatic discovery of contracts from an external source is a deferred follow-up — see Out of Scope).
2. **Serving** scores *any* wallet address generically — no contract address or date required at request time. Feature extraction becomes fully wallet-intrinsic, anchored on the wallet's own transaction/token history instead of an external event.
3. **Shared-funder detection** becomes a persistent, cross-airdrop signal instead of being scoped to one seed run's cohort and frozen into that run's model artifact.

---

## Architecture

```
configs/airdrop-farmer.yaml
  airdrop:
    contract_addresses: [list of strings]   # was: single contract_address + date

src/blockchain_ai/
  database/
    job_store.py                ← unchanged
    funder_ledger.py             ← NEW: persistent sqlite store (same style as job_store.py)
  feature/
    airdrop_features.py          ← rewritten: drops contract_address/airdrop_date params;
                                    takes a funder_count_lookup predicate instead
  model/
    gmm_wrapper.py                ← drop funding_address_set param/field entirely
  server/
    router_airdrop_farmer.py     ← _run_job also records the wallet's funder into the ledger
  workflow/
    run_airdrop_farmer_seed.py   ← loops over contract_addresses, unions+dedupes callers,
                                    records funders into the ledger, no per-row contract coupling

app.py                           ← drop _cfg.airdrop.date / contract_address wiring;
                                    AirdropFeatureExtractor(client, funder_ledger) only
```

**Key shift:** contract addresses are now used *only* by the seed step, to build a candidate pool of wallets to train on. Once a wallet's feature vector is computed, nothing in the feature set, the fitted model, or the serving path references any contract address — scoring is purely a function of the wallet's own on-chain history. This is what allows `POST /airdrop-farmer/analyze/{address}` to score any address, not just claimants of one specific airdrop.

The funder ledger is the other half of "cross-airdrop": instead of being baked into one model's `.joblib` artifact (frozen at fit time, scoped to one cohort), it is a standalone, ever-growing sqlite table updated by both the seed step and live serving traffic, so the shared-funder signal keeps improving over time regardless of how many airdrops have been seeded.

---

## Configuration (`configs/airdrop-farmer.yaml`)

```yaml
airdrop:
  contract_addresses:
    - "0x..."     # any number of known airdrop contracts, used to build the training pool
    - "0x..."

etherscan:
  base_url: https://api.etherscan.io/v2/api
  chain_id: 1
  rate_limit_per_sec: 5
  timeout_sec: 30

ingest:
  feature_cols:
    - wallet_age_days
    - tx_count_before_first_inflow
    - token_type_diversity
    - inflow_to_outflow_hours
    - shared_funder_score
    - inter_tx_time_variance
    - unique_counterparty_count
  fill_zero_cols:
    - inflow_to_outflow_hours
    - inter_tx_time_variance

train:
  model_type: gmm
  hyperparameters:
    n_components: 4
    covariance_type: full
    random_state: 42

serve:
  model_path: models/airdrop_farmer_gmm.joblib
  db_path: data/jobs.db          # FunderLedger reuses this sqlite file (new table)
```

`AirdropConfig.date` is removed — nothing downstream needs an airdrop date anymore, since no feature is computed relative to an external event date.

---

## Feature Engineering

| Feature | Description | Limitation |
|---|---|---|
| `wallet_age_days` | Days between wallet's first transaction and now. Unchanged. | — |
| `tx_count_before_first_inflow` | Count of transactions strictly before the wallet's **first-ever inbound ERC-20 token transfer** (any token, not a specific airdrop contract). If the wallet has never received a token, falls back to total tx count (neutral — not treated as suspicious by itself). | A wallet whose very first action is a token receipt (no prior tx history at all) reads identically to a long-lived wallet receiving its first token after years of activity, if the prior activity happens to be small; magnitude still distinguishes these in practice via this feature combined with `wallet_age_days`. |
| `token_type_diversity` | Count of unique token contract addresses appearing in the wallet's token-transfer history. Unchanged. | — |
| `inflow_to_outflow_hours` | For each distinct token the wallet received, compute hours between that token's first inbound transfer and its first outbound transfer of the *same* token afterward. Take the **minimum** across all tokens received (fastest flip = most suspicious). No qualifying token at all → imputed to 0 via `fill_zero_cols` (absence of any token activity already reads as a farmer signal, consistent with prior convention). | If a wallet receives many unrelated tokens (e.g. spam airdrops it never asked for) and dumps just one of them quickly, this feature will read as "fast flip" even if the wallet is otherwise a long-term holder of its main assets. |
| `shared_funder_score` | `log1p(N)` where `N` = number of *other* distinct wallets funded by this wallet's funding address (the `from` of its earliest inbound value-bearing transaction), looked up from the persistent `FunderLedger` across all airdrops and live traffic seen so far. Higher = stronger evidence the wallet is one of many controlled by the same operator. Log-transformed so a funder of 5 wallets and a funder of 5,000 don't sit at wildly different scales relative to typical sybil-ring sizes. | Does not distinguish a sybil operator's funding wallet from a legitimate high-fan-out source (e.g. an exchange withdrawal hot wallet) — both produce a high `funded_wallet_count`. This will inflate scores for wallets funded directly from exchanges. Out of scope for now; a future fix would exclude known exchange/CEX addresses from the ledger. |
| `inter_tx_time_variance` | Population variance of gaps between consecutive tx timestamps. Unchanged. | — |
| `unique_counterparty_count` | Count of unique counterparty addresses across the wallet's tx history. Unchanged. | — |

`compute_airdrop_features` signature:

```python
def compute_airdrop_features(
    address: str,
    txs: list[dict],
    token_txs: list[dict],
    funder_count_lookup: Callable[[str], int],
) -> dict[str, float]:
```

`funder_count_lookup` is called with the wallet's own derived funder address and returns the count of *other* wallets that funder has funded (per `FunderLedger.funded_count`). Keeping this as an injected callable (rather than the function reaching into the ledger directly) keeps `compute_airdrop_features` pure and unit-testable with stub lookups.

---

## Funder Ledger

`src/blockchain_ai/database/funder_ledger.py`, mirroring the existing `JobStore` sqlite pattern:

```python
class FunderLedger:
    def __init__(self, db_path: str): ...

    def record(self, funder: str, wallet: str) -> None
        # INSERT OR IGNORE INTO funder_ledger(funder, wallet) — PRIMARY KEY(funder, wallet)

    def funded_count(self, funder: str, exclude_wallet: str) -> int
        # SELECT COUNT(DISTINCT wallet) FROM funder_ledger WHERE funder = ? AND wallet != ?
```

Table schema:

```sql
CREATE TABLE IF NOT EXISTS funder_ledger (
    funder TEXT NOT NULL,
    wallet TEXT NOT NULL,
    PRIMARY KEY (funder, wallet)
)
```

Both the offline seed step and the online `_run_job` call `record()` for every wallet they process (the funder is derived the same way as today: the `from` of the wallet's earliest inbound value-bearing transaction), then call `funded_count()` to compute `shared_funder_score`. This preserves the existing two-pass shape in `run_airdrop_farmer_seed.py` (record all funders first, then compute the shared-funder feature for every row), just against a persistent table instead of an in-run dict — so a funder seen in one seeded airdrop correctly contributes to `shared_funder_score` when one of its other funded wallets is scored later, whether via a later seed run or a live `/analyze` call.

The ledger lives in the same sqlite file as `JobStore` (`serve.db_path`), as a separate table, to avoid managing an extra database file.

---

## Model Changes

- `GMMWrapper.fit()` drops the `funding_address_set` parameter; `self.funding_address_set` field is removed entirely. The funder ledger lives outside the model artifact — it's infrastructure state, not a statistical fit parameter — so the `.joblib` now contains purely the scaler, GMM, farmer cluster indices, and BIC scores.
- `_identify_farmer_clusters` is unchanged in logic — `shared_funder_score`'s direction is still `"high"` in the `farmer_signals` map, and the per-cluster-mean min-max normalization already handles continuous-valued features the same way it handled the old boolean.

---

## Serving (`router_airdrop_farmer.py`, `app.py`)

- `app.py` no longer requires `_cfg.airdrop.date` or a single `contract_address`. `AirdropFeatureExtractor` is constructed as `AirdropFeatureExtractor(client, funder_ledger)` — no contract/date args.
- `_run_job` in `router_airdrop_farmer.py` additionally calls `funder_ledger.record(funder, address)` after extracting features for a wallet, so live traffic keeps contributing to the cross-airdrop signal.

---

## Workflow (`run_airdrop_farmer_seed.py`)

1. For each `contract_address` in `cfg.airdrop.contract_addresses`, fetch its transaction list and extract caller addresses (same as today, per contract).
2. Union and dedupe caller addresses across all configured contracts into a single wallet pool — a wallet that claimed two different airdrops is processed once.
3. Pass 1: for each wallet, fetch `txs`/`token_txs`, derive its funder, call `funder_ledger.record(funder, wallet)`.
4. Pass 2: for each wallet, compute `shared_funder_score` via `funder_ledger.funded_count(funder, wallet)`, then call `compute_airdrop_features(...)` to get the full feature row.
5. Fit `GMMWrapper` on the combined pool exactly as today (BIC loop, then final fit).
6. Save model + write `wallet_scores.csv`, unchanged in shape aside from the renamed feature columns.

---

## Migration

The existing `models/airdrop_farmer_gmm.joblib` is incompatible with this change (different feature semantics for two columns, dropped `funding_address_set` field). There is no backward-compatibility shim — the seed step must be re-run to regenerate the model artifact after this change ships.

---

## Testing

- **`test_airdrop_features.py`** — rewritten for the new `compute_airdrop_features` signature; new cases cover: multi-token min-gap `inflow_to_outflow_hours` logic, the "no inbound transfers" fallback for `tx_count_before_first_inflow`, and `shared_funder_score` via a stub `funder_count_lookup`.
- **`test_funder_ledger.py`** (new) — `record`/`funded_count` round-trip, including the case where a funder has only funded the wallet being queried (count excludes self → 0).
- **`test_router_airdrop_farmer.py`** — updated `feature_cols` list; new case asserting `_run_job` records the wallet's funder into the ledger.
- **`run_airdrop_farmer_seed.py`** integration-level check — a wallet appearing in two configured contracts' caller lists is only processed/scored once.

## Error Handling

Unchanged from the original spec: per-wallet fetch failures during the seed step are logged and skipped (abort if >50% fail); missing-claim and single-tx-variance defaults follow the same imputation conventions as before, just under the renamed features.

---

## Out of Scope

- **Automatic discovery of airdrop contract addresses** from an external tracking source (to replace the manually configured `contract_addresses` list). Deferred to a follow-up spec once a specific provider is researched and verified.
- Excluding known exchange/CEX addresses from the funder ledger to reduce `shared_funder_score` false positives.
- Multi-chain support (chain is configured in YAML, as today).
- UI / dashboard for results.
