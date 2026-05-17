# Ethereum Gas Price Predictor

An ML service that predicts Ethereum's next-block base fee in Gwei, served via a FastAPI backend and a Streamlit UI.

---

## Why ML when EIP-1559 has a fixed formula?

EIP-1559 defines the next block's base fee deterministically:

```
next_base_fee = current_base_fee × (1 + (gas_used_ratio − 0.5) / 4)
```

Given a closed block's data, this formula is exact — no model needed. So the honest question is: **when does ML add value here?**

### The formula's free variable is the prediction problem

`current_base_fee` is always known. `gas_used_ratio` is the only stochastic input. Predicting next base fee is therefore equivalent to predicting gas_used_ratio — just expressed in more useful units. The model isn't competing with the formula; it is estimating the formula's unknown input, then outputting the result in Gwei that a user can act on directly.

### Two cases where ML earns its keep

**1. Before the current block closes**
A user submitting a transaction right now doesn't know the current block's final `gas_used_ratio` yet. The model predicts the next base fee from partial block data and historical demand patterns. The formula cannot help here.

**2. N blocks ahead**
The formula is exact for one step but requires `gas_used_ratio[t+1]`, `gas_used_ratio[t+2]`, ... for steps 2 and beyond. These are genuinely unknown. Network demand follows learnable patterns — weekday mornings are more congested, late-night hours are cheaper, utilisation cycles repeat. ML captures these patterns and translates them into a base fee forecast in Gwei that a user can act on: "should I submit now or wait 10 minutes?"

### What this means for the model design

- **Output stays `base_fee_gwei`** — good UX, no domain knowledge required from the user.
- **Internally**, the model approximates the formula's free variable using time-of-day, day-of-week, recent congestion trend, and current utilisation signals.
- **For 1-block-ahead on a closed block**, the formula wins. For pre-close timing and multi-step planning, ML adds genuine value.

The current implementation covers both single-step and multi-step (next N blocks) prediction.

## Multi-step prediction (N blocks ahead)

`GET /predict/latest?n_blocks=N` returns predictions for the next N blocks.

### Step 1 is exact — no model needed

The EIP-1559 base fee for the next block is a **protocol-enforced value**, determined entirely by the latest closed block's data:

```
base_fee[t+1] = base_fee[t] × (1 + (gas_used_ratio[t] − 0.5) / 4)
```

Both inputs are known from the latest finalised block. Step 1 is computed with this formula — zero uncertainty.

### Steps 2+ use auto-regression

From step 2 onward, `gas_used_ratio[t+1], gas_used_ratio[t+2], ...` are unknown. The ML model takes over:

- Predicted `base_fee[t+k]` becomes the `base_fee_gwei` input for step `k+1`
- `gas_used_ratio` is held constant at the last known value
- `hour_of_day` and `day_of_week` are advanced by 12 seconds per block, so the model's time-of-day demand patterns apply
- `base_fee_trend` is recomputed from the growing rolling window of predictions

Accuracy is reasonable for ~5–10 blocks (~1–2 minutes) and degrades beyond that as the gas_used_ratio assumption diverges from reality. The UI signals this with a dashed line for model-predicted steps vs a solid line for the formula step.

---

## Architecture

```
Etherscan API
     │
     ▼
collect_blocks.py   ←── fetches raw block data, derives features, applies target shift
     │
     ▼
run_pipeline.py     ←── ingest → [HPO] → train (XGBoost) → evaluate
     │
     ▼
models/model.joblib + reports/report.json
     │
     ├──► app.py (FastAPI)      ←── /predict, /predict/batch, /predict/latest
     └──► ui/streamlit_app.py   ←── calls FastAPI, renders chart + metrics
```

## Quickstart (local)

```bash
cp .env.example .env          # add ETHERSCAN_API_KEY
./scripts/run_local.sh        # collect → train → serve
```

Streamlit UI: `http://localhost:8501`
FastAPI docs: `http://localhost:8000/docs`

## Deploy to Google Cloud Run

```bash
./scripts/deploy_cloudrun.sh [PROJECT_ID] [REGION] GCS_BUCKET
```

Deploys:
- `gas-predictor` — FastAPI prediction service
- `gas-predictor-ui` — Streamlit UI
- `gas-predictor-retrain` — daily Cloud Run Job (retrains + uploads model + report to GCS)

## Features used by the model

| Feature | Description |
|---|---|
| `base_fee_gwei` | Current block's base fee |
| `gas_used_ratio` | Fraction of block gas limit consumed (0–1) |
| `hour_of_day` | UTC hour — captures intra-day demand cycles |
| `day_of_week` | 0 = Monday — captures weekly demand patterns |
| `base_fee_trend` | 10-block momentum: `(current − 10_blocks_ago) / 10_blocks_ago` |

Target: `base_fee_gwei` of the **next** block (via target shift during data collection).
