# Streamlit UI Design — Ethereum Gas Price Predictor

**Date:** 2026-05-01
**Status:** Approved

## Overview

A single-file Streamlit frontend (`ui/streamlit_app.py`) that calls the existing FastAPI backend to expose two prediction modes: live on-chain prediction and manual feature input.

## Architecture

Single file: `ui/streamlit_app.py`. Reads `API_URL` from env (default `http://localhost:8000`). Uses `requests` to call the FastAPI backend. Loads model metrics from `reports/report.json` at startup.

```
┌─────────────────────────────────────────────┐
│  Sidebar                                    │
│  ─ App title + description                  │
│  ─ Model Metrics (R², RMSE, MAE)            │
│    loaded from reports/report.json          │
│  ─ API URL config (editable text input)     │
├─────────────────────────────────────────────┤
│  Tab 1: Live Prediction                     │
│  ─ "Fetch latest block & predict" button    │
│  ─ Shows predicted base fee in Gwei + Wei   │
│  ─ Shows block number fetched               │
├─────────────────────────────────────────────┤
│  Tab 2: Manual Prediction                   │
│  ─ 5 input fields (sliders/number inputs)   │
│  ─ "Predict" button                         │
│  ─ Shows predicted base fee in Gwei + Wei   │
└─────────────────────────────────────────────┘
```

## Components

### Sidebar
- Loads `reports/report.json` once at startup; renders R², RMSE, MAE
- If `report.json` is missing: shows `st.warning` instead of crashing
- Editable `API_URL` text input stored in `st.session_state`

### Tab 1 — Live Prediction
- Button: "Fetch latest block & predict"
- On click: `GET {API_URL}/predict/latest` with spinner
- Displays: predicted base fee in Gwei and Wei, plus block number
- On error: `st.error` with message

### Tab 2 — Manual Prediction
- 5 input widgets matching `serve.fields` from `configs/ethereum-gas-price-predictor.yaml`:
  - `base_fee_gwei` → `st.number_input`, min > 0, default 15.0
  - `gas_used_ratio` → `st.slider`, 0.0–1.0, default 0.5
  - `hour_of_day` → `st.slider`, 0–23, default 14
  - `day_of_week` → `st.slider`, 0–6, default 1
  - `base_fee_trend` → `st.number_input`, default 0.02
- Button: "Predict"
- On click: `POST {API_URL}/predict` with field values as JSON, spinner
- Displays: predicted base fee in Gwei and Wei
- On error: `st.error` with message

## Error Handling

| Scenario | Behavior |
|---|---|
| FastAPI not reachable | `st.error("Could not connect to API. Is the FastAPI server running?")` |
| Non-200 API response | `st.error(f"API error {status_code}: {detail}")` |
| `report.json` missing | `st.warning("Model metrics not available")` in sidebar |

## Deployment

- Run locally: `streamlit run ui/streamlit_app.py` (FastAPI must be running separately)
- Override API URL via `API_URL` env var or the sidebar input
- New dependencies: `streamlit`, `requests` — added to `pyproject.toml`
- No Docker changes required for local use

## Files Changed

| File | Change |
|---|---|
| `ui/streamlit_app.py` | New file — the Streamlit app |
| `pyproject.toml` | Add `streamlit` and `requests` dependencies |
