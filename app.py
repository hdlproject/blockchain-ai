#!/usr/bin/env python3
"""
FastAPI server built dynamically from the serve section of the pipeline YAML config.
Usage: CONFIG=configs/ethereum-gas-price.yaml uvicorn app:app --reload
"""
import io
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import Field, create_model

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))
from blockchain_ai.config import FieldConfig, ServeConfig, load_config
from blockchain_ai.etherscan import EtherscanClient

_CONFIG_PATH = os.environ.get("CONFIG", "configs/ethereum-gas-price.yaml")
_cfg = load_config(_CONFIG_PATH)

if _cfg.serve is None:
    raise RuntimeError(f"Config at {_CONFIG_PATH} is missing a 'serve' section.")

serve: ServeConfig = _cfg.serve
feature_cols: list[str] = _cfg.ingest.feature_cols

app = FastAPI(title=serve.title, description=serve.description, version="0.1.0")

_etherscan_client = None
if _cfg.etherscan is not None:
    try:
        _etherscan_client = EtherscanClient.from_config(_cfg.etherscan)
    except Exception as exc:
        print(f"WARNING: Etherscan client could not be initialized ({exc}). /predict/latest will return 503.")
_raw_model_path = os.environ.get("MODEL_PATH", serve.model_path)
model = None
try:
    if _raw_model_path.startswith("gs://"):
        import tempfile
        from google.cloud import storage as gcs
        from google.api_core.exceptions import NotFound
        _tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
        _bucket_name, _, _blob_path = _raw_model_path[5:].partition("/")
        gcs.Client().bucket(_bucket_name).blob(_blob_path).download_to_filename(_tmp.name)
        _model_path = _tmp.name
    else:
        _model_path = _raw_model_path
    model = joblib.load(_model_path)
except Exception as exc:
    print(f"WARNING: model could not be loaded ({exc}). Prediction endpoints will return 503.")


_TREND_LOOKBACK = 10


def _fetch_latest_features() -> pd.DataFrame:
    if _etherscan_client is None:
        raise HTTPException(
            status_code=503,
            detail="Etherscan client not available. Check ETHERSCAN_API_KEY and etherscan config.",
        )
    latest = _etherscan_client.get_latest_block_number()
    rows = []
    for block_num in range(latest - _TREND_LOOKBACK, latest + 1):
        row = _etherscan_client.get_block(block_num)
        if row:
            rows.append(row)
    if not rows:
        raise HTTPException(status_code=503, detail="Could not fetch recent blocks from Etherscan.")
    df = pd.DataFrame(rows)
    df["base_fee_gwei"] = df["base_fee_per_gas"] / 1e9
    df["hour_of_day"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.hour
    df["day_of_week"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.dayofweek
    shifted = df["base_fee_gwei"].shift(_TREND_LOOKBACK)
    df["base_fee_trend"] = ((df["base_fee_gwei"] - shifted) / shifted).fillna(0.0)
    return df


# --- dynamic Pydantic model built from serve.fields ---

_TYPE_MAP = {"float": float, "int": int}

def _pydantic_field(fc: FieldConfig) -> Any:
    constraints: dict[str, Any] = {"description": fc.description, "examples": [fc.example]}
    if fc.ge is not None:
        constraints["ge"] = fc.ge
    if fc.gt is not None:
        constraints["gt"] = fc.gt
    if fc.le is not None:
        constraints["le"] = fc.le
    if fc.lt is not None:
        constraints["lt"] = fc.lt
    return (_TYPE_MAP[fc.type], Field(**constraints))

TransactionModel = create_model(
    "Transaction",
    **{name: _pydantic_field(fc) for name, fc in serve.fields.items()},
)


def _to_response(gwei: float) -> dict:
    result = {
        f"predicted_{serve.target_description.lower().replace(' ', '_')}_wei": gwei * 1e9,
        f"predicted_{serve.target_description.lower().replace(' ', '_')}_gwei": gwei,
    }
    return result


def _predict_df(df: pd.DataFrame) -> np.ndarray:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available yet. The retrain job may not have run.")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required columns: {missing}")
    raw = model.predict(df[feature_cols])
    return np.expm1(raw) if serve.log_transform else raw


def _predict_n_blocks(df: pd.DataFrame, n: int) -> list[dict]:
    last = df.iloc[-1]
    last_block = int(last["block_number"])
    last_timestamp = float(last["timestamp"])
    gas_used_ratio = float(last["gas_used_ratio"])

    # rolling fee window — seeded with historical values for trend lookback
    rolling_fees: list[float] = list(df["base_fee_gwei"].values)

    predictions = []
    for step in range(1, n + 1):
        prev_fee = rolling_fees[-1]
        block_number = last_block + step
        timestamp = last_timestamp + step * 12

        if step == 1:
            base_fee = prev_fee * (1 + (gas_used_ratio - 0.5) / 4)
            method = "formula"
        else:
            dt = pd.Timestamp(timestamp, unit="s", tz="UTC")
            lookback = len(rolling_fees) - 1 - _TREND_LOOKBACK
            if lookback >= 0 and rolling_fees[lookback] > 0:
                trend = (prev_fee - rolling_fees[lookback]) / rolling_fees[lookback]
            else:
                trend = 0.0
            features = pd.DataFrame([{
                "base_fee_gwei": prev_fee,
                "gas_used_ratio": gas_used_ratio,
                "hour_of_day": dt.hour,
                "day_of_week": dt.dayofweek,
                "base_fee_trend": trend,
            }])
            base_fee = float(_predict_df(features)[0])
            method = "model"

        rolling_fees.append(base_fee)
        predictions.append({
            "step": step,
            "block_number": block_number,
            "base_fee_gwei": base_fee,
            "base_fee_wei": base_fee * 1e9,
            "method": method,
        })

    return predictions


# --- endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/predict",
    summary=f"Predict {serve.target_description} for a single transaction",
)
def predict_json(tx: TransactionModel):  # type: ignore[valid-type]
    df = pd.DataFrame([tx.model_dump()])[feature_cols]
    preds = _predict_df(df)
    return _to_response(float(preds[0]))


@app.post(
    "/predict/batch",
    summary=f"Predict {serve.target_description} for multiple transactions via CSV",
    description=(
        f"Upload a CSV with columns: `{'`, `'.join(feature_cols)}`. "
        "Returns predictions in the same row order."
    ),
)
async def predict_csv(file: UploadFile = File(..., description="CSV file with transaction rows.")):
    if not (file.filename or "").endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV: {e}")
    preds = _predict_df(df)
    return JSONResponse({
        "count": len(preds),
        "predictions": [_to_response(float(w)) for w in preds.tolist()],
    })


@app.get(
    "/predict/latest",
    summary=f"Predict {serve.target_description} using live on-chain data",
    description=(
        "Fetches the latest block from Etherscan, computes all features automatically, "
        "and returns a prediction. No input required."
    ),
)
def predict_latest(n_blocks: int = Query(default=1, ge=1, le=50)):
    df = _fetch_latest_features()
    return {
        "block_number": int(df["block_number"].iloc[-1]),
        "block_history": df[["block_number", "base_fee_gwei"]]
            .rename(columns={"block_number": "block"})
            .to_dict(orient="records"),
        "predictions": _predict_n_blocks(df, n_blocks),
    }
