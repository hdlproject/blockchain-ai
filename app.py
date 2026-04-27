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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import Field, create_model

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))
from blockchain_ai.config import FieldConfig, ServeConfig, load_config

_CONFIG_PATH = os.environ.get("CONFIG", "configs/ethereum-gas-price.yaml")
_cfg = load_config(_CONFIG_PATH)

if _cfg.serve is None:
    raise RuntimeError(f"Config at {_CONFIG_PATH} is missing a 'serve' section.")

serve: ServeConfig = _cfg.serve
feature_cols: list[str] = _cfg.ingest.feature_cols

app = FastAPI(title=serve.title, description=serve.description, version="0.1.0")

_model_path = os.environ.get("MODEL_PATH", serve.model_path)
if _model_path.startswith("gs://"):
    import subprocess, tempfile
    _tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    subprocess.run(["gsutil", "cp", _model_path, _tmp.name], check=True)
    _model_path = _tmp.name
model = joblib.load(_model_path)


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


def _to_response(wei: float) -> dict:
    result = {
        f"predicted_{serve.target_description.lower().replace(' ', '_')}_wei": wei,
        f"predicted_{serve.target_description.lower().replace(' ', '_')}_gwei": wei / 1e9,
    }
    return result


def _predict_df(df: pd.DataFrame) -> np.ndarray:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required columns: {missing}")
    raw = model.predict(df[feature_cols])
    return np.expm1(raw) if serve.log_transform else raw


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
