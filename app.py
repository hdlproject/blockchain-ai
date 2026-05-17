#!/usr/bin/env python3
"""
FastAPI server built from a pipeline YAML config.
Usage:
  CONFIG=configs/ethereum-gas-price-predictor.yaml uvicorn app:app --reload
  CONFIG=configs/address-classifier.yaml uvicorn app:app --reload
"""
import os
import sys
from pathlib import Path

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))
from blockchain_ai.config import ServeConfig, load_config
from blockchain_ai.connector.etherscan import EtherscanClient

_CONFIG_PATH = os.environ.get("CONFIG", "configs/ethereum-gas-price-predictor.yaml")
_cfg = load_config(_CONFIG_PATH)

if _cfg.serve is None:
    raise RuntimeError(f"Config at {_CONFIG_PATH} is missing a 'serve' section.")

task = _cfg.task
serve: ServeConfig = _cfg.serve
feature_cols: list[str] = _cfg.ingest.feature_cols

app = FastAPI(title=serve.title, description=serve.description, version="0.1.0")

def _load_model(raw_path: str, label: str = "model") -> object | None:
    try:
        if raw_path.startswith("gs://"):
            import tempfile
            from google.cloud import storage as gcs
            tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
            bucket_name, _, blob_path = raw_path[5:].partition("/")
            gcs.Client().bucket(bucket_name).blob(blob_path).download_to_filename(tmp.name)
            local_path = tmp.name
        else:
            local_path = raw_path
        return joblib.load(local_path)
    except Exception as exc:
        print(f"WARNING: {label} could not be loaded ({exc}). Prediction endpoints will be unavailable.")
        return None


_etherscan_client = None
if _cfg.etherscan is not None:
    try:
        _etherscan_client = EtherscanClient.from_config(_cfg.etherscan)
    except Exception as exc:
        print(f"WARNING: Etherscan client could not be initialized ({exc}).")

model = _load_model(os.environ.get("MODEL_PATH", serve.model_path))


@app.get("/health")
def health():
    return {"status": "ok"}


if model is not None and task == "regression":
    from blockchain_ai.server.router_gas_price import create_router
    app.include_router(create_router(serve, feature_cols, model, _etherscan_client))

elif model is not None and task == "classification":
    from blockchain_ai.feature.address_features import AddressFeatureExtractor
    from blockchain_ai.database.job_store import JobStore
    from blockchain_ai.server.router_address import create_router
    _job_store = JobStore(serve.db_path)
    _feature_extractor = AddressFeatureExtractor(_etherscan_client) if _etherscan_client else None
    app.include_router(create_router(_job_store, model, _feature_extractor, feature_cols, serve.confidence_threshold))

_CONFIG_V2_PATH = os.environ.get("CONFIG_V2")
if _CONFIG_V2_PATH:
    _cfg_v2 = load_config(_CONFIG_V2_PATH)
    if _cfg_v2.serve is None:
        print(f"WARNING: CONFIG_V2 config at {_CONFIG_V2_PATH} has no 'serve' section — v2 router skipped.")
    else:
        model_v2 = _load_model(os.environ.get("MODEL_PATH_V2", _cfg_v2.serve.model_path), label="v2 model")
        if model_v2 is not None:
            from blockchain_ai.server.router_gas_price_v2 import create_router as create_router_v2
            app.include_router(
                create_router_v2(
                    _cfg_v2.serve,
                    _cfg_v2.ingest.feature_cols,
                    model_v2,
                    _etherscan_client,
                )
            )
