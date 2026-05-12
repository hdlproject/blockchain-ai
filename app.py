#!/usr/bin/env python3
"""
FastAPI server built from a pipeline YAML config.
Usage:
  CONFIG=configs/ethereum-gas-price.yaml uvicorn app:app --reload
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

_CONFIG_PATH = os.environ.get("CONFIG", "configs/ethereum-gas-price.yaml")
_cfg = load_config(_CONFIG_PATH)

if _cfg.serve is None:
    raise RuntimeError(f"Config at {_CONFIG_PATH} is missing a 'serve' section.")

task = _cfg.task
serve: ServeConfig = _cfg.serve
feature_cols: list[str] = _cfg.ingest.feature_cols

app = FastAPI(title=serve.title, description=serve.description, version="0.1.0")

_etherscan_client = None
if _cfg.etherscan is not None:
    try:
        _etherscan_client = EtherscanClient.from_config(_cfg.etherscan)
    except Exception as exc:
        print(f"WARNING: Etherscan client could not be initialized ({exc}).")

_raw_model_path = os.environ.get("MODEL_PATH", serve.model_path)
model = None
try:
    if _raw_model_path.startswith("gs://"):
        import tempfile
        from google.cloud import storage as gcs
        _tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
        _bucket_name, _, _blob_path = _raw_model_path[5:].partition("/")
        gcs.Client().bucket(_bucket_name).blob(_blob_path).download_to_filename(_tmp.name)
        _model_path = _tmp.name
    else:
        _model_path = _raw_model_path
    model = joblib.load(_model_path)
except Exception as exc:
    print(f"WARNING: model could not be loaded ({exc}). Prediction endpoints will return 503.")


@app.get("/health")
def health():
    return {"status": "ok"}


if task == "regression":
    from blockchain_ai.router_gas_price import create_router
    app.include_router(create_router(serve, feature_cols, model, _etherscan_client))

elif task == "classification":
    from blockchain_ai.feature.address_features import AddressFeatureExtractor
    from blockchain_ai.job_store import JobStore
    from blockchain_ai.router_address import create_router
    _job_store = JobStore(serve.db_path)
    _feature_extractor = AddressFeatureExtractor(_etherscan_client) if _etherscan_client else None
    app.include_router(create_router(_job_store, model, _feature_extractor, feature_cols, serve.confidence_threshold))
