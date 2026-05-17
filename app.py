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
    from blockchain_ai.server.router_gas_price import create_router
    app.include_router(create_router(serve, feature_cols, model, _etherscan_client))

elif task == "classification":
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
        _raw_model_path_v2 = os.environ.get("MODEL_PATH_V2", _cfg_v2.serve.model_path)
        model_v2 = None
        try:
            if _raw_model_path_v2.startswith("gs://"):
                import tempfile
                from google.cloud import storage as gcs
                _tmp_v2 = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
                _bucket_v2, _, _blob_v2 = _raw_model_path_v2[5:].partition("/")
                gcs.Client().bucket(_bucket_v2).blob(_blob_v2).download_to_filename(_tmp_v2.name)
                _model_path_v2 = _tmp_v2.name
            else:
                _model_path_v2 = _raw_model_path_v2
            model_v2 = joblib.load(_model_path_v2)
        except Exception as exc:
            print(f"WARNING: v2 model could not be loaded ({exc}). v2 endpoint will return 503.")
        from blockchain_ai.server.router_gas_price_v2 import create_router as create_router_v2
        app.include_router(
            create_router_v2(
                _cfg_v2.serve,
                _cfg_v2.ingest.feature_cols,
                model_v2,
                _etherscan_client,
            )
        )
