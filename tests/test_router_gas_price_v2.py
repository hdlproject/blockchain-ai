import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blockchain_ai.config import ServeConfig
from blockchain_ai.server.router_gas_price_v2 import create_router

_FEATURE_COLS = ["base_fee_gwei", "gas_used_ratio", "hour_of_day", "day_of_week", "base_fee_trend"]
_SEQ_LEN = 5

_SERVE = ServeConfig(
    model_path="models/gas_price_predictor_v2.joblib",
    title="v2",
    description="test",
    target_description="base fee",
    target_unit="Gwei",
    log_transform=False,
    fields={},
)


def _make_rows(n=40, start_fee=15.0):
    return [
        {
            "block_number": 21_000_000 + i,
            "base_fee_per_gas": int((start_fee + i * 0.01) * 1e9),
            "gas_used_ratio": 0.5,
            "timestamp": 1_700_000_000 + i * 12,
        }
        for i in range(n)
    ]


def _make_model(seq_len=_SEQ_LEN, pred=16.0):
    model = MagicMock()
    model.sequence_length = seq_len
    model.predict.return_value = np.full(seq_len, pred, dtype=np.float32)
    return model


def _make_client(n_rows=40):
    rows = _make_rows(n_rows)
    client = MagicMock()
    client.get_latest_block_number.return_value = rows[-1]["block_number"]
    client.get_block.side_effect = lambda n: next(
        (r for r in rows if r["block_number"] == n), None
    )
    return client


def _app(model=None, client=None, serve=_SERVE):
    app = FastAPI()
    app.include_router(create_router(serve, _FEATURE_COLS, model, client))
    return TestClient(app)


def test_predict_v2_returns_200():
    resp = _app(_make_model(), _make_client()).get("/predict/gas-price/v2/latest")
    assert resp.status_code == 200


def test_predict_v2_response_shape():
    data = _app(_make_model(), _make_client()).get("/predict/gas-price/v2/latest").json()
    assert "block_number" in data
    assert "block_history" in data
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    p = data["predictions"][0]
    assert set(p.keys()) == {"step", "block_number", "base_fee_gwei", "base_fee_wei", "method"}
    assert p["method"] == "lstm"
    assert p["step"] == 1


def test_predict_v2_n_blocks_2_returns_2_predictions():
    data = _app(_make_model(), _make_client()).get("/predict/gas-price/v2/latest?n_blocks=2").json()
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["step"] == 1
    assert data["predictions"][1]["step"] == 2


def test_predict_v2_no_model_returns_503():
    resp = _app(model=None, client=_make_client()).get("/predict/gas-price/v2/latest")
    assert resp.status_code == 503


def test_predict_v2_no_client_returns_503():
    resp = _app(model=_make_model(), client=None).get("/predict/gas-price/v2/latest")
    assert resp.status_code == 503


def test_predict_v2_log_transform_applied():
    serve = ServeConfig(
        model_path="m.joblib",
        title="v",
        description="d",
        target_description="fee",
        target_unit="Gwei",
        log_transform=True,
        fields={},
    )
    raw_pred = 2.0
    model = MagicMock()
    model.sequence_length = _SEQ_LEN
    model.predict.return_value = np.full(_SEQ_LEN, raw_pred, dtype=np.float32)

    data = _app(model, _make_client(), serve).get("/predict/gas-price/v2/latest").json()
    expected = float(np.expm1(raw_pred))
    assert abs(data["predictions"][0]["base_fee_gwei"] - expected) < 1e-4


def test_predict_v2_block_number_is_last_historical():
    data = _app(_make_model(), _make_client()).get("/predict/gas-price/v2/latest").json()
    assert data["block_number"] == 21_000_039
    assert data["predictions"][0]["block_number"] == 21_000_040
