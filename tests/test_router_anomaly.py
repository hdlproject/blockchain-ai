import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from blockchain_ai.server.router_anomaly import create_router
from blockchain_ai.config import ServeConfig, FieldConfig
from fastapi import FastAPI


def _make_app():
    model = MagicMock()
    model.eps = 0.5
    model.anomaly_score.return_value = np.array([0.8])  # > eps → anomaly

    serve = ServeConfig(
        model_path="models/test.joblib",
        title="Test",
        description="Test",
        fields={
            "value_eth": FieldConfig(type="float", description="value", example=0.5, ge=0.0),
        },
    )
    feature_cols = ["value_eth"]
    app = FastAPI()
    app.include_router(create_router(serve, feature_cols, model))
    return TestClient(app), model


def test_detect_transaction_anomaly():
    client, _ = _make_app()
    resp = client.post("/detect/transaction", json={"value_eth": 9999.0})
    assert resp.status_code == 200
    data = resp.json()
    assert data["anomaly"] is True
    assert data["label"] == "anomaly"
    assert isinstance(data["score"], float)


def test_detect_transaction_normal():
    client, model = _make_app()
    model.anomaly_score.return_value = np.array([0.1])  # < eps → normal
    resp = client.post("/detect/transaction", json={"value_eth": 0.5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["anomaly"] is False
    assert data["label"] == "normal"
