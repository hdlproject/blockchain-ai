import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from blockchain_ai.job_store import JobStore
from blockchain_ai.router_address import create_router

_FEATURE_COLS = ["tx_count", "account_age_days"]


def _app(tmp_path, model=None, feature_extractor=None):
    store = JobStore(str(tmp_path / "jobs.db"))
    app = FastAPI()
    router = create_router(
        job_store=store,
        model=model,
        feature_extractor=feature_extractor,
        feature_cols=_FEATURE_COLS,
        threshold=0.5,
    )
    app.include_router(router)
    return app, store


def test_new_address_returns_202(tmp_path):
    app, _ = _app(tmp_path)
    resp = TestClient(app).get("/predict/address/0x1234567890abcdef1234567890abcdef12345678")
    assert resp.status_code == 202
    assert resp.json()["status"] == "pending"


def test_pending_address_returns_202(tmp_path):
    app, store = _app(tmp_path)
    store.create_pending("0xabc")
    resp = TestClient(app).get("/predict/address/0xabc")
    assert resp.status_code == 202


def test_done_address_returns_200_with_result(tmp_path):
    app, store = _app(tmp_path)
    store.create_pending("0xdone")
    result = {"label": "scammer", "probabilities": {"sanctioned": 0.1, "scammer": 0.8, "phishing": 0.1}}
    store.mark_done("0xdone", result)
    resp = TestClient(app).get("/predict/address/0xdone")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "done"
    assert data["label"] == "scammer"
    assert "probabilities" in data


def test_failed_address_returns_200_with_error(tmp_path):
    app, store = _app(tmp_path)
    store.create_pending("0xfail")
    store.mark_failed("0xfail", "Etherscan timeout")
    resp = TestClient(app).get("/predict/address/0xfail")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert "Etherscan timeout" in data["error"]


def test_address_normalized_to_lowercase(tmp_path):
    app, store = _app(tmp_path)
    TestClient(app).get("/predict/address/0xABCDEF1234567890ABCDEF1234567890ABCDEF12")
    job = store.get("0xabcdef1234567890abcdef1234567890abcdef12")
    assert job is not None
