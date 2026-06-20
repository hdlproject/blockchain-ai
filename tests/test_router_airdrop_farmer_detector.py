import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from blockchain_ai.database.job_store import JobStore
from blockchain_ai.server.router_airdrop_farmer_detector import create_router

_FEATURE_COLS = [
    "wallet_age_days", "tx_count_before_first_inflow", "token_type_diversity",
    "inflow_to_outflow_hours", "shared_funder_score",
    "inter_tx_time_variance", "unique_counterparty_count",
]

_RESULT = {
    "farmer_score": 0.85,
    "priority_tier": "deprioritize",
    "bic_scores": [{"k": k, "bic": float(1000 - k * 10)} for k in range(2, 9)],
    "wallet_age_days": 5.0,
    "tx_count_before_first_inflow": 0.0,
    "token_type_diversity": 1.0,
    "inflow_to_outflow_hours": 0.5,
    "shared_funder_score": 1.1,
    "inter_tx_time_variance": 100.0,
    "unique_counterparty_count": 2.0,
}


def _app(tmp_path, model=None, feature_extractor=None):
    store = JobStore(str(tmp_path / "jobs.db"))
    app = FastAPI()
    router = create_router(
        job_store=store,
        model=model,
        feature_extractor=feature_extractor,
        feature_cols=_FEATURE_COLS,
    )
    app.include_router(router)
    return app, store


def test_new_address_returns_202(tmp_path):
    app, _ = _app(tmp_path)
    resp = TestClient(app).get("/airdrop-farmer-detector/analyze/0xabc")
    assert resp.status_code == 202
    assert resp.json()["status"] == "pending"


def test_pending_address_returns_202(tmp_path):
    app, store = _app(tmp_path)
    store.create_pending("0xabc")
    resp = TestClient(app).get("/airdrop-farmer-detector/analyze/0xabc")
    assert resp.status_code == 202


def test_done_address_returns_200_with_result(tmp_path):
    app, store = _app(tmp_path)
    store.create_pending("0xdone")
    store.mark_done("0xdone", _RESULT)
    resp = TestClient(app).get("/airdrop-farmer-detector/analyze/0xdone")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "done"
    assert data["farmer_score"] == pytest.approx(0.85)
    assert data["priority_tier"] == "deprioritize"
    assert len(data["bic_scores"]) == 7


def test_failed_address_returns_200_with_error(tmp_path):
    app, store = _app(tmp_path)
    store.create_pending("0xfail")
    store.mark_failed("0xfail", "Etherscan timeout")
    resp = TestClient(app).get("/airdrop-farmer-detector/analyze/0xfail")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert "Etherscan timeout" in data["error"]


def test_address_normalized_to_lowercase(tmp_path):
    app, store = _app(tmp_path)
    TestClient(app).get("/airdrop-farmer-detector/analyze/0xABCDEF")
    job = store.get("0xabcdef")
    assert job is not None
