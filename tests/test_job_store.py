import pytest
from blockchain_ai.database.job_store import JobStore


def test_get_returns_none_for_unknown_address(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    assert store.get("0xunknown") is None


def test_create_pending_stores_job(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    store.create_pending("0xabc")
    job = store.get("0xabc")
    assert job is not None
    assert job["status"] == "pending"
    assert job["address"] == "0xabc"


def test_create_pending_is_idempotent(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    store.create_pending("0xabc")
    store.create_pending("0xabc")  # second call must not raise
    assert store.get("0xabc")["status"] == "pending"


def test_mark_done_stores_result(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    store.create_pending("0xdone")
    result = {"label": "scammer", "probabilities": {"sanctioned": 0.1, "scammer": 0.8, "phishing": 0.1}}
    store.mark_done("0xdone", result)
    job = store.get("0xdone")
    assert job["status"] == "done"
    import json
    stored = json.loads(job["result"])
    assert stored["label"] == "scammer"


def test_mark_failed_stores_error(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    store.create_pending("0xfail")
    store.mark_failed("0xfail", "Etherscan timeout")
    job = store.get("0xfail")
    assert job["status"] == "failed"
    assert "Etherscan timeout" in job["error"]


def test_job_store_creates_parent_dirs(tmp_path):
    store = JobStore(str(tmp_path / "nested" / "dir" / "jobs.db"))
    store.create_pending("0xtest")
    assert store.get("0xtest") is not None
