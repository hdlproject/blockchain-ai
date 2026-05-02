import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from ui.streamlit_app import load_metrics, predict_latest, predict_manual


def test_load_metrics_returns_dict(tmp_path):
    report = {"rmse": 0.01, "mae": 0.005, "r2": 0.999}
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report))
    result = load_metrics(p)
    assert result == report


def test_load_metrics_missing_file(tmp_path):
    result = load_metrics(tmp_path / "nonexistent.json")
    assert result is None


def test_load_metrics_invalid_json(tmp_path):
    p = tmp_path / "report.json"
    p.write_text("not valid json")
    result = load_metrics(p)
    assert result is None


def test_predict_latest_calls_correct_endpoint():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "block_number": 12345678,
        "block_history": [{"block": 12345668 + i, "base_fee_gwei": 15.0 + i * 0.1} for i in range(11)],
        "predictions": [
            {"step": 1, "block_number": 12345679, "base_fee_gwei": 15.5,
             "base_fee_wei": 15500000000.0, "method": "formula"},
        ],
    }
    mock_resp.raise_for_status = MagicMock()
    with patch("ui.streamlit_app.requests.get", return_value=mock_resp) as mock_get:
        result = predict_latest("http://localhost:8000")
    mock_get.assert_called_once_with(
        "http://localhost:8000/predict/latest", params={"n_blocks": 1}, timeout=60
    )
    assert result["block_number"] == 12345678
    assert result["predictions"][0]["base_fee_gwei"] == 15.5
    assert result["predictions"][0]["method"] == "formula"


def test_predict_latest_passes_n_blocks():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "block_number": 12345678,
        "block_history": [],
        "predictions": [
            {"step": i + 1, "block_number": 12345679 + i, "base_fee_gwei": 15.0,
             "base_fee_wei": 15000000000.0, "method": "formula" if i == 0 else "model"}
            for i in range(5)
        ],
    }
    mock_resp.raise_for_status = MagicMock()
    with patch("ui.streamlit_app.requests.get", return_value=mock_resp) as mock_get:
        result = predict_latest("http://localhost:8000", n_blocks=5)
    mock_get.assert_called_once_with(
        "http://localhost:8000/predict/latest", params={"n_blocks": 5}, timeout=60
    )
    assert len(result["predictions"]) == 5


def test_predict_manual_calls_correct_endpoint():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "predicted_predicted_next-block_base_fee_gwei": 16.0,
        "predicted_predicted_next-block_base_fee_wei": 16000000000.0,
    }
    mock_resp.raise_for_status = MagicMock()
    payload = {
        "base_fee_gwei": 15.0,
        "gas_used_ratio": 0.5,
        "hour_of_day": 14,
        "day_of_week": 1,
        "base_fee_trend": 0.02,
    }
    with patch("ui.streamlit_app.requests.post", return_value=mock_resp) as mock_post:
        result = predict_manual("http://localhost:8000", payload)
    mock_post.assert_called_once_with(
        "http://localhost:8000/predict", json=payload, timeout=10
    )
    assert result["predicted_predicted_next-block_base_fee_gwei"] == 16.0
