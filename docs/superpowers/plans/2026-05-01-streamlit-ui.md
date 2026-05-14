# Streamlit UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-file Streamlit UI (`ui/streamlit_app.py`) that calls the existing FastAPI backend for live and manual Ethereum gas price predictions.

**Architecture:** Single-page tabbed layout with a sidebar showing model metrics. Two tabs: Live Prediction (calls `GET /predict/latest`) and Manual Prediction (calls `POST /predict`). Helper functions are pure (no Streamlit calls) so they can be unit-tested with mocked `requests`.

**Tech Stack:** Python 3.12, Streamlit, requests (already in deps), pytest + unittest.mock

---

### Task 1: Add streamlit dependency and expand pytest pythonpath

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `streamlit` to dependencies and `"."` to pythonpath**

In `pyproject.toml`, add `streamlit` to `[project].dependencies`:

```toml
dependencies = [
    "pandas>=2.0.0",
    "scikit-learn>=1.4.0",
    "joblib>=1.3.0",
    "xgboost (>=2.0.0)",
    "pyyaml (>=6.0)",
    "optuna (>=3.0)",
    "plotext (>=5.3.2,<6.0.0)",
    "fastapi (>=0.111.0)",
    "uvicorn[standard] (>=0.29.0)",
    "python-multipart (>=0.0.9)",
    "requests (>=2.33.1,<3.0.0)",
    "python-dotenv (>=1.2.2,<2.0.0)",
    "google-cloud-storage (>=2.0.0)",
    "streamlit (>=1.35.0)",
]
```

And update `[tool.pytest.ini_options]` to include the project root:

```toml
[tool.pytest.ini_options]
pythonpath = ["src", "."]
```

- [ ] **Step 2: Install the new dependency**

Run: `poetry install`
Expected: resolves and installs streamlit and its transitive deps without errors.

- [ ] **Step 3: Verify streamlit is importable**

Run: `python -c "import streamlit; print(streamlit.__version__)"`
Expected: prints a version string like `1.35.0`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml poetry.lock
git commit -m "chore: add streamlit dependency"
```

---

### Task 2: Write failing tests for helper functions

**Files:**
- Create: `ui/__init__.py`
- Create: `tests/test_streamlit_ui.py`

- [ ] **Step 1: Create the `ui` package**

Create `ui/__init__.py` as an empty file.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_streamlit_ui.py`:

```python
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
        "predicted_predicted_next-block_base_fee_gwei": 15.5,
        "predicted_predicted_next-block_base_fee_wei": 15500000000.0,
        "block_number": 12345678,
    }
    mock_resp.raise_for_status = MagicMock()
    with patch("ui.streamlit_app.requests.get", return_value=mock_resp) as mock_get:
        result = predict_latest("http://localhost:8000")
    mock_get.assert_called_once_with("http://localhost:8000/predict/latest", timeout=10)
    assert result["block_number"] == 12345678
    assert result["predicted_predicted_next-block_base_fee_gwei"] == 15.5


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_streamlit_ui.py -v`
Expected: `ModuleNotFoundError: No module named 'ui.streamlit_app'`

---

### Task 3: Implement `ui/streamlit_app.py`

**Files:**
- Create: `ui/streamlit_app.py`

- [ ] **Step 1: Create the app file**

Create `ui/streamlit_app.py`:

```python
import json
import os
from pathlib import Path

import requests
import streamlit as st

API_URL_DEFAULT = os.environ.get("API_URL", "http://localhost:8000")
REPORT_PATH = Path(__file__).parent.parent / "reports" / "report.json"
_TARGET_KEY = "predicted_next-block_base_fee"


def load_metrics(report_path: Path) -> dict | None:
    try:
        with open(report_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def predict_latest(api_url: str) -> dict:
    resp = requests.get(f"{api_url}/predict/latest", timeout=10)
    resp.raise_for_status()
    return resp.json()


def predict_manual(api_url: str, payload: dict) -> dict:
    resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _render_prediction(result: dict) -> None:
    col1, col2 = st.columns(2)
    col1.metric("Predicted Base Fee (Gwei)", f"{result[f'predicted_{_TARGET_KEY}_gwei']:.4f}")
    col2.metric("Predicted Base Fee (Wei)", f"{result[f'predicted_{_TARGET_KEY}_wei']:.0f}")


def main() -> None:
    st.set_page_config(page_title="Ethereum Gas Price Predictor", layout="wide")

    with st.sidebar:
        st.title("Ethereum Gas Price Predictor")
        st.caption("Predicts the next block's base fee using recent network congestion signals.")

        if "api_url" not in st.session_state:
            st.session_state.api_url = API_URL_DEFAULT
        st.session_state.api_url = st.text_input("API URL", value=st.session_state.api_url)

        st.divider()
        st.subheader("Model Metrics")
        metrics = load_metrics(REPORT_PATH)
        if metrics:
            st.metric("R²", f"{metrics['r2']:.4f}")
            st.metric("RMSE", f"{metrics['rmse']:.6f}")
            st.metric("MAE", f"{metrics['mae']:.6f}")
        else:
            st.warning("Model metrics not available")

    tab1, tab2 = st.tabs(["Live Prediction", "Manual Prediction"])

    with tab1:
        st.header("Live Prediction")
        st.write("Fetches the latest Ethereum block and predicts the next base fee automatically.")
        if st.button("Fetch latest block & predict"):
            with st.spinner("Fetching latest block..."):
                try:
                    result = predict_latest(st.session_state.api_url)
                    st.success("Prediction complete")
                    _render_prediction(result)
                    st.info(f"Block number: {result.get('block_number', 'N/A')}")
                except requests.ConnectionError:
                    st.error("Could not connect to API. Is the FastAPI server running?")
                except requests.HTTPError as e:
                    detail = e.response.json().get("detail", str(e)) if e.response else str(e)
                    st.error(f"API error {e.response.status_code}: {detail}")

    with tab2:
        st.header("Manual Prediction")
        st.write("Enter block features manually to get a prediction.")
        base_fee_gwei = st.number_input(
            "Base Fee (Gwei)", min_value=0.001, value=15.0, step=0.1,
            help="Current block's base fee in Gwei (> 0)",
        )
        gas_used_ratio = st.slider(
            "Gas Used Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            help="Fraction of block gas limit consumed (0.0–1.0)",
        )
        hour_of_day = st.slider(
            "Hour of Day (UTC)", min_value=0, max_value=23, value=14,
            help="UTC hour of block timestamp (0–23)",
        )
        day_of_week = st.slider(
            "Day of Week", min_value=0, max_value=6, value=1,
            help="0 = Monday, 6 = Sunday",
        )
        base_fee_trend = st.number_input(
            "Base Fee Trend", value=0.02, step=0.001, format="%.4f",
            help="10-block momentum: (current - 10_blocks_ago) / 10_blocks_ago. Positive = fees rising.",
        )

        if st.button("Predict"):
            payload = {
                "base_fee_gwei": base_fee_gwei,
                "gas_used_ratio": gas_used_ratio,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "base_fee_trend": base_fee_trend,
            }
            with st.spinner("Predicting..."):
                try:
                    result = predict_manual(st.session_state.api_url, payload)
                    st.success("Prediction complete")
                    _render_prediction(result)
                except requests.ConnectionError:
                    st.error("Could not connect to API. Is the FastAPI server running?")
                except requests.HTTPError as e:
                    detail = e.response.json().get("detail", str(e)) if e.response else str(e)
                    st.error(f"API error {e.response.status_code}: {detail}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_streamlit_ui.py -v`
Expected: all 5 tests PASS

```
PASSED tests/test_streamlit_ui.py::test_load_metrics_returns_dict
PASSED tests/test_streamlit_ui.py::test_load_metrics_missing_file
PASSED tests/test_streamlit_ui.py::test_load_metrics_invalid_json
PASSED tests/test_streamlit_ui.py::test_predict_latest_calls_correct_endpoint
PASSED tests/test_streamlit_ui.py::test_predict_manual_calls_correct_endpoint
```

- [ ] **Step 3: Run full test suite to check for regressions**

Run: `pytest --tb=short -q`
Expected: all existing tests still pass, 5 new tests added.

- [ ] **Step 4: Commit**

```bash
git add ui/__init__.py ui/streamlit_app.py tests/test_streamlit_ui.py
git commit -m "feat: add Streamlit UI for gas price predictor"
```

---

### Task 4: Smoke test the UI

**Files:** none (manual verification)

- [ ] **Step 1: Start the FastAPI backend**

In one terminal:
```bash
CONFIG=configs/ethereum-gas-price-predictor.yaml uvicorn app:app --reload
```
Expected: `Uvicorn running on http://127.0.0.1:8000`

- [ ] **Step 2: Start the Streamlit app**

In another terminal:
```bash
streamlit run ui/streamlit_app.py
```
Expected: opens browser at `http://localhost:8501`

- [ ] **Step 3: Verify sidebar**

Check that the sidebar shows:
- Title "Ethereum Gas Price Predictor"
- API URL input defaulting to `http://localhost:8000`
- Model Metrics section with R², RMSE, MAE values (should show ~0.9988, ~0.0106, ~0.0050)

- [ ] **Step 4: Test Live Prediction tab**

Click "Fetch latest block & predict". With a valid `ETHERSCAN_API_KEY` in `.env`, expect:
- Spinner while fetching
- Two metric cards showing Gwei and Wei values
- Block number info box

Without an API key, expect an `st.error` message (503 from FastAPI).

- [ ] **Step 5: Test Manual Prediction tab**

Click "Manual Prediction" tab, leave defaults, click "Predict". Expect:
- Spinner
- Two metric cards with predicted values

- [ ] **Step 6: Test error state**

Stop the FastAPI server. Click either predict button. Expect:
- `st.error("Could not connect to API. Is the FastAPI server running?")`
