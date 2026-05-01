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
