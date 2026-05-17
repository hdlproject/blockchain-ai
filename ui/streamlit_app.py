import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL_DEFAULT = os.environ.get("API_URL", "http://localhost:8000")
_GCS_BUCKET = os.environ.get("GCS_BUCKET")
REPORT_PATH = Path(__file__).parent.parent / "reports" / "report.json"
_TARGET_KEY = "predicted_next-block_base_fee"


def load_metrics(report_path: Path) -> dict | None:
    try:
        with open(report_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_metrics_gcs(bucket_name: str) -> dict | None:
    try:
        from google.cloud import storage
        blob = storage.Client().bucket(bucket_name).blob("report.json")
        return json.loads(blob.download_as_text())
    except Exception:
        return None


def predict_latest(api_url: str, n_blocks: int = 1) -> dict:
    resp = requests.get(f"{api_url}/predict/gas-price/latest", params={"n_blocks": n_blocks}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def predict_manual(api_url: str, payload: dict) -> dict:
    resp = requests.post(f"{api_url}/predict/gas-price", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _render_prediction(result: dict) -> None:
    step1 = result["predictions"][0]
    col1, col2 = st.columns(2)
    col1.metric("Next Block Base Fee (Gwei)", f"{step1['base_fee_gwei']:.4f}")
    col2.metric("Next Block Base Fee (Wei)", f"{step1['base_fee_wei']:.0f}")


def _render_base_fee_chart(result: dict) -> None:
    history = result.get("block_history")
    predictions = result.get("predictions")
    if not history or not predictions:
        return

    last_block = history[-1]["block"]

    hist_df = pd.DataFrame(history)
    pred_df = pd.DataFrame([{
        "block": last_block + p["step"],
        "base_fee_gwei": p["base_fee_gwei"],
        "method": "Exact (EIP-1559)" if p["method"] == "formula" else "Model estimate",
    } for p in predictions])

    # bridge: connect last historical point to first prediction for visual continuity
    bridge_df = pd.concat([
        hist_df.iloc[[-1]][["block", "base_fee_gwei"]],
        pred_df[["block", "base_fee_gwei"]],
    ], ignore_index=True)

    color_scale = alt.Scale(
        domain=["Exact (EIP-1559)", "Model estimate"],
        range=["#e05c5c", "#e08c3a"],
    )

    hist_line = (
        alt.Chart(hist_df)
        .mark_line(color="steelblue")
        .encode(x=alt.X("block:O", title="Block Number"), y=alt.Y("base_fee_gwei:Q", title="Base Fee (Gwei)"))
    )
    hist_dots = (
        alt.Chart(hist_df)
        .mark_point(filled=True, size=60, color="steelblue")
        .encode(
            x="block:O", y="base_fee_gwei:Q",
            tooltip=["block:O", alt.Tooltip("base_fee_gwei:Q", format=".4f")],
        )
    )
    pred_line = (
        alt.Chart(bridge_df)
        .mark_line(strokeDash=[5, 3], color="#e05c5c")
        .encode(x="block:O", y="base_fee_gwei:Q")
    )
    pred_dots = (
        alt.Chart(pred_df)
        .mark_point(filled=True, size=80)
        .encode(
            x="block:O", y="base_fee_gwei:Q",
            color=alt.Color("method:N", scale=color_scale, title=""),
            tooltip=["block:O", alt.Tooltip("base_fee_gwei:Q", format=".4f"), "method:N"],
        )
    )

    st.altair_chart(hist_line + hist_dots + pred_line + pred_dots, use_container_width=True)


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
        metrics = _load_metrics_gcs(_GCS_BUCKET) if _GCS_BUCKET else load_metrics(REPORT_PATH)
        if metrics:
            st.metric("R²", f"{metrics['r2']:.4f}")
            st.metric("RMSE", f"{metrics['rmse']:.6f}")
            st.metric("MAE", f"{metrics['mae']:.6f}")
        else:
            st.warning("Model metrics not available")

    tab1, tab2 = st.tabs(["Live Prediction", "Manual Prediction"])

    with tab1:
        st.header("Live Prediction")
        st.write("Fetches the latest Ethereum block and predicts the next N base fees automatically.")
        st.caption("Fetches 11 blocks from Etherscan — typically takes 15–30 seconds.")
        n_blocks = st.slider("Blocks to predict", min_value=1, max_value=20, value=1,
                             help="Step 1 uses the exact EIP-1559 formula. Steps 2+ use auto-regression.")
        if st.button("Fetch latest block & predict"):
            with st.status("Fetching latest block data...", expanded=True) as status:
                try:
                    st.write("Calling Etherscan API for the last 10 blocks...")
                    result = predict_latest(st.session_state.api_url, n_blocks=n_blocks)
                    st.write("Running inference...")
                    status.update(label="Done!", state="complete", expanded=False)
                    _render_prediction(result)
                    _render_base_fee_chart(result)
                    st.info(f"Latest block: {result.get('block_number', 'N/A')}")
                except requests.ConnectionError:
                    status.update(label="Connection failed", state="error")
                    st.error(
                        f"Could not connect to the API at `{st.session_state.api_url}`. "
                        "Is the FastAPI server running?"
                    )
                except requests.Timeout:
                    status.update(label="Request timed out", state="error")
                    st.error(
                        "The request timed out (60s). Etherscan may be slow right now — please try again."
                    )
                except requests.HTTPError as e:
                    code = e.response.status_code if e.response else "?"
                    detail = e.response.json().get("detail", str(e)) if e.response else str(e)
                    status.update(label=f"API error {code}", state="error")
                    st.error(f"**HTTP {code}:** {detail}")
                    if code == 503:
                        st.warning("Check that `ETHERSCAN_API_KEY` is set and the model has been trained.")
                except Exception as e:
                    status.update(label="Unexpected error", state="error")
                    with st.expander("Error details"):
                        st.exception(e)

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
            with st.spinner("Running prediction..."):
                try:
                    result = predict_manual(st.session_state.api_url, payload)
                    st.success("Prediction complete")
                    _render_prediction(result)
                except requests.ConnectionError:
                    st.error(
                        f"Could not connect to the API at `{st.session_state.api_url}`. "
                        "Is the FastAPI server running?"
                    )
                except requests.Timeout:
                    st.error("The request timed out. Please try again.")
                except requests.HTTPError as e:
                    code = e.response.status_code if e.response else "?"
                    detail = e.response.json().get("detail", str(e)) if e.response else str(e)
                    st.error(f"**HTTP {code}:** {detail}")
                    if code == 503:
                        st.warning("The model may not be loaded yet. Check that the retrain job has run.")
                except Exception as e:
                    with st.expander("Error details"):
                        st.exception(e)


if __name__ == "__main__":
    main()
