import json
import os
import time
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

_GCS_BUCKET = os.environ.get("GCS_BUCKET")
REPORT_PATH = Path(__file__).parent.parent / "reports" / "report.json"
ADDRESS_REPORT_PATH = Path(__file__).parent.parent / "reports" / "address_classifier.json"
GAS_V2_REPORT_PATH = Path(__file__).parent.parent / "reports" / "gas_price_predictor_v2.json"

_DEFAULT_GAS_V1_URL = os.environ.get("GAS_V1_API_URL", "http://localhost:8000")
_DEFAULT_GAS_V2_URL = os.environ.get("GAS_V2_API_URL", "http://localhost:8000")
_DEFAULT_ADDR_URL = os.environ.get("ADDR_API_URL", "http://localhost:8000")
_DEFAULT_TXANOMALY_URL = os.environ.get("TXANOMALY_API_URL", "http://localhost:8000")
TXANOMALY_REPORT_PATH = Path(__file__).parent.parent / "reports" / "transaction_anomaly.json"


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


def predict_latest_v2(api_url: str, n_blocks: int = 1) -> dict:
    resp = requests.get(f"{api_url}/predict/gas-price/v2/latest", params={"n_blocks": n_blocks}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def classify_address(api_url: str, address: str, max_wait: int = 60) -> dict:
    resp = requests.get(f"{api_url}/predict/address/{address}", timeout=10)
    resp.raise_for_status()
    result = resp.json()
    deadline = time.time() + max_wait
    while result["status"] == "pending" and time.time() < deadline:
        time.sleep(2)
        resp = requests.get(f"{api_url}/predict/address/{address}", timeout=10)
        resp.raise_for_status()
        result = resp.json()
    return result


def detect_transaction(api_url: str, payload: dict) -> dict:
    resp = requests.post(f"{api_url}/detect/transaction", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _render_anomaly_result(result: dict) -> None:
    score = result.get("score", 0.0)
    if result.get("anomaly"):
        st.error("Anomaly detected")
    else:
        st.success("Normal transaction")
    st.metric("Anomaly Score", f"{score:.4f}", help="Distance to nearest normal cluster. Higher = more unusual.")


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


def _render_address_result(result: dict) -> None:
    label = result.get("label", "unknown")
    probabilities = result.get("probabilities", {})

    if label == "unknown":
        st.info(f"Label: **unknown** (confidence below threshold)")
    elif label in ("sanctioned", "scammer"):
        st.error(f"Label: **{label}**")
    else:
        st.warning(f"Label: **{label}**")

    if probabilities:
        st.subheader("Probabilities")
        for cls, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
            col1, col2 = st.columns([1, 4])
            col1.write(f"**{cls}**")
            col2.progress(float(prob), text=f"{prob:.1%}")


def _handle_request_errors(exc: Exception, status: object) -> None:
    api_url = st.session_state.get("gas_v1_url", "")
    if isinstance(exc, requests.ConnectionError):
        status.update(label="Connection failed", state="error")
        st.error(f"Could not connect to the API. Is the backend running?")
    elif isinstance(exc, requests.Timeout):
        status.update(label="Request timed out", state="error")
        st.error("The request timed out. Please try again.")
    elif isinstance(exc, requests.HTTPError):
        code = exc.response.status_code if exc.response else "?"
        detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
        status.update(label=f"API error {code}", state="error")
        st.error(f"**HTTP {code}:** {detail}")
    else:
        status.update(label="Unexpected error", state="error")
        with st.expander("Error details"):
            st.exception(exc)


def main() -> None:
    st.set_page_config(page_title="Ethereum AI", layout="wide")

    with st.sidebar:
        st.title("Ethereum AI")

        st.subheader("API Endpoints")
        for key, label, default in [
            ("gas_v1_url", "Gas Price v1", _DEFAULT_GAS_V1_URL),
            ("gas_v2_url", "Gas Price v2 (LSTM)", _DEFAULT_GAS_V2_URL),
            ("addr_url", "Address Classifier", _DEFAULT_ADDR_URL),
            ("txanomaly_url", "Transaction Anomaly", _DEFAULT_TXANOMALY_URL),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default
            st.session_state[key] = st.text_input(label, value=st.session_state[key])

        st.divider()
        st.subheader("Model Metrics")

        metrics = _load_metrics_gcs(_GCS_BUCKET) if _GCS_BUCKET else load_metrics(REPORT_PATH)
        if metrics:
            st.caption("Gas Price v1")
            st.metric("R²", f"{metrics['r2']:.4f}")
            st.metric("RMSE", f"{metrics['rmse']:.6f}")
            st.metric("MAE", f"{metrics['mae']:.6f}")

        v2_metrics = load_metrics(GAS_V2_REPORT_PATH)
        if v2_metrics:
            st.caption("Gas Price v2 (LSTM)")
            st.metric("R²", f"{v2_metrics['r2']:.4f}")
            st.metric("RMSE", f"{v2_metrics['rmse']:.6f}")
            st.metric("MAE", f"{v2_metrics['mae']:.6f}")

        addr_metrics = load_metrics(ADDRESS_REPORT_PATH)
        if addr_metrics:
            st.caption("Address Classifier")
            st.metric("Accuracy", f"{addr_metrics['accuracy']:.4f}")
            st.metric("F1 Macro", f"{addr_metrics['f1_macro']:.4f}")
            if "f1_sanctioned" in addr_metrics:
                st.metric("F1 Sanctioned", f"{addr_metrics['f1_sanctioned']:.4f}")
            if "f1_phishing" in addr_metrics:
                st.metric("F1 Phishing", f"{addr_metrics['f1_phishing']:.4f}")

        txanomaly_metrics = load_metrics(TXANOMALY_REPORT_PATH)
        if txanomaly_metrics:
            st.caption("Transaction Anomaly")
            st.metric("Anomaly Ratio", f"{txanomaly_metrics['anomaly_ratio']:.4f}")
            st.metric("Clusters Found", str(txanomaly_metrics['n_clusters']))
            st.metric("Noise Points", str(txanomaly_metrics['n_noise']))

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Gas Price: Live", "Gas Price: Manual", "Gas Price v2 (LSTM)",
        "Address Classifier", "Transaction Anomaly",
    ])

    with tab1:
        st.header("Gas Price: Live Prediction")
        st.write("Fetches the latest Ethereum block and predicts the next N base fees automatically.")
        st.caption("Fetches 11 blocks from Etherscan — typically takes 15–30 seconds.")
        n_blocks = st.slider("Blocks to predict", min_value=1, max_value=20, value=1,
                             help="Step 1 uses the exact EIP-1559 formula. Steps 2+ use auto-regression.")
        if st.button("Fetch latest block & predict", key="btn_v1_live"):
            with st.status("Fetching latest block data...", expanded=True) as status:
                try:
                    st.write("Calling Etherscan API for the last 10 blocks...")
                    result = predict_latest(st.session_state.gas_v1_url, n_blocks=n_blocks)
                    st.write("Running inference...")
                    status.update(label="Done!", state="complete", expanded=False)
                    _render_prediction(result)
                    _render_base_fee_chart(result)
                    st.info(f"Latest block: {result.get('block_number', 'N/A')}")
                except Exception as e:
                    _handle_request_errors(e, status)

    with tab2:
        st.header("Gas Price: Manual Prediction")
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
        if st.button("Predict", key="btn_v1_manual"):
            payload = {
                "base_fee_gwei": base_fee_gwei,
                "gas_used_ratio": gas_used_ratio,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "base_fee_trend": base_fee_trend,
            }
            with st.spinner("Running prediction..."):
                try:
                    result = predict_manual(st.session_state.gas_v1_url, payload)
                    st.success("Prediction complete")
                    _render_prediction(result)
                except requests.ConnectionError:
                    st.error("Could not connect to the Gas Price v1 API. Is the backend running?")
                except requests.Timeout:
                    st.error("The request timed out. Please try again.")
                except requests.HTTPError as e:
                    code = e.response.status_code if e.response else "?"
                    detail = e.response.json().get("detail", str(e)) if e.response else str(e)
                    st.error(f"**HTTP {code}:** {detail}")
                except Exception as e:
                    with st.expander("Error details"):
                        st.exception(e)

    with tab3:
        st.header("Gas Price v2: Live Prediction (LSTM)")
        st.write("Feeds recent block history into the LSTM sequence model.")
        st.caption("Fetches 40 blocks from Etherscan — typically takes 30–60 seconds.")
        n_blocks_v2 = st.slider("Blocks to predict", min_value=1, max_value=20, value=1, key="slider_v2")
        if st.button("Fetch latest blocks & predict", key="btn_v2_live"):
            with st.status("Fetching block sequence...", expanded=True) as status:
                try:
                    st.write("Calling Etherscan API...")
                    result = predict_latest_v2(st.session_state.gas_v2_url, n_blocks=n_blocks_v2)
                    st.write("Running LSTM inference...")
                    status.update(label="Done!", state="complete", expanded=False)
                    _render_prediction(result)
                    _render_base_fee_chart(result)
                    st.info(f"Latest block: {result.get('block_number', 'N/A')}")
                except Exception as e:
                    _handle_request_errors(e, status)

    with tab4:
        st.header("Address Classifier")
        st.write("Classify an Ethereum address as sanctioned, phishing, or scammer.")
        st.caption(
            "On-chain features are fetched from Etherscan on first request — typically takes 15–30 seconds."
        )
        address = st.text_input(
            "Ethereum Address",
            placeholder="0x...",
            help="A 42-character hex address starting with 0x",
        )
        if st.button("Classify", key="btn_addr"):
            if not address:
                st.warning("Please enter an Ethereum address.")
            elif not (address.startswith("0x") and len(address) == 42):
                st.error("Invalid address format. Must be 42 characters starting with 0x.")
            else:
                with st.status("Classifying address...", expanded=True) as status:
                    try:
                        st.write(f"Submitting {address}...")
                        result = classify_address(st.session_state.addr_url, address)
                        if result["status"] == "done":
                            status.update(label="Classification complete", state="complete", expanded=False)
                            _render_address_result(result)
                        elif result["status"] == "pending":
                            status.update(label="Timed out", state="error")
                            st.error("Classification is still running. Try again in a moment.")
                        else:
                            status.update(label="Classification failed", state="error")
                            st.error(f"Error: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        _handle_request_errors(e, status)


    with tab5:
        st.header("Transaction Anomaly Detector")
        st.write("Check whether a transaction looks unusual before submitting it.")

        col1, col2 = st.columns(2)
        with col1:
            value_eth = st.number_input("Value (ETH)", min_value=0.0, value=0.5, step=0.1,
                                         help="Transaction value in ETH")
            gas_price_gwei = st.number_input("Gas Price (Gwei)", min_value=0.0, value=20.0, step=1.0)
            gas_used = st.number_input("Gas Used", min_value=0.0, value=21000.0, step=1000.0)
            is_contract_call = st.selectbox("Contract Call?", options=[0.0, 1.0],
                                             format_func=lambda x: "Yes" if x == 1.0 else "No")
            input_data_len = st.number_input("Calldata Length (bytes)", min_value=0.0, value=0.0, step=4.0,
                                              help="0 = simple ETH transfer")
        with col2:
            hour_of_day = st.slider("Hour of Day (UTC)", min_value=0.0, max_value=23.0, value=14.0, step=1.0)
            sender_tx_count_window = st.number_input("Sender Tx Count (window)", min_value=1.0, value=5.0, step=1.0)
            sender_avg_value_eth = st.number_input("Sender Avg Value ETH (window)", min_value=0.0, value=0.5, step=0.1)
            receiver_tx_count_window = st.number_input("Receiver Tx Count (window)", min_value=1.0, value=3.0, step=1.0)

        if st.button("Check Transaction", key="btn_anomaly"):
            payload = {
                "value_eth": value_eth,
                "gas_price_gwei": gas_price_gwei,
                "gas_used": gas_used,
                "input_data_len": input_data_len,
                "is_contract_call": is_contract_call,
                "hour_of_day": hour_of_day,
                "sender_tx_count_window": sender_tx_count_window,
                "sender_avg_value_eth": sender_avg_value_eth,
                "receiver_tx_count_window": receiver_tx_count_window,
            }
            with st.spinner("Checking transaction..."):
                try:
                    result = detect_transaction(st.session_state.txanomaly_url, payload)
                    _render_anomaly_result(result)
                except requests.ConnectionError:
                    st.error("Could not connect to the Transaction Anomaly API. Is the backend running?")
                except requests.Timeout:
                    st.error("The request timed out. Please try again.")
                except requests.HTTPError as e:
                    code = e.response.status_code if e.response else "?"
                    detail = e.response.json().get("detail", str(e)) if e.response else str(e)
                    st.error(f"**HTTP {code}:** {detail}")
                except Exception as e:
                    with st.expander("Error details"):
                        st.exception(e)


if __name__ == "__main__":
    main()
