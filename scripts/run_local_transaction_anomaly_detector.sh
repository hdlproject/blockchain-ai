#!/usr/bin/env bash
# Run the full transaction anomaly detection pipeline locally.
# Collects transactions from recent blocks, trains DBSCAN, and starts the API + UI.
#
# Usage:
#   ./scripts/run_local_transaction_anomaly_detector.sh [CONFIG]
#
# Example:
#   ./scripts/run_local_transaction_anomaly_detector.sh configs/transaction-anomaly-detector.yaml
set -euo pipefail

CONFIG="${1:-configs/transaction-anomaly-detector.yaml}"

echo "==> Config : ${CONFIG}"
echo ""

echo "[1/3] Collecting transactions from Etherscan..."
poetry run python src/blockchain_ai/workflow/collect_transactions.py --config "${CONFIG}"

echo "[2/3] Running training pipeline..."
poetry run python src/blockchain_ai/workflow/run_pipeline.py \
  --config "${CONFIG}"

echo "[3/3] Starting API + Streamlit UI..."
echo "      FastAPI  → http://localhost:8000"
echo "      Streamlit → http://localhost:8501"
echo "      Press Ctrl+C to stop."
echo ""

trap 'kill ${API_PID} 2>/dev/null; exit 0' INT TERM

CONFIG="${CONFIG}" poetry run uvicorn app:app --host 0.0.0.0 --port 8000 &
API_PID=$!

poetry run streamlit run ui/streamlit_app.py

kill "${API_PID}" 2>/dev/null
