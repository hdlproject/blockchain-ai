#!/usr/bin/env bash
# Run the full Airdrop Farmer Detector pipeline locally for testing.
# Fetches callers of configured airdrop contracts from Etherscan, trains
# the GMM model, and starts the API server.
#
# Usage:
#   ./scripts/run_local_airdrop_farmer_detector.sh [CONFIG]
#
# Example:
#   ./scripts/run_local_airdrop_farmer_detector.sh configs/airdrop-farmer-detector.yaml
set -euo pipefail

CONFIG="${1:-configs/airdrop-farmer-detector.yaml}"

echo "==> Config : ${CONFIG}"
echo ""

echo "[1/2] Running seed step (fetch wallets, record funders, fit GMM)..."
poetry run python src/blockchain_ai/workflow/run_airdrop_farmer_detector_seed.py \
  --config "${CONFIG}"

echo "[2/2] Starting API + Streamlit UI..."
echo "      FastAPI   → http://localhost:8000"
echo "      Streamlit → http://localhost:8501"
echo "      Press Ctrl+C to stop."
echo ""

trap 'kill ${API_PID} 2>/dev/null; exit 0' INT TERM

CONFIG="${CONFIG}" poetry run uvicorn app:app --host 0.0.0.0 --port 8000 &
API_PID=$!

poetry run streamlit run ui/streamlit_app.py

kill "${API_PID}" 2>/dev/null
