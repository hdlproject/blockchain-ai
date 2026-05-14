#!/usr/bin/env bash
# Run the full Ethereum gas price pipeline locally for testing.
# Collects fresh blocks from Etherscan, trains, and evaluates.
#
# Usage:
#   ./scripts/run_local_ethereum_gas_price_predictor.sh [CONFIG]
#
# Example:
#   ./scripts/run_local_ethereum_gas_price_predictor.sh configs/ethereum-gas-price-predictor.yaml
set -euo pipefail

CONFIG="${1:-configs/ethereum-gas-price-predictor.yaml}"

echo "==> Config : ${CONFIG}"
echo ""

echo "[1/3] Collecting blocks from Etherscan..."
poetry run python src/blockchain_ai/workflow/collect_blocks.py --config "${CONFIG}"

echo "[2/3] Running training pipeline..."
poetry run python src/blockchain_ai/workflow/ethereum_gas_price_pipeline.py \
  --input data/raw/ethereum-blocks.csv \
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
