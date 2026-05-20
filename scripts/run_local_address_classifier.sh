#!/usr/bin/env bash
# Run the full address classification pipeline locally for testing.
# Collects labels and on-chain features, trains, and evaluates.
#
# Usage:
#   ./scripts/run_local_address_classifier.sh [CONFIG]
#
# Example:
#   ./scripts/run_local_address_classifier.sh configs/address-classifier.yaml
set -euo pipefail

CONFIG="${1:-configs/address-classifier.yaml}"

echo "==> Config : ${CONFIG}"
echo ""

echo "[1/4] Collecting address labels (GoPlus, OFAC, Forta)..."
poetry run python src/blockchain_ai/workflow/collect_labels.py --config "${CONFIG}"

echo "[2/4] Collecting on-chain features from Etherscan..."
poetry run python src/blockchain_ai/workflow/collect_address_features.py --config "${CONFIG}"

echo "[3/4] Running training pipeline..."
poetry run python src/blockchain_ai/workflow/run_pipeline.py \
  --config "${CONFIG}"

echo "[4/4] Starting API + Streamlit UI..."
echo "      FastAPI  → http://localhost:8000"
echo "      Streamlit → http://localhost:8501"
echo "      Press Ctrl+C to stop."
echo ""

trap 'kill ${API_PID} 2>/dev/null; exit 0' INT TERM

CONFIG="${CONFIG}" poetry run uvicorn app:app --host 0.0.0.0 --port 8000 &
API_PID=$!

poetry run streamlit run ui/streamlit_app.py

kill "${API_PID}" 2>/dev/null
