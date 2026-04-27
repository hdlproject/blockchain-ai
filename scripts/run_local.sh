#!/usr/bin/env bash
# Run the full pipeline locally for testing.
# Collects fresh blocks from Etherscan, trains, and evaluates.
#
# Usage:
#   ./scripts/run_local.sh [CONFIG]
#
# Example:
#   ./scripts/run_local.sh configs/ethereum-gas-price.yaml
set -euo pipefail

CONFIG="${1:-configs/ethereum-gas-price.yaml}"

echo "==> Config : ${CONFIG}"
echo ""

echo "[1/2] Collecting blocks from Etherscan..."
poetry run python scripts/collect_blocks.py --config "${CONFIG}"

echo "[2/2] Running training pipeline..."
poetry run python scripts/run_pipeline.py \
  --input data/raw/ethereum-blocks.csv \
  --config "${CONFIG}"
