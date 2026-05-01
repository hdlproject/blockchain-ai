#!/usr/bin/env bash
set -euo pipefail

BUCKET="${GCS_BUCKET:?GCS_BUCKET env var is required}"
CONFIG="${CONFIG:-configs/ethereum-gas-price.yaml}"

echo "[1/3] Collecting blocks from Etherscan..."
python scripts/collect_blocks.py --config "${CONFIG}"

echo "[2/3] Running training pipeline..."
python scripts/run_pipeline.py \
  --input data/raw/ethereum-blocks.csv \
  --config "${CONFIG}"

echo "[3/3] Uploading model to gs://${BUCKET}/model.joblib ..."
python -c "
from google.cloud import storage; import os
storage.Client().bucket(os.environ['GCS_BUCKET']).blob('model.joblib').upload_from_filename('models/model.joblib')
"
echo "Done."
