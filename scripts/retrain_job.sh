#!/usr/bin/env bash
set -euo pipefail

BUCKET="${GCS_BUCKET:?GCS_BUCKET env var is required}"
CONFIG="${CONFIG:-configs/ethereum-gas-price-predictor.yaml}"
DATA_FILE="data/raw/ethereum-blocks.csv"
GCS_DATA="gs://${BUCKET}/ethereum-blocks.csv"

mkdir -p "$(dirname "${DATA_FILE}")"

echo "[1/4] Restoring block history from ${GCS_DATA} ..."
python -c "
from google.cloud import storage
from google.api_core.exceptions import NotFound
import os
bucket = storage.Client().bucket(os.environ['GCS_BUCKET'])
blob = bucket.blob('ethereum-blocks.csv')
try:
    blob.download_to_filename('${DATA_FILE}')
    print('      Restored existing block history.')
except NotFound:
    print('      No prior history found — starting fresh.')
"

echo "[2/4] Collecting new blocks from Etherscan..."
python src/blockchain_ai/workflow/collect_blocks.py --config "${CONFIG}"

echo "[3/4] Uploading updated block history to ${GCS_DATA} ..."
python -c "
from google.cloud import storage; import os
storage.Client().bucket(os.environ['GCS_BUCKET']).blob('ethereum-blocks.csv').upload_from_filename('${DATA_FILE}')
"

echo "[4/4] Running training pipeline and uploading model + report..."
python src/blockchain_ai/workflow/ethereum_gas_price_pipeline.py \
  --input "${DATA_FILE}" \
  --config "${CONFIG}"
python -c "
from google.cloud import storage; import os
bucket = storage.Client().bucket(os.environ['GCS_BUCKET'])
bucket.blob('gas_price_predictor.joblib').upload_from_filename('models/gas_price_predictor.joblib')
bucket.blob('gas_price_predictor.json').upload_from_filename('reports/gas_price_predictor.json')
"
echo "Done."
