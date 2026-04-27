#!/usr/bin/env bash
# Deploy blockchain-ai to GCP Cloud Run.
# Usage: ./scripts/deploy_cloudrun.sh [PROJECT_ID] [REGION]
set -euo pipefail

PROJECT_ID="${1:-$(gcloud config get-value project)}"
REGION="${2:-us-central1}"
SERVICE="blockchain-ai"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"

echo "==> Project : ${PROJECT_ID}"
echo "==> Region  : ${REGION}"
echo "==> Image   : ${IMAGE}"
echo ""

echo "[1/3] Building and pushing Docker image via Cloud Build..."
gcloud builds submit \
  --tag "${IMAGE}" \
  --project "${PROJECT_ID}" \
  .

echo "[2/3] Deploying to Cloud Run..."
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --allow-unauthenticated \
  --project "${PROJECT_ID}"

echo "[3/3] Done."
SERVICE_URL=$(gcloud run services describe "${SERVICE}" \
  --platform managed \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --format "value(status.url)")
echo ""
echo "Service URL : ${SERVICE_URL}"
echo "Health check: curl ${SERVICE_URL}/health"
echo "Predict     : curl -X POST ${SERVICE_URL}/predict -F 'file=@your_data.csv'"
