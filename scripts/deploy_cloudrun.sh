#!/usr/bin/env bash
# Deploy blockchain-ai to GCP Cloud Run.
#
# Prerequisites:
#   - gcloud CLI authenticated and project set
#   - Trained model artifact at models/model.joblib (committed to repo)
#   - ETHERSCAN_API_KEY stored in Secret Manager:
#       echo -n "your_key" | gcloud secrets create ETHERSCAN_API_KEY --data-file=-
#
# Usage:
#   ./scripts/deploy_cloudrun.sh [PROJECT_ID] [REGION]
set -euo pipefail

PROJECT_ID="${1:-$(gcloud config get-value project)}"
REGION="${2:-us-central1}"
SERVICE="blockchain-ai"
REPO="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy"
IMAGE="${REPO}/${SERVICE}"

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
  --set-secrets "ETHERSCAN_API_KEY=ETHERSCAN_API_KEY:latest" \
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
echo "Predict     : curl -s -X POST ${SERVICE_URL}/predict \\"
echo "                -H 'Content-Type: application/json' \\"
echo "                -d '{\"base_fee_gwei\":15.0,\"gas_used_ratio\":0.5,\"hour_of_day\":14,\"day_of_week\":1,\"base_fee_trend\":0.02}'"
