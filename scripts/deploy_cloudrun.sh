#!/usr/bin/env bash
# Deploy blockchain-ai serving service + daily retraining job to GCP Cloud Run.
#
# Prerequisites (one-time setup):
#   gsutil mb -l REGION gs://BUCKET_NAME
#   gsutil cp models/model.joblib gs://BUCKET_NAME/model.joblib   # initial model
#   echo -n "YOUR_KEY" | gcloud secrets create ETHERSCAN_API_KEY --data-file=-
#   gcloud services enable run.googleapis.com cloudscheduler.googleapis.com \
#     cloudbuild.googleapis.com secretmanager.googleapis.com
#
# Usage:
#   ./scripts/deploy_cloudrun.sh [PROJECT_ID] [REGION] GCS_BUCKET
set -euo pipefail

PROJECT_ID="${1:-$(gcloud config get-value project)}"
REGION="${2:-us-central1}"
BUCKET="${3:?Usage: $0 [PROJECT_ID] [REGION] GCS_BUCKET}"
SERVICE="blockchain-ai"
UI_SERVICE="blockchain-ai-ui"
JOB="blockchain-ai-retrain"
REPO="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy"
IMAGE="${REPO}/${SERVICE}"
UI_IMAGE="${REPO}/${UI_SERVICE}"
JOB_IMAGE="${REPO}/${SERVICE}-job"
SA="blockchain-ai-scheduler@${PROJECT_ID}.iam.gserviceaccount.com"

echo "==> Project : ${PROJECT_ID}"
echo "==> Region  : ${REGION}"
echo "==> Bucket  : gs://${BUCKET}"
echo ""

echo "[1/6] Building and deploying serving image..."
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}" .
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --memory 512Mi --cpu 1 \
  --min-instances 0 --max-instances 3 \
  --allow-unauthenticated \
  --set-env-vars "MODEL_PATH=gs://${BUCKET}/model.joblib" \
  --set-secrets "ETHERSCAN_API_KEY=ETHERSCAN_API_KEY:latest" \
  --project "${PROJECT_ID}"

API_URL=$(gcloud run services describe "${SERVICE}" \
  --platform managed --region "${REGION}" \
  --project "${PROJECT_ID}" --format "value(status.url)")

echo "[2/6] Building and deploying Streamlit UI..."
gcloud builds submit \
  --config cloudbuild.ui.yaml \
  --substitutions "_IMAGE=${UI_IMAGE}" \
  --project "${PROJECT_ID}" \
  .
gcloud run deploy "${UI_SERVICE}" \
  --image "${UI_IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --memory 512Mi --cpu 1 \
  --min-instances 0 --max-instances 2 \
  --allow-unauthenticated \
  --set-env-vars "API_URL=${API_URL},GCS_BUCKET=${BUCKET}" \
  --project "${PROJECT_ID}"

echo "[3/6] Building retrain job image..."
gcloud builds submit \
  --config cloudbuild.job.yaml \
  --substitutions "_IMAGE=${JOB_IMAGE}" \
  --project "${PROJECT_ID}" \
  .

echo "[4/6] Deploying retrain Cloud Run Job..."
gcloud run jobs deploy "${JOB}" \
  --image "${JOB_IMAGE}" \
  --region "${REGION}" \
  --memory 1Gi --cpu 1 \
  --task-timeout 3600 \
  --set-env-vars "GCS_BUCKET=${BUCKET},CONFIG=configs/ethereum-gas-price.yaml" \
  --set-secrets "ETHERSCAN_API_KEY=ETHERSCAN_API_KEY:latest" \
  --project "${PROJECT_ID}"

echo "[5/6] Setting up scheduler service account..."
gcloud iam service-accounts create blockchain-ai-scheduler \
  --display-name "Blockchain AI Scheduler" \
  --project "${PROJECT_ID}" 2>/dev/null || true
gcloud run jobs add-iam-policy-binding "${JOB}" \
  --region "${REGION}" \
  --member "serviceAccount:${SA}" \
  --role "roles/run.invoker" \
  --project "${PROJECT_ID}"

echo "[6/6] Setting up Cloud Scheduler (daily at midnight UTC)..."
gcloud services enable cloudscheduler.googleapis.com --project "${PROJECT_ID}"
SCHEDULE_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB}:run"
EXISTING_SCHEDULE=$(gcloud scheduler jobs describe "${JOB}-schedule" \
  --location "${REGION}" --project "${PROJECT_ID}" \
  --format "value(name)" 2>/dev/null || true)
if [[ -n "${EXISTING_SCHEDULE}" ]]; then
  echo "      Scheduler job exists — updating."
  gcloud scheduler jobs update http "${JOB}-schedule" \
    --location "${REGION}" \
    --schedule "0 0 * * *" \
    --uri "${SCHEDULE_URI}" \
    --http-method POST \
    --oauth-service-account-email "${SA}" \
    --project "${PROJECT_ID}"
else
  echo "      Scheduler job not found — creating."
  gcloud scheduler jobs create http "${JOB}-schedule" \
    --location "${REGION}" \
    --schedule "0 0 * * *" \
    --uri "${SCHEDULE_URI}" \
    --http-method POST \
    --oauth-service-account-email "${SA}" \
    --project "${PROJECT_ID}"
fi

echo ""
echo "Done."
SERVICE_URL=$(gcloud run services describe "${SERVICE}" \
  --platform managed --region "${REGION}" \
  --project "${PROJECT_ID}" --format "value(status.url)")
UI_URL=$(gcloud run services describe "${UI_SERVICE}" \
  --platform managed --region "${REGION}" \
  --project "${PROJECT_ID}" --format "value(status.url)")
echo "API URL     : ${SERVICE_URL}"
echo "UI URL      : ${UI_URL}"
echo "Health check: curl ${SERVICE_URL}/health"
echo "Run job now : gcloud run jobs execute ${JOB} --region ${REGION}"
echo "Predict     : curl -s -X POST ${SERVICE_URL}/predict \\"
echo "                -H 'Content-Type: application/json' \\"
echo "                -d '{\"base_fee_gwei\":15.0,\"gas_used_ratio\":0.5,\"hour_of_day\":14,\"day_of_week\":1,\"base_fee_trend\":0.02}'"
