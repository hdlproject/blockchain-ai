#!/usr/bin/env bash
# Create the Artifact Registry repository required by deploy_cloudrun.sh.
#
# Usage:
#   ./scripts/setup_artifact_registry.sh [PROJECT_ID] [REGION]
#
# Idempotent: safe to re-run; skips creation if the repository already exists.
set -euo pipefail

PROJECT_ID="${1:-$(gcloud config get-value project)}"
REGION="${2:-us-central1}"
REPO_NAME="cloud-run-source-deploy"

echo "==> Project : ${PROJECT_ID}"
echo "==> Region  : ${REGION}"
echo "==> Repo    : ${REPO_NAME}"
echo ""

echo "[1/2] Enabling Artifact Registry API..."
gcloud services enable artifactregistry.googleapis.com --project "${PROJECT_ID}"

echo "[2/2] Creating Docker repository '${REPO_NAME}'..."
if gcloud artifacts repositories describe "${REPO_NAME}" \
     --location "${REGION}" \
     --project "${PROJECT_ID}" \
     --format "value(name)" &>/dev/null; then
  echo "      Repository already exists — skipping."
else
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format docker \
    --location "${REGION}" \
    --description "Docker images for gas-predictor Cloud Run service and retrain job" \
    --project "${PROJECT_ID}"
  echo "      Repository created."
fi

echo ""
echo "Done."
echo "Image prefix: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"
