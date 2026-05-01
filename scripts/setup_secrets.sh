#!/usr/bin/env bash
# Enable Secret Manager API and provision secrets required by deploy_cloudrun.sh.
#
# Usage:
#   ./scripts/setup_secrets.sh [PROJECT_ID] [ETHERSCAN_API_KEY_VALUE]
#
# If ETHERSCAN_API_KEY_VALUE is omitted you will be prompted interactively.
# Idempotent: safe to re-run; adds a new secret version if the secret already exists.
set -euo pipefail

PROJECT_ID="${1:-$(gcloud config get-value project)}"
SECRET_NAME="ETHERSCAN_API_KEY"

echo "==> Project : ${PROJECT_ID}"
echo "==> Secret  : ${SECRET_NAME}"
echo ""

echo "[1/3] Enabling Secret Manager API..."
gcloud services enable secretmanager.googleapis.com --project "${PROJECT_ID}"

# Read key value — from arg or prompt
if [[ -n "${2:-}" ]]; then
  KEY_VALUE="$2"
else
  read -rsp "Enter Etherscan API key (input hidden): " KEY_VALUE
  echo ""
fi

echo "[2/3] Creating / updating secret '${SECRET_NAME}'..."
if gcloud secrets describe "${SECRET_NAME}" \
     --project "${PROJECT_ID}" \
     --format "value(name)" &>/dev/null; then
  echo "      Secret exists — adding new version."
  printf '%s' "${KEY_VALUE}" | \
    gcloud secrets versions add "${SECRET_NAME}" \
      --data-file=- \
      --project "${PROJECT_ID}"
else
  echo "      Secret not found — creating."
  printf '%s' "${KEY_VALUE}" | \
    gcloud secrets create "${SECRET_NAME}" \
      --data-file=- \
      --replication-policy automatic \
      --project "${PROJECT_ID}"
fi

echo "[3/3] Granting Cloud Run service account access..."
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format "value(projectNumber)")
# Cloud Run services run as the Compute Engine default SA unless a custom SA is specified.
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding "${SECRET_NAME}" \
  --member "serviceAccount:${COMPUTE_SA}" \
  --role "roles/secretmanager.secretAccessor" \
  --project "${PROJECT_ID}"

echo ""
echo "Done. Secret '${SECRET_NAME}' is ready for Cloud Run."
