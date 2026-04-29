#!/usr/bin/env bash
#
# One-shot deploy: upload weights → Cloud Build image → IAM → three Cloud Run services.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated:  gcloud auth login
#   - gcloud auth application-default login
#   - Billing linked to the target project (Console → Billing → link account); required for Run/Build/Artifact Registry
#   - From repo root, trained weights exist:
#       Final_Source_Model/model.safetensors (or pytorch_model.bin)
#       Phase2_Outputs/fewshot_twi_5/  and  fewshot_pcm_5/  (after Experiment 2)
#
# Usage (defaults match your course project):
#   ./deploy/gcp/deploy_all.sh
#
# Environment overrides (optional):
#   GCP_PROJECT_ID   default: zero-shot-494819
#   GCP_REGION       default: us-central1
#   GCP_AR_REPO      default: afrihate  (Artifact Registry docker repo name)
#   GCS_MODEL_BUCKET default: ${GCP_PROJECT_ID}-afrihate-models  (must be globally unique)
#   IMAGE_TAG        default: latest
#
# Flags:
#   --skip-upload   Skip gsutil rsync (weights already in GCS at expected prefixes)
#   --skip-build    Skip Cloud Build (image already in Artifact Registry)
#   --help          Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SKIP_UPLOAD=0
SKIP_BUILD=0

usage() {
  cat <<'EOF'
Usage: ./deploy/gcp/deploy_all.sh [--skip-upload] [--skip-build]

  One-shot: upload weights → Cloud Build → IAM → 3× Cloud Run (E4 + few-shot Twi/Pidgin k=5).

  Before running:
    gcloud auth login
    gcloud auth application-default login

  Optional env:
    GCP_PROJECT_ID   (default: zero-shot-494819)
    GCP_REGION       (default: us-central1)
    GCP_AR_REPO      (default: afrihate)
    GCS_MODEL_BUCKET (default: ${GCP_PROJECT_ID}-afrihate-models)
    IMAGE_TAG        (default: latest)

  Flags:
    --skip-upload   Weights already under gs://BUCKET/models/{e4,fewshot_*}
    --skip-build    Reuse existing image in Artifact Registry
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-upload) SKIP_UPLOAD=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

PROJECT_ID="${GCP_PROJECT_ID:-zero-shot-494819}"
REGION="${GCP_REGION:-us-central1}"
AR_REPO="${GCP_AR_REPO:-afrihate}"
BUCKET="${GCS_MODEL_BUCKET:-${PROJECT_ID}-afrihate-models}"
TAG="${IMAGE_TAG:-latest}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/afrihate-api:${TAG}"

echo "==> Repo root:     ${REPO_ROOT}"
echo "==> Project:       ${PROJECT_ID}"
echo "==> Region:        ${REGION}"
echo "==> Bucket:        ${BUCKET}"
echo "==> Image:         ${IMAGE}"
echo "==> Skip upload:   ${SKIP_UPLOAD}"
echo "==> Skip build:    ${SKIP_BUILD}"
echo

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud not found. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install" >&2
  exit 1
fi

echo "==> gcloud project + auth check"
gcloud config set project "${PROJECT_ID}"
gcloud auth application-default print-access-token >/dev/null 2>&1 || {
  echo "Run: gcloud auth login && gcloud auth application-default login" >&2
  exit 1
}

# Align ADC quota/billing metadata with the project you are deploying (avoids warnings
# when your default quota project was another GCP project, e.g. zephyrmobile-492023).
if gcloud auth application-default set-quota-project "${PROJECT_ID}" 2>/dev/null; then
  echo "==> Application Default Credentials quota project set to ${PROJECT_ID}"
else
  echo "==> (Optional) Run: gcloud auth application-default set-quota-project ${PROJECT_ID}" >&2
fi

echo "==> Billing check (Cloud Run / Build / Artifact Registry require an active billing account)"
if ! BILLING_ACCOUNT="$(gcloud billing projects describe "${PROJECT_ID}" --format='value(billingAccountName)' 2>/dev/null)" || [[ -z "${BILLING_ACCOUNT}" ]]; then
  cat <<EOF >&2

ERROR: Project '${PROJECT_ID}' has no billing account linked (or you lack permission to read it).

Fix in the console (log in as a project Owner):
  https://console.cloud.google.com/billing/linkedaccount?project=${PROJECT_ID}

Link any active billing account, wait a minute, then re-run:
  ./deploy/gcp/deploy_all.sh

(GCP error UREQ_PROJECT_BILLING_NOT_FOUND / "Billing account ... is not found" means the same thing.)

EOF
  exit 1
fi
echo "    billingAccountName=${BILLING_ACCOUNT}"

echo "==> Enable APIs"
if ! gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  --project="${PROJECT_ID}"; then
  cat <<'EOF' >&2

ERROR: gcloud services enable failed.
If the message mentions billing, finish linking a billing account (see URL above) and retry.

EOF
  exit 1
fi

echo "==> Artifact Registry repo: ${AR_REPO}"
if ! gcloud artifacts repositories describe "${AR_REPO}" --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${AR_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --description="AfriHate inference API"
fi

echo "==> GCS bucket: gs://${BUCKET}/"
if ! gcloud storage buckets describe "gs://${BUCKET}/" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET}/" --project="${PROJECT_ID}" --location="${REGION}" --uniform-bucket-level-access
fi

if [[ "${SKIP_UPLOAD}" -eq 0 ]]; then
  E4="${REPO_ROOT}/Final_Source_Model"
  if [[ ! -f "${E4}/config.json" ]]; then
    echo "Missing ${E4}/config.json" >&2
    exit 1
  fi
  if [[ ! -f "${E4}/model.safetensors" && ! -f "${E4}/pytorch_model.bin" ]]; then
    echo "Missing weights under ${E4}/ (model.safetensors or pytorch_model.bin). Train Experiment 1 first." >&2
    exit 1
  fi
  for fs in fewshot_twi_5 fewshot_pcm_5; do
    d="${REPO_ROOT}/Phase2_Outputs/${fs}"
    if [[ ! -d "${d}" || ! -f "${d}/config.json" ]]; then
      echo "Missing few-shot folder ${d} (run Experiment 2 first)." >&2
      exit 1
    fi
  done

  echo "==> Upload E4 weights → gs://${BUCKET}/models/e4/"
  gcloud storage rsync --recursive "${E4}/" "gs://${BUCKET}/models/e4/"
  echo "==> Upload few-shot Twi k=5"
  gcloud storage rsync --recursive "${REPO_ROOT}/Phase2_Outputs/fewshot_twi_5/" "gs://${BUCKET}/models/fewshot_twi_5/"
  echo "==> Upload few-shot Pidgin k=5"
  gcloud storage rsync --recursive "${REPO_ROOT}/Phase2_Outputs/fewshot_pcm_5/" "gs://${BUCKET}/models/fewshot_pcm_5/"
else
  echo "==> Skipping upload (--skip-upload)"
fi

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
CLOUD_BUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

echo "==> Grant Cloud Run default SA read on bucket (${COMPUTE_SA})"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}/" \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/storage.objectViewer" \
  --project="${PROJECT_ID}" 2>/dev/null || true

echo "==> Artifact Registry: grant Cloud Build push (${CLOUD_BUILD_SA})"
gcloud artifacts repositories add-iam-policy-binding "${AR_REPO}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}" \
  --member="serviceAccount:${CLOUD_BUILD_SA}" \
  --role="roles/artifactregistry.writer" 2>/dev/null || true

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  echo "==> Cloud Build: ${IMAGE}"
  gcloud builds submit "${REPO_ROOT}" \
    --project="${PROJECT_ID}" \
    --config="${SCRIPT_DIR}/cloudbuild.yaml" \
    --substitutions=_IMAGE="${IMAGE}"
else
  echo "==> Skipping Cloud Build (--skip-build); using existing image: ${IMAGE}"
fi

run_deploy() {
  local name="$1" uri="$2" variant="$3"
  echo "==> Cloud Run deploy: ${name}"
  gcloud run deploy "${name}" \
    --image="${IMAGE}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --allow-unauthenticated \
    --memory=8Gi \
    --cpu=2 \
    --timeout=900 \
    --max-instances=3 \
    --set-env-vars="GCS_MODEL_URI=${uri},NLP_SERVE_VARIANT=${variant}"
}

run_deploy "afrihate-e4-zero" "gs://${BUCKET}/models/e4" "e4_zero"
run_deploy "afrihate-fewshot-twi-5" "gs://${BUCKET}/models/fewshot_twi_5" "fewshot_twi_5"
run_deploy "afrihate-fewshot-pcm-5" "gs://${BUCKET}/models/fewshot_pcm_5" "fewshot_pcm_5"

echo
echo "==> Done. Service URLs:"
for svc in afrihate-e4-zero afrihate-fewshot-twi-5 afrihate-fewshot-pcm-5; do
  url="$(gcloud run services describe "${svc}" --region="${REGION}" --project="${PROJECT_ID}" --format='value(status.url)' 2>/dev/null || true)"
  echo "    ${svc}: ${url}"
done
echo
echo "First cold start may download ~2 GiB per revision (can take several minutes)."
echo "Optional: add --min-instances=1 to each deploy above for faster demos (extra cost)."
