#!/usr/bin/env bash
set -euo pipefail
cd /app

# Cloud Run sets PORT; default 8080 for local docker
export PORT="${PORT:-8080}"

if [[ -n "${GCS_MODEL_URI:-}" ]]; then
  echo "Fetching model from ${GCS_MODEL_URI} -> /app/served_model"
  rm -rf /app/served_model
  mkdir -p /app/served_model
  python /app/deploy/gcp/fetch_model_from_gcs.py "${GCS_MODEL_URI}" /app/served_model
  export NLP_SERVE_MODEL_DIR=/app/served_model
fi

if [[ -z "${NLP_SERVE_MODEL_DIR:-}" ]]; then
  echo "Set GCS_MODEL_URI (Cloud Run) or NLP_SERVE_MODEL_DIR (local path in image)." >&2
  exit 1
fi

exec python -m uvicorn api.main:app --host 0.0.0.0 --port "${PORT}"
