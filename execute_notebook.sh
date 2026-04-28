#!/usr/bin/env bash
# Headless notebook execution using THIS folder's .venv.
# Usage: ./execute_notebook.sh [notebook.ipynb]   (default: Source_Model_FineTuning.ipynb)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${ROOT}/.venv/bin/python"
KERNEL_NAME="nlp-afrihate-finalproject"
NOTEBOOK="${1:-${ROOT}/Source_Model_FineTuning.ipynb}"
if [[ "${NOTEBOOK}" != /* ]]; then
  NOTEBOOK="${ROOT}/${NOTEBOOK}"
fi

if [[ ! -x "$PY" ]]; then
  echo "Missing ${PY}. Create the venv and install deps:"
  echo "  cd \"${ROOT}\" && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

"$PY" -m ipykernel install --prefix="${ROOT}/.venv" --name="${KERNEL_NAME}" --display-name="NLP AfriHate (FinalProject .venv)" >/dev/null
"$PY" "${ROOT}/normalize_notebook_json.py" "${NOTEBOOK}"

NBCONVERT="${ROOT}/.venv/bin/jupyter-nbconvert"
exec "${NBCONVERT}" \
  --to notebook \
  --execute "${NOTEBOOK}" \
  --inplace \
  --ExecutePreprocessor.kernel_name="${KERNEL_NAME}" \
  --ExecutePreprocessor.timeout=-1
