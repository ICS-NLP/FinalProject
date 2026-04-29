# Build from repository root:
#   docker build -f deploy/gcp/Dockerfile -t afrihate-api:local .
#
# Image does not contain weights; at runtime set GCS_MODEL_URI (see README.md).
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image smaller for Cloud Run (default is CPU).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY deploy/gcp/requirements-docker.txt /app/deploy/gcp/requirements-docker.txt
RUN pip install --no-cache-dir -r /app/deploy/gcp/requirements-docker.txt

COPY project_paths.py /app/project_paths.py
COPY api /app/api
COPY deploy/gcp/fetch_model_from_gcs.py /app/deploy/gcp/fetch_model_from_gcs.py
COPY deploy/gcp/entrypoint.sh /app/deploy/gcp/entrypoint.sh
RUN chmod +x /app/deploy/gcp/entrypoint.sh

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

ENTRYPOINT ["/app/deploy/gcp/entrypoint.sh"]
