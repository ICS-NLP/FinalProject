# AfriHate E4 — HTTP API

Serves the **best cross-lingual checkpoint** (Experiment 1, E4: `Davlan/afro-xlmr-large-76L` on Hausa + Amharic + Yoruba) from `Final_Source_Model/`.

## Prerequisites

- Repo `.venv` with `requirements.txt` + API deps installed (see root `README.md`).
- **`Final_Source_Model/model.safetensors`** (or `pytorch_model.bin`) present — train Experiment 1 or copy weights.

## Run locally

From **repository root** (`FinalProject/`):

```bash
.venv/bin/pip install 'fastapi>=0.115' 'uvicorn[standard]>=0.32'
export NLP_SERVE_MODEL_DIR="$PWD/Final_Source_Model"   # optional if default path is fine
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8080
```

- Docs: [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs)  
- Health: `GET /health`  
- Classify one text: `POST /predict` JSON `{"text":"..."}`  
- Classify many: `POST /predict/batch` JSON `{"texts":["...","..."]}` (max 32 by default; override with `NLP_SERVE_MAX_BATCH`)

### Example

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Me ma me ho atuu"}' | python3 -m json.tool
```

### CORS (browser frontends)

By default all origins are allowed. For production set a comma-separated allowlist:

```bash
export NLP_SERVE_CORS_ORIGINS="https://your-app.example,http://localhost:3000"
```

## Integrate in a web app

1. **Same machine / private network:** call `http://<server-ip>:8080/predict` from your backend (Node, Django, etc.) with `fetch` or `axios` — keep the model server private; do not expose without auth in production.
2. **Production:** put **nginx** or a cloud load balancer in front, add **HTTPS**, **API keys** or **OAuth** on your own gateway (this service does not ship authentication).

Response shape for `POST /predict`:

```json
{
  "text": "...",
  "label": "Normal",
  "label_id": 2,
  "scores": { "Abuse": 0.12, "Hate": 0.03, "Normal": 0.85 }
}
```

See also **[../EXPERIMENTS.md](../EXPERIMENTS.md#recommended-model-for-deployment-e4)** for model provenance and ethics notes.
