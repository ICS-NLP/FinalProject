# AfriHate E4 — HTTP API

Serves the **best cross-lingual checkpoint** (Experiment 1, E4: `Davlan/afro-xlmr-large-76L` on Hausa + Amharic + Yoruba) from `Final_Source_Model/`.

## Why a clearly toxic **English** line can still be scored `Normal`

The E4 checkpoint is **not** an English toxicity model. It was **fine-tuned only on AfriHate posts in Hausa, Amharic, and Yoruba**, then studied for **zero-shot transfer to Twi and Nigerian Pidgin**. Your sentence is **out-of-distribution**: the head never saw English training distribution or English-style threats, so logits can look arbitrary or wrongly “safe” even when a human reader is alarmed.

AfriHate **label definitions** (Abuse vs Hate vs Normal) also follow the dataset’s guidelines, which do not match every intuitive English reading.

**Takeaway:** treat **`/predict` on English** as **unreliable** unless you validate separately. For demos, use **Twi, Pidgin, or the source languages** the project was built for; for production English, use a model trained on English (or multilingual data that includes your target register) and keep **human review** for high-stakes decisions. See **`EXPERIMENTS.md`** (qualitative error analysis) for known failure modes even **on** target languages.

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

Response shape for `POST /predict` (every response includes **`model_scope`** so UIs can show a banner):

```json
{
  "model_scope": {
    "base_encoder": "Davlan/afro-xlmr-large-76L",
    "fine_tune_languages": ["hau", "amh", "yor"],
    "zero_shot_eval_languages": ["twi", "pcm"],
    "label_schema": "AfriHate 3-class (Abuse / Hate / Normal)",
    "out_of_scope_warning": "…"
  },
  "text": "...",
  "label": "Normal",
  "label_id": 2,
  "scores": { "Abuse": 0.12, "Hate": 0.03, "Normal": 0.85 }
}
```

See also **[../EXPERIMENTS.md](../EXPERIMENTS.md#recommended-model-for-deployment-e4)** for model provenance and ethics notes.
