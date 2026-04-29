# AfriHate E4 — HTTP API

Serves the **best cross-lingual checkpoint** (Experiment 1, E4: `Davlan/afro-xlmr-large-76L` on Hausa + Amharic + Yoruba) from `Final_Source_Model/`.

## Why a clearly toxic **English** line can still be scored `Normal`

The E4 checkpoint is **not** an English toxicity model. It was **fine-tuned only on AfriHate posts in Hausa, Amharic, and Yoruba**, then studied for **zero-shot transfer to Twi and Nigerian Pidgin**. Your sentence is **out-of-distribution**: the head never saw English training distribution or English-style threats, so logits can look arbitrary or wrongly “safe” even when a human reader is alarmed.

AfriHate **label definitions** (Abuse vs Hate vs Normal) also follow the dataset’s guidelines, which do not match every intuitive English reading.

**Takeaway:** treat **`/predict` on English** as **unreliable** unless you validate separately. For demos, use **Twi, Pidgin, or the source languages** the project was built for; for production English, use a model trained on English (or multilingual data that includes your target register) and keep **human review** for high-stakes decisions. See **`EXPERIMENTS.md`** (qualitative error analysis) for known failure modes even **on** target languages.

## Models you can deploy for side‑by‑side comparison

Everything below is a **3‑class AfriHate head** (Abuse / Hate / Normal) unless noted. The FastAPI app loads **one folder** at a time (`NLP_SERVE_MODEL_DIR`). To compare several models in a web app, run **multiple `uvicorn` processes** on different ports (or put a small gateway in front) and point your UI at each base URL.

| Variant | `experiment_id` (see `Checkpoints/experiment_log.csv`) | Typical on-disk folder | Role |
|--------|----------------------------------------------------------|-------------------------|------|
| **E4 (recommended cross‑lingual)** | `E4_large76L_hau_amh_yor` | `Final_Source_Model/` | **76L** encoder, trained **hau+amh+yor** — best zero‑shot on **Twi + Pidgin** in this repo. Default for `api/main.py`. |
| **E3** | `E3_base_hau_amh_yor` | Not kept by default; re‑train E3 and **save to** e.g. `deploy_models/E3_base_hau_amh_yor/` | **Base** encoder + **Yoruba** in the mix — stronger Twi zero‑shot than hau+amh‑only, weaker Pidgin than E4. |
| **E2** | `E2_large76L_hau_amh` | Optional local copy e.g. `Final_Source_Model_E2_76L/` (see `.gitignore`) | **76L** but **no Yoruba** — good Pidgin, weak Twi vs E4. |
| **E1** | `E1_base_hau_amh` | Save to e.g. `deploy_models/E1_base_hau_amh/` after training | **Base** encoder, **hau+amh** only — baseline ablation. |
| **Few‑shot adapters** (Phase 2) | Rows in `Phase2_Outputs/few_shot_results.csv` | `Phase2_Outputs/fewshot_twi_5`, `fewshot_pcm_20`, … (see Experiment 2) | Same E4 backbone then adapted on **k ∈ {5,10,20}** examples per class on **one** target — compare vs zero‑shot on that target only. |
| **T1 supervised ceiling** | `T1_supervised_twi` | `Phase3_Outputs/runs_T1_supervised_twi/checkpoint-*` | **`afro-xlmr-base`** trained **only on Twi** — ceiling / diagnostic for **Twi**; **not** for Pidgin or cross‑lingual use. |
| **T2 supervised ceiling** | `T2_supervised_pcm` | `Phase3_Outputs/runs_T2_supervised_pcm/checkpoint-*` | **`afro-xlmr-base`** trained **only on Pidgin** — ceiling for **Pidgin** only. |
| **LLM (`gpt‑4o‑mini`)** | `LLM0_*` rows | *No local folder* | Prompt API from Experiment 3 — integrate via **OpenAI** in your app, not this encoder service. |

**Reality check:** After training, **Experiment 1 always writes the latest run to `Final_Source_Model/`**. To keep **E1 vs E2 vs E3 vs E4** on disk at once, save each run to a **different directory** (copy the whole HF folder after each Phase‑1 job, or change `FINAL_MODEL_PATH` in the notebook once per ablation). Metrics for all E‑runs are already in `experiment_log.csv` even if you only keep E4 weights today.

**Non‑deployable “models” in the log:** `E4_encoder_matchedLLMsubset_*` rows are the **same E4 weights** evaluated on a fixed 200‑example slice — not a separate checkpoint.

### Example: two encoders on two ports

```bash
# Terminal A — E4 cross-lingual (default path)
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8080

# Terminal B — e.g. few-shot Twi k=5 adapter (after Experiment 2 has written this folder)
export NLP_SERVE_MODEL_DIR="$PWD/Phase2_Outputs/fewshot_twi_5"
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8081
```

Your web app can call `http://localhost:8080/predict` vs `http://localhost:8081/predict` with the same JSON body.

## Prerequisites

- Repo `.venv` with `requirements.txt` + API deps installed (see root `README.md`).
- Each model directory must contain **`config.json`**, tokenizer files, and **`model.safetensors`** (or `pytorch_model.bin`) — same layout Hugging Face `save_pretrained` produces.

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
