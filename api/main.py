"""HTTP API for the E4 AfriHate classifier (fine-tuned XLM-R-large in Final_Source_Model/).

Run from repository root:
  .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8080

Env:
  NLP_SERVE_MODEL_DIR — folder with config.json + tokenizer + model.safetensors (default: <repo>/Final_Source_Model)
  NLP_SERVE_MAX_BATCH — max texts per /predict/batch (default: 32)
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import repo_root  # noqa: E402

_model = None
_tokenizer = None
_device: torch.device | None = None
_id2label: dict[int, str] = {}
_max_batch = 32


def _model_dir() -> Path:
    override = os.environ.get("NLP_SERVE_MODEL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "Final_Source_Model"


def _load():
    global _model, _tokenizer, _device, _id2label, _max_batch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _max_batch = max(1, int(os.environ.get("NLP_SERVE_MAX_BATCH", "32")))
    path = _model_dir()
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json under {path}")
    if not (path / "model.safetensors").is_file() and not (path / "pytorch_model.bin").is_file():
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin in {path}. Train Experiment 1 (E4) or copy weights."
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(str(path))
    _model = AutoModelForSequenceClassification.from_pretrained(str(path))
    _model.eval()
    _model.to(_device)
    _id2label = {int(k): v for k, v in _model.config.id2label.items()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load()
    except FileNotFoundError as e:
        # Still start server so /health can report not ready
        app.state.load_error = str(e)
    else:
        app.state.load_error = None
    yield
    global _model, _tokenizer
    _model = None
    _tokenizer = None


app = FastAPI(
    title="AfriHate E4 classifier API",
    description="Cross-lingual hate speech (Abuse / Hate / Normal) using the fine-tuned Davlan/afro-xlmr-large-76L checkpoint.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("NLP_SERVE_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000, description="Single post to classify")


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Batch of posts (length capped at runtime by NLP_SERVE_MAX_BATCH)",
    )


def _scores_from_logits(logits: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    return {_id2label[i]: float(probs[i]) for i in range(len(probs))}


def _predict_one(text: str) -> dict[str, Any]:
    assert _model is not None and _tokenizer is not None and _device is not None
    enc = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    enc = {k: v.to(_device) for k, v in enc.items()}
    with torch.inference_mode():
        logits = _model(**enc).logits
    pred_id = int(logits.argmax(dim=-1).item())
    return {
        "label": _id2label[pred_id],
        "label_id": pred_id,
        "scores": _scores_from_logits(logits),
    }


@app.get("/health")
def health():
    err = getattr(app.state, "load_error", None)
    if err:
        return {"status": "unhealthy", "detail": err}
    return {"status": "ok", "device": str(_device), "model_dir": str(_model_dir())}


@app.post("/predict")
def predict(body: PredictRequest):
    err = getattr(app.state, "load_error", None)
    if err:
        raise HTTPException(status_code=503, detail=err)
    out = _predict_one(body.text.strip())
    return {"text": body.text, **out}


@app.post("/predict/batch")
def predict_batch(body: PredictBatchRequest):
    err = getattr(app.state, "load_error", None)
    if err:
        raise HTTPException(status_code=503, detail=err)
    texts = [t.strip() for t in body.texts if t and t.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="No non-empty texts")
    if len(texts) > _max_batch:
        raise HTTPException(
            status_code=400,
            detail=f"Too many texts (max {_max_batch}). Raise NLP_SERVE_MAX_BATCH if needed.",
        )
    return {"predictions": [{"text": t, **_predict_one(t)} for t in texts]}
