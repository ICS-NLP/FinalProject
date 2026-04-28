#!/usr/bin/env python3
"""Fair encoder vs LLM comparison on the *identical* random test subset.

Phase3 LLM cell uses:
  test_df = DataFrame(...); test_df.sample(N, random_state=42).reset_index(drop=True)
with N = int(os.environ.get("NLP_LLM_MAX_EXAMPLES", "200")).

This script rebuilds that subset for `twi` and `pcm`, runs `Final_Source_Model/`
(the saved E4 checkpoint), and writes:
  - Phase3_Outputs/matched_subset_indices_{twi,pcm}.json  (original row indices)
  - Phase3_Outputs/matched_subset_encoder_vs_llm.csv   (side-by-side metrics)
It also appends two rows to `Checkpoints/experiment_log.csv` (dedup on
experiment_id + setting + subset) so the report can cite them next to
`LLM0_*_zeroshot`.

Requires: HF_TOKEN (or huggingface-cli login), network for dataset load.

Run:
    .venv/bin/python scripts/compare_encoder_llm_matched_subset.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=r".*pin_memory.*")

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from project_paths import ROOT
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from huggingface_hub import get_token
from huggingface_hub.utils import disable_progress_bars
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers import logging as transformers_logging

disable_progress_bar()
disable_progress_bars()
transformers_logging.set_verbosity_error()

CHECKPOINT_DIR = ROOT / "Checkpoints"
OUT_DIR = ROOT / "Phase3_Outputs"
OUT_DIR.mkdir(exist_ok=True)

LABEL_NAMES = ["Abuse", "Hate", "Normal"]
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
SEED = 42
MAX_LEN = int(os.environ.get("NLP_MAX_LENGTH", "128"))
N_SAMPLE = int(os.environ.get("NLP_LLM_MAX_EXAMPLES", "200"))

MODEL_DIR = ROOT / "Final_Source_Model"
if not (MODEL_DIR / "config.json").is_file():
    raise FileNotFoundError(f"Missing {MODEL_DIR}/config.json — train Phase 1 E4 first.")

hf_token = os.getenv("HF_TOKEN") or get_token()
if not hf_token:
    raise RuntimeError("Set HF_TOKEN or run `hf auth login`.")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def evaluate_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict:
    labels = np.asarray(labels).astype(np.int64).ravel()
    predictions = np.asarray(predictions).astype(np.int64).ravel()
    n_classes = len(LABEL_NAMES)
    class_idx = list(range(n_classes))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", labels=class_idx, zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", labels=class_idx, zero_division=0
    )
    _, _, f1_per, _ = precision_recall_fscore_support(
        labels, predictions, average=None, labels=class_idx, zero_division=0
    )
    pred_counts = np.bincount(predictions, minlength=n_classes)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_w),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_Abuse": float(f1_per[0]),
        "f1_Hate": float(f1_per[1]),
        "f1_Normal": float(f1_per[2]),
        "pred_majority_frac": float(pred_counts.max()) / max(1, len(predictions)),
        "num_pred_classes_used": float((pred_counts > 0).sum()),
    }


def build_matched_frame(target_lang: str) -> tuple[pd.DataFrame, list[int]]:
    """Same construction + sample as Phase3 `llm_baseline_run` (before LLM calls)."""
    raw = load_dataset("afrihate/afrihate", target_lang, token=hf_token)["test"]
    raw = raw.filter(lambda ex: ex["label"] in LABEL2ID)
    text_col = "tweet" if "tweet" in raw.column_names else "text"
    full = pd.DataFrame({"text": list(raw[text_col]), "label": list(raw["label"])})
    if len(full) > N_SAMPLE:
        sampled = full.sample(N_SAMPLE, random_state=SEED)
        indices = sampled.index.astype(int).tolist()
        out = sampled.reset_index(drop=True)
    else:
        indices = full.index.astype(int).tolist()
        out = full.reset_index(drop=True)
    return out, indices


def predict_encoder(texts: list[str], labels_str: list[str]) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    init: dict = {}
    if device == "mps":
        init["attn_implementation"] = "eager"
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, **init)

    enc = tok(texts, padding="max_length", truncation=True, max_length=MAX_LEN)
    enc = {k: torch.tensor(v) for k, v in enc.items()}
    enc["labels"] = torch.tensor([LABEL2ID[x] for x in labels_str], dtype=torch.long)

    class TinyDs(torch.utils.data.Dataset):
        def __len__(self):
            return len(texts)

        def __getitem__(self, i):
            return {k: enc[k][i] for k in enc}

    ds = TinyDs()
    args = TrainingArguments(
        output_dir=str(ROOT / "tmp_matched_subset_eval"),
        per_device_eval_batch_size=16,
        dataloader_pin_memory=False,
        report_to="none",
        disable_tqdm=True,
        use_cpu=device == "cpu",
    )
    trainer = Trainer(model=mdl, args=args, data_collator=DataCollatorWithPadding(tok))
    pred = trainer.predict(ds)
    return np.argmax(pred.predictions.astype(np.float64), axis=-1)


def append_experiment_log(rows: list[dict]) -> None:
    log_path = CHECKPOINT_DIR / "experiment_log.csv"
    log_columns = [
        "experiment_id",
        "model",
        "source_languages",
        "num_epochs",
        "learning_rate",
        "setting",
        "subset",
        "n_eval",
        "accuracy",
        "balanced_accuracy",
        "mcc",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "pred_majority_frac",
        "num_pred_classes_used",
        "notes",
    ]
    new_df = pd.DataFrame(rows)
    for col in log_columns:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[log_columns]
    if log_path.exists():
        prev = pd.read_csv(log_path)
        for col in log_columns:
            if col not in prev.columns:
                prev[col] = ""
        prev = prev[log_columns]
        keys = set(zip(new_df["experiment_id"], new_df["setting"], new_df["subset"]))
        keep_mask = ~prev.apply(
            lambda r: (r["experiment_id"], r["setting"], r["subset"]) in keys, axis=1
        )
        combined = pd.concat([prev[keep_mask], new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(log_path, index=False)


def main() -> None:
    # Match Phase3 config.json base model name for logging
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    base_model = cfg.get("_name_or_path") or cfg.get("model_type") or "Final_Source_Model"
    if not str(base_model).startswith("Davlan"):
        base_model = "Davlan/afro-xlmr-large-76L"

    log_df = pd.read_csv(CHECKPOINT_DIR / "experiment_log.csv") if (CHECKPOINT_DIR / "experiment_log.csv").exists() else pd.DataFrame()

    comparison_rows: list[dict] = []
    log_rows: list[dict] = []

    for target in ("twi", "pcm"):
        frame, orig_indices = build_matched_frame(target)
        (OUT_DIR / f"matched_subset_indices_{target}.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "seed": SEED,
                    "n_requested": N_SAMPLE,
                    "n_actual": len(frame),
                    "afrihate_test_row_indices": orig_indices,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        preds = predict_encoder(frame["text"].astype(str).tolist(), frame["label"].astype(str).tolist())
        y = np.array([LABEL2ID[x] for x in frame["label"]])
        enc_m = evaluate_predictions(y, preds)

        subset_label = f"{target} (sampled test, n={len(frame)})"
        llm_row = None
        if not log_df.empty:
            hit = log_df[
                (log_df["experiment_id"] == f"LLM0_{target}_zeroshot")
                & (log_df["setting"] == "llm_zeroshot")
            ]
            if not hit.empty:
                llm_row = hit.iloc[-1].to_dict()

        comparison_rows.append(
            {
                "target": target,
                "n": len(frame),
                "subset_label": subset_label,
                "encoder_f1_macro": enc_m["f1_macro"],
                "encoder_accuracy": enc_m["accuracy"],
                "encoder_balanced_accuracy": enc_m["balanced_accuracy"],
                "encoder_mcc": enc_m["mcc"],
                "encoder_precision_macro": enc_m["precision_macro"],
                "encoder_recall_macro": enc_m["recall_macro"],
                "encoder_pred_majority_frac": enc_m["pred_majority_frac"],
                "encoder_num_pred_classes_used": enc_m["num_pred_classes_used"],
                "llm_f1_macro": float(llm_row["f1_macro"]) if llm_row else None,
                "llm_accuracy": float(llm_row["accuracy"]) if llm_row else None,
                "llm_balanced_accuracy": float(llm_row["balanced_accuracy"]) if llm_row else None,
                "llm_mcc": float(llm_row["mcc"]) if llm_row else None,
                "llm_precision_macro": float(llm_row["precision_macro"]) if llm_row else None,
                "llm_recall_macro": float(llm_row["recall_macro"]) if llm_row else None,
                "llm_pred_majority_frac": float(llm_row["pred_majority_frac"]) if llm_row else None,
                "llm_num_pred_classes_used": float(llm_row["num_pred_classes_used"]) if llm_row else None,
            }
        )

        log_rows.append(
            {
                "experiment_id": f"E4_encoder_matchedLLMsubset_{target}",
                "model": base_model,
                "source_languages": "hau+amh+yor",
                "num_epochs": 4,
                "learning_rate": "2e-05",
                "setting": "zero_shot_matched_llm_subset",
                "subset": subset_label,
                "n_eval": int(len(frame)),
                **{k: enc_m[k] for k in (
                    "accuracy", "balanced_accuracy", "mcc", "f1_macro", "f1_weighted",
                    "precision_macro", "recall_macro", "pred_majority_frac", "num_pred_classes_used",
                )},
                "notes": f"Encoder E4 checkpoint on same {len(frame)}-row subset as LLM (seed={SEED}, NLP_LLM_MAX_EXAMPLES={N_SAMPLE}).",
            }
        )

    cmp = pd.DataFrame(comparison_rows)
    out_csv = OUT_DIR / "matched_subset_encoder_vs_llm.csv"
    cmp.to_csv(out_csv, index=False)
    print(cmp.to_string(index=False))
    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {OUT_DIR}/matched_subset_indices_twi.json")
    print(f"Wrote: {OUT_DIR}/matched_subset_indices_pcm.json")

    append_experiment_log(log_rows)
    print(f"Appended/replaced rows in {CHECKPOINT_DIR / 'experiment_log.csv'}:")
    for r in log_rows:
        print(f"  {r['experiment_id']} | {r['subset']} | f1_macro={r['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
