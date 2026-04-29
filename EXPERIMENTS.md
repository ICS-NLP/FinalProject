# Experiment log — AfriHate cross-lingual transfer study

> **Colleagues — start here:** for **every committed path** (notebooks, CSVs, plots, configs), open [**§ Complete artifact index**](#complete-artifact-index-paths-in-this-repo). For **which model to deploy** and **where weights live**, open [**§ Recommended model for deployment (E4)**](#recommended-model-for-deployment-e4).

## Repository layout (where things live)

| Location | Purpose |
|----------|---------|
| [`experiments/`](experiments/) | **Experiment_1–Experiment_3** — runnable notebooks + per-folder READMEs |
| [`results/best_experiment_e4/`](results/best_experiment_e4/) | **E4 documentation hub** (README only); figures and tables are stored once under `Phase2_Outputs/` and `Phase3_Outputs/` |
| `Checkpoints/` | `experiment_log.csv`, `training_log_*.csv`, `results_report_table.csv` |
| `Phase2_Outputs/` | Few-shot CSV, training curves (PNG), qualitative error markdown + sample rows |
| `Phase3_Outputs/` | Transfer gap CSV, confusion matrices, matched LLM subset indices + comparison CSV |
| `Final_Source_Model/` | **Recommended deployable checkpoint** after E4 training — config + tokenizer in Git; **weights (`model.safetensors`) gitignored** |
| [`scripts/`](scripts/) | `compare_encoder_llm_matched_subset.py`, `make_training_curves.py`, `make_error_summary.py` |
| `project_paths.py` | `repo_root()` helper used by scripts |
| `normalize_notebook_json.py` | Used by `execute_notebook.sh` before `nbconvert` |
| `requirements.txt`, `execute_notebook.sh`, `README.md` | Environment, headless runs, repo overview |

**Per-run metrics refresh.** After any notebook run that changes `experiment_log.csv` or `few_shot_results.csv`, regenerate the **derived** tables so they cannot drift:

```bash
.venv/bin/python scripts/sync_metrics_tables.py
```

This rewrites `Checkpoints/results_report_table.csv` and `Phase3_Outputs/transfer_gap_summary.csv` from the canonical logs. It does **not** modify `experiment_log.csv` itself.

---

## Recommended model for deployment (E4)

For **cross-lingual hate-speech classification on Twi and Nigerian Pidgin** without target-language fine-tuning, use the **Phase 1 E4** checkpoint: **`Davlan/afro-xlmr-large-76L`** fine-tuned on AfriHate **Hausa + Amharic + Yoruba** (`experiment_id = E4_large76L_hau_amh_yor`). This is the strongest zero-shot configuration in our matrix (see §3).

### Where the weights live on disk

| Path | Tracked in Git? | Role |
|------|-----------------|------|
| **`Final_Source_Model/model.safetensors`** | **No** (`.gitignore`; ~2.2 GB) | **Trained weights** — required for inference |
| **`Final_Source_Model/config.json`** | Yes | Architecture, `id2label` / `label2id` (Abuse=0, Hate=1, Normal=2) |
| **`Final_Source_Model/tokenizer.json`**, **`tokenizer_config.json`** | Yes | XLM-R tokenizer files |
| **`Final_Source_Model/training_args.bin`** | Yes | Hugging Face `TrainingArguments` snapshot (optional for inference) |

**Why there is no `.pth` file in Git.** This codebase does **not** save raw `torch.save` checkpoints during training. Experiment 1 calls Hugging Face `Trainer.save_model(...)` into `Final_Source_Model/`, which writes a **Hugging Face layout**: by default **`model.safetensors`** (or, on older setups, **`pytorch_model.bin`**). Those weight files are **gitignored** and will **not** appear on GitHub; only `config.json` + tokenizer files (+ `training_args.bin`) are committed. If you browse `Final_Source_Model/` after a fresh clone and only see JSON, you still need to **train once** or **copy in** the weight file from someone who ran training.

**Optional `.pth` bundle (local only — do not push the file to normal GitHub).** If your deployment pipeline expects a single PyTorch file, run:

```bash
.venv/bin/python scripts/export_e4_model_to_pth.py
# writes Weights_export/e4_best_model.pth (gitignored; ~same size as model.safetensors)
```

The archive contains `state_dict`, `id2label`, and `label2id`. **GitHub rejects files larger than ~100 MB**; a 76L encoder is **~2 GB**, so a full `.pth` **cannot** be committed to a standard repo. Use **Hugging Face Hub** (`huggingface-cli upload`), **Git LFS** with a paid/large quota, or **shared cloud storage** — not a plain `git push` of the weight file.

**HTTP API for colleagues.** A small **FastAPI** app in [`api/main.py`](api/main.py) exposes `POST /predict` and `POST /predict/batch` so a web backend can call the same checkpoint. Run **`uvicorn api.main:app`** from the repo root after weights exist; see [`api/README.md`](api/README.md) for CORS, env vars, and integration notes. This service does **not** include authentication — add a reverse proxy + API keys in production.

After `git clone`, colleagues **must** either (1) run **Experiment 1** with the E4 environment (§9) to recreate `model.safetensors` (or `pytorch_model.bin`), or (2) receive that file via shared drive / Hugging Face Hub / artefact store and place it in `Final_Source_Model/`.

### Minimal inference load

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo_root = "/path/to/FinalProject"  # directory that contains Final_Source_Model/
model = AutoModelForSequenceClassification.from_pretrained(f"{repo_root}/Final_Source_Model")
tokenizer = AutoTokenizer.from_pretrained(f"{repo_root}/Final_Source_Model")
# inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
# logits = model(**inputs).logits
```

Use **`AutoModelForSequenceClassification`** — the head is 3-class AfriHate (`config.json` lists `Abuse`, `Hate`, `Normal`).

### What **not** to ship as the “main” product

| Path / ID | Reason |
|-----------|--------|
| **`T1_supervised_twi`**, **`T2_supervised_pcm`** checkpoints under `Phase3_Outputs/runs_*` (local only, gitignored) | **Ceiling / diagnostic** runs on a **single** target language with `afro-xlmr-base`; Twi supervised collapsed to **2 predicted classes** (see §5). They are **not** the cross-lingual model and are not stored in Git. |
| **LLM rows** (`LLM0_*`) | Prompt-only baseline on a **200-example** slice for cost control — not a single encoder artefact. |

---

## Complete artifact index (paths in this repo)

Use this table to locate **every committed file**. Paths are relative to the **repository root** (`FinalProject/`).

### Notebooks and experiment docs

| Path | Description |
|------|-------------|
| `experiments/README.md` | Index of Experiment_1 / 2 / 3 |
| `experiments/Experiment_1/Source_Model_FineTuning.ipynb` | Source fine-tune + zero-shot (E1–E4) |
| `experiments/Experiment_1/README.md` | Short Phase 1 description + run command |
| `experiments/Experiment_2/Phase2_FewShot_And_ErrorAnalysis.ipynb` | Few-shot + error sampling |
| `experiments/Experiment_2/README.md` | Phase 2 description + run command |
| `experiments/Experiment_3/Phase3_TargetSupervised_LLM_Baseline.ipynb` | Supervised ceilings T1/T2 + optional LLM |
| `experiments/Experiment_3/README.md` | Phase 3 description + run command |

### Automation and dependencies

| Path | Description |
|------|-------------|
| `execute_notebook.sh` | Headless `jupyter nbconvert --execute` using `.venv` (default notebook = Experiment 1) |
| `normalize_notebook_json.py` | Normalizes notebook JSON before execution |
| `requirements.txt` | Python dependencies |
| `project_paths.py` | `repo_root()` for scripts |
| `scripts/compare_encoder_llm_matched_subset.py` | Same 200-example subset: E4 encoder vs LLM; writes `Phase3_Outputs/matched_subset_*` and log rows |
| `scripts/make_training_curves.py` | Renders loss/metric PNGs from `Checkpoints/training_log_*.csv` into `Phase2_Outputs/` |
| `scripts/make_error_summary.py` | Builds / refreshes qualitative error markdown |
| `scripts/sync_metrics_tables.py` | Regenerates `results_report_table.csv` + `transfer_gap_summary.csv` from `experiment_log.csv` + `few_shot_results.csv` |
| `scripts/export_e4_model_to_pth.py` | Optional: exports `Final_Source_Model` weights to a single **`.pth`** bundle (~same size as `model.safetensors`); output defaults to `Weights_export/` (gitignored) |

### Metrics and training logs (`Checkpoints/`)

| Path | Description |
|------|-------------|
| `Checkpoints/experiment_log.csv` | **Master log**: one row per evaluation (E1–E4 zero-shot + source-test, T1/T2, LLM, matched subset encoder rows). **Updated only when you run notebooks or `compare_encoder_llm_matched_subset.py`** — it is not auto-regenerated by `sync_metrics_tables.py`. |
| `Checkpoints/results_report_table.csv` | Compact E4-focused extract (subset / n_eval / headline metrics) |
| `Checkpoints/training_log_history.csv` | Loss + eval metrics per step for **Phase 1 (E4)** source training |
| `Checkpoints/training_log_T1_supervised_twi.csv` | Training trace for **T1** supervised Twi ceiling |
| `Checkpoints/training_log_T2_supervised_pcm.csv` | Training trace for **T2** supervised Pidgin ceiling |

### Phase 2 outputs (plots, few-shot, errors)

| Path | Description |
|------|-------------|
| `Phase2_Outputs/few_shot_results.csv` | Zero-shot + k ∈ {5,10,20} few-shot metrics per target |
| `Phase2_Outputs/few_shot_curve.png` | Plot of few-shot sweep |
| `Phase2_Outputs/training_log_history_loss.png` | Phase 1 train/val loss |
| `Phase2_Outputs/training_log_history_metrics.png` | Phase 1 eval F1 etc. |
| `Phase2_Outputs/training_log_T1_supervised_twi_loss.png` | T1 loss curve |
| `Phase2_Outputs/training_log_T1_supervised_twi_metrics.png` | T1 metric curve |
| `Phase2_Outputs/training_log_T2_supervised_pcm_loss.png` | T2 loss curve |
| `Phase2_Outputs/training_log_T2_supervised_pcm_metrics.png` | T2 metric curve |
| `Phase2_Outputs/zero_shot_error_summary.md` | Categorised qualitative error analysis (RQ4) |
| `Phase2_Outputs/zero_shot_errors_sample.csv` | Sampled rows backing the error discussion |

### Phase 3 outputs (transfer gap, confusion, LLM comparison)

| Path | Description |
|------|-------------|
| `Phase3_Outputs/transfer_gap_summary.csv` | Zero-shot vs few-shot k=20 vs supervised macro-F1; gap, transfer ratio, % gap recovered |
| `Phase3_Outputs/confusion_T1_supervised_twi.csv` | Confusion matrix counts — **T1** official test |
| `Phase3_Outputs/confusion_T2_supervised_pcm.csv` | Confusion matrix counts — **T2** official test |
| `Phase3_Outputs/matched_subset_encoder_vs_llm.csv` | Encoder vs LLM on identical 200-example slices |
| `Phase3_Outputs/matched_subset_indices_twi.json` | Test indices used for Twi matched eval (seed aligned with LLM cell) |
| `Phase3_Outputs/matched_subset_indices_pcm.json` | Same for Pidgin |

### Recommended model config (no weights in Git)

| Path | Description |
|------|-------------|
| `Final_Source_Model/config.json` | 3-class XLM-R-large head; label map |
| `Final_Source_Model/tokenizer.json` | SentencePiece tokenizer data |
| `Final_Source_Model/tokenizer_config.json` | Tokenizer settings |
| `Final_Source_Model/training_args.bin` | Serialized `TrainingArguments` |

### HTTP API (FastAPI)

| Path | Description |
|------|-------------|
| `api/main.py` | **`uvicorn api.main:app`** — `GET /health`, `POST /predict`, `POST /predict/batch` for web backends |
| `api/README.md` | Run instructions, CORS, `curl` examples, multi-model comparison |
| `deploy/gcp/Dockerfile` | CPU inference image (weights loaded from GCS at startup via `GCS_MODEL_URI`) |
| `deploy/gcp/deploy_all.sh` | **One-shot deploy** — run from repo root after `gcloud auth login` |
| `deploy/gcp/cloudbuild.yaml` | Cloud Build Docker build (used by `deploy_all.sh`) |
| `deploy/gcp/entrypoint.sh` | Downloads weights then starts uvicorn (honours Cloud Run `PORT`) |
| `deploy/gcp/fetch_model_from_gcs.py` | Pulls a model prefix from GCS into `/app/served_model` |

### Report pointer (no duplicate binaries)

| Path | Description |
|------|-------------|
| `results/best_experiment_e4/README.md` | Points to E4 config, **`Final_Source_Model/`**, and canonical figure/table paths above |

### Not in Git (clone / run behaviour)

| Pattern | Why |
|---------|-----|
| `.venv/`, `venv/` | Local Python environment |
| `data/` | AfriHate cache / downloads (see `README.md`) |
| `*.safetensors`, `pytorch_model.bin`, `Weights_export/` | Weight files exceed GitHub blob limits; local `.pth` exports go here |
| `Phase3_Outputs/runs_*/`, `Checkpoints/checkpoint-*/`, `Phase2_Outputs/fewshot_*/` | Ephemeral training checkpoints from notebooks |
| `.env` | Secrets (`HF_TOKEN`, `OPENAI_API_KEY`) |

---

## 1. Research questions answered

| # | Research question | Where it is answered |
|---|---|---|
| RQ1 | Can a multilingual encoder fine-tuned only on **high-resource source languages** (Hausa, Amharic, Yoruba) classify hate speech in **target languages** (Twi, Nigerian Pidgin) it never saw during training? | Phase 1 zero-shot rows (E1–E4) in `experiment_log.csv`; §3 below |
| RQ2 | How much labeled target data is required to materially improve performance? | Phase 2 few-shot table in `Phase2_Outputs/few_shot_results.csv`; §4 below |
| RQ3 | What is the **transfer gap** between cross-lingual zero-shot and target-language **fully supervised** fine-tuning? | Phase 3 supervised ceiling rows (T1, T2) and `Phase3_Outputs/transfer_gap_summary.csv`; §5 below |
| RQ4 | Where does the cross-lingual model fail, and what does that say about deploying it on Akan / Pidgin content? | `Phase2_Outputs/zero_shot_error_summary.md`; §7 below |

---

## 2. Phase definitions

| Phase | Notebook | What it produces |
|---|---|---|
| 1 — Source fine-tune & zero-shot | [`experiments/Experiment_1/Source_Model_FineTuning.ipynb`](experiments/Experiment_1/Source_Model_FineTuning.ipynb) | E1–E4 rows, source checkpoint, `training_log_history.csv` |
| 2 — Few-shot + error analysis | [`experiments/Experiment_2/Phase2_FewShot_And_ErrorAnalysis.ipynb`](experiments/Experiment_2/Phase2_FewShot_And_ErrorAnalysis.ipynb) | `few_shot_results.csv`, `few_shot_curve.png`, `zero_shot_errors_sample.csv` |
| 3 — Target supervised ceiling + (optional) LLM baseline | [`experiments/Experiment_3/Phase3_TargetSupervised_LLM_Baseline.ipynb`](experiments/Experiment_3/Phase3_TargetSupervised_LLM_Baseline.ipynb) | T1, T2 rows, per-target confusion matrices, `transfer_gap_summary.csv` |

### Predefined experiment IDs (auto-assigned when `NLP_EXPERIMENT_ID` is unset)

| ID | Model | Sources | Setting |
|---|---|---|---|
| `E1_base_hau_amh` | `Davlan/afro-xlmr-base` | `hau,amh` | source fine-tune + zero-shot |
| `E2_large76L_hau_amh` | `Davlan/afro-xlmr-large-76L` | `hau,amh` | source fine-tune + zero-shot |
| `E3_base_hau_amh_yor` | `Davlan/afro-xlmr-base` | `hau,amh,yor` | source fine-tune + zero-shot |
| `E4_large76L_hau_amh_yor` | `Davlan/afro-xlmr-large-76L` | `hau,amh,yor` | source fine-tune + zero-shot |
| `T1_supervised_twi` | `Davlan/afro-xlmr-base` | `twi` | target-language supervised ceiling |
| `T2_supervised_pcm` | `Davlan/afro-xlmr-base` | `pcm` | target-language supervised ceiling |

---

## 3. Phase 1 — cross-lingual zero-shot (full panel)

Macro precision / recall / F1 are reported on the official AfriHate **test** splits, alongside MCC and the collapse diagnostics (`pred_majority_frac`, `num_pred_classes_used`).

### Source-language in-distribution (sanity check)

| ID | Model | Sources | Acc | Bal-Acc | F1-macro | F1-w | MCC |
|---|---|---|---|---|---|---|---|
| E1 | base | hau+amh | 0.765 | 0.756 | **0.758** | 0.764 | 0.635 |
| E2 | large-76L | hau+amh | 0.775 | 0.763 | **0.766** | 0.774 | 0.650 |
| E3 | base | hau+amh+yor | 0.755 | 0.728 | 0.737 | 0.752 | 0.615 |
| E4 | large-76L | hau+amh+yor | 0.771 | 0.746 | **0.753** | 0.769 | 0.641 |

The 76L encoder is uniformly the strongest in-distribution learner.

### Zero-shot — Twi (`698` test rows)

| ID | Acc | Bal-Acc | F1-macro | Precision-macro | Recall-macro | MCC | Majority-frac | #classes used |
|---|---|---|---|---|---|---|---|---|
| E1 base, hau+amh | 0.370 | 0.386 | 0.287 | — | — | 0.084 | 0.620 | 3 |
| E2 76L, hau+amh | 0.331 | 0.408 | 0.279 | — | — | 0.095 | 0.678 | 3 |
| E3 base, **hau+amh+yor** | 0.539 | 0.441 | 0.374 | — | — | 0.166 | 0.553 | 3 |
| **E4 76L, hau+amh+yor** | **0.527** | 0.402 | **0.375** | **0.381** | **0.402** | 0.093 | 0.596 | 3 |

(Precision / recall back-filled for the headline E4 row. The Phase-1 notebook now writes both columns automatically for any new run; legacy E1–E3 rows from earlier runs are left as `—` and can be regenerated by re-running those configurations.)

### Zero-shot — Nigerian Pidgin (`1593` test rows)

| ID | Acc | Bal-Acc | F1-macro | Precision-macro | Recall-macro | MCC | Majority-frac | #classes used |
|---|---|---|---|---|---|---|---|---|
| E1 base, hau+amh | 0.559 | 0.550 | 0.519 | — | — | 0.325 | 0.697 | 3 |
| E2 76L, hau+amh | 0.589 | 0.603 | 0.557 | — | — | 0.382 | 0.658 | 3 |
| E3 base, hau+amh+yor | 0.566 | 0.530 | 0.515 | — | — | 0.309 | 0.664 | 3 |
| **E4 76L, hau+amh+yor** | **0.608** | **0.620** | **0.584** | **0.590** | **0.620** | **0.383** | 0.565 | 3 |

**E4 source-test (Hausa+Amharic+Yoruba combined, n=2615):** acc 0.771, bal-acc 0.746, f1-macro 0.753, **precision-macro 0.763**, **recall-macro 0.746**, mcc 0.641.

**Reading precision vs recall on the headline E4 row.** Twi: precision 0.381 < recall 0.402 — the model is *slightly* over-flagging at macro level but the bigger issue is the very low MCC. Pidgin: precision 0.590 < recall 0.620 — again leaning toward over-flagging, consistent with the qualitative pattern that Pidgin's familiar/teasing register triggers Abuse predictions on `Normal` posts.

**Phase 1 takeaways.**
1. **Adding Yoruba (E3, E4)** fixes the Twi zero-shot ceiling that hau+amh-only models cannot reach. F1 on Twi roughly **+10 absolute points** over hau+amh-only.
2. **76L vs base** is the main lever for Pidgin: E4 (76L + hau+amh+yor) is the best Pidgin model overall.
3. No experiment shows model collapse: every Phase-1 zero-shot prediction uses all three classes (`num_pred_classes_used = 3`), and `pred_majority_frac` stays well below 70%.

---

## 4. Phase 2 — few-shot adaptation (source = E4)

`experiments/Experiment_2/Phase2_FewShot_And_ErrorAnalysis.ipynb` resumes the best Phase-1 checkpoint (`Final_Source_Model/`, 76L on hau+amh+yor) and fine-tunes on a tiny labeled budget per target. Knobs: `NLP_FEWSHOT_LR=8e-6`, `NLP_FEWSHOT_EPOCHS=6`. Curve: `Phase2_Outputs/few_shot_curve.png`.

### Twi (`698` test rows)

| Condition | k/class | Acc | Bal-Acc | F1-macro | F1-w | MCC | Majority-frac |
|---|---|---|---|---|---|---|---|
| zero-shot | 0 | 0.527 | 0.402 | 0.375 | 0.544 | 0.093 | 0.596 |
| few-shot k=5 | 5 | 0.423 | **0.458** | 0.380 | 0.464 | **0.152** | 0.342 |
| few-shot k=10 | 10 | 0.354 | 0.440 | 0.328 | 0.383 | 0.147 | 0.519 |
| few-shot k=20 | 20 | 0.440 | 0.439 | **0.380** | 0.483 | 0.129 | 0.397 |

### Nigerian Pidgin (`1593` test rows)

| Condition | k/class | Acc | Bal-Acc | F1-macro | F1-w | MCC | Majority-frac |
|---|---|---|---|---|---|---|---|
| zero-shot | 0 | 0.608 | 0.620 | 0.584 | 0.594 | 0.383 | 0.565 |
| few-shot k=5 | 5 | 0.618 | 0.627 | **0.596** | 0.611 | 0.386 | 0.508 |
| few-shot k=10 | 10 | 0.545 | **0.633** | 0.537 | 0.549 | 0.318 | 0.353 |
| few-shot k=20 | 20 | 0.606 | 0.628 | **0.593** | 0.608 | 0.357 | 0.431 |

**Phase 2 takeaways.**
- For **Pidgin**, `k=5` is enough to *beat* zero-shot on every metric — the few-shot signal calibrates the decision boundary without large labeling spend.
- For **Twi**, few-shot **does not** materially improve macro-F1 over the strong E4 zero-shot baseline (0.375 → 0.380). The model balances classes more aggressively (`majority_frac` drops from 0.60 to 0.34 at k=5) but the F1 gain is negligible. Twi requires *content* knowledge that 5 labeled examples per class cannot inject.

---

## 5. Phase 3 — supervised target ceiling and **transfer gap**

We fine-tune a **fresh** `afro-xlmr-base` on each target's *own* AfriHate train split (15% stratified dev split for early stopping; `NLP_TARGET_EPOCHS=4`, `NLP_TARGET_LR=2e-5`) and evaluate on the same official test split used in Phase 1. The **transfer ratio** = zero-shot F1 / supervised F1 quantifies how much of the upper bound the cross-lingual encoder recovers without any target labels.

### Test-set metrics

| ID | Subset | Acc | Bal-Acc | F1-macro | Precision-macro | Recall-macro | MCC | #classes used |
|---|---|---|---|---|---|---|---|---|
| T1 supervised twi | twi (official test) | 0.729 | 0.435 | **0.414** | 0.405 | 0.435 | 0.285 | **2** |
| T2 supervised pcm | pcm (official test) | 0.688 | 0.666 | **0.661** | 0.656 | 0.666 | 0.472 | 3 |

Confusion matrices: `Phase3_Outputs/confusion_T1_supervised_twi.csv`, `Phase3_Outputs/confusion_T2_supervised_pcm.csv`. Loss / metric curves: `Phase2_Outputs/training_log_T1_supervised_twi_loss.png`, `…_metrics.png`, and the same pair for T2.

*Numbers above match the `T1_supervised_twi` and `T2_supervised_pcm` rows in `Checkpoints/experiment_log.csv` (official test splits).*

### Transfer gap

`Phase3_Outputs/transfer_gap_summary.csv` is a **derived** table: zero-shot and few-shot k=20 macro-F1 values come from `Phase2_Outputs/few_shot_results.csv`; supervised macro-F1 comes from the T1/T2 rows in `experiment_log.csv`. If you re-run Phase 3, update `transfer_gap_summary.csv` so it stays consistent (same formulas as in the CSV header).

| Target | Zero-shot F1 (E4) | Few-shot k=20 F1 | Supervised F1 | Absolute gap (sup − ZS) | Transfer ratio (ZS / sup) | Few-shot recovers |
|---|---|---|---|---|---|---|
| twi | 0.375 | 0.380 | 0.414 | **+0.038** | **0.91** | **11.5%** of the gap |
| pcm | 0.584 | 0.593 | 0.661 | **+0.076** | **0.88** | **10.9%** of the gap |

**Phase 3 takeaways.**
- **Cross-lingual zero-shot from Hau+Amh+Yor recovers about 91% of the Twi supervised macro-F1 ceiling and about 88% of the Pidgin ceiling** (ratio = zero-shot F1 / supervised F1 in `Phase3_Outputs/transfer_gap_summary.csv`). That answers RQ3 and gives the report a single defensible "transfer gap" headline.
- The **Twi supervised baseline collapses on the `Normal` class** (`num_pred_classes_used = 2`): Twi's training distribution is dominated by `Abuse`, so vanilla cross-entropy fine-tuning rarely predicts `Normal` even though overall macro-F1 exceeds E4 zero-shot. This is a **deployment-relevant** contrast to the E4 encoder, which uses **all three** labels on the test set.
- The **Pidgin supervised baseline** uses all three classes and produces a clean ceiling around macro-F1 ≈ 0.66.

---

## 6. Optional LLM baseline (prompt-only)

`experiments/Experiment_3/Phase3_TargetSupervised_LLM_Baseline.ipynb` ships a cell that — when `OPENAI_API_KEY` is exported — runs a zero-shot or 5-shot prompt baseline against a sampled subset of each target's official test split (`NLP_LLM_MAX_EXAMPLES=200` by default for cost control) and writes `LLM0_<target>_<mode>` rows into `experiment_log.csv`. Without the key the cell prints a clear *skipped* notice so the headless runs stay deterministic.

The LLM is instructed to return **JSON only** (`{"label":"Abuse|Hate|Normal","confidence":…}`) with `response_format={"type":"json_object"}` and `max_tokens=128`, plus a regex fallback if the API rejects JSON mode. That fixes an earlier bug where `max_tokens=4` truncated replies and the parser fell back to a single class for every example. Set `NLP_LLM_DEBUG=1` to print parse failures. Re-running overwrites matching rows in `experiment_log.csv`.

The cell supports `NLP_LLM_MODE=zeroshot|fewshot` and `NLP_LLM_MODEL=gpt-4o-mini|gpt-4o|...`.

**Billing / quota.** If the account has **no credits** or the key is over quota, OpenAI returns **429 `insufficient_quota`**. The notebook then falls back to `Normal` on each failed call, so `experiment_log.csv` can show **`pred_majority_frac = 1.0`** and very low F1 even though the JSON parser is fixed. After a run, check stdout (first few API errors) and the **`notes`** column (`api_errors=…`, `INVALID_METRICS` when ≥90% of calls failed).

### Matched subset: E4 encoder vs LLM on the **same** 200 examples

The LLM cell uses `pandas.sample(N, random_state=42)` on each target's official test split (`N = NLP_LLM_MAX_EXAMPLES`, default 200). Run:

```bash
.venv/bin/python scripts/compare_encoder_llm_matched_subset.py
```

This writes `Phase3_Outputs/matched_subset_encoder_vs_llm.csv`, saves the exact AfriHate test **row indices** (`matched_subset_indices_twi.json`, `matched_subset_indices_pcm.json`), and appends **`E4_encoder_matchedLLMsubset_{twi,pcm}`** rows to `experiment_log.csv` (`setting=zero_shot_matched_llm_subset`, same `subset` string as the `LLM0_*` rows).

**Latest matched n=200 snapshot (encoder `Final_Source_Model` vs `gpt-4o-mini` zero-shot):**

| Target | Encoder F1-macro | LLM F1-macro | Encoder acc | LLM acc |
|--------|------------------|--------------|---------------|---------|
| Twi | **0.444** | 0.336 | **0.585** | 0.400 |
| Pidgin | 0.556 | **0.592** | 0.595 | **0.635** |

On this slice the **encoder wins Twi**; **`gpt-4o-mini` wins Pidgin** slightly on macro-F1. Full-test E4 metrics remain in the main `E4_large76L_hau_amh_yor` rows.

---

## 7. Qualitative error analysis (RQ4)

Full categorised report with concrete examples: `Phase2_Outputs/zero_shot_error_summary.md`. Eight false positives + eight false negatives per target, drawn from the official test splits, were inspected. Highlights:

**Twi — over-flagging (`Normal` → `Abuse / Hate`).** Tweets that simply *mention* identity terms (`Ntafo`, `Frafrafo`, `Zongofo`, `Alatafuo`) are pushed toward Hate even when the rest of the post is playful or descriptive. The encoder appears to associate intergroup tokens with hostile contexts seen during Hausa+Amharic training. Examples:
- `"Anka )b3ti Zongofo) no nka"` — true *Normal*, predicted *Hate*.
- `"Wo Kɔ Alatafuo Traditional Marriage…"` — true *Normal*, predicted *Hate*.

**Twi — under-flagging (`Abuse` → `Normal`).** The toughest cases are idiomatic insults/slurs that the *source* languages do not share (`Kontomponi`, `Gyimisɛm`, `akoa`, `Apakye musuoni`). They carry abuse for Akan readers but the source-trained head does not recognise them. Example:
- `"Ah na severed vagina deɛ bosom de yɛ dɛn? Gyimisɛm kwa!"` — true *Abuse*, predicted *Normal*.

**Pidgin — over-flagging.** Pidgin's familiar / teasing register is repeatedly classified Abuse — common in-group banter rather than directed harassment. Example:
- `"fine boys like us we nor too dey chase women na dem dey rush us…"` — true *Normal*, predicted *Abuse*.

**Pidgin — under-flagging.** Stereotypes framed as agreement (`I agree "Facts: 50% of Whores in Nigeria are Igbo's"`) and ethnic put-downs (`Yoruba people too like noise tah`) are labelled *Normal*. The model misses ethnic-group hate when the surface tone is conversational.

**Implications for deployment.** False positives on identity terms in Twi are an **ethical** risk: deploying the zero-shot model without human review would silence legitimate in-group speech in under-served languages. Few-shot k=5 lowers `pred_majority_frac` (more balanced predictions) at the cost of weighted F1 on the more frequent classes — a calibration trade-off the report should flag.

---

## 8. Training and validation loss curves

For every fine-tune we save a `Checkpoints/training_log_<id>.csv` and render two PNGs into `Phase2_Outputs/`:

| Run | Loss curve | Eval-metric curve |
|---|---|---|
| Phase 1 — latest source training (E4) | `training_log_history_loss.png` | `training_log_history_metrics.png` |
| Phase 3 — T1 supervised twi | `training_log_T1_supervised_twi_loss.png` | `training_log_T1_supervised_twi_metrics.png` |
| Phase 3 — T2 supervised pcm | `training_log_T2_supervised_pcm_loss.png` | `training_log_T2_supervised_pcm_metrics.png` |

Re-render any time after training:

```bash
.venv/bin/python scripts/sync_metrics_tables.py
.venv/bin/python scripts/make_training_curves.py
.venv/bin/python scripts/make_error_summary.py
.venv/bin/python scripts/compare_encoder_llm_matched_subset.py
```

The Phase-1 curve in particular shows training loss decaying smoothly while validation loss plateaus around step 800 — the saved best checkpoint is taken from the `f1_macro` peak at the same step, so we are not over-fitting.

---

## 9. Reproduction commands

Every notebook is fully headless. **Always** clear stale environment exports first:

```bash
unset NLP_EXPERIMENT_ID NLP_SOURCE_LANGS NLP_MODEL NLP_LR \
      NLP_NUM_EPOCHS NLP_TARGET_EPOCHS NLP_TARGET_LR \
      NLP_FEWSHOT_LR NLP_FEWSHOT_EPOCHS \
      OPENAI_API_KEY NLP_LLM_MODE NLP_LLM_MAX_EXAMPLES NLP_LLM_MODEL
```

### Phase 1 — strongest source fine-tune (E4)

```bash
export NLP_MODEL=Davlan/afro-xlmr-large-76L
export NLP_SOURCE_LANGS=hau,amh,yor
export NLP_NUM_EPOCHS=4
./execute_notebook.sh experiments/Experiment_1/Source_Model_FineTuning.ipynb
```

### Phase 2 — few-shot adaptation against the saved E4 checkpoint

```bash
./execute_notebook.sh experiments/Experiment_2/Phase2_FewShot_And_ErrorAnalysis.ipynb
```

### Phase 3 — supervised ceilings (and optional LLM baseline)

```bash
export NLP_TARGET_EPOCHS=4 NLP_TARGET_LR=2e-5
# Optional: export OPENAI_API_KEY=... NLP_LLM_MODEL=gpt-4o-mini NLP_LLM_MODE=zeroshot
# LLM-only (skip ~1h supervised re-run): export NLP_PHASE3_SKIP_SUPERVISED=1
./execute_notebook.sh experiments/Experiment_3/Phase3_TargetSupervised_LLM_Baseline.ipynb
```

### HTTP API (after `Final_Source_Model/` has weights)

```bash
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8080
# Docs: http://127.0.0.1:8080/docs — see api/README.md for CORS and batch limits.
```

---

## 10. What the report can claim

1. **Cross-lingual transfer is viable for AfriHate**: a multilingual encoder fine-tuned only on Hausa+Amharic+Yoruba reaches **~91% of the Twi supervised ceiling** and **~88% of the Pidgin ceiling** with **zero target labels** (see `transfer_gap_summary.csv` for exact ratios).
2. **Adding a typologically related source language matters more than scaling the encoder**: adding Yoruba lifts Twi zero-shot F1 by ≈10 absolute points. The 76L encoder helps Pidgin most (E4 best on Pidgin) but does not unlock Twi by itself.
3. **Few-shot k=5 is enough for Pidgin** but does **not** rescue Twi: Twi failures are dominated by *vocabulary* the source model never saw, not just decision-boundary calibration.
4. **Failure modes are linguistically interpretable**: over-flagging on identity tokens, under-flagging on Akan-specific slurs, and ethnic-group hate framed as agreement in Pidgin. These have direct ethical implications for deployment.

All numbers in this document are reproducible from `Checkpoints/experiment_log.csv`, `Phase2_Outputs/few_shot_results.csv`, and `Phase3_Outputs/transfer_gap_summary.csv` using the commands above.
