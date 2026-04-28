# FinalProject — AfriHate cross-lingual hate speech

Multilingual fine-tuning (**Hausa, Amharic**, optional **Yoruba**) and **zero-shot** evaluation on **Twi** and **Nigerian Pidgin** using `Davlan/afro-xlmr-base` or `Davlan/afro-xlmr-large-76L`.

**Upstream repo:** [https://github.com/ICS-NLP/FinalProject](https://github.com/ICS-NLP/FinalProject)

## What’s in this repo

| Path | Contents |
|------|----------|
| `Source_Model_FineTuning.ipynb` | Phase 1 — source fine-tune + zero-shot evaluation, writes `experiment_log.csv` |
| `Phase2_FewShot_And_ErrorAnalysis.ipynb` | Phase 2 — k=5/10/20 few-shot adaptation + error sampling |
| `Phase3_TargetSupervised_LLM_Baseline.ipynb` | Phase 3 — supervised target ceiling (Twi, Pidgin) + optional LLM baseline (skipped without `OPENAI_API_KEY`) |
| `make_training_curves.py` | Renders train/val loss + eval-metric PNGs from any `Checkpoints/training_log_*.csv` |
| `make_error_summary.py` | Categorises `zero_shot_errors_sample.csv` into `zero_shot_error_summary.md` |
| `Checkpoints/` | `experiment_log.csv`, `results_report_table.csv`, `training_log_*.csv` |
| `Phase2_Outputs/` | `few_shot_results.csv`, `few_shot_curve.png`, `zero_shot_error_summary.md`, training-curve PNGs |
| `Phase3_Outputs/` | `transfer_gap_summary.csv`, per-target confusion matrices |
| `Final_Source_Model/` | Config + tokenizer only on GitHub (see below) |
| `EXPERIMENTS.md` | Full run matrix, transfer-gap analysis, error patterns, reproduction commands |
| `execute_notebook.sh` | Headless run with project `.venv` kernel |

## Model weights (not stored on GitHub)

`*.safetensors` files are **gitignored** (100 MB limit on GitHub). After cloning:

1. Create a Hugging Face token, accept the [AfriHate dataset](https://huggingface.co/datasets/afrihate/afrihate) terms, set `HF_TOKEN` in `.env` or `huggingface-cli login`.
2. Run `Source_Model_FineTuning.ipynb` (or `./execute_notebook.sh Source_Model_FineTuning.ipynb`) to fine-tune and write `model.safetensors` into `Final_Source_Model/`.
3. Optional: push your trained folder to the [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/guides/upload) as a model repo.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# .env with HF_TOKEN …
unset NLP_EXPERIMENT_ID NLP_SOURCE_LANGS NLP_MODEL NLP_LR NLP_NUM_EPOCHS
./execute_notebook.sh Source_Model_FineTuning.ipynb
./execute_notebook.sh Phase2_FewShot_And_ErrorAnalysis.ipynb
./execute_notebook.sh Phase3_TargetSupervised_LLM_Baseline.ipynb
.venv/bin/python make_training_curves.py
.venv/bin/python make_error_summary.py
```

See `EXPERIMENTS.md` for experiment IDs (E1–E4 zero-shot + T1/T2 supervised), the headline transfer-gap numbers, and recommended `NLP_MODEL` / `NLP_SOURCE_LANGS` settings.
