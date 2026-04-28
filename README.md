# FinalProject — AfriHate cross-lingual hate speech

Multilingual fine-tuning (**Hausa, Amharic**, optional **Yoruba**) and **zero-shot** evaluation on **Twi** and **Nigerian Pidgin** using `Davlan/afro-xlmr-base` or `Davlan/afro-xlmr-large-76L`.

**Upstream repo:** [https://github.com/ICS-NLP/FinalProject](https://github.com/ICS-NLP/FinalProject)

## Repository layout

| Path | Contents |
|------|----------|
| [`experiments/`](experiments/) | **Experiment_1–Experiment_3** folders (notebooks + README each) |
| [`results/best_experiment_e4/`](results/best_experiment_e4/) | **Headline configuration (E4)** — key figures + summary for the report |
| [`scripts/`](scripts/) | Helper CLIs (`compare_encoder_llm_matched_subset.py`, `make_training_curves.py`, `make_error_summary.py`) |
| `project_paths.py` | Shared `repo_root()` for scripts |
| `Checkpoints/` | `experiment_log.csv`, `training_log_*.csv` (all phases write here) |
| `Phase2_Outputs/` | Few-shot CSVs, plots, qualitative error markdown |
| `Phase3_Outputs/` | Transfer gap, confusion matrices, matched LLM subset artefacts |
| `Final_Source_Model/` | Best Phase 1 checkpoint (weights are **gitignored**; see below) |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Full experiment matrix, metrics, reproduction commands |
| `execute_notebook.sh` | Headless notebook execution (default = Experiment 1 notebook) |

## Model weights (not stored on GitHub)

`*.safetensors` files are **gitignored** (GitHub 100 MB blob limit). After cloning:

1. Create a Hugging Face token, accept the [AfriHate dataset](https://huggingface.co/datasets/afrihate/afrihate) terms, set `HF_TOKEN` in `.env` or `huggingface-cli login`.
2. Run Phase 1 (see below) to fine-tune and write `model.safetensors` into `Final_Source_Model/`.
3. Optional: upload weights to the [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/guides/upload).

## Quick start (from this directory)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# .env with HF_TOKEN …
unset NLP_EXPERIMENT_ID NLP_SOURCE_LANGS NLP_MODEL NLP_LR NLP_NUM_EPOCHS

# Default: Experiment 1 (Phase 1) E4-style run (override with env vars — see EXPERIMENTS.md)
./execute_notebook.sh

./execute_notebook.sh experiments/Experiment_2/Phase2_FewShot_And_ErrorAnalysis.ipynb
./execute_notebook.sh experiments/Experiment_3/Phase3_TargetSupervised_LLM_Baseline.ipynb

.venv/bin/python scripts/make_training_curves.py
.venv/bin/python scripts/make_error_summary.py
.venv/bin/python scripts/compare_encoder_llm_matched_subset.py
```

Notebooks call `os.chdir` to the **repo root** on startup, so output paths (`Checkpoints/`, `Phase2_Outputs/`, …) stay stable even though `.ipynb` files live under `experiments/`.

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for experiment IDs (E1–E4, T1–T2, LLM, matched subset), transfer-gap headline, and recommended `NLP_MODEL` / `NLP_SOURCE_LANGS` settings.
