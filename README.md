# FinalProject — AfriHate cross-lingual hate speech

Multilingual fine-tuning (**Hausa, Amharic**, optional **Yoruba**) and **zero-shot** evaluation on **Twi** and **Nigerian Pidgin** using `Davlan/afro-xlmr-base` or `Davlan/afro-xlmr-large-76L`.

**Upstream repo:** [https://github.com/ICS-NLP/FinalProject](https://github.com/ICS-NLP/FinalProject)

## What’s in this repo

| Path | Contents |
|------|----------|
| `Source_Model_FineTuning.ipynb` | Main training + zero-shot + `experiment_log.csv` |
| `Phase2_FewShot_And_ErrorAnalysis.ipynb` | Few-shot curves + error samples |
| `Checkpoints/` | `experiment_log.csv`, `results_report_table.csv`, `training_log_history.csv` |
| `Phase2_Outputs/` | `few_shot_results.csv`, `zero_shot_errors_sample.csv`, plots |
| `Final_Source_Model/` | **Config + tokenizer** only on GitHub (see below) |
| `EXPERIMENTS.md` | Run matrix, env hygiene (`unset NLP_SOURCE_LANGS`), recommended commands |
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
./execute_notebook.sh Source_Model_FineTuning.ipynb
./execute_notebook.sh Phase2_FewShot_And_ErrorAnalysis.ipynb
```

See `EXPERIMENTS.md` for experiment IDs (E1–E4) and recommended `NLP_MODEL` / `NLP_SOURCE_LANGS` settings.
