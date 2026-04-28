# Phase 1 — Source fine-tuning and zero-shot evaluation

**Notebook:** `Source_Model_FineTuning.ipynb`

Fine-tunes `Davlan/afro-xlmr-base` or `Davlan/afro-xlmr-large-76L` on AfriHate **Hausa ± Amharic ± Yoruba** training data, evaluates on the combined source test split, then **zero-shot** evaluates on **Twi** and **Nigerian Pidgin** test sets.

**Primary outputs (repo root):**

- `Checkpoints/experiment_log.csv`, `results_report_table.csv`, `training_log_history.csv`
- `Final_Source_Model/` — best checkpoint (weights are gitignored; config + tokenizer may be present)

**Run** (current working directory must be the **repository root** — the folder that contains `execute_notebook.sh` and `requirements.txt`):

```bash
./execute_notebook.sh experiments/01_phase1_source_finetuning_zeroshot/Source_Model_FineTuning.ipynb
```
