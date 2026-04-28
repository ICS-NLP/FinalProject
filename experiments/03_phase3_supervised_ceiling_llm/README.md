# Phase 3 — Target supervised ceiling and optional LLM baseline

**Notebook:** `Phase3_TargetSupervised_LLM_Baseline.ipynb`

Fine-tunes `afro-xlmr-base` **on each target’s own training data** (ceiling), writes transfer-gap summaries, and optionally runs an OpenAI **JSON** few-/zero-shot baseline when `OPENAI_API_KEY` is set.

**Primary outputs (repo root):**

- `Checkpoints/training_log_T1_supervised_twi.csv`, `training_log_T2_supervised_pcm.csv`
- `Phase3_Outputs/transfer_gap_summary.csv`, confusion matrices, matched-subset artefacts

**Run** (shell cwd = repository root):

```bash
export NLP_PHASE3_SKIP_SUPERVISED=1   # optional: LLM-only refresh
./execute_notebook.sh experiments/03_phase3_supervised_ceiling_llm/Phase3_TargetSupervised_LLM_Baseline.ipynb
```
