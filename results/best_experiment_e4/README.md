# Best experiment — **E4** (`Davlan/afro-xlmr-large-76L`, Hausa + Amharic + Yoruba)

This folder is the **documentation hub** for the configuration we treat as the team’s primary cross-lingual result (**`E4_large76L_hau_amh_yor`** in `Checkpoints/experiment_log.csv`). We **do not duplicate** large binaries here; figures and comparison tables live next to the notebooks’ outputs so there is a single canonical copy in Git.

## Where to look (canonical paths)

| What you need | Path |
|----------------|------|
| **Deployable checkpoint (weights + config + tokenizer)** | **`Final_Source_Model/`** at repo root — see **[../../EXPERIMENTS.md](../../EXPERIMENTS.md#recommended-model-for-deployment-e4)**. **`model.safetensors` is gitignored**; after a fresh clone you must train E4 or obtain weights out-of-band. |
| Phase 1 training / validation curves | `Phase2_Outputs/training_log_history_loss.png`, `Phase2_Outputs/training_log_history_metrics.png` |
| Few-shot sweep plot | `Phase2_Outputs/few_shot_curve.png` |
| Matched subset encoder vs LLM table | `Phase3_Outputs/matched_subset_encoder_vs_llm.csv` |
| Full metrics log | `Checkpoints/experiment_log.csv` |

## Configuration (E4)

| Field | Value |
|-------|--------|
| Model | `Davlan/afro-xlmr-large-76L` |
| Source languages | Hausa, Amharic, Yoruba |
| Experiment ID | `E4_large76L_hau_amh_yor` |
| Notebook | `experiments/Experiment_1/Source_Model_FineTuning.ipynb` |

## Headline numbers (official test splits)

See `Checkpoints/experiment_log.csv` rows with `experiment_id=E4_large76L_hau_amh_yor`. At a glance:

- **Zero-shot Twi** — macro-F1 ≈ **0.38**, full test *n* = 698  
- **Zero-shot Nigerian Pidgin** — macro-F1 ≈ **0.58**, *n* = 1593  
- **Source in-distribution** (combined hau+amh+yor test) — macro-F1 ≈ **0.75**, *n* = 2615  

(Exact floats are in the CSV; re-run Experiment 1 if you regenerate weights.)

## Regenerate plots and side artefacts

From repo root (with `.venv` installed):

```bash
.venv/bin/python scripts/make_training_curves.py
.venv/bin/python scripts/make_error_summary.py
.venv/bin/python scripts/compare_encoder_llm_matched_subset.py
```

For the full methodology, Phase 2/3 artefacts, and qualitative error analysis, see **[../EXPERIMENTS.md](../EXPERIMENTS.md)**.
