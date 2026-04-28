# Best experiment — **E4** (`Davlan/afro-xlmr-large-76L`, Hausa + Amharic + Yoruba)

This folder is a **curated snapshot** for the final report: the configuration we treat as the team’s primary result (**E4** in `Checkpoints/experiment_log.csv`).

## Configuration

| Field | Value |
|-------|--------|
| Model | `Davlan/afro-xlmr-large-76L` |
| Source languages | Hausa, Amharic, Yoruba |
| Experiment ID | `E4_large76L_hau_amh_yor` |
| Notebook | `experiments/Experiment_1/Source_Model_FineTuning.ipynb` |

## Headline numbers (official test splits)

See `../Checkpoints/experiment_log.csv` rows with `experiment_id=E4_large76L_hau_amh_yor`. At a glance:

- **Zero-shot Twi** — macro-F1 ≈ **0.38**, full test *n* = 698  
- **Zero-shot Nigerian Pidgin** — macro-F1 ≈ **0.58**, *n* = 1593  
- **Source in-distribution** (combined hau+amh+yor test) — macro-F1 ≈ **0.75**, *n* = 2615  

(Exact floats are in the CSV; re-run Phase 1 if you regenerate weights.)

## Figures in `figures/`

Static copies of the main plots (regenerate anytime with `scripts/make_training_curves.py`):

- `training_log_history_loss.png` / `training_log_history_metrics.png` — Phase 1 training / validation curves  
- `few_shot_curve.png` — Phase 2 few-shot sweep (uses `Final_Source_Model` from E4)

## Tables in `tables/`

- `matched_subset_encoder_vs_llm.csv` — **same 200-example** random slice: fine-tuned encoder vs `gpt-4o-mini` zero-shot (rebuild with `scripts/compare_encoder_llm_matched_subset.py` after updating the LLM rows).

For the full methodology, Phase 2/3 artefacts, and qualitative error analysis, see **[../EXPERIMENTS.md](../EXPERIMENTS.md)**.
