# Experiment log — AfriHate cross-lingual study

Aggregated metrics live in `Checkpoints/experiment_log.csv`. Each run **replaces** rows that share the same
`(experiment_id, model, source_languages)` triple.

## Shell hygiene (important)

`jupyter nbconvert` inherits **all** exported variables from the shell. If you once ran:

```bash
export NLP_SOURCE_LANGS=hau,amh,yor
```

then a later command **only** setting `NLP_MODEL=Davlan/afro-xlmr-large-76L` will still train on three source
languages unless you unset the variable. **Cell 1** of `Source_Model_FineTuning.ipynb` prints a **RUN_MANIFEST**
so you can verify `NLP_MODEL` and source languages before a long run.

```bash
cd FinalProject
unset NLP_EXPERIMENT_ID NLP_SOURCE_LANGS NLP_MODEL NLP_LR
```

## Predefined experiment IDs (auto when `NLP_EXPERIMENT_ID` is unset)

| ID | Model | Sources |
|----|------|---------|
| `E1_base_hau_amh` | `Davlan/afro-xlmr-base` | `hau,amh` |
| `E2_large76L_hau_amh` | `Davlan/afro-xlmr-large-76L` | `hau,amh` |
| `E3_base_hau_amh_yor` | `Davlan/afro-xlmr-base` | `hau,amh,yor` |
| `E4_large76L_hau_amh_yor` | `Davlan/afro-xlmr-large-76L` | `hau,amh,yor` |

## Snapshot (macro-F1 on zero-shot targets; source = official combined test)

| Run | Twi | Pidgin | Source test |
|-----|-----|--------|-------------|
| E1 base, hau+amh | 0.287 | 0.519 | 0.758 |
| E2 76L, hau+amh | 0.279 | **0.557** | **0.766** |
| E3 base, hau+amh+yor | **0.374** | 0.515 | 0.737 |
| **E4 76L, hau+amh+yor** (4 epochs, best eval) | **0.375** | **0.584** | **0.753** |

**Takeaway:** Adding Yoruba (E3/E4) fixes the **Twi** zero-shot ceiling relative to hau+amh-only training. The **76L**
encoder (E4) then recovers **Pidgin** performance vs E3 and beats E2 on Pidgin while keeping Twi near the E3 peak.

## Recommended full command (E4)

```bash
unset NLP_EXPERIMENT_ID NLP_SOURCE_LANGS NLP_LR
export NLP_MODEL=Davlan/afro-xlmr-large-76L
export NLP_SOURCE_LANGS=hau,amh,yor
export NLP_NUM_EPOCHS=4   # optional; default in notebook is 3 if unset in kernel
./execute_notebook.sh Source_Model_FineTuning.ipynb
```

Default learning rate for `large-76L` is **1.5e-5** unless `NLP_LR` is set. On Apple MPS the notebook uses a
smaller per-device batch and higher gradient accumulation for large models to reduce memory pressure.

## Phase 2

```bash
./execute_notebook.sh Phase2_FewShot_And_ErrorAnalysis.ipynb
```

Knobs: `NLP_FEWSHOT_LR` (default `8e-6`), `NLP_FEWSHOT_EPOCHS` (default `6`), `NLP_SOURCE_MODEL_PATH`.
