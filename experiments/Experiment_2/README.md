# Experiment 2 — Phase 2: few-shot adaptation and qualitative error analysis

**Notebook:** `Phase2_FewShot_And_ErrorAnalysis.ipynb`

Loads the Phase 1 checkpoint from `Final_Source_Model/`, runs k ∈ {5, 10, 20} few-shot fine-tunes per target language, and samples zero-shot errors for qualitative analysis.

**Primary outputs (repo root):**

- `Phase2_Outputs/few_shot_results.csv`, `few_shot_curve.png`, `zero_shot_errors_sample.csv`

**Run** (shell cwd = repository root):

```bash
./execute_notebook.sh experiments/Experiment_2/Phase2_FewShot_And_ErrorAnalysis.ipynb
```
