# Experiments (notebooks)

Each numbered folder holds one runnable notebook and a short README. All notebooks `os.chdir` to the **repository root** on startup, so paths like `Checkpoints/` and `Final_Source_Model/` keep working.

| Folder | Phase | Notebook |
|--------|--------|----------|
| [`Experiment_1`](Experiment_1/) | Phase 1 — source fine-tune + zero-shot | `Source_Model_FineTuning.ipynb` |
| [`Experiment_2`](Experiment_2/) | Phase 2 — few-shot + error samples | `Phase2_FewShot_And_ErrorAnalysis.ipynb` |
| [`Experiment_3`](Experiment_3/) | Phase 3 — target supervised ceiling + optional LLM | `Phase3_TargetSupervised_LLM_Baseline.ipynb` |

Headless run (from repo root):

```bash
./execute_notebook.sh experiments/Experiment_1/Source_Model_FineTuning.ipynb
```

See [../EXPERIMENTS.md](../EXPERIMENTS.md) for the full experiment matrix (E1–E4, T1–T2, LLM, matched subset).
