# Experiments (notebooks)

Each phase has its own folder with the runnable notebook and a short README. All notebooks `os.chdir` to the **repository root** on startup, so paths like `Checkpoints/` and `Final_Source_Model/` keep working.

| Folder | Phase | Notebook |
|--------|--------|----------|
| [01_phase1_source_finetuning_zeroshot](01_phase1_source_finetuning_zeroshot/) | Phase 1 — source fine-tune + zero-shot | `Source_Model_FineTuning.ipynb` |
| [02_phase2_fewshot_error_analysis](02_phase2_fewshot_error_analysis/) | Phase 2 — few-shot + error samples | `Phase2_FewShot_And_ErrorAnalysis.ipynb` |
| [03_phase3_supervised_ceiling_llm](03_phase3_supervised_ceiling_llm/) | Phase 3 — target supervised ceiling + optional LLM | `Phase3_TargetSupervised_LLM_Baseline.ipynb` |

Headless run (from repo root):

```bash
./execute_notebook.sh experiments/01_phase1_source_finetuning_zeroshot/Source_Model_FineTuning.ipynb
```

See [../EXPERIMENTS.md](../EXPERIMENTS.md) for the full experiment matrix (E1–E4, T1–T2, LLM, matched subset).
