#!/usr/bin/env python3
"""Export the fine-tuned Hugging Face checkpoint to a single PyTorch `.pth` bundle.

Training saves `Final_Source_Model/model.safetensors` (+ config). Some teams want
`torch.load("*.pth")` instead. This script loads the HF folder and writes:

  {
    "state_dict": <full model, CPU>,
    "id2label": ...,
    "label2id": ...,
    "export_note": "E4 AfriHate cross-lingual encoder (same weights as model.safetensors)",
  }

**Size:** roughly the same as `model.safetensors` (~2 GB for 76L). GitHub blocks
blobs > 100 MB — do **not** commit this file to a normal Git remote without
Git LFS or an artefact store / Hugging Face Hub.

Usage:
  .venv/bin/python scripts/export_e4_model_to_pth.py
  .venv/bin/python scripts/export_e4_model_to_pth.py --model-dir /path/to/Final_Source_Model \\
      --output ./Weights_export/e4_best_model.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from project_paths import repo_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="HF save folder (default: <repo>/Final_Source_Model)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .pth path (default: <repo>/Weights_export/e4_best_model.pth)",
    )
    args = parser.parse_args()
    root = repo_root()
    model_dir = (args.model_dir or (root / "Final_Source_Model")).resolve()
    out = (args.output or (root / "Weights_export" / "e4_best_model.pth")).resolve()

    if not (model_dir / "config.json").is_file():
        raise SystemExit(f"Missing config.json under {model_dir}")
    weights = model_dir / "model.safetensors"
    legacy = model_dir / "pytorch_model.bin"
    if not weights.is_file() and not legacy.is_file():
        raise SystemExit(
            f"No weights found in {model_dir}.\n"
            "Train Experiment 1 (E4) first, or copy model.safetensors / pytorch_model.bin here."
        )

    import torch
    from transformers import AutoModelForSequenceClassification

    print(f"Loading model from {model_dir} …")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.cpu()
    payload = {
        "state_dict": model.state_dict(),
        "id2label": dict(model.config.id2label),
        "label2id": dict(model.config.label2id),
        "export_note": "E4 AfriHate cross-lingual encoder; load with torch.load then model.load_state_dict",
        "source_hf_folder": str(model_dir),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote: {out} ({mb:.1f} MiB)")


if __name__ == "__main__":
    main()
