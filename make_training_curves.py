#!/usr/bin/env python3
"""Plot training and validation curves for the report.

Uses any `Checkpoints/training_log_history.csv` (latest Phase-1 run) and
`Checkpoints/training_log_<experiment_id>.csv` files written by Phase 3.
Produces two PNGs per log:
  * `<basename>_loss.png`   — train loss vs eval loss against optimizer step
  * `<basename>_metrics.png` — eval F1-macro and balanced accuracy

Run:
    .venv/bin/python make_training_curves.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "Checkpoints"
OUT_DIR = ROOT / "Phase2_Outputs"
OUT_DIR.mkdir(exist_ok=True)


def plot_log(csv_path: Path, label: str) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    if "step" not in df.columns:
        return

    # Train loss = rows where `loss` is set and `eval_loss` is NaN
    train_df = df.dropna(subset=["loss"]) if "loss" in df.columns else pd.DataFrame()
    eval_df = df.dropna(subset=["eval_loss"]) if "eval_loss" in df.columns else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not train_df.empty:
        ax.plot(train_df["step"], train_df["loss"], label="train loss", marker=".", linewidth=1.4)
    if not eval_df.empty:
        ax.plot(
            eval_df["step"],
            eval_df["eval_loss"],
            label="eval loss",
            marker="o",
            linewidth=1.6,
            color="C3",
        )
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Loss (cross-entropy)")
    ax.set_title(f"Training vs validation loss — {label}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{csv_path.stem}_loss.png", dpi=150)
    plt.close(fig)

    if not eval_df.empty and {"eval_f1_macro", "eval_balanced_accuracy"} <= set(eval_df.columns):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(
            eval_df["step"], eval_df["eval_f1_macro"], marker="o", color="C2", label="eval f1_macro"
        )
        ax.plot(
            eval_df["step"],
            eval_df["eval_balanced_accuracy"],
            marker="s",
            color="C4",
            label="eval balanced_accuracy",
        )
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Score")
        ax.set_title(f"Validation metrics — {label}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{csv_path.stem}_metrics.png", dpi=150)
        plt.close(fig)


def main() -> None:
    plotted = 0
    for csv in sorted(CHECKPOINTS.glob("training_log*.csv")):
        if csv.stem == "training_log_history":
            label = "Phase 1 (latest run, source training)"
        else:
            label = csv.stem.replace("training_log_", "Phase 3 — ")
        plot_log(csv, label)
        plotted += 1
        print(f"plotted: {csv.name}")
    print(f"done; {plotted} log file(s) plotted under {OUT_DIR}")


if __name__ == "__main__":
    main()
