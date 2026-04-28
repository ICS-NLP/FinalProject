#!/usr/bin/env python3
"""Regenerate derived CSVs from canonical logs (single source of truth).

Reads:
  - Checkpoints/experiment_log.csv  (master eval rows)
  - Phase2_Outputs/few_shot_results.csv  (zero-shot + few-shot k)

Writes:
  - Checkpoints/results_report_table.csv  (E4 headline rows only)
  - Phase3_Outputs/transfer_gap_summary.csv  (transfer headline)

Run after re-running notebooks so colleagues always see consistent numbers:
  .venv/bin/python scripts/sync_metrics_tables.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Allow `python scripts/sync_metrics_tables.py` from repo root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from project_paths import repo_root  # noqa: E402

E4_ID = "E4_large76L_hau_amh_yor"
RESULT_COLS = [
    "setting",
    "subset",
    "n_eval",
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "pred_majority_frac",
    "num_pred_classes_used",
    "notes",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def build_results_report(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        if r.get("experiment_id") != E4_ID:
            continue
        if r.get("setting") not in ("zero_shot", "source_test_in_distribution"):
            continue
        out.append({k: r.get(k, "") for k in RESULT_COLS})
    out.sort(key=lambda r: (0 if r.get("subset") == "twi" else 1 if r.get("subset") == "pcm" else 2, r.get("subset", "")))
    return out


def few_shot_lookup(rows: list[dict], target: str, condition: str) -> dict | None:
    for r in rows:
        if r.get("target") == target and r.get("condition") == condition:
            return r
    return None


def build_transfer_gap(few_rows: list[dict], exp_rows: list[dict]) -> list[dict]:
    def sup_f1(target_lang: str) -> float:
        eid = "T1_supervised_twi" if target_lang == "twi" else "T2_supervised_pcm"
        for r in exp_rows:
            if r.get("experiment_id") == eid:
                return float(r["f1_macro"])
        raise KeyError(f"missing {eid} in experiment_log.csv")

    header = [
        "target",
        "zero_shot_f1_macro",
        "few_shot_k20_f1_macro",
        "supervised_f1_macro",
        "absolute_gap_zs_to_sup",
        "transfer_ratio_zs_over_sup",
        "few_shot_recovers_pct_of_gap",
    ]
    built: list[dict] = []
    for target in ("twi", "pcm"):
        zr = few_shot_lookup(few_rows, target, "zero-shot")
        fr = few_shot_lookup(few_rows, target, "few-shot k=20")
        if not zr or not fr:
            raise KeyError(f"missing few_shot row for {target}")
        zs = float(zr["f1_macro"])
        fk = float(fr["f1_macro"])
        sup = sup_f1(target)
        gap = sup - zs
        ratio = zs / sup if sup else 0.0
        rec = (fk - zs) / gap * 100 if gap else 0.0
        built.append(
            {
                "target": target,
                "zero_shot_f1_macro": zs,
                "few_shot_k20_f1_macro": fk,
                "supervised_f1_macro": sup,
                "absolute_gap_zs_to_sup": gap,
                "transfer_ratio_zs_over_sup": ratio,
                "few_shot_recovers_pct_of_gap": rec,
            }
        )
    return built


def main() -> None:
    root = repo_root()
    exp_path = root / "Checkpoints" / "experiment_log.csv"
    few_path = root / "Phase2_Outputs" / "few_shot_results.csv"
    report_path = root / "Checkpoints" / "results_report_table.csv"
    gap_path = root / "Phase3_Outputs" / "transfer_gap_summary.csv"

    exp_rows = _read_csv(exp_path)
    few_rows = _read_csv(few_path)

    report = build_results_report(exp_rows)
    if len(report) != 3:
        raise SystemExit(
            f"expected 3 E4 rows in experiment_log for results_report_table, got {len(report)}. "
            "Check experiment_id / setting columns."
        )
    _write_csv(report_path, RESULT_COLS, report)
    print(f"Wrote: {report_path}")

    gap_rows = build_transfer_gap(few_rows, exp_rows)
    gap_header = [
        "target",
        "zero_shot_f1_macro",
        "few_shot_k20_f1_macro",
        "supervised_f1_macro",
        "absolute_gap_zs_to_sup",
        "transfer_ratio_zs_over_sup",
        "few_shot_recovers_pct_of_gap",
    ]
    _write_csv(gap_path, gap_header, gap_rows)
    print(f"Wrote: {gap_path}")


if __name__ == "__main__":
    main()
