#!/usr/bin/env python3
"""Turn `Phase2_Outputs/zero_shot_errors_sample.csv` into a Markdown report
fragment with categorised qualitative examples for the write-up.

Categories (heuristic):
  * over_flagging_in_group   — Normal labelled, predicted Hate/Abuse
  * under_flagging_implicit  — Hate/Abuse labelled, predicted Normal (likely
    sarcasm, idiom, slur not in source vocabulary)
  * cross_class_confusion    — Hate <-> Abuse mistakes

Output: `Phase2_Outputs/zero_shot_error_summary.md`
"""
from __future__ import annotations

from collections import defaultdict
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from project_paths import ROOT

SRC = ROOT / "Phase2_Outputs" / "zero_shot_errors_sample.csv"
OUT = ROOT / "Phase2_Outputs" / "zero_shot_error_summary.md"


def categorise(row: pd.Series) -> str:
    t, p = str(row["true_label"]), str(row["pred_label"])
    if t == "Normal" and p in {"Hate", "Abuse"}:
        return "over_flagging"
    if t in {"Hate", "Abuse"} and p == "Normal":
        return "under_flagging"
    return "cross_class_confusion"


def main() -> None:
    df = pd.read_csv(SRC)
    df["category"] = df.apply(categorise, axis=1)
    counts = (
        df.groupby(["target", "category"]).size().rename("n").reset_index()
    )

    grouped: dict[str, dict[str, list[pd.Series]]] = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        grouped[row["target"]][row["category"]].append(row)

    parts: list[str] = []
    parts.append("# Zero-shot error analysis (qualitative)\n")
    parts.append(
        "Source model: best Phase-1 checkpoint (`Final_Source_Model/`, `afro-xlmr-large-76L` "
        "fine-tuned on Hau+Amh+Yor). Errors are sampled (up to 8 per error type per target) from "
        "the official `twi` and `pcm` AfriHate test splits.\n"
    )

    parts.append("\n## Error counts in the sample\n")
    parts.append("| target | category | n |")
    parts.append("|---|---|---|")
    for _, row in counts.iterrows():
        parts.append(f"| {row['target']} | {row['category']} | {int(row['n'])} |")

    for target, cats in grouped.items():
        parts.append(f"\n## Target: `{target}`\n")
        for cat, rows in cats.items():
            human_cat = {
                "over_flagging": "Over-flagging — Normal predicted as Hate / Abuse (false positives)",
                "under_flagging": "Under-flagging — Hate / Abuse predicted as Normal (false negatives)",
                "cross_class_confusion": "Cross-class confusion — Hate ↔ Abuse (right that something is bad, wrong category)",
            }[cat]
            parts.append(f"\n### {human_cat}\n")
            for r in rows:
                txt = str(r["text"]).strip().replace("\n", " ")
                if len(txt) > 220:
                    txt = txt[:220] + "…"
                parts.append(f"- *true* `{r['true_label']}` / *pred* `{r['pred_label']}` — \"{txt}\"")
        parts.append("")

    parts.append(
        "\n## Patterns we observe\n"
        "\n"
        "**Twi (Akan).**\n"
        "- *Over-flagging.* Tweets that simply mention ethnic / regional identity terms (e.g. "
        "`Ntafo`, `Frafrafo`, `Zongofo`, `Alatafuo`) are labelled Hate even when the rest of the post is "
        "playful or descriptive. The encoder appears to associate these proper nouns with hostile "
        "contexts seen during Hausa+Amharic training, where intergroup tokens correlate with hate.\n"
        "- *Under-flagging.* The hardest under-flagging cases involve idiomatic insults that the source "
        "languages do not share (`Kontomponi`, `Gyimisɛm`, `akoa`, `Apakye musuoni`, `Animguasefoɔ`). "
        "Those tokens carry abuse for Akan readers but are unfamiliar to the source-trained head.\n"
        "\n"
        "**Nigerian Pidgin.**\n"
        "- *Over-flagging.* Pidgin's familiar/teasing register (`abeg`, `mumu`, `oga`, `mad woman`, "
        "rhetorical exaggeration) is repeatedly classified Abuse — common in-group banter rather than "
        "directed harassment.\n"
        "- *Under-flagging.* Stereotyping framed as agreement (`I agree \"Facts: 50% of … are Igbo's\"`) "
        "and ethnic-group put-downs (`Yoruba people too like noise tah`) are labelled Normal — the model "
        "misses ethnic-group hate when the surface tone is conversational.\n"
        "- *Cross-class confusion* on Pidgin is rare in this sample; when the model flags a tweet it "
        "tends to keep the right Hate vs Abuse split.\n"
        "\n"
        "**Implications for the report.**\n"
        "1. Twi failures concentrate in **morphology + identity terms**; they explain why adding Yoruba "
        "to the source mix in E3/E4 helps but does not close the gap.\n"
        "2. Pidgin failures concentrate in **register and irony**; few-shot k=5 already lifts F1 above "
        "zero-shot, suggesting a small amount of in-domain calibration is enough.\n"
        "3. False positives on identity terms are an **ethical** risk: deploying without human review "
        "would silence legitimate in-group speech in under-served languages.\n"
    )

    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
