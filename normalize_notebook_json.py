#!/usr/bin/env python3
"""Repair common nbformat v4 issues so `jupyter nbconvert` can read the notebook.

Some editors / older runs omit required fields on stream or execute_result outputs.
Also assigns stable cell `id`s (nbformat ≥5.1).
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path


def normalize(path: Path) -> int:
    import nbformat
    from nbformat.validator import normalize as nb_normalize

    nb = nbformat.read(path, as_version=4)
    for cell in nb.get("cells", []):
        if not cell.get("id"):
            cell["id"] = str(uuid.uuid4())
    n = 0
    for cell in nb.get("cells", []):
        ec = cell.get("execution_count")
        if ec is None:
            ec = 0
        for out in cell.get("outputs") or []:
            ot = out.get("output_type")
            if ot == "stream" and "name" not in out:
                out["name"] = "stdout"
                n += 1
            if ot in ("display_data", "execute_result"):
                if "metadata" not in out or out["metadata"] is None:
                    out["metadata"] = {}
                    n += 1
            if ot == "execute_result":
                if "execution_count" not in out or out["execution_count"] is None:
                    out["execution_count"] = ec
                    n += 1
            if ot == "error":
                tb = out.get("traceback")
                if isinstance(tb, str):
                    out["traceback"] = [tb]
                    n += 1
    nb_normalize(nb)
    nbformat.write(nb, path)
    return n


def main() -> None:
    paths = [Path(p) for p in (sys.argv[1:] or [])]
    if not paths:
        print("usage: normalize_notebook_json.py <notebook.ipynb> [...]", file=sys.stderr)
        sys.exit(2)
    total = 0
    for p in paths:
        if not p.is_file():
            print(f"skip (not a file): {p}", file=sys.stderr)
            continue
        k = normalize(p)
        total += k
        print(f"{p}: {k} field(s) normalized")
    print(f"total: {total}")


if __name__ == "__main__":
    main()
