"""Single place to resolve the FinalProject root (repo root).

Notebooks may execute with cwd inside experiments/; scripts may live in
scripts/. This module finds the directory that contains requirements.txt
and execute_notebook.sh.
"""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for candidate in (p, *p.parents):
        if (candidate / "requirements.txt").is_file() and (candidate / "execute_notebook.sh").is_file():
            return candidate
    return p


ROOT = repo_root()
