from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


REQUIRED_FILES = [
    "metrics.json",
    "predictions.parquet",
    "regimes.parquet",
    "explanations.parquet",
    "ablations.json",
    "placebo.json",
]


def latest_run_path(runs_root: str | Path = "outputs/runs") -> Path:
    root = Path(runs_root)
    if not root.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {root}")
    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found under: {root}")
    return runs[-1]


def load_run_artifacts(run_dir: str | Path) -> Dict[str, Any]:
    run_path = Path(run_dir)
    out: Dict[str, Any] = {"run_dir": str(run_path)}
    for name in REQUIRED_FILES:
        p = run_path / name
        if not p.exists():
            continue
        if p.suffix == ".json":
            out[name] = json.loads(p.read_text(encoding="utf-8"))
        elif p.suffix == ".parquet":
            out[name] = pd.read_parquet(p)
    return out
