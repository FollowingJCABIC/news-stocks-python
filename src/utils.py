from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


@dataclass
class RunPaths:
    root: Path
    plots: Path


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_deterministic(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_run_paths(cfg: Dict[str, Any], run_name: str | None = None) -> RunPaths:
    base = Path(cfg["project"]["output_dir"])
    ensure_dir(base)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    prefix = cfg["project"].get("run_name_prefix", "run")
    name = run_name or f"{prefix}_{stamp}"
    root = ensure_dir(base / name)
    plots = ensure_dir(root / "plots")
    return RunPaths(root=root, plots=plots)


def list_run_dirs(base_dir: str | Path) -> List[Path]:
    base = Path(base_dir)
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def prune_old_runs(base_dir: str | Path, max_runs_to_keep: int) -> List[str]:
    if max_runs_to_keep <= 0:
        return []
    runs = list_run_dirs(base_dir)
    if len(runs) <= max_runs_to_keep:
        return []
    victims = runs[: len(runs) - max_runs_to_keep]
    removed: List[str] = []
    for v in victims:
        shutil.rmtree(v)
        removed.append(str(v))
    return removed


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
