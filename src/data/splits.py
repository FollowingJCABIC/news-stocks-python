from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def build_walk_forward_splits(
    sample_date_idx: np.ndarray,
    num_dates: int,
    train_days: int,
    val_days: int,
    test_days: int,
    step_days: int,
    purge_days: int,
    embargo_days: int,
    min_train_samples: int,
) -> List[Dict[str, Any]]:
    if num_dates <= 0:
        return []

    splits: List[Dict[str, Any]] = []
    cursor = 0
    fold = 0

    while True:
        train_start = cursor
        train_end = train_start + train_days - 1

        val_start = train_end + 1 + embargo_days
        val_end = val_start + val_days - 1

        test_start = val_end + 1 + embargo_days
        test_end = test_start + test_days - 1

        if test_end >= num_dates:
            break

        train_hi = max(train_start, train_end - purge_days)
        val_lo = min(val_end, val_start + purge_days)
        val_hi = max(val_lo, val_end - purge_days)
        test_lo = min(test_end, test_start + purge_days)

        train_idx = np.where((sample_date_idx >= train_start) & (sample_date_idx <= train_hi))[0]
        val_idx = np.where((sample_date_idx >= val_lo) & (sample_date_idx <= val_hi))[0]
        test_idx = np.where((sample_date_idx >= test_lo) & (sample_date_idx <= test_end))[0]

        if len(train_idx) >= min_train_samples and len(val_idx) > 0 and len(test_idx) > 0:
            splits.append(
                {
                    "fold": fold,
                    "train_range": [int(train_start), int(train_hi)],
                    "val_range": [int(val_lo), int(val_hi)],
                    "test_range": [int(test_lo), int(test_end)],
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "test_idx": test_idx,
                }
            )
            fold += 1

        cursor += step_days

    return splits


def save_splits(splits: List[Dict[str, Any]], out_dir: str | Path = "data/processed") -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    npz_payload: Dict[str, np.ndarray] = {}
    summary: List[Dict[str, Any]] = []
    for s in splits:
        f = s["fold"]
        npz_payload[f"fold_{f}_train_idx"] = s["train_idx"]
        npz_payload[f"fold_{f}_val_idx"] = s["val_idx"]
        npz_payload[f"fold_{f}_test_idx"] = s["test_idx"]
        summary.append(
            {
                "fold": f,
                "train_range": s["train_range"],
                "val_range": s["val_range"],
                "test_range": s["test_range"],
                "n_train": int(len(s["train_idx"])),
                "n_val": int(len(s["val_idx"])),
                "n_test": int(len(s["test_idx"])),
            }
        )

    npz_path = out / "splits_indices.npz"
    summary_path = out / "splits_summary.json"
    np.savez_compressed(npz_path, **npz_payload)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {"indices": str(npz_path), "summary": str(summary_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe walk-forward splits")
    parser.add_argument("--dataset", default="data/processed/dataset.npz")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--train-days", type=int, default=756)
    parser.add_argument("--val-days", type=int, default=126)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=126)
    parser.add_argument("--purge-days", type=int, default=2)
    parser.add_argument("--embargo-days", type=int, default=2)
    parser.add_argument("--min-train-samples", type=int, default=500)
    args = parser.parse_args()

    ds = np.load(args.dataset, allow_pickle=False)
    date_idx = ds["date_idx"]
    num_dates = len(ds["dates"])

    splits = build_walk_forward_splits(
        sample_date_idx=date_idx,
        num_dates=num_dates,
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        step_days=args.step_days,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
        min_train_samples=args.min_train_samples,
    )
    paths = save_splits(splits, out_dir=args.out_dir)
    print(json.dumps({"splits": len(splits), "saved": paths}, indent=2))


if __name__ == "__main__":
    main()
