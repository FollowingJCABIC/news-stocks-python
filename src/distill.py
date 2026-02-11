from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV


def _feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c.startswith("regime_vec_") or c in {
            "resonance",
            "market_ret_mean",
            "market_vol_mean",
            "calendar_shift_delta",
            "event_zero_delta",
            "event_top_type_delta",
        }:
            cols.append(c)
    return cols


def run_distillation(run_dir: str | Path, alpha: float = 0.001, cv_folds: int = 3, max_iter: int = 5000) -> Dict[str, Any]:
    run_path = Path(run_dir)
    pred_path = run_path / "predictions.parquet"
    regime_path = run_path / "regimes.parquet"
    expl_path = run_path / "explanations.parquet"

    if not pred_path.exists() or not regime_path.exists() or not expl_path.exists():
        raise FileNotFoundError("Missing predictions/regimes/explanations parquet for distillation")

    pred = pd.read_parquet(pred_path)
    regimes = pd.read_parquet(regime_path)
    expl = pd.read_parquet(expl_path)

    merged = pred.merge(regimes, on=["date", "ticker", "fold"], how="left")
    merged = merged.merge(
        expl[
            [
                "date",
                "ticker",
                "fold",
                "event_zero_delta",
                "event_top_type_delta",
                "calendar_shift_delta",
                "resonance",
            ]
        ],
        on=["date", "ticker", "fold"],
        how="left",
    )

    feature_cols = _feature_columns(merged)
    if not feature_cols:
        raise ValueError("No features available for distillation")

    X = merged[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y = merged["ret_hat"].to_numpy(dtype=np.float32)

    model = LassoCV(alphas=np.array([alpha], dtype=np.float32), cv=cv_folds, max_iter=max_iter, random_state=42)
    model.fit(X, y)

    y_hat = model.predict(X)
    mse = float(np.mean((y_hat - y) ** 2))
    r2 = float(1.0 - np.sum((y - y_hat) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12))

    coef_map = {f: float(c) for f, c in zip(feature_cols, model.coef_)}
    sparse = {k: v for k, v in coef_map.items() if abs(v) > 1e-8}

    report = {
        "features": feature_cols,
        "coefficients": coef_map,
        "nonzero_coefficients": sparse,
        "intercept": float(model.intercept_),
        "mse": mse,
        "r2": r2,
    }

    (run_path / "distill.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    out_df = merged[["date", "ticker", "fold", "ret_hat"]].copy()
    out_df["ret_hat_distilled"] = y_hat
    out_df.to_parquet(run_path / "distill_predictions.parquet", index=False)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc sparse distillation for interpretability")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=5000)
    args = parser.parse_args()

    report = run_distillation(
        run_dir=args.run_dir,
        alpha=args.alpha,
        cv_folds=args.cv_folds,
        max_iter=args.max_iter,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
