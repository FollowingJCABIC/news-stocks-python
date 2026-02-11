from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class RegressionMetrics:
    mae: float
    mse: float


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    return RegressionMetrics(mae=mae, mse=mse)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def backtest_long_short(
    predictions: pd.DataFrame,
    top_n: int,
    allow_short: bool,
    transaction_cost_bps: float,
    annualization: int = 252,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if predictions.empty:
        empty_curve = pd.DataFrame(columns=["date", "daily_return", "equity"])
        return {
            "mean_daily_return": 0.0,
            "vol_daily_return": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
        }, empty_curve

    tc = transaction_cost_bps / 10000.0
    rows = []

    for date, g in predictions.groupby("date", sort=True):
        g = g.sort_values("ret_hat", ascending=False)
        longs = g.head(top_n)
        long_ret = float(longs["ret_true"].mean()) if not longs.empty else 0.0

        if allow_short:
            shorts = g.tail(top_n)
            short_ret = float(shorts["ret_true"].mean()) if not shorts.empty else 0.0
            gross = long_ret - short_ret
            costs = 2.0 * tc
        else:
            gross = long_ret
            costs = tc

        net = gross - costs
        rows.append({"date": pd.to_datetime(date), "daily_return": net})

    curve = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    curve["equity"] = np.exp(curve["daily_return"].cumsum())

    daily = curve["daily_return"].to_numpy(dtype=np.float64)
    mean_ret = float(np.mean(daily))
    vol_ret = float(np.std(daily) + 1e-12)
    sharpe = float((mean_ret / vol_ret) * np.sqrt(annualization))

    peak = curve["equity"].cummax()
    dd = curve["equity"] / peak - 1.0
    max_dd = float(dd.min())

    metrics = {
        "mean_daily_return": mean_ret,
        "vol_daily_return": vol_ret,
        "sharpe_like": sharpe,
        "max_drawdown": max_dd,
        "total_return": float(curve["equity"].iloc[-1] - 1.0),
    }
    return metrics, curve


def summarize_predictions(predictions: pd.DataFrame, eval_cfg: Dict[str, Any]) -> Dict[str, Any]:
    ret_m = regression_metrics(
        y_true=predictions["ret_true"].to_numpy(dtype=np.float32),
        y_pred=predictions["ret_hat"].to_numpy(dtype=np.float32),
    )
    vol_m = regression_metrics(
        y_true=predictions["vol_true"].to_numpy(dtype=np.float32),
        y_pred=predictions["vol_hat"].to_numpy(dtype=np.float32),
    )

    da = directional_accuracy(
        y_true=predictions["ret_true"].to_numpy(dtype=np.float32),
        y_pred=predictions["ret_hat"].to_numpy(dtype=np.float32),
    )

    bt_metrics, _ = backtest_long_short(
        predictions=predictions,
        top_n=int(eval_cfg["top_n"]),
        allow_short=bool(eval_cfg["allow_short"]),
        transaction_cost_bps=float(eval_cfg["transaction_cost_bps"]),
        annualization=int(eval_cfg["annualization"]),
    )

    out = {
        "ret_mae": ret_m.mae,
        "ret_mse": ret_m.mse,
        "vol_mae": vol_m.mae,
        "vol_mse": vol_m.mse,
        "directional_accuracy": da,
    }
    out.update({f"backtest_{k}": v for k, v in bt_metrics.items()})
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate saved predictions parquet")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--annualization", type=int, default=252)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    pred = pd.read_parquet(args.predictions)
    metrics = summarize_predictions(
        pred,
        {
            "top_n": args.top_n,
            "allow_short": args.allow_short,
            "transaction_cost_bps": args.transaction_cost_bps,
            "annualization": args.annualization,
        },
    )
    txt = json.dumps(metrics, indent=2)
    print(txt)
    if args.out_json:
        Path(args.out_json).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
