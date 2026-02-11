from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yfinance as yf


def _normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    rename_map = {"adj_close": "adj_close"}
    out = out.rename(columns=rename_map)
    if "adj_close" not in out.columns and "close" in out.columns:
        out["adj_close"] = out["close"]
    out["ticker"] = ticker
    out = out.reset_index().rename(columns={"Date": "date", "index": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    keep = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    return out[keep]


def fetch_prices(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    out_path: str | Path,
) -> pd.DataFrame:
    symbols = list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()]))
    if not symbols:
        raise ValueError("No tickers provided")

    raw = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    frames: List[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        for ticker in symbols:
            if ticker not in level0:
                continue
            block = raw[ticker].dropna(how="all")
            if block.empty:
                continue
            frames.append(_normalize_ohlcv(block, ticker))
    else:
        if len(symbols) != 1:
            raise RuntimeError("Unexpected yfinance format for multi-symbol download")
        frames.append(_normalize_ohlcv(raw.dropna(how="all"), symbols[0]))

    if not frames:
        raise RuntimeError("No price data downloaded")

    prices = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out_path, index=False)
    return prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch daily OHLCV using yfinance")
    parser.add_argument("--tickers", required=True, help="Comma separated symbols")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--out", default="data/raw/prices.parquet", help="Output parquet path")
    args = parser.parse_args()

    tickers = [s.strip() for s in args.tickers.split(",") if s.strip()]
    prices = fetch_prices(tickers=tickers, start_date=args.start, end_date=args.end, out_path=args.out)
    print(f"Saved {len(prices):,} rows to {args.out}")


if __name__ == "__main__":
    main()
