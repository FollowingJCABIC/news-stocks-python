from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests


EXPECTED_COLUMNS = ["date", "event_text", "event_type", "severity", "sentiment"]


@dataclass
class GDELTConfig:
    query: str
    max_records_per_day: int
    request_sleep_sec: float
    cache_dir: str


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns or "event_text" not in out.columns:
        raise ValueError("Events must include at least: date,event_text")
    for c in EXPECTED_COLUMNS:
        if c not in out.columns:
            out[c] = None
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=["date", "event_text"])
    out["event_text"] = out["event_text"].astype(str).str.strip()
    out = out[out["event_text"] != ""]
    out["event_type"] = out["event_type"].fillna("unknown").astype(str)
    out["severity"] = pd.to_numeric(out["severity"], errors="coerce")
    out["sentiment"] = pd.to_numeric(out["sentiment"], errors="coerce")
    return out[EXPECTED_COLUMNS].sort_values("date").reset_index(drop=True)


def load_events_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Events CSV not found: {p}")
    df = pd.read_csv(p)
    return _ensure_schema(df)


def _daterange(start_date: str, end_date: str) -> List[datetime]:
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    days: List[datetime] = []
    cur = s
    while cur <= e:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _gdelt_cache_file(cache_dir: Path, day: datetime) -> Path:
    return cache_dir / f"{day.strftime('%Y-%m-%d')}.json"


def _fetch_gdelt_day(day: datetime, cfg: GDELTConfig, timeout: int = 30) -> Dict[str, Any]:
    start_ts = day.strftime("%Y%m%d000000")
    end_ts = day.strftime("%Y%m%d235959")
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": cfg.query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": cfg.max_records_per_day,
        "startdatetime": start_ts,
        "enddatetime": end_ts,
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        return {"articles": []}
    return data


def fetch_gdelt_events(start_date: str, end_date: str, cfg: GDELTConfig) -> pd.DataFrame:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for day in _daterange(start_date, end_date):
        cache_file = _gdelt_cache_file(cache_dir, day)
        payload: Dict[str, Any]
        if cache_file.exists():
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            try:
                payload = _fetch_gdelt_day(day, cfg)
            except Exception:
                payload = {"articles": []}
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
            if cfg.request_sleep_sec > 0:
                time.sleep(cfg.request_sleep_sec)

        articles = payload.get("articles", [])
        if not isinstance(articles, list):
            continue
        for a in articles:
            title = (a.get("title") or "").strip()
            if not title:
                continue
            tone = a.get("tone")
            try:
                sentiment = float(tone) / 100.0 if tone is not None else None
            except Exception:
                sentiment = None
            themes = a.get("themes") or []
            event_type = "unknown"
            if isinstance(themes, list) and themes:
                event_type = str(themes[0])
            rows.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "event_text": title,
                    "event_type": event_type,
                    "severity": None,
                    "sentiment": sentiment,
                }
            )
    if not rows:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    return _ensure_schema(pd.DataFrame(rows))


def load_events_from_config(cfg: Dict[str, Any]) -> pd.DataFrame:
    events_cfg = cfg["data"]["events"]
    backend = str(events_cfg.get("backend", "csv")).lower().strip()
    if backend == "csv":
        return load_events_csv(events_cfg["csv_path"])
    if backend == "gdelt":
        gdelt_cfg = GDELTConfig(**events_cfg["gdelt"])
        return fetch_gdelt_events(
            start_date=cfg["data"]["start_date"],
            end_date=cfg["data"]["end_date"],
            cfg=gdelt_cfg,
        )
    raise ValueError(f"Unsupported events backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load or fetch event data")
    parser.add_argument("--backend", choices=["csv", "gdelt"], required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--csv", default="data/raw/events.csv")
    parser.add_argument("--out", default="data/raw/events.csv")
    parser.add_argument("--gdelt-query", default="(economy OR inflation OR rates OR oil)")
    parser.add_argument("--gdelt-max", type=int, default=250)
    parser.add_argument("--gdelt-sleep", type=float, default=0.4)
    parser.add_argument("--gdelt-cache", default="data/raw/gdelt_cache")
    args = parser.parse_args()

    if args.backend == "csv":
        df = load_events_csv(args.csv)
    else:
        cfg = GDELTConfig(
            query=args.gdelt_query,
            max_records_per_day=args.gdelt_max,
            request_sleep_sec=args.gdelt_sleep,
            cache_dir=args.gdelt_cache,
        )
        df = fetch_gdelt_events(start_date=args.start, end_date=args.end, cfg=cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} events to {out}")


if __name__ == "__main__":
    main()
