from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.load_events import load_events_from_config


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_") or "unknown"


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_price_features(prices: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    seq_windows = cfg["features"]["asset_rolling_windows"]
    rsi_period = int(cfg["features"].get("rsi_period", 14))

    frames: List[pd.DataFrame] = []
    for ticker, g in df.groupby("ticker", sort=True):
        g = g.copy()
        g["log_close"] = np.log(g["adj_close"].clip(lower=1e-9))
        g["ret_1d"] = g["log_close"].diff()
        g["log_volume"] = np.log(g["volume"].clip(lower=1) + 1.0)
        g["rsi"] = compute_rsi(g["adj_close"], period=rsi_period)
        for w in seq_windows:
            g[f"vol_{w}"] = g["ret_1d"].rolling(w, min_periods=w).std()
            g[f"ma_ratio_{w}"] = g["adj_close"] / g["adj_close"].rolling(w, min_periods=w).mean() - 1.0
        g["ret_fwd_1d"] = g["ret_1d"].shift(-1)
        g["vol_fwd_1d"] = g["ret_fwd_1d"].abs()
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


def build_market_daily_features(price_features: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    proxies = [p.upper() for p in cfg["data"].get("market_proxies", [])]
    frame = price_features[price_features["ticker"].isin(proxies)].copy()
    if frame.empty:
        raise ValueError("No market proxy data found for configured proxies")

    cols = ["ret_1d"] + [f"vol_{w}" for w in cfg["features"]["market_rolling_windows"]]
    pivot_blocks: List[pd.DataFrame] = []
    for p in proxies:
        g = frame[frame["ticker"] == p][["date", *cols]].copy()
        g = g.rename(columns={c: f"{p}_{c}" for c in cols})
        pivot_blocks.append(g)

    daily = pivot_blocks[0]
    for b in pivot_blocks[1:]:
        daily = daily.merge(b, on="date", how="outer")

    proxy_ret_cols = [c for c in daily.columns if c.endswith("_ret_1d")]
    daily["market_ret_mean"] = daily[proxy_ret_cols].mean(axis=1)
    vol_cols = [c for c in daily.columns if "_vol_" in c]
    daily["market_vol_mean"] = daily[vol_cols].mean(axis=1)
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def _build_text_embeddings(
    event_texts: List[str],
    asset_texts: List[str],
    cfg: Dict[str, Any],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    emb_cfg = cfg["features"]["event_embedding"]
    use_st = bool(emb_cfg.get("prefer_sentence_transformer", True))
    model_name = emb_cfg.get("model_name", "all-MiniLM-L6-v2")
    if use_st:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            event_emb = model.encode(event_texts, convert_to_numpy=True, show_progress_bar=False)
            asset_emb = model.encode(asset_texts, convert_to_numpy=True, show_progress_bar=False)
            return event_emb.astype(np.float32), asset_emb.astype(np.float32), f"sentence_transformer:{model_name}"
        except Exception:
            pass

    max_features = int(emb_cfg.get("tfidf_max_features", 512))
    svd_dim = int(emb_cfg.get("svd_dim", 32))
    corpus = event_texts + asset_texts
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), lowercase=True)
    X = vectorizer.fit_transform(corpus)
    if X.shape[1] <= 1:
        event_emb = np.zeros((len(event_texts), svd_dim), dtype=np.float32)
        asset_emb = np.zeros((len(asset_texts), svd_dim), dtype=np.float32)
        return event_emb, asset_emb, "tfidf_zero"

    dim = max(1, min(svd_dim, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    Z = svd.fit_transform(X)

    event_emb = Z[: len(event_texts)]
    asset_emb = Z[len(event_texts) :]

    if dim < svd_dim:
        event_emb = np.pad(event_emb, ((0, 0), (0, svd_dim - dim)))
        asset_emb = np.pad(asset_emb, ((0, 0), (0, svd_dim - dim)))

    return event_emb.astype(np.float32), asset_emb.astype(np.float32), f"tfidf_svd:{dim}"


def build_event_daily_features(
    events: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    cfg: Dict[str, Any],
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[int], List[str], str]:
    e = events.copy()
    e["date"] = pd.to_datetime(e["date"]).dt.tz_localize(None).dt.normalize()

    if e.empty:
        out = pd.DataFrame({"date": trading_dates})
        out["event_count"] = 0.0
        out["event_sentiment_mean"] = 0.0
        out["event_severity_mean"] = 0.0
        event_texts = ["" for _ in range(len(trading_dates))]
        asset_texts: List[str] = []
        event_emb = np.zeros((len(trading_dates), cfg["features"]["event_embedding"]["svd_dim"]), dtype=np.float32)
        feature_names = [c for c in out.columns if c != "date"] + [f"event_emb_{i}" for i in range(event_emb.shape[1])]
        matrix = np.concatenate([out[["event_count", "event_sentiment_mean", "event_severity_mean"]].to_numpy(), event_emb], axis=1)
        top_idx = [-1] * len(trading_dates)
        return out, matrix.astype(np.float32), feature_names, top_idx, [], "none"

    e["event_type"] = e["event_type"].fillna("unknown").astype(str)
    e["event_type_clean"] = e["event_type"].map(_sanitize_name)

    counts = (
        e.assign(_n=1)
        .pivot_table(index="date", columns="event_type_clean", values="_n", aggfunc="sum", fill_value=0)
        .sort_index()
    )
    top_types = counts.sum(axis=0).sort_values(ascending=False).head(12).index.tolist()
    counts = counts[top_types] if top_types else counts
    counts.columns = [f"event_type_count_{c}" for c in counts.columns]

    agg = (
        e.groupby("date")
        .agg(
            event_count=("event_text", "count"),
            event_sentiment_mean=("sentiment", "mean"),
            event_severity_mean=("severity", "mean"),
            event_text_join=("event_text", lambda s: " || ".join([str(x) for x in s if str(x).strip()][:40])),
        )
        .reset_index()
    )

    daily = pd.DataFrame({"date": trading_dates}).merge(agg, on="date", how="left")
    daily = daily.merge(counts.reset_index(), on="date", how="left")
    daily["event_count"] = daily["event_count"].fillna(0.0)
    daily["event_sentiment_mean"] = daily["event_sentiment_mean"].fillna(0.0)
    daily["event_severity_mean"] = daily["event_severity_mean"].fillna(0.0)
    daily["event_text_join"] = daily["event_text_join"].fillna("")

    type_cols = [c for c in daily.columns if c.startswith("event_type_count_")]
    for c in type_cols:
        daily[c] = daily[c].fillna(0.0)

    event_texts = daily["event_text_join"].astype(str).tolist()
    asset_texts = [f"{t} {n}" for t, n in cfg["data"].get("asset_names", {}).items()]
    event_emb, asset_emb, emb_backend = _build_text_embeddings(event_texts, asset_texts, cfg=cfg, seed=seed)

    structured_cols = ["event_count", "event_sentiment_mean", "event_severity_mean", *type_cols]
    structured = daily[structured_cols].to_numpy(dtype=np.float32)
    matrix = np.concatenate([structured, event_emb], axis=1)

    event_feature_names = structured_cols + [f"event_emb_{i}" for i in range(event_emb.shape[1])]
    type_col_idx = [event_feature_names.index(c) for c in type_cols]

    top_type_idx: List[int] = []
    for _, row in daily[type_cols].iterrows():
        if not type_cols or float(row.max()) <= 0:
            top_type_idx.append(-1)
            continue
        best_col = row.idxmax()
        top_type_idx.append(event_feature_names.index(best_col))

    out = daily[["date", *structured_cols, "event_text_join"]].copy()
    out["embedding_backend"] = emb_backend

    asset_names_ordered = [
        f"{t} {cfg['data']['asset_names'].get(t, t)}" for t in cfg["data"]["universe"] if t in cfg["data"].get("asset_names", {})
    ]
    if asset_names_ordered and len(asset_names_ordered) != asset_emb.shape[0]:
        asset_emb = np.zeros((len(cfg["data"]["universe"]), event_emb.shape[1]), dtype=np.float32)

    return out, matrix.astype(np.float32), event_feature_names, top_type_idx, type_cols, emb_backend


def build_calendar_features(trading_dates: pd.DatetimeIndex, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.DataFrame({"date": trading_dates})

    for window in cfg["features"].get("windows", []):
        label = _sanitize_name(window["label"])
        start = pd.Timestamp(window["start"]).normalize()
        end = pd.Timestamp(window["end"]).normalize()

        in_window = ((df["date"] >= start) & (df["date"] <= end)).astype(np.float32)

        before = (start - df["date"]).dt.days.clip(lower=0)
        after = (df["date"] - end).dt.days.clip(lower=0)
        days_to_window = (before + after).astype(np.float32)

        df[f"window_{label}_in"] = in_window
        df[f"window_{label}_days_to"] = days_to_window

    periodic = cfg["features"].get("periodic_terms", [])
    if "day_of_week" in periodic:
        dow = df["date"].dt.dayofweek.astype(np.float32)
        df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    if "day_of_year" in periodic:
        doy = df["date"].dt.dayofyear.astype(np.float32)
        df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    names = [c for c in df.columns if c != "date"]
    return df, names


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return (a @ b.T) / (a_norm * b_norm.T)


def _clip_extremes(arr: np.ndarray, clip_std: float) -> np.ndarray:
    if arr.size == 0:
        return arr
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sd = np.nanstd(arr, axis=0, keepdims=True) + 1e-9
    lo = mu - clip_std * sd
    hi = mu + clip_std * sd
    return np.clip(arr, lo, hi)


@dataclass
class FeatureBundle:
    arrays: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


def build_feature_bundle(cfg: Dict[str, Any]) -> FeatureBundle:
    prices_path = Path(cfg["data"]["prices_path"])
    if not prices_path.exists():
        raise FileNotFoundError(f"Prices parquet not found: {prices_path}")

    prices = pd.read_parquet(prices_path)
    price_features = build_price_features(prices, cfg)

    events = load_events_from_config(cfg)
    trading_dates = pd.DatetimeIndex(sorted(price_features["date"].dropna().unique()))

    market_daily = build_market_daily_features(price_features, cfg)
    market_daily = pd.DataFrame({"date": trading_dates}).merge(market_daily, on="date", how="left").fillna(0.0)

    (
        event_daily_table,
        event_daily_matrix,
        event_feature_names,
        top_event_type_idx_by_date,
        event_type_cols,
        embedding_backend,
    ) = build_event_daily_features(events=events, trading_dates=trading_dates, cfg=cfg, seed=cfg["project"]["seed"])

    calendar_daily, calendar_feature_names = build_calendar_features(trading_dates, cfg)

    asset_cols = [
        "ret_1d",
        "log_volume",
        "rsi",
        *[f"vol_{w}" for w in cfg["features"]["asset_rolling_windows"]],
        *[f"ma_ratio_{w}" for w in cfg["features"]["asset_rolling_windows"]],
    ]
    market_cols = [c for c in market_daily.columns if c != "date"]

    price_features = price_features.merge(market_daily[["date", *market_cols]], on="date", how="left")

    tickers = [t for t in cfg["data"]["universe"] if t in set(price_features["ticker"].unique())]
    if not tickers:
        raise ValueError("No configured tickers present in price data")
    ticker_to_id = {t: i for i, t in enumerate(tickers)}

    seq_len = int(cfg["features"]["sequence_length"])
    clip_std = float(cfg["features"].get("clip_std", 6.0))

    market_matrix = market_daily[market_cols].to_numpy(dtype=np.float32)
    calendar_matrix = calendar_daily[calendar_feature_names].to_numpy(dtype=np.float32) if calendar_feature_names else np.zeros((len(trading_dates), 0), dtype=np.float32)

    date_to_idx = {d: i for i, d in enumerate(trading_dates)}

    asset_name_map = cfg["data"].get("asset_names", {})
    asset_name_texts = [f"{t} {asset_name_map.get(t, t)}" for t in tickers]
    daily_event_texts = event_daily_table["event_text_join"].astype(str).tolist()
    event_emb_daily, asset_name_emb, _ = _build_text_embeddings(
        event_texts=daily_event_texts,
        asset_texts=asset_name_texts,
        cfg=cfg,
        seed=cfg["project"]["seed"],
    )
    resonance_daily = _cosine_similarity_matrix(event_emb_daily, asset_name_emb)

    samples_event_seq: List[np.ndarray] = []
    samples_market: List[np.ndarray] = []
    samples_asset: List[np.ndarray] = []
    samples_calendar: List[np.ndarray] = []
    samples_resonance: List[np.ndarray] = []
    samples_asset_id: List[int] = []
    samples_date_idx: List[int] = []
    samples_y_ret: List[float] = []
    samples_y_vol: List[float] = []
    samples_top_event_type_idx: List[int] = []

    typed = price_features.copy()
    typed[asset_cols] = typed[asset_cols].replace([np.inf, -np.inf], np.nan)

    for ticker, g in typed.groupby("ticker", sort=False):
        if ticker not in ticker_to_id:
            continue
        g = g.sort_values("date").reset_index(drop=True)
        for row in g.itertuples(index=False):
            date = pd.Timestamp(row.date).normalize()
            di = date_to_idx.get(date)
            if di is None or di < seq_len - 1:
                continue

            y_ret = getattr(row, "ret_fwd_1d")
            y_vol = getattr(row, "vol_fwd_1d")
            if pd.isna(y_ret) or pd.isna(y_vol):
                continue

            asset_vals = np.array([getattr(row, c) for c in asset_cols], dtype=np.float32)
            if np.any(np.isnan(asset_vals)):
                continue

            event_seq = event_daily_matrix[di - seq_len + 1 : di + 1]
            if event_seq.shape[0] != seq_len:
                continue

            samples_event_seq.append(event_seq.astype(np.float32))
            samples_market.append(market_matrix[di].astype(np.float32))
            samples_asset.append(asset_vals)
            samples_calendar.append(calendar_matrix[di].astype(np.float32))
            samples_resonance.append(np.array([resonance_daily[di, ticker_to_id[ticker]]], dtype=np.float32))
            samples_asset_id.append(ticker_to_id[ticker])
            samples_date_idx.append(di)
            samples_y_ret.append(float(y_ret))
            samples_y_vol.append(float(y_vol))
            samples_top_event_type_idx.append(int(top_event_type_idx_by_date[di]))

    if not samples_asset:
        raise RuntimeError("No training samples built. Check date range and feature windows.")

    arr_event_seq = _clip_extremes(np.stack(samples_event_seq).astype(np.float32), clip_std)
    arr_market = _clip_extremes(np.stack(samples_market).astype(np.float32), clip_std)
    arr_asset = _clip_extremes(np.stack(samples_asset).astype(np.float32), clip_std)
    arr_calendar = _clip_extremes(np.stack(samples_calendar).astype(np.float32), clip_std) if samples_calendar else np.zeros((len(samples_asset), 0), dtype=np.float32)
    arr_resonance = np.stack(samples_resonance).astype(np.float32)

    arrays: Dict[str, np.ndarray] = {
        "event_seq": arr_event_seq,
        "market_x": arr_market,
        "asset_x": arr_asset,
        "calendar_x": arr_calendar,
        "resonance_x": arr_resonance,
        "asset_id": np.asarray(samples_asset_id, dtype=np.int64),
        "date_idx": np.asarray(samples_date_idx, dtype=np.int64),
        "y_ret": np.asarray(samples_y_ret, dtype=np.float32),
        "y_vol": np.asarray(samples_y_vol, dtype=np.float32),
        "top_event_type_idx": np.asarray(samples_top_event_type_idx, dtype=np.int64),
        "calendar_daily": calendar_matrix.astype(np.float32),
        "event_daily": event_daily_matrix.astype(np.float32),
        "market_daily": market_matrix.astype(np.float32),
        "dates": np.array([d.strftime("%Y-%m-%d") for d in trading_dates]),
        "tickers": np.array(tickers),
    }

    metadata: Dict[str, Any] = {
        "market_feature_names": market_cols,
        "asset_feature_names": asset_cols,
        "event_feature_names": event_feature_names,
        "calendar_feature_names": calendar_feature_names,
        "event_type_feature_names": event_type_cols,
        "embedding_backend": embedding_backend,
        "sequence_length": seq_len,
        "num_samples": int(len(samples_asset)),
        "num_dates": int(len(trading_dates)),
        "num_assets": int(len(tickers)),
    }

    return FeatureBundle(arrays=arrays, metadata=metadata)


def save_feature_bundle(bundle: FeatureBundle, out_dir: str | Path = "data/processed") -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    npz_path = out / "dataset.npz"
    meta_path = out / "metadata.json"

    np.savez_compressed(npz_path, **bundle.arrays)
    meta_path.write_text(json.dumps(bundle.metadata, indent=2), encoding="utf-8")

    return {"dataset": str(npz_path), "metadata": str(meta_path)}


def load_feature_bundle(npz_path: str | Path, meta_path: str | Path) -> FeatureBundle:
    data = np.load(npz_path, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return FeatureBundle(arrays=arrays, metadata=meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aligned market/event feature arrays")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bundle = build_feature_bundle(cfg)
    paths = save_feature_bundle(bundle, out_dir=args.out_dir)
    print(json.dumps({"saved": paths, "samples": bundle.metadata["num_samples"]}, indent=2))


if __name__ == "__main__":
    main()
