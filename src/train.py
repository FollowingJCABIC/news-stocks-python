from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.build_features import FeatureBundle, build_feature_bundle, save_feature_bundle
from src.data.fetch_prices import fetch_prices
from src.data.splits import build_walk_forward_splits, save_splits
from src.distill import run_distillation
from src.eval import backtest_long_short, summarize_predictions
from src.models.asset_model import EventDrivenMarketModel, ModelDims
from src.utils import (
    ensure_dir,
    load_config,
    make_run_paths,
    prune_old_runs,
    save_json,
    set_deterministic,
    to_numpy,
)


@dataclass
class FeatureScalers:
    event_mean: np.ndarray
    event_std: np.ndarray
    market_mean: np.ndarray
    market_std: np.ndarray
    asset_mean: np.ndarray
    asset_std: np.ndarray
    calendar_mean: np.ndarray
    calendar_std: np.ndarray
    resonance_mean: np.ndarray
    resonance_std: np.ndarray


def _fit_scale_2d(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x_train.size == 0:
        return np.zeros((1,), dtype=np.float32), np.ones((1,), dtype=np.float32)
    mu = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    sd = x_train.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    return mu, sd


def _fit_scale_3d(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x_train.size == 0:
        return np.zeros((1, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32)
    flat = x_train.reshape(-1, x_train.shape[-1])
    mu = flat.mean(axis=0, keepdims=True).astype(np.float32)
    sd = flat.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    return mu, sd


def _apply_scale_2d(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    return ((x - mu) / sd).astype(np.float32)


def _apply_scale_3d(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    return ((x - mu.reshape(1, 1, -1)) / sd.reshape(1, 1, -1)).astype(np.float32)


def normalize_arrays(arrays: Dict[str, np.ndarray], train_idx: np.ndarray, include_events: bool) -> Tuple[Dict[str, np.ndarray], FeatureScalers]:
    raw_event = arrays["event_seq"].copy()
    raw_market = arrays["market_x"].copy()
    raw_asset = arrays["asset_x"].copy()
    raw_calendar = arrays["calendar_x"].copy()
    raw_resonance = arrays["resonance_x"].copy()

    event_mu, event_sd = _fit_scale_3d(raw_event[train_idx])
    market_mu, market_sd = _fit_scale_2d(raw_market[train_idx])
    asset_mu, asset_sd = _fit_scale_2d(raw_asset[train_idx])
    cal_mu, cal_sd = _fit_scale_2d(raw_calendar[train_idx]) if raw_calendar.shape[1] > 0 else (np.zeros((1, 0), dtype=np.float32), np.ones((1, 0), dtype=np.float32))
    rho_mu, rho_sd = _fit_scale_2d(raw_resonance[train_idx])

    event_x = _apply_scale_3d(raw_event, event_mu, event_sd)
    if not include_events:
        event_x[:] = 0.0

    scaled = {
        "event_seq": event_x,
        "market_x": _apply_scale_2d(raw_market, market_mu, market_sd),
        "asset_x": _apply_scale_2d(raw_asset, asset_mu, asset_sd),
        "calendar_x": _apply_scale_2d(raw_calendar, cal_mu, cal_sd) if raw_calendar.shape[1] > 0 else raw_calendar,
        "resonance_x": _apply_scale_2d(raw_resonance, rho_mu, rho_sd),
        "asset_id": arrays["asset_id"],
        "date_idx": arrays["date_idx"],
        "y_ret": arrays["y_ret"],
        "y_vol": arrays["y_vol"],
        "top_event_type_idx": arrays["top_event_type_idx"],
    }

    scalers = FeatureScalers(
        event_mean=event_mu,
        event_std=event_sd,
        market_mean=market_mu,
        market_std=market_sd,
        asset_mean=asset_mu,
        asset_std=asset_sd,
        calendar_mean=cal_mu,
        calendar_std=cal_sd,
        resonance_mean=rho_mu,
        resonance_std=rho_sd,
    )
    return scaled, scalers


def mode_flags(mode: str, cfg: Dict[str, Any]) -> Dict[str, bool]:
    if mode == "A_price_only":
        return {"use_events": False, "use_resonance": False, "use_calendar": False}
    if mode == "B_price_events":
        return {"use_events": True, "use_resonance": False, "use_calendar": False}
    return {
        "use_events": True,
        "use_resonance": bool(cfg["model"].get("use_resonance", True)),
        "use_calendar": bool(cfg["model"].get("use_calendar", True)),
    }


def make_tensor_loader(
    arrays: Dict[str, np.ndarray],
    idx: np.ndarray,
    regime_label: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(arrays["event_seq"][idx], dtype=torch.float32),
        torch.tensor(arrays["market_x"][idx], dtype=torch.float32),
        torch.tensor(arrays["asset_x"][idx], dtype=torch.float32),
        torch.tensor(arrays["calendar_x"][idx], dtype=torch.float32),
        torch.tensor(arrays["resonance_x"][idx], dtype=torch.float32),
        torch.tensor(arrays["asset_id"][idx], dtype=torch.long),
        torch.tensor(arrays["y_ret"][idx], dtype=torch.float32).unsqueeze(-1),
        torch.tensor(arrays["y_vol"][idx], dtype=torch.float32).unsqueeze(-1),
        torch.tensor(regime_label[idx], dtype=torch.long),
        torch.tensor(arrays["top_event_type_idx"][idx], dtype=torch.long),
        torch.tensor(arrays["date_idx"][idx], dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def compute_regime_labels(market_x: np.ndarray, train_idx: np.ndarray, num_regimes: int, seed: int) -> np.ndarray:
    model = KMeans(n_clusters=num_regimes, random_state=seed, n_init=10)
    model.fit(market_x[train_idx])
    labels = model.predict(market_x)
    return labels.astype(np.int64)


def train_one_fold(
    cfg: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    split: Dict[str, Any],
    flags: Dict[str, bool],
    device: torch.device,
    seed: int,
    epochs_override: int | None = None,
) -> Tuple[EventDrivenMarketModel, Dict[str, float], np.ndarray]:
    train_cfg = cfg["training"]
    fold = int(split["fold"])
    set_deterministic(seed + fold, deterministic=bool(cfg["project"].get("deterministic", True)))

    regime_labels = compute_regime_labels(
        market_x=arrays["market_x"],
        train_idx=split["train_idx"],
        num_regimes=int(cfg["model"]["num_regimes"]),
        seed=seed + fold,
    )

    dims = ModelDims(
        event_dim=arrays["event_seq"].shape[-1],
        market_dim=arrays["market_x"].shape[-1],
        asset_dim=arrays["asset_x"].shape[-1],
        calendar_dim=arrays["calendar_x"].shape[-1],
        num_assets=int(arrays["asset_id"].max()) + 1,
    )

    model_cfg = copy.deepcopy(cfg)
    model_cfg["model"]["use_resonance"] = flags["use_resonance"]
    model_cfg["model"]["use_calendar"] = flags["use_calendar"]

    model = EventDrivenMarketModel(dims=dims, cfg=model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    huber = nn.HuberLoss(delta=float(train_cfg["huber_delta"]))
    ce = nn.CrossEntropyLoss()

    epochs = int(epochs_override or train_cfg["epochs"])
    train_loader = make_tensor_loader(arrays, split["train_idx"], regime_labels, int(train_cfg["batch_size"]), shuffle=True)
    val_loader = make_tensor_loader(arrays, split["val_idx"], regime_labels, int(train_cfg["batch_size"]), shuffle=False)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    patience = int(train_cfg["early_stopping_patience"])

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            (
                event_seq,
                market_x,
                asset_x,
                calendar_x,
                resonance_x,
                asset_id,
                y_ret,
                y_vol,
                reg_lbl,
                _,
                _,
            ) = [t.to(device) for t in batch]

            if not flags["use_resonance"]:
                resonance_x = torch.zeros_like(resonance_x)
            if not flags["use_calendar"]:
                calendar_x = torch.zeros_like(calendar_x)

            out = model(event_seq, market_x, asset_x, calendar_x, resonance_x, asset_id)
            ret_loss = huber(out["ret_hat"], y_ret)
            vol_loss = huber(out["vol_hat"], y_vol)
            loss = ret_loss + vol_loss

            if out["regime_logits"] is not None:
                reg_loss = ce(out["regime_logits"], reg_lbl)
                loss = loss + float(train_cfg["regime_loss_weight"]) * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                (
                    event_seq,
                    market_x,
                    asset_x,
                    calendar_x,
                    resonance_x,
                    asset_id,
                    y_ret,
                    y_vol,
                    reg_lbl,
                    _,
                    _,
                ) = [t.to(device) for t in batch]

                if not flags["use_resonance"]:
                    resonance_x = torch.zeros_like(resonance_x)
                if not flags["use_calendar"]:
                    calendar_x = torch.zeros_like(calendar_x)

                out = model(event_seq, market_x, asset_x, calendar_x, resonance_x, asset_id)
                ret_loss = huber(out["ret_hat"], y_ret)
                vol_loss = huber(out["vol_hat"], y_vol)
                loss = ret_loss + vol_loss
                if out["regime_logits"] is not None:
                    reg_loss = ce(out["regime_logits"], reg_lbl)
                    loss = loss + float(train_cfg["regime_loss_weight"]) * reg_loss
                val_losses.append(float(loss.item()))

        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_loss": best_val}, regime_labels


def _shift_calendar_for_samples(arrays: Dict[str, np.ndarray], shift_days: int) -> np.ndarray:
    cal_daily = arrays["calendar_daily"]
    idx = arrays["date_idx"]
    shift_idx = np.clip(idx + shift_days, 0, cal_daily.shape[0] - 1)
    return cal_daily[shift_idx].astype(np.float32)


def predict_fold(
    model: EventDrivenMarketModel,
    arrays: Dict[str, np.ndarray],
    scalers: FeatureScalers,
    split: Dict[str, Any],
    regime_labels: np.ndarray,
    flags: Dict[str, bool],
    cfg: Dict[str, Any],
    metadata: Dict[str, Any],
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model.eval()
    test_idx = split["test_idx"]

    tcfg = cfg["training"]
    loader = make_tensor_loader(arrays, test_idx, regime_labels, int(tcfg["batch_size"]), shuffle=False)

    shift_days = int(cfg["features"].get("calendar_shift_days_for_counterfactual", 10))
    shifted_cal_raw = _shift_calendar_for_samples({**arrays, "calendar_daily": arrays["calendar_daily"], "date_idx": arrays["date_idx"]}, shift_days=shift_days)
    shifted_cal = _apply_scale_2d(shifted_cal_raw, scalers.calendar_mean, scalers.calendar_std) if arrays["calendar_x"].shape[1] > 0 else shifted_cal_raw

    date_labels = arrays["dates"]
    tickers = arrays["tickers"]
    event_feature_names = metadata["event_feature_names"]

    pred_rows: List[Dict[str, Any]] = []
    regime_rows: List[Dict[str, Any]] = []
    explain_rows: List[Dict[str, Any]] = []

    start = 0
    with torch.no_grad():
        for batch in loader:
            (
                event_seq,
                market_x,
                asset_x,
                calendar_x,
                resonance_x,
                asset_id,
                y_ret,
                y_vol,
                _,
                top_event_idx,
                date_idx,
            ) = [t.to(device) for t in batch]

            bs = event_seq.shape[0]
            slice_idx = test_idx[start : start + bs]
            start += bs

            if not flags["use_resonance"]:
                resonance_x = torch.zeros_like(resonance_x)
            if not flags["use_calendar"]:
                calendar_x = torch.zeros_like(calendar_x)

            base = model(event_seq, market_x, asset_x, calendar_x, resonance_x, asset_id)

            event_zero_seq = torch.zeros_like(event_seq)
            out_zero = model(event_zero_seq, market_x, asset_x, calendar_x, resonance_x, asset_id)

            event_top_removed = event_seq.clone()
            idx_np = top_event_idx.detach().cpu().numpy()
            for bi, feat_idx in enumerate(idx_np.tolist()):
                if feat_idx >= 0 and feat_idx < event_top_removed.shape[-1]:
                    event_top_removed[bi, :, feat_idx] = 0.0
            out_top = model(event_top_removed, market_x, asset_x, calendar_x, resonance_x, asset_id)

            cal_shift_batch = torch.tensor(shifted_cal[slice_idx], dtype=torch.float32, device=device)
            if not flags["use_calendar"]:
                cal_shift_batch = torch.zeros_like(calendar_x)
            out_cal = model(event_seq, market_x, asset_x, cal_shift_batch, resonance_x, asset_id)

            ret_hat = to_numpy(base["ret_hat"]).reshape(-1)
            vol_hat = to_numpy(base["vol_hat"]).reshape(-1)
            ret_true = to_numpy(y_ret).reshape(-1)
            vol_true = to_numpy(y_vol).reshape(-1)

            ww = to_numpy(base["world_weights"])
            reg_vec = to_numpy(base["regime_vec"])
            reg_id = to_numpy(base["regime_id"]).reshape(-1) if base["regime_id"] is not None else np.full(bs, -1)

            delta_zero = ret_hat - to_numpy(out_zero["ret_hat"]).reshape(-1)
            delta_top = ret_hat - to_numpy(out_top["ret_hat"]).reshape(-1)
            delta_cal = ret_hat - to_numpy(out_cal["ret_hat"]).reshape(-1)

            aid = to_numpy(asset_id).reshape(-1)
            did = to_numpy(date_idx).reshape(-1)
            rho = to_numpy(resonance_x).reshape(-1)
            mkt = to_numpy(market_x)
            mkt_names = metadata["market_feature_names"]
            mkt_mean_ret_idx = mkt_names.index("market_ret_mean") if "market_ret_mean" in mkt_names else None
            mkt_mean_vol_idx = mkt_names.index("market_vol_mean") if "market_vol_mean" in mkt_names else None

            for j in range(bs):
                date = str(date_labels[did[j]])
                ticker = str(tickers[aid[j]])
                fold = int(split["fold"])

                pred_row = {
                    "date": date,
                    "ticker": ticker,
                    "fold": fold,
                    "ret_true": float(ret_true[j]),
                    "ret_hat": float(ret_hat[j]),
                    "vol_true": float(vol_true[j]),
                    "vol_hat": float(vol_hat[j]),
                    "market_ret_mean": float(mkt[j, mkt_mean_ret_idx]) if mkt_mean_ret_idx is not None else 0.0,
                    "market_vol_mean": float(mkt[j, mkt_mean_vol_idx]) if mkt_mean_vol_idx is not None else 0.0,
                }
                pred_rows.append(pred_row)

                reg_row = {
                    "date": date,
                    "ticker": ticker,
                    "fold": fold,
                    "regime_id": int(reg_id[j]),
                }
                for k, v in enumerate(reg_vec[j].tolist()):
                    reg_row[f"regime_vec_{k}"] = float(v)
                for k, v in enumerate(ww[j].tolist()):
                    reg_row[f"world_weight_{k}"] = float(v)
                regime_rows.append(reg_row)

                feat_idx = int(idx_np[j])
                feat_name = event_feature_names[feat_idx] if 0 <= feat_idx < len(event_feature_names) else "none"

                explain_rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "fold": fold,
                        "top_event_feature": feat_name,
                        "event_zero_delta": float(delta_zero[j]),
                        "event_top_type_delta": float(delta_top[j]),
                        "calendar_shift_delta": float(delta_cal[j]),
                        "resonance": float(rho[j]),
                    }
                )

    pred_df = pd.DataFrame(pred_rows).sort_values(["date", "ticker", "fold"]).reset_index(drop=True)
    regime_df = pd.DataFrame(regime_rows).sort_values(["date", "ticker", "fold"]).reset_index(drop=True)
    explain_df = pd.DataFrame(explain_rows).sort_values(["date", "ticker", "fold"]).reset_index(drop=True)
    return pred_df, regime_df, explain_df


def run_mode(
    mode: str,
    cfg: Dict[str, Any],
    bundle: FeatureBundle,
    splits: List[Dict[str, Any]],
    device: torch.device,
    seed_offset: int = 0,
    epochs_override: int | None = None,
    arrays_override: Dict[str, np.ndarray] | None = None,
) -> Dict[str, Any]:
    flags = mode_flags(mode, cfg)
    arrays_raw = arrays_override or bundle.arrays

    all_pred: List[pd.DataFrame] = []
    all_reg: List[pd.DataFrame] = []
    all_exp: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, float]] = []

    for split in splits:
        arrays_scaled, scalers = normalize_arrays(arrays=arrays_raw, train_idx=split["train_idx"], include_events=flags["use_events"])
        model, train_info, regime_labels = train_one_fold(
            cfg=cfg,
            arrays=arrays_scaled,
            split=split,
            flags=flags,
            device=device,
            seed=int(cfg["project"]["seed"]) + seed_offset,
            epochs_override=epochs_override,
        )

        pred_df, reg_df, exp_df = predict_fold(
            model=model,
            arrays={**arrays_scaled, "dates": arrays_raw["dates"], "tickers": arrays_raw["tickers"], "calendar_daily": arrays_raw["calendar_daily"]},
            scalers=scalers,
            split=split,
            regime_labels=regime_labels,
            flags=flags,
            cfg=cfg,
            metadata=bundle.metadata,
            device=device,
        )

        metrics = summarize_predictions(pred_df, cfg["evaluation"])
        metrics.update(train_info)
        metrics["fold"] = int(split["fold"])
        fold_metrics.append(metrics)

        all_pred.append(pred_df)
        all_reg.append(reg_df)
        all_exp.append(exp_df)

    if not all_pred:
        raise RuntimeError(f"No predictions generated for mode: {mode}")

    pred = pd.concat(all_pred, ignore_index=True)
    reg = pd.concat(all_reg, ignore_index=True)
    exp = pd.concat(all_exp, ignore_index=True)

    agg = pd.DataFrame(fold_metrics).drop(columns=["fold"]).mean(numeric_only=True).to_dict()
    agg = {k: float(v) for k, v in agg.items()}
    agg["num_folds"] = len(fold_metrics)

    bt_metrics, equity_curve = backtest_long_short(
        pred,
        top_n=int(cfg["evaluation"]["top_n"]),
        allow_short=bool(cfg["evaluation"]["allow_short"]),
        transaction_cost_bps=float(cfg["evaluation"]["transaction_cost_bps"]),
        annualization=int(cfg["evaluation"]["annualization"]),
    )

    return {
        "mode": mode,
        "flags": flags,
        "metrics": agg,
        "fold_metrics": fold_metrics,
        "predictions": pred,
        "regimes": reg,
        "explanations": exp,
        "equity_curve": equity_curve,
        "backtest": bt_metrics,
    }


def apply_placebo(arrays: Dict[str, np.ndarray], test_name: str, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in arrays.items()}

    if test_name == "shuffle_event_dates":
        dates = pd.to_datetime(out["dates"])
        date_idx = out["date_idx"]
        months = pd.Series(dates[date_idx]).dt.to_period("M").astype(str).to_numpy()
        for m in np.unique(months):
            idx = np.where(months == m)[0]
            shuffled = rng.permutation(idx)
            out["event_seq"][idx] = out["event_seq"][shuffled]
            out["resonance_x"][idx] = out["resonance_x"][shuffled]
    elif test_name == "shift_windows":
        shift = int(rng.integers(5, 30))
        out["calendar_x"] = np.roll(out["calendar_x"], shift=shift, axis=0)
        out["calendar_daily"] = np.roll(out["calendar_daily"], shift=shift, axis=0)
    elif test_name == "random_name_embeddings":
        out["resonance_x"] = rng.normal(0.0, 1.0, size=out["resonance_x"].shape).astype(np.float32)

    return out


def write_run_manifest(run_dir: Path, cfg: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    files = []
    total = 0
    for p in sorted(run_dir.glob("*")):
        if p.is_file():
            sz = p.stat().st_size
            files.append({"name": p.name, "size_bytes": sz})
            total += sz

    payload = {
        "run_dir": str(run_dir),
        "total_size_bytes": total,
        "files": files,
        "key_metrics": metrics,
    }

    storage_cfg = cfg.get("storage", {})
    manifest_dir = Path(storage_cfg.get("manifest_dir", "reports/run_manifests"))
    ensure_dir(manifest_dir)
    target = manifest_dir / f"{run_dir.name}.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train event-driven market model with walk-forward and ablations")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-placebo", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_deterministic(seed=int(cfg["project"]["seed"]), deterministic=bool(cfg["project"].get("deterministic", True)))

    prices_path = Path(cfg["data"]["prices_path"])
    if not prices_path.exists():
        fetch_prices(
            tickers=cfg["data"]["universe"],
            start_date=cfg["data"]["start_date"],
            end_date=cfg["data"]["end_date"],
            out_path=prices_path,
        )

    bundle = build_feature_bundle(cfg)
    save_feature_bundle(bundle, out_dir="data/processed")

    splits = build_walk_forward_splits(
        sample_date_idx=bundle.arrays["date_idx"],
        num_dates=len(bundle.arrays["dates"]),
        train_days=int(cfg["training"]["train_days"]),
        val_days=int(cfg["training"]["val_days"]),
        test_days=int(cfg["training"]["test_days"]),
        step_days=int(cfg["training"]["step_days"]),
        purge_days=int(cfg["training"]["purge_days"]),
        embargo_days=int(cfg["training"]["embargo_days"]),
        min_train_samples=int(cfg["training"]["min_train_samples"]),
    )
    save_splits(splits, out_dir="data/processed")
    if not splits:
        raise RuntimeError("No valid walk-forward splits produced")

    run_paths = make_run_paths(cfg, run_name=args.run_name)
    run_dir = run_paths.root
    Path(run_dir / "config_used.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    requested_device = str(cfg["training"].get("device", "cpu")).lower().strip()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("Requested CUDA but no GPU is available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    ablation_cfg = cfg.get("ablation", {})
    modes = ["C_full"]
    if bool(ablation_cfg.get("enabled", True)) and not args.skip_ablation:
        modes = list(ablation_cfg.get("modes", ["A_price_only", "B_price_events", "C_full"]))

    ablation_results: Dict[str, Any] = {}
    full_result: Dict[str, Any] | None = None

    for mode in modes:
        result = run_mode(mode=mode, cfg=cfg, bundle=bundle, splits=splits, device=device)
        ablation_results[mode] = {
            "metrics": result["metrics"],
            "backtest": result["backtest"],
            "flags": result["flags"],
        }
        if mode == "C_full":
            full_result = result

    if full_result is None:
        full_result = run_mode(mode="C_full", cfg=cfg, bundle=bundle, splits=splits, device=device)

    placebo_cfg = cfg.get("placebo", {})
    placebo_results: Dict[str, Any] = {}
    if bool(placebo_cfg.get("enabled", True)) and not args.skip_placebo:
        quick_epochs = int(placebo_cfg.get("quick_epochs", 4))
        tests = placebo_cfg.get("tests", [])
        pseed = int(placebo_cfg.get("random_seed", 17))
        for n, test_name in enumerate(tests):
            arrays_mod = apply_placebo(bundle.arrays, test_name=test_name, seed=pseed + n)
            result = run_mode(
                mode="C_full",
                cfg=cfg,
                bundle=bundle,
                splits=splits,
                device=device,
                seed_offset=100 + n,
                epochs_override=quick_epochs,
                arrays_override=arrays_mod,
            )
            placebo_results[test_name] = {
                "metrics": result["metrics"],
                "backtest": result["backtest"],
            }

    pred = full_result["predictions"]
    reg = full_result["regimes"]
    exp = full_result["explanations"]
    eq = full_result["equity_curve"]

    pred.to_parquet(run_dir / "predictions.parquet", index=False)
    reg.to_parquet(run_dir / "regimes.parquet", index=False)
    exp.to_parquet(run_dir / "explanations.parquet", index=False)
    eq.to_parquet(run_dir / "equity_curve.parquet", index=False)

    save_json(run_dir / "ablations.json", ablation_results)
    save_json(run_dir / "placebo.json", placebo_results)

    metrics = {
        "full_mode_metrics": full_result["metrics"],
        "full_mode_backtest": full_result["backtest"],
        "num_splits": len(splits),
        "num_samples": int(bundle.metadata["num_samples"]),
        "embedding_backend": bundle.metadata.get("embedding_backend", "unknown"),
    }
    save_json(run_dir / "metrics.json", metrics)

    distill_cfg = cfg.get("distillation", {})
    if bool(distill_cfg.get("enabled", True)):
        run_distillation(
            run_dir=run_dir,
            alpha=float(distill_cfg.get("alpha", 0.001)),
            cv_folds=int(distill_cfg.get("cv_folds", 3)),
            max_iter=int(distill_cfg.get("max_iter", 5000)),
        )

    if bool(cfg.get("storage", {}).get("write_run_manifest", True)):
        write_run_manifest(run_dir=run_dir, cfg=cfg, metrics=metrics)

    max_runs = int(cfg.get("storage", {}).get("max_runs_to_keep", 5))
    removed = prune_old_runs(cfg["project"]["output_dir"], max_runs)

    save_json(
        run_dir / "run_summary.json",
        {
            "run_dir": str(run_dir),
            "removed_old_runs": removed,
            "modes": modes,
            "placebo_tests": list(placebo_results.keys()),
        },
    )

    print(json.dumps({"run_dir": str(run_dir), "modes": modes, "removed_old_runs": removed}, indent=2))


if __name__ == "__main__":
    main()
