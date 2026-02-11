# News + Stocks Python

This repository now has two tracks:

1. `app.py`: the original Streamlit news/stocks dashboard.
2. Research MVP: an event-driven deep learning market pipeline in `src/` with walk-forward evaluation, purging/embargo, ablations, placebo tests, distillation, and a Dash artifacts viewer.

## Research MVP Scope

The model outputs:

1. Next-day return forecasts (`ret_hat`)
2. Next-day volatility forecasts (`vol_hat`, using `|r_{t+1}|` target)
3. Inferred regime/state (continuous regime vector + optional discrete regime id)
4. Explanation artifacts (world weights, regime vectors, counterfactual deltas)

Key properties:

- CPU-first (macOS friendly)
- Daily frequency
- Deterministic seed control via config
- Time-leakage protection with walk-forward + purge/embargo

## Project Layout

- `configs/default.yaml`: all configuration (data/model/training/splits/ablations/placebo/storage)
- `data/raw/`: raw inputs (ignored by git except `.gitkeep` and sample CSV)
- `data/processed/`: aligned arrays and split metadata
- `src/data/`: fetching/loading/building features/splits
- `src/models/`: event worlds, gate/MoE, regime, asset predictor
- `src/train.py`: end-to-end training + ablation/placebo + artifacts
- `src/eval.py`: metrics/backtest on predictions parquet
- `src/distill.py`: sparse distillation for interpretability
- `dashboards/app.py`: Dash UI for latest run artifacts
- `outputs/runs/<run_name>/`: run artifacts
- `reports/run_manifests/`: lightweight metadata summaries safe to commit

## Install

```bash
cd /Users/usernew/.codex/workspaces/default/news-stocks-python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional text encoder upgrade:

```bash
pip install sentence-transformers
```

If not installed, event text uses TF-IDF + TruncatedSVD fallback.

## Google Colab (Recommended for More Compute)

Use the notebook at `notebooks/train_colab.ipynb`.

What it does:

1. Mounts Google Drive.
2. Clones/pulls your GitHub repo.
3. Installs dependencies.
4. Uses `configs/colab.yaml` (Drive-backed paths + `training.device: cuda`).
5. Runs training and writes outputs to:
   - `/content/drive/MyDrive/news-stocks-python/outputs/runs`

Important:

- Set `REPO_URL` in the notebook before running.
- Put your events file at `/content/drive/MyDrive/news-stocks-python/data/raw/events.csv`.
- If Colab runtime has no GPU, `src/train.py` now auto-falls back to CPU.

## Data Inputs

### 1) Prices (yfinance)

`src/train.py` auto-fetches prices if `data/raw/prices.parquet` is missing.

Manual fetch:

```bash
python -m src.data.fetch_prices \
  --tickers SPY,QQQ,AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA \
  --start 2018-01-01 \
  --end 2026-01-31 \
  --out data/raw/prices.parquet
```

### 2) Events

Default backend is CSV (`data/raw/events.csv`) with columns:

- `date`
- `event_text`
- `event_type` (optional)
- `severity` (optional)
- `sentiment` (optional)

Sample file: `data/raw/events.sample.csv`

Optional GDELT backend (set in `configs/default.yaml`):

```bash
python -m src.data.load_events \
  --backend gdelt \
  --start 2018-01-01 \
  --end 2026-01-31 \
  --out data/raw/events.csv
```

## Train (Walk-Forward + Ablations + Placebo)

```bash
python -m src.train --config configs/default.yaml
```

Outputs are written to `outputs/runs/<timestamp>/`:

- `metrics.json`
- `predictions.parquet`
- `regimes.parquet`
- `explanations.parquet`
- `ablations.json`
- `placebo.json`
- `equity_curve.parquet`
- `distill.json`
- `distill_predictions.parquet`

## Evaluate Saved Predictions

```bash
python -m src.eval \
  --predictions outputs/runs/<run>/predictions.parquet \
  --top-n 2 \
  --allow-short \
  --transaction-cost-bps 5
```

## Distillation (Post-hoc, Sparse)

```bash
python -m src.distill --run-dir outputs/runs/<run>
```

## Dash Dashboard

```bash
python dashboards/app.py
```

Or specify a run:

```bash
python dashboards/app.py --run-dir outputs/runs/<run>
```

Dashboard panels:

- Equity curve
- Inferred regime over time
- World weights over time
- Per-asset prediction table
- Explanation card (top event feature + counterfactual deltas)
- Ablation mode summary viewer

## Adding Assets / Events / Timeline Windows

- Assets: update `data.universe` and `data.asset_names` in `configs/default.yaml`
- Events CSV path/backend: update `data.events.*`
- Timeline windows: update `features.windows`
- Calendar block toggle: `model.use_calendar`
- Resonance block toggle: `model.use_resonance`

## Placebo and Ablation Behavior

- Ablations (`ablation.modes`):
  - `A_price_only`
  - `B_price_events`
  - `C_full`
- Placebo tests (`placebo.tests`):
  - `shuffle_event_dates`
  - `shift_windows`
  - `random_name_embeddings`

If extras fail to beat placebo, treat them as unsupported signal.

## Storage + GitHub Strategy (Low Footprint)

Configured in `.gitignore` and `configs/default.yaml`:

- Raw data and run artifacts are excluded from git
- Only code, config, and compact manifests should be committed
- `storage.max_runs_to_keep` prunes older local runs automatically
- `reports/run_manifests/*.json` stores lightweight run metadata for GitHub history

Recommended workflow:

1. Commit code/config/manifests to GitHub.
2. Keep heavy data/artifacts local or on external object storage.
3. Reproduce any run from `config_used.yaml` + manifest + source commit.

## Existing Streamlit App

Legacy Streamlit dashboard still runs as before:

```bash
streamlit run app.py
```

See environment variables/API notes in `app.py` and old sections in commit history.
