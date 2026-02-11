from __future__ import annotations

import argparse
import json
from pathlib import Path

import dash
import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html
from dash.dash_table import DataTable

from src.artifacts import latest_run_path


def load_run(run_dir: Path) -> dict:
    required = {
        "predictions": run_dir / "predictions.parquet",
        "regimes": run_dir / "regimes.parquet",
        "explanations": run_dir / "explanations.parquet",
        "equity": run_dir / "equity_curve.parquet",
        "ablations": run_dir / "ablations.json",
        "placebo": run_dir / "placebo.json",
    }

    out = {"run_dir": str(run_dir)}
    for k, p in required.items():
        if not p.exists():
            out[k] = pd.DataFrame() if p.suffix == ".parquet" else {}
            continue
        if p.suffix == ".parquet":
            out[k] = pd.read_parquet(p)
        else:
            out[k] = json.loads(p.read_text(encoding="utf-8"))
    return out


def build_app(run_dir: Path) -> dash.Dash:
    data = load_run(run_dir)

    pred = data["predictions"].copy()
    reg = data["regimes"].copy()
    exp = data["explanations"].copy()
    eq = data["equity"].copy()

    if not pred.empty:
        pred["date"] = pd.to_datetime(pred["date"])
    if not reg.empty:
        reg["date"] = pd.to_datetime(reg["date"])
    if not exp.empty:
        exp["date"] = pd.to_datetime(exp["date"])
    if not eq.empty:
        eq["date"] = pd.to_datetime(eq["date"])

    dates = sorted(pred["date"].dt.strftime("%Y-%m-%d").unique().tolist()) if not pred.empty else []
    tickers = sorted(pred["ticker"].unique().tolist()) if not pred.empty else []

    ablation_modes = sorted(list(data["ablations"].keys())) if data["ablations"] else ["C_full"]

    app = dash.Dash(__name__)

    app.layout = html.Div(
        style={"fontFamily": "Arial, sans-serif", "padding": "16px"},
        children=[
            html.H2("Event-Driven Market Model Dashboard"),
            html.Div(f"Run: {run_dir}", style={"marginBottom": "12px", "color": "#444"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px", "marginBottom": "12px"},
                children=[
                    html.Div(
                        [
                            html.Label("Date"),
                            dcc.Dropdown(id="date-dd", options=[{"label": d, "value": d} for d in dates], value=dates[-1] if dates else None),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Ticker"),
                            dcc.Dropdown(id="ticker-dd", options=[{"label": t, "value": t} for t in tickers], value=tickers[0] if tickers else None),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Ablation Mode"),
                            dcc.Dropdown(
                                id="ablation-dd",
                                options=[{"label": m, "value": m} for m in ablation_modes],
                                value=ablation_modes[-1] if ablation_modes else None,
                            ),
                        ]
                    ),
                ],
            ),
            dcc.Graph(id="equity-graph"),
            dcc.Graph(id="regime-graph"),
            dcc.Graph(id="world-weights-graph"),
            html.H4("Per-Asset Predictions"),
            DataTable(
                id="pred-table",
                columns=[
                    {"name": "ticker", "id": "ticker"},
                    {"name": "ret_hat", "id": "ret_hat"},
                    {"name": "ret_true", "id": "ret_true"},
                    {"name": "vol_hat", "id": "vol_hat"},
                    {"name": "vol_true", "id": "vol_true"},
                ],
                page_size=12,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "fontSize": 13},
            ),
            html.H4("Explanation Card"),
            html.Div(id="explanation-card", style={"padding": "12px", "background": "#f7f7f7", "borderRadius": "8px"}),
            html.H4("Ablation Summary"),
            html.Pre(id="ablation-summary", style={"background": "#f7f7f7", "padding": "12px", "borderRadius": "8px", "whiteSpace": "pre-wrap"}),
        ],
    )

    @app.callback(Output("equity-graph", "figure"), Input("ablation-dd", "value"))
    def update_equity(_mode: str):
        if eq.empty:
            return px.line(title="No equity curve data")
        fig = px.line(eq, x="date", y="equity", title="Backtest Equity Curve")
        fig.update_layout(height=360)
        return fig

    @app.callback(Output("regime-graph", "figure"), Input("ticker-dd", "value"))
    def update_regime(ticker: str):
        if reg.empty:
            return px.line(title="No regime data")
        g = reg.copy()
        if ticker:
            g = g[g["ticker"] == ticker]
        if g.empty:
            return px.line(title="No regime data for selection")
        fig = px.line(g, x="date", y="regime_id", title="Inferred Regime Over Time")
        fig.update_layout(height=320)
        return fig

    @app.callback(Output("world-weights-graph", "figure"), Input("ticker-dd", "value"))
    def update_worlds(ticker: str):
        if reg.empty:
            return px.area(title="No world weights")
        g = reg.copy()
        if ticker:
            g = g[g["ticker"] == ticker]
        ww_cols = [c for c in g.columns if c.startswith("world_weight_")]
        if g.empty or not ww_cols:
            return px.area(title="No world weights for selection")
        grouped = g.groupby("date", as_index=False)[ww_cols].mean()
        melt = grouped.melt(id_vars="date", value_vars=ww_cols, var_name="world", value_name="weight")
        fig = px.area(melt, x="date", y="weight", color="world", title="World Weights Over Time")
        fig.update_layout(height=360)
        return fig

    @app.callback(Output("pred-table", "data"), Input("date-dd", "value"), Input("ticker-dd", "value"))
    def update_pred_table(date_str: str, ticker: str):
        if pred.empty:
            return []
        g = pred.copy()
        if date_str:
            g = g[g["date"].dt.strftime("%Y-%m-%d") == date_str]
        if ticker:
            g = g[g["ticker"] == ticker]
        keep = ["ticker", "ret_hat", "ret_true", "vol_hat", "vol_true"]
        g = g[keep].round(6)
        return g.to_dict("records")

    @app.callback(Output("explanation-card", "children"), Input("date-dd", "value"), Input("ticker-dd", "value"))
    def update_explanation(date_str: str, ticker: str):
        if exp.empty:
            return "No explanations available"
        g = exp.copy()
        if date_str:
            g = g[g["date"].dt.strftime("%Y-%m-%d") == date_str]
        if ticker:
            g = g[g["ticker"] == ticker]
        if g.empty:
            return "No explanation for this selection"
        row = g.iloc[0]
        return html.Div(
            [
                html.Div(f"Top event feature: {row['top_event_feature']}") ,
                html.Div(f"Delta (remove event block): {row['event_zero_delta']:.6f}"),
                html.Div(f"Delta (remove top event type): {row['event_top_type_delta']:.6f}"),
                html.Div(f"Delta (shift calendar windows): {row['calendar_shift_delta']:.6f}"),
                html.Div(f"Resonance: {row['resonance']:.6f}"),
            ]
        )

    @app.callback(Output("ablation-summary", "children"), Input("ablation-dd", "value"))
    def update_ablation(mode: str):
        payload = data["ablations"].get(mode, {}) if data["ablations"] else {}
        return json.dumps(payload, indent=2)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Dash dashboard for event model artifacts")
    parser.add_argument("--runs-root", default="outputs/runs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = latest_run_path(args.runs_root)

    app = build_app(run_dir)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
