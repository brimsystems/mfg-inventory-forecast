"""Baselines and the rolling-origin backtest harness.

The forecast target is demand over the item's lead time, not a fixed horizon,
because that is what a reorder point needs. Each item's horizon is its corrected
lead time in months. The harness evaluates every baseline over at least twelve
monthly origins and reports WAPE, MASE and bias by demand segment and by ABC
class. The same origins and horizons are reused by the model so the comparison is
like for like.

Baselines: naive, seasonal naive, moving average (what the shop effectively does),
Croston/SBA for intermittent demand, and simple exponential smoothing.

Run:  python -m ml.src.baselines
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .resolution import REPO

MARTS = REPO / "ml" / "data" / "marts"
OUT = REPO / "ml" / "data" / "backtest"
MIN_ORIGINS = 12
BASELINES = ["naive", "snaive", "ma3", "ma6", "ses", "croston"]


def _ses(h, alpha=0.3):
    lvl = h[0]
    for x in h[1:]:
        lvl = alpha * x + (1 - alpha) * lvl
    return lvl


def _croston(h, alpha=0.1, sba=True):
    z = p = None; q = 1
    for x in h:
        if x > 0:
            z = x if z is None else alpha * x + (1 - alpha) * z
            p = q if p is None else alpha * q + (1 - alpha) * p
            q = 1
        else:
            q += 1
    if not z or not p:
        return 0.0
    return (z / p) * (1 - alpha / 2 if sba else 1)


def _rate_forecasts(hist: np.ndarray, t: int, full: np.ndarray) -> dict:
    h = hist[:t]
    r = {"naive": h[-1], "ma3": h[-3:].mean(), "ma6": h[-6:].mean(),
         "ses": _ses(h), "croston": _croston(h)}
    return r


def backtest_series(series: np.ndarray, horizon: int, n_origins: int = MIN_ORIGINS):
    T = len(series)
    last = T - horizon
    origins = [t for t in range(max(12, last - n_origins + 1), last + 1)]
    rows = []
    for t in origins:
        actual = series[t:t + horizon].sum()
        rates = _rate_forecasts(series, t, series)
        fc = {b: rates[b] * horizon for b in ["naive", "ma3", "ma6", "ses", "croston"]}
        fc["snaive"] = series[t - 12:t - 12 + horizon].sum() if t >= 12 else rates["naive"] * horizon
        rows.append({"origin": t, "actual": float(actual), "horizon": horizon, **fc})
    return rows


def _agg(df: pd.DataFrame, by: str) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(by):
        denom = g["actual"].sum()
        naive_mae = np.abs(g["actual"] - g["naive"]).mean()
        row = {by: key, "items": g["item"].nunique(), "n": len(g)}
        for b in BASELINES:
            row[f"wape_{b}"] = np.abs(g["actual"] - g[b]).sum() / denom if denom else np.nan
        # best baseline by WAPE
        best = min(BASELINES, key=lambda b: row[f"wape_{b}"])
        row["best_baseline"] = best
        row["wape_best"] = row[f"wape_{best}"]
        row["mase_best"] = np.abs(g["actual"] - g[best]).mean() / naive_mae if naive_mae else np.nan
        row["bias_best"] = (g[best] - g["actual"]).sum() / denom if denom else np.nan
        out.append(row)
    return pd.DataFrame(out)


def run():
    monthly = pd.read_parquet(MARTS / "consumption_monthly.parquet")
    attrs = pd.read_parquet(MARTS / "item_attributes.parquet").set_index("canonical_item_number")
    wide = (monthly.pivot(index="canonical", columns="month", values="consumption")
            .fillna(0).sort_index(axis=1))

    rows = []
    for item, series in wide.iterrows():
        lead = attrs.loc[item, "corrected_lead_days"] if item in attrs.index else 14
        horizon = max(1, int(round(lead / 30.0)))
        for r in backtest_series(series.to_numpy(float), horizon):
            r["item"] = item
            rows.append(r)
    bt = pd.DataFrame(rows)
    bt["segment"] = bt["item"].map(attrs["segment"])
    bt["abc"] = bt["item"].map(attrs["abc"])

    OUT.mkdir(parents=True, exist_ok=True)
    bt.to_parquet(OUT / "baseline_backtest.parquet", index=False)

    print("\n=== Baseline backtest (demand over lead time, rolling origin) ===")
    print(f"  {bt['item'].nunique()} items, {len(bt):,} item-origins, "
          f"horizons {sorted(bt['horizon'].unique())}")
    seg = _agg(bt, "segment").set_index("segment")
    print("\n  WAPE by segment (best baseline):")
    for s in ["smooth", "erratic", "lumpy", "intermittent"]:
        if s in seg.index:
            print(f"    {s:<13} WAPE {seg.loc[s,'wape_best']*100:5.1f}%  "
                  f"MASE {seg.loc[s,'mase_best']:.2f}  best={seg.loc[s,'best_baseline']}")
    abc = _agg(bt, "abc").set_index("abc")
    print("\n  WAPE by ABC (best baseline):")
    for a in ["A", "B", "C"]:
        if a in abc.index:
            print(f"    {a}  WAPE {abc.loc[a,'wape_best']*100:5.1f}%  MASE {abc.loc[a,'mase_best']:.2f}")
    denom = bt["actual"].sum()
    overall = {b: np.abs(bt['actual'] - bt[b]).sum() / denom for b in BASELINES}
    print("\n  Overall WAPE by method: " + ", ".join(f"{b} {v*100:.0f}%" for b, v in overall.items()))
    print()


if __name__ == "__main__":
    run()
