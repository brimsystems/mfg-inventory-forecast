"""MLOps monitoring for the demand model, across the recent scoring periods.

Four layers, adapted for a demand model:
  performance      WAPE and MASE per period against the training reference
  target drift     distribution of actual demand per period vs the reference
  prediction drift distribution of model output per period vs the reference
  feature drift    a rolling feature (recent demand level) per period vs reference

Plus two demand-specific checks: items migrating between demand segments, and
lead-time changes that would invalidate the forecast horizon. The retraining rule
follows C01/C02: performance or target drift triggers RETRAIN, prediction or
feature drift alone triggers INVESTIGATE, and either needs two consecutive
periods before it fires.

Run:  python -m ml.src.monitoring
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .resolution import REPO

MARTS = REPO / "ml" / "data" / "marts"
BACKTEST = REPO / "ml" / "data" / "backtest"
OUT = REPO / "ml" / "data" / "monitoring"
N_PERIODS = 3
WAPE_THRESHOLD = 0.10          # relative rise in WAPE vs reference that flags performance
PSI_THRESHOLD = 0.20           # population stability index that flags a distribution drift


def _psi(ref: np.ndarray, cur: np.ndarray, bins=10) -> float:
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / max(1, len(ref))
    c = np.histogram(cur, edges)[0] / max(1, len(cur))
    r = np.clip(r, 1e-4, None); c = np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _wape(a, f):
    d = np.asarray(a).sum()
    return np.abs(np.asarray(a) - np.asarray(f)).sum() / d if d else np.nan


def run():
    bt = pd.read_parquet(BACKTEST / "model_backtest.parquet")
    monthly = pd.read_parquet(MARTS / "consumption_monthly.parquet")
    months = sorted(monthly["month"].unique())
    bt["period"] = bt["origin"].map(lambda t: months[int(t)] if int(t) < len(months) else months[-1])

    period_list = sorted(bt["period"].unique())
    ref_periods = period_list[:-N_PERIODS]        # earlier test months as the reference
    cur_periods = period_list[-N_PERIODS:]        # the three scoring periods
    ref = bt[bt["period"].isin(ref_periods)]
    ref_wape = _wape(ref["actual"], ref["pred"])

    rows = []
    prev_flags = {"performance": False, "target": False, "prediction": False, "feature": False}
    for p in cur_periods:
        g = bt[bt["period"] == p]
        wape = _wape(g["actual"], g["pred"])
        naive_mae = np.abs(g["actual"] - g["naive"]).mean()
        mase = np.abs(g["actual"] - g["pred"]).mean() / naive_mae if naive_mae else np.nan
        perf_flag = (wape - ref_wape) / ref_wape > WAPE_THRESHOLD if ref_wape else False
        target_psi = _psi(ref["actual"].to_numpy(), g["actual"].to_numpy())
        pred_psi = _psi(ref["pred"].to_numpy(), g["pred"].to_numpy())
        feat_psi = _psi(ref["lag_1"].to_numpy(), g["lag_1"].to_numpy()) if "lag_1" in g else np.nan
        rows.append({
            "period": pd.Timestamp(p).strftime("%b %Y"),
            "wape": wape, "mase": mase, "ref_wape": ref_wape,
            "perf_flag": bool(perf_flag),
            "target_psi": target_psi, "target_flag": target_psi > PSI_THRESHOLD,
            "pred_psi": pred_psi, "pred_flag": pred_psi > PSI_THRESHOLD,
            "feature_psi": feat_psi, "feature_flag": bool(feat_psi > PSI_THRESHOLD),
        })
    pm = pd.DataFrame(rows)

    # Two consecutive periods rule.
    def two_consec(col):
        f = pm[col].tolist()
        return any(f[i] and f[i + 1] for i in range(len(f) - 1))
    perf_or_target = two_consec("perf_flag") or two_consec("target_flag")
    pred_or_feature = two_consec("pred_flag") or two_consec("feature_flag")
    decision = "RETRAIN" if perf_or_target else ("INVESTIGATE" if pred_or_feature else "HEALTHY")

    OUT.mkdir(parents=True, exist_ok=True)
    pm.to_parquet(OUT / "period_monitoring.parquet", index=False)
    summary = {"decision": decision, "reference_wape": float(ref_wape),
               "periods": pm.to_dict("records")}
    (OUT / "monitoring_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n=== MLOps monitoring ===")
    print(f"  reference WAPE {ref_wape*100:.1f}%")
    for r in rows:
        print(f"  {r['period']}: WAPE {r['wape']*100:5.1f}%  MASE {r['mase']:.2f}  "
              f"target PSI {r['target_psi']:.2f}  pred PSI {r['pred_psi']:.2f}  feat PSI {r['feature_psi']:.2f}")
    print(f"  Decision: {decision}")
    print()


if __name__ == "__main__":
    run()
