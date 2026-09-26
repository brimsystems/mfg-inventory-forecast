"""Explainability and diagnostics for the weekly demand model, for the technical report.

Rebuilds the weekly training frame exactly as forward_policy does, refits the
selected model on train + validation, and writes:

  ml/data/backtest/explain/split_summary.json   window, rows, items and target stats per split
  ml/data/backtest/explain/shap_importance.csv  mean |SHAP| per feature on a sample of the 2025 test year
  ml/data/backtest/explain/learning_curve.csv   train and held-out WAPE as the training set grows
  ml/data/backtest/explain/feature_corr.csv     Spearman correlation of each feature with the target
  ml/data/backtest/explain/feature_corr_matrix.csv  Spearman correlations among the numeric features
  ml/data/backtest/explain/target_sample.parquet    target by split, for the distribution chart

Run:  python -m ml.src.explain_weekly
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .forward_policy import build_frame, FEATS, _monday, MARTS, BACKTEST
from . import training as tr
from .training import _make, _wape
from data_source.generate import config as C

OUT = BACKTEST / "explain"
NUMERIC = ["s4", "s13", "s26", "s52", "nz13", "nz52", "since_nz", "cv13", "trend", "ly", "mean_all",
           "std_cost", "lead", "annual", "h"]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    weekly = pd.read_parquet(MARTS / "consumption_weekly.parquet")
    attrs = pd.read_parquet(MARTS / "item_attributes.parquet").set_index("canonical_item_number")
    metrics = json.loads((BACKTEST / "weekly_metrics.json").read_text())
    winner, params = metrics["winner"], metrics["candidates"][metrics["winner"]]["params"]

    frame, weeks = build_frame(weekly, attrs)
    frame[FEATS] = frame[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0)
    widx = {w: i for i, w in enumerate(weeks)}
    t_fwd = widx[pd.Timestamp(_monday(C.FORWARD_START))]
    t_test0 = widx[pd.Timestamp(_monday(date(2025, 1, 1) + timedelta(days=7)))]
    t_val0 = t_test0 - 26
    known = frame["t"] + frame["h"] <= t_fwd
    test = frame[known & (frame["t"] >= t_test0)].copy()
    val = frame[(frame["t"] >= t_val0) & (frame["t"] + frame["h"] <= t_test0)]
    trn = frame[frame["t"] + frame["h"] <= t_val0]
    tv = pd.concat([trn, val])
    tr._CAP = float(tv["target"].max()) * 4

    def span(d):
        return f"{d['week'].min().date()} to {d['week'].max().date()}"
    summary = {name: {"window": span(d), "rows": int(len(d)), "items": int(d["item"].nunique()),
                      "target_mean": float(d["target"].mean()), "target_median": float(d["target"].median()),
                      "zero_share": float((d["target"] == 0).mean())}
               for name, d in [("train", trn), ("validation", val), ("test", test)]}
    summary["winner"], summary["params"] = winner, params
    (OUT / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print("  splits:", {k: v["rows"] for k, v in summary.items() if isinstance(v, dict) and "rows" in v})

    pd.concat([trn.assign(split="Train"), val.assign(split="Validation"), test.assign(split="Test (2025)")])[
        ["split", "target"]].to_parquet(OUT / "target_sample.parquet", index=False)

    # feature relationships, on the training rows
    fc = trn[FEATS + ["target"]].corr(method="spearman")["target"].drop("target")
    fc.rename("spearman").to_csv(OUT / "feature_corr.csv")
    trn[NUMERIC].corr(method="spearman").to_csv(OUT / "feature_corr_matrix.csv")

    # the selected model, refit as in the backtest
    model = _make(winner, params)
    model.fit(tv[FEATS], np.log1p(tv["target"]))
    pred = np.clip(np.expm1(model.predict(test[FEATS])), 0, tr._CAP)
    print(f"  refit {winner}: held-out 2025 WAPE {_wape(test['target'], pred) * 100:.1f}%")

    # SHAP on a sample of the held-out year (log-usage units)
    import shap
    sample = test.sample(n=min(1500, len(test)), random_state=7)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(sample[FEATS], check_additivity=False)
    imp = pd.DataFrame({"feature": FEATS, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False)
    imp.to_csv(OUT / "shap_importance.csv", index=False)
    print("  top features:", imp.head(5)["feature"].tolist())

    # learning curve: the same model on growing random shares of train + validation
    rng = np.random.default_rng(11)
    rows = []
    for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
        idx = rng.choice(len(tv), size=int(len(tv) * frac), replace=False)
        sub = tv.iloc[idx]
        m_ = _make(winner, params)
        m_.fit(sub[FEATS], np.log1p(sub["target"]))
        fit_in = np.clip(np.expm1(m_.predict(sub[FEATS])), 0, tr._CAP)
        fit_out = np.clip(np.expm1(m_.predict(test[FEATS])), 0, tr._CAP)
        rows.append({"train_rows": len(sub), "train_wape": _wape(sub["target"], fit_in),
                     "test_wape": _wape(test["target"], fit_out)})
        print(f"  learning curve {frac:.0%}: train {rows[-1]['train_wape']*100:.1f}%  held-out {rows[-1]['test_wape']*100:.1f}%")
    pd.DataFrame(rows).to_csv(OUT / "learning_curve.csv", index=False)


if __name__ == "__main__":
    run()
