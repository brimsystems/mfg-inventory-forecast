"""The reorder points the shop runs on in the forward window, refreshed monthly
from the demand model.

At the start of every forward month the model is retrained on the cleaned
history to date, forecasts each item's demand over its lead time, and sets

    reorder point = forecast over the lead time + safety stock
    safety stock  = k(ABC) x the item's forecast error over the lead time

where k is calibrated, not assumed: the quantile of the 2025 backtest errors,
in units of each item's own error, at the class's service level (98 / 95 / 90).
Forecast errors are skewed, so the normal z would oversize the buffer.

A 20% hysteresis band keeps a point where it is unless the fresh value moves
it by more than a fifth, so the buyers see a number that changes when demand
does rather than one that jitters with the model. Both the raw and the applied
points are recorded, so the churn the band removes can be reported.

Writes ml/data/policy/rop_schedule.json, which the generator applies in the
forward window. Run:  python -m ml.src.forward_policy
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from .resolution import REPO
from .features import build_feature_frame, FEATURE_COLS, _row_features
from . import training as tr
from .training import _make, _fit_predict
from data_source.generate import config as C

MARTS = REPO / "ml" / "data" / "marts"
BACKTEST = REPO / "ml" / "data" / "backtest"
OUT = REPO / "ml" / "data" / "policy"
Z_BY_ABC = {"A": 2.05, "B": 1.64, "C": 1.28}      # 98 / 95 / 90 percent service
HYSTERESIS = 0.20


def _forward_months():
    m = C.FORWARD_START
    out = []
    for k in range(C.FORWARD_MONTHS):
        y, mo = m.year + (m.month - 1 + k) // 12, (m.month - 1 + k) % 12 + 1
        out.append(pd.Timestamp(date(y, mo, 1)))
    return out


def run():
    monthly = pd.read_parquet(MARTS / "consumption_monthly.parquet")
    attrs = pd.read_parquet(MARTS / "item_attributes.parquet").set_index("canonical_item_number")
    metrics = json.loads((BACKTEST / "model_metrics.json").read_text())
    bt = pd.read_parquet(BACKTEST / "model_backtest.parquet")
    winner, params = metrics["winner"], metrics["candidates"][metrics["winner"]]["params"]

    wide = monthly.pivot(index="canonical", columns="month", values="consumption").fillna(0).sort_index(axis=1)
    all_months = list(wide.columns)
    fwd = _forward_months()

    # the model is trained on log demand, so its forecast sits nearer the median
    # than the mean and runs low; a reorder point needs the mean. Correct it per
    # demand pattern with the ratio of actual to forecast on the 2025 backtest,
    # which ends before the forward window, then size safety stock on the
    # corrected residuals.
    bias_factor = (bt.groupby("segment")["actual"].sum() / bt.groupby("segment")["pred"].sum()).to_dict()
    bt = bt.assign(pred_c=bt["pred"] * bt["segment"].map(bias_factor))
    resid = (bt["actual"] - bt["pred_c"]).groupby(bt["item"]).std()
    seg_resid = (bt["actual"] - bt["pred_c"]).groupby(bt["segment"]).std()
    print("  bias correction by segment (actual / forecast on the 2025 backtest):",
          {k: round(v, 3) for k, v in bias_factor.items()})
    # calibrate the safety factor per ABC class on the same backtest
    sd_item = bt["item"].map(resid)
    std_err = ((bt["actual"] - bt["pred_c"]) / sd_item).replace([np.inf, -np.inf], np.nan)
    service = {"A": 0.98, "B": 0.95, "C": 0.90}
    k_abc = {a: float(np.nanquantile(std_err[bt["abc"] == a], q)) for a, q in service.items()}
    print("  safety factor k by ABC (calibrated) vs normal z:",
          {a: (round(k_abc[a], 2), Z_BY_ABC[a]) for a in k_abc})

    schedule, raw_points, applied_points = {}, {}, {}
    forecasts = []
    prev = {}
    for m in fwd:
        hist_months = [x for x in all_months if x < m]
        hist = wide[hist_months]
        frame = build_feature_frame(hist, attrs, {}, hist_months)
        frame[FEATURE_COLS] = frame[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
        frame[FEATURE_COLS] = frame[FEATURE_COLS].fillna(frame[FEATURE_COLS].median())
        train = frame[frame["split"] != "gap"]
        tr._CAP = float(train["target"].max()) * 4
        model = _make(winner, params)
        # one row per item at the forecast origin: features from the history before m
        rows = []
        T = len(hist_months)
        for item, series in hist.iterrows():
            if item not in attrs.index:
                continue
            a = attrs.loc[item]
            h = max(1, int(round(a["corrected_lead_days"] / 30.0)))
            f = _row_features(series.to_numpy(float), T, m.month)
            f.update({"item": item, "horizon": h, "std_cost": a["standard_cost"], "corrected_lead": a["corrected_lead_days"],
                      "annual": a["annual_consumption"], "price": a["standard_cost"],
                      "seg_code": {"smooth": 0, "erratic": 1, "lumpy": 2, "intermittent": 3}.get(a["segment"], 0),
                      "abc_code": {"A": 0, "B": 1, "C": 2}.get(a["abc"], 2),
                      "cls_code": frame.loc[frame["item"] == item, "cls_code"].iloc[0] if (frame["item"] == item).any() else 0})
            rows.append(f)
        origin = pd.DataFrame(rows)
        origin[FEATURE_COLS] = origin[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(train[FEATURE_COLS].median())
        origin["pred"] = _fit_predict(model, train[FEATURE_COLS], train["target"], origin[FEATURE_COLS])
        seg_of = {i: attrs.loc[i, "segment"] for i in origin["item"]}
        origin["pred"] = origin["pred"] * origin["item"].map(lambda i: bias_factor.get(seg_of[i], 1.0))

        for r in origin.itertuples(index=False):
            a = attrs.loc[r.item]
            lead_days = float(a["corrected_lead_days"]); h = int(r.horizon)
            fc_lead = float(r.pred) * lead_days / (30.0 * h)          # demand over the lead time, in units
            sd_h = float(resid.get(r.item, np.nan))
            if np.isnan(sd_h):
                sd_h = float(seg_resid.get(a["segment"], resid.median()))
            ss = max(0.0, k_abc.get(a["abc"], 1.0)) * sd_h * np.sqrt(lead_days / (30.0 * h))
            rop_raw = fc_lead + ss
            last = prev.get(r.item)
            rop = rop_raw if (last is None or last <= 0 or abs(rop_raw - last) / last > HYSTERESIS) else last
            prev[r.item] = rop
            schedule.setdefault(r.item, []).append([m.date().isoformat(), round(rop, 2), round(ss, 2),
                                                    round(float(r.pred), 3), int(round(30 * h))])
            raw_points.setdefault(r.item, []).append(rop_raw)
            applied_points.setdefault(r.item, []).append(rop)
            forecasts.append({"item": r.item, "month": m.date().isoformat(), "horizon_months": h,
                              "pred_over_horizon": float(r.pred), "segment": a["segment"], "abc": a["abc"]})
        print(f"  {m.date()}  trained on {len(train):,} rows, {len(origin):,} points set")

    def churn(points):
        moves, n = 0, 0
        for seq in points.values():
            for a_, b_ in zip(seq[:-1], seq[1:]):
                n += 1
                if a_ > 0 and abs(b_ - a_) / a_ > 0.20:
                    moves += 1
        return moves / n if n else 0.0

    out = {"months": [m.date().isoformat() for m in fwd], "hysteresis": HYSTERESIS, "model": winner,
           "bias_correction": bias_factor, "safety_factor": k_abc, "normal_z": Z_BY_ABC,
           "items": schedule, "forecasts": forecasts,
           "churn": {"without_hysteresis": churn(raw_points), "with_hysteresis": churn(applied_points)}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rop_schedule.json").write_text(json.dumps(out, indent=1))
    print(f"\n  Reorder-point churn month to month (share of items moving >20%): "
          f"raw {out['churn']['without_hysteresis']*100:.0f}%, applied {out['churn']['with_hysteresis']*100:.0f}%")
    print(f"  -> {OUT / 'rop_schedule.json'}")


if __name__ == "__main__":
    run()
