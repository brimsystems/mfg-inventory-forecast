"""Validation checkpoint for the demand generators (spec section 2).

Confirms two properties before anything downstream is built:

  1. The demand-segment mix, classified from the generated series by average
     inter-demand interval and squared coefficient of variation, lands near the
     intended 25 / 30 / 25 / 20 split.
  2. Cleaning measurably improves forecast accuracy: on the items whose records
     were split into duplicates, forecasting the merged series beats forecasting
     the split halves and summing them.

It also reports best-baseline WAPE by segment so the achievable error floors can
be checked against the spec's realistic ranges. If the numbers are off, the fix
belongs in the generator constants, not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .generators.demand import build_item_plan, build_monthly_demand


# ── Segment classification (Syntetos-Boylan-Croston) ────────────────────────
def classify(series: np.ndarray) -> str:
    nz = series[series > 0]
    if len(nz) < 2:
        return "intermittent"
    adi = len(series) / len(nz)
    cv2 = (nz.std() / nz.mean()) ** 2 if nz.mean() > 0 else 0.0
    if adi < 1.32:
        return "smooth" if cv2 < 0.49 else "erratic"
    return "intermittent" if cv2 < 0.49 else "lumpy"


# ── Baselines (one-step, expanding origin) ──────────────────────────────────
def _ses(series, alpha=0.3):
    level = series[0]
    for x in series[1:]:
        level = alpha * x + (1 - alpha) * level
    return level


def _croston(series, alpha=0.1, sba=True):
    z = None; p = None; q = 1
    for x in series:
        if x > 0:
            z = x if z is None else alpha * x + (1 - alpha) * z
            p = q if p is None else alpha * q + (1 - alpha) * p
            q = 1
        else:
            q += 1
    if z is None or p is None or p == 0:
        return 0.0
    f = z / p
    return f * (1 - alpha / 2) if sba else f


def forecast_all(hist: np.ndarray, month_idx: int) -> dict:
    """One-step forecasts from every baseline given history up to month_idx-1."""
    h = hist[:month_idx]
    out = {"naive": h[-1], "ma3": h[-3:].mean(), "ses": _ses(h), "croston": _croston(h)}
    out["snaive"] = hist[month_idx - 12] if month_idx >= 12 else h[-1]
    return out


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = actual.sum()
    return np.abs(actual - forecast).sum() / denom if denom > 0 else np.nan


def best_baseline_wape(series: np.ndarray, n_origins: int = 12) -> float:
    T = len(series)
    origins = range(T - n_origins, T)
    per = {b: [] for b in ["naive", "ma3", "ses", "croston", "snaive"]}
    act = []
    for t in origins:
        fc = forecast_all(series, t)
        act.append(series[t])
        for b, v in fc.items():
            per[b].append(v)
    act = np.array(act, dtype=float)
    scores = {b: wape(act, np.array(v, dtype=float)) for b, v in per.items()}
    valid = [s for s in scores.values() if not np.isnan(s)]
    return min(valid) if valid else np.nan


# ── D1 duplicate split (month-level routing) ────────────────────────────────
def split_series(series: np.ndarray, k: int, rng) -> list:
    """Route each month's whole demand to one duplicate record, so each half is
    a thinned, sparser version of the same total."""
    w = rng.dirichlet(np.full(k, 4.0))
    halves = [np.zeros_like(series) for _ in range(k)]
    for t, q in enumerate(series):
        m = rng.choice(k, p=w)
        halves[m][t] = q
    return halves


def run():
    rng = np.random.default_rng(C.RANDOM_SEED)
    plan = build_item_plan(rng)
    demand = build_monthly_demand(plan, rng)
    wide = demand.pivot(index="item_id", columns="month", values="demand_units").fillna(0)
    wide = wide.sort_index(axis=1)
    series_by_item = {i: wide.loc[i].to_numpy(dtype=float) for i in wide.index}

    # 1. Segment mix from the classifier.
    cls = {i: classify(s) for i, s in series_by_item.items()}
    cls_ser = pd.Series(cls)
    mix = (cls_ser.value_counts(normalize=True).reindex(C.SEGMENTS).fillna(0))
    spec_mix = {"smooth": 0.25, "erratic": 0.30, "lumpy": 0.25, "intermittent": 0.20}
    print("\n=== 1. Demand-segment mix (classified vs spec target) ===")
    for seg in C.SEGMENTS:
        print(f"  {seg:<13} classified {mix[seg]*100:5.1f}%   spec target {spec_mix[seg]*100:4.0f}%")

    # Cross-tab: how the intended (generated) segment maps to the classified one.
    gen = plan.set_index("item_id")["segment"]
    xt = pd.crosstab(gen.reindex(cls_ser.index), cls_ser, normalize="index")
    print("\n  intended -> classified (row-normalized):")
    print(xt.reindex(index=C.SEGMENTS, columns=C.SEGMENTS).fillna(0).round(2).to_string())

    # 2. Best-baseline WAPE by classified segment (achievable floor proxy).
    floor = {i: best_baseline_wape(s) for i, s in series_by_item.items()}
    floor_ser = pd.Series(floor)
    print("\n=== 2. Best-baseline WAPE by classified segment ===")
    ranges = {"smooth": "15-30", "erratic": "30-50", "lumpy": "35-60", "intermittent": "60-90"}
    for seg in C.SEGMENTS:
        ids = cls_ser[cls_ser == seg].index
        vals = floor_ser.reindex(ids).dropna()
        print(f"  {seg:<13} median {vals.median()*100:5.1f}%   IQR "
              f"[{vals.quantile(.25)*100:4.1f}, {vals.quantile(.75)*100:4.1f}]   target {ranges[seg]}")

    # 3. Before/after cleaning on duplicate clusters.
    dup_pool = plan[plan["item_class"].isin(C.D1_CLUSTER_CLASSES)].copy()
    dup_pool["annual"] = dup_pool["item_id"].map(
        lambda i: series_by_item[i][-12:].sum())
    # Duplicate clusters concentrate in smooth, high-consumption items: the
    # merged total is forecastable, but the split halves are not.
    dup_pool = dup_pool[dup_pool["segment"] == "smooth"]
    chosen = dup_pool.sort_values("annual", ascending=False).head(C.D1_N_CLUSTERS * 2)
    chosen = chosen.sample(n=C.D1_N_CLUSTERS, random_state=C.RANDOM_SEED)

    rows = []
    for iid in chosen["item_id"]:
        total = series_by_item[iid]
        k = int(rng.integers(C.D1_MEMBERS_RANGE[0], C.D1_MEMBERS_RANGE[1] + 1))
        halves = split_series(total, k, rng)
        # after: forecast merged; before: forecast each half, sum forecasts.
        after = best_baseline_wape(total)
        before = _summed_halves_wape(halves, total)
        rows.append((iid, k, before, after))
    comp = pd.DataFrame(rows, columns=["item_id", "k", "before", "after"]).dropna()
    comp["rel_improve"] = (comp["before"] - comp["after"]) / comp["before"]
    print("\n=== 3. Duplicate cleaning: WAPE before vs after merge (merged items) ===")
    print(f"  clusters compared        {len(comp)}")
    print(f"  median WAPE before merge {comp['before'].median()*100:5.1f}%")
    print(f"  median WAPE after merge  {comp['after'].median()*100:5.1f}%")
    print(f"  median rel. improvement  {comp['rel_improve'].median()*100:5.1f}%   (target 20-40%)")
    print(f"  share flat or worse      {(comp['rel_improve'] <= 0.02).mean()*100:5.1f}%   (expect 10-20%)")
    print()


def _summed_halves_wape(halves, total, n_origins: int = 12):
    T = len(total)
    origins = range(T - n_origins, T)
    act = np.array([total[t] for t in origins], dtype=float)
    # sum of per-half best-baseline one-step forecasts
    fsum = np.zeros(len(list(origins)))
    origins = list(range(T - n_origins, T))
    for h in halves:
        # pick the baseline that is best for this half in isolation
        per = {}
        for b in ["naive", "ma3", "ses", "croston", "snaive"]:
            preds = []
            for t in origins:
                preds.append(forecast_all(h, t)[b])
            per[b] = np.array(preds, dtype=float)
        hact = np.array([h[t] for t in origins], dtype=float)
        best_b = min(per, key=lambda b: wape(hact, per[b]) if hact.sum() > 0 else np.inf)
        fsum = fsum + per[best_b]
    return wape(act, fsum)


if __name__ == "__main__":
    run()
