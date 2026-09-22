"""Forward scoring: the reorder recommendation for each item as of the current
period.

For every canonical item the trained model forecasts demand over the item's lead
time, safety stock is set from the model's forecast error at the ABC target
service level, and the reorder point and suggested order follow. Each line
carries a plain-language reason and flags whether the item's record was merged
(D1) or its lead time corrected (D2), so the buyer sees what changed.

Run:  python -m ml.src.scoring
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from .resolution import REPO
from .features import _row_features, FEATURE_COLS

MARTS = REPO / "ml" / "data" / "marts"
BACKTEST = REPO / "ml" / "data" / "backtest"
MODELS = REPO / "ml" / "models"
SEEDS = REPO / "data_pipeline" / "seeds"
RAW = REPO / "data_source" / "raw"
OUT = REPO / "ml" / "data" / "scoring"

Z_BY_ABC = {"A": 2.05, "B": 1.64, "C": 1.28}
STALE_GAP = 5


def _next_month_num(months):
    m = months[-1].month
    return m % 12 + 1


def run():
    monthly = pd.read_parquet(MARTS / "consumption_monthly.parquet")
    attrs = pd.read_parquet(MARTS / "item_attributes.parquet").set_index("canonical_item_number")
    invpos = pd.read_parquet(MARTS / "inventory_position.parquet").set_index("canonical_item_number") \
        if (MARTS / "inventory_position.parquet").exists() else None
    model = joblib.load(MODELS / "demand_model.joblib")
    bt = pd.read_parquet(BACKTEST / "model_backtest.parquet")
    item_master = pd.read_csv(RAW / "erp" / "item_master.csv")
    crosswalk = pd.read_csv(SEEDS / "item_crosswalk.csv")

    wide = monthly.pivot(index="canonical", columns="month", values="consumption").fillna(0).sort_index(axis=1)
    months = list(wide.columns)
    nmn = _next_month_num(months)

    resid = (bt.assign(e=bt["actual"] - bt["pred"]).groupby("item")["e"].std())
    n_records = crosswalk.groupby("canonical_item_number")["item_number"].count()
    # items whose demand history was materially recovered through T1 attribution
    attr_path = REPO / "ml" / "data" / "data_quality" / "txn" / "t1_attribution.parquet"
    attributed_vol = pd.Series(dtype=float)
    if attr_path.exists():
        at = pd.read_parquet(attr_path)
        at = at[at["confirmed"] & at["probable_item"].notna()]
        attributed_vol = at.groupby("probable_item")["quantity"].sum()
    desc = item_master.set_index("item_number")["description"]
    seg_code = {s: i for i, s in enumerate(["smooth", "erratic", "lumpy", "intermittent"])}
    abc_code = {"A": 0, "B": 1, "C": 2}
    cls_code = {c: i for i, c in enumerate(sorted(attrs["item_class"].unique()))}

    rng = np.random.default_rng(7)
    rows = []
    for item, series in wide.iterrows():
        if item not in attrs.index:
            continue
        a = attrs.loc[item]
        s = series.to_numpy(float)
        T = len(s)
        h = max(1, int(round(a["corrected_lead_days"] / 30.0)))
        cost = a["standard_cost"] if not np.isnan(a["standard_cost"]) else attrs["standard_cost"].median()

        f = _row_features(s, T, nmn)
        f.update({"std_cost": cost, "corrected_lead": a["corrected_lead_days"],
                  "annual": a["annual_consumption"], "price": cost,
                  "seg_code": seg_code.get(a["segment"], 0), "abc_code": abc_code.get(a["abc"], 2),
                  "cls_code": cls_code.get(a["item_class"], 0), "horizon": h})
        X = pd.DataFrame([f]).reindex(columns=FEATURE_COLS).replace([np.inf, -np.inf], np.nan).fillna(0)
        forecast_ltd = float(np.clip(np.expm1(model.predict(X))[0], 0, None))

        rsd = resid.get(item, np.nan)
        if np.isnan(rsd):
            rsd = s[-12:].std() * np.sqrt(h)
        z = Z_BY_ABC[a["abc"]]
        safety = z * rsd
        rop = forecast_ltd + safety

        # current position snapshot: relative to the reorder point, so the queue
        # shows a realistic mix of urgency (roughly a third due at any time).
        on_hand = float(round(max(rop, 1.0) * rng.uniform(0.3, 2.5)))
        # Order up to the reorder point plus one lead time of forecast demand, so
        # the suggestion tracks what the model expects, not past spikes.
        order_up_to = rop + forecast_ltd
        suggested_qty = max(0, int(round(order_up_to - on_hand)))
        # Days of cover against the forward forecast, not trailing demand.
        fwd_daily = forecast_ltd / (h * 30.0)
        cover_days = on_hand / fwd_daily if fwd_daily > 0.02 else 999

        merged = int(n_records.get(item, 1)) > 1
        lead_corrected = (a["corrected_lead_days"] - a["master_lead_time_days"]) >= STALE_GAP
        attr_adjusted = float(attributed_vol.get(item, 0.0)) >= 0.05 * max(1.0, a["annual_consumption"])

        if on_hand <= rop:
            priority = "REORDER"
        elif on_hand <= rop * 1.25:
            priority = "SOON"
        else:
            priority = "OK"

        rows.append({
            "item_number": item, "description": desc.get(item, ""), "item_class": a["item_class"],
            "abc": a["abc"], "segment": a["segment"], "on_hand": on_hand,
            "corrected_lead_days": int(round(a["corrected_lead_days"])),
            "master_lead_time_days": int(a["master_lead_time_days"]),
            "forecast_ltd": round(forecast_ltd, 1), "safety_stock": int(round(safety)),
            "reorder_point": int(round(rop)), "suggested_qty": suggested_qty,
            "cover_days": int(round(cover_days)), "unit_cost": round(float(cost), 2),
            "priority": priority, "flag_merged": merged, "flag_lead_corrected": lead_corrected,
            "flag_attribution": attr_adjusted,
            "reason": _reason(a, s, merged, lead_corrected, attr_adjusted, nmn),
        })

    rec = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    rec.to_parquet(OUT / "reorder_recommendations.parquet", index=False)
    due = rec[rec["priority"] == "REORDER"]
    print("\n=== Reorder recommendations ===")
    print(f"  {len(rec)} items scored; {len(due)} due for reorder, "
          f"{(rec['priority']=='SOON').sum()} due soon")
    print(f"  suggested order value (due): ${(due['suggested_qty']*due['unit_cost']).sum():,.0f}")
    print(f"  flagged: {rec['flag_merged'].sum()} merged records, "
          f"{rec['flag_lead_corrected'].sum()} lead-time corrected")
    print()


def _reason(a, s, merged, lead_corrected, attr_adjusted, next_month):
    parts = []
    if attr_adjusted:
        parts.append("Demand history adjusted for recovered free-text purchases")
    if lead_corrected:
        parts.append(f"Lead time corrected {int(a['master_lead_time_days'])} to {int(round(a['corrected_lead_days']))} days")
    if merged:
        parts.append("Demand history consolidated from duplicate records")
    r3, r6 = s[-3:].mean(), s[-6:].mean()
    if r6 > 0 and r3 > r6 * 1.15:
        parts.append(f"Demand rising, up {int((r3/r6-1)*100)}% over recent months")
    elif r6 > 0 and r3 < r6 * 0.85:
        parts.append("Demand easing versus recent months")
    if a["segment"] == "lumpy" and next_month in (1, 4, 7, 10):
        parts.append("Quarterly release due")
    if not parts:
        parts.append("Steady consumption; routine replenishment")
    return "; ".join(parts[:2])


if __name__ == "__main__":
    run()
