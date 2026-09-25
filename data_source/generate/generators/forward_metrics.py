"""What a replay delivered over a window, in the terms the shop and the buyers use.

Computed from a replay's outputs for any date window: fill rate in total and by
ABC class, stockout episodes and days short, jobs held for material and the
days lost, rush lines and rush spend, average inventory value and days of
supply, safety stock against cycle stock, excess stock, order lines placed,
purchases by month, and, where a forecast schedule drove the reorder points,
forecast bias by demand pattern and the month-to-month churn of the points.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .. import config as C

EXCESS_MONTHS = 12


def window_metrics(sim, start, end, cost_by_item, abc_by_item, primary, production_orders,
                   ss_by_item=None, schedule=None, segment_by_item=None, tier_by_item=None):
    days = sim["days"]
    mask = np.asarray((days.date >= start) & (days.date <= end))
    n_days = int(mask.sum())
    ndays_month = pd.Series(days[mask]).dt.to_period("M")

    rows = []
    inv_daily = np.zeros(n_days)
    for iid, phys in sim["physical"].items():
        cost = float(cost_by_item.get(iid, 0.0))
        inv_daily += np.clip(phys[mask], 0, None) * cost
        dem = sim["demand"].get(iid, np.zeros(len(days)))[mask]
        unmet = sim["unmet"].get(iid, np.zeros(len(days)))[mask]
        short = sim["short_days"].get(iid, np.zeros(len(days), dtype=bool))[mask]
        avg_phys = float(np.clip(phys[mask], 0, None).mean())
        rows.append({"iid": iid, "abc": abc_by_item.get(iid, "C"), "cost": cost,
                     "demand": float(dem.sum()), "unmet": float(unmet.sum()), "short_days": int(short.sum()),
                     "avg_phys": avg_phys, "avg_value": avg_phys * cost,
                     "demand_value": float(dem.sum()) * cost,
                     "ss_units": float((ss_by_item or {}).get(iid, 0.0))})
    it = pd.DataFrame(rows)

    def fill(df):
        return float(1 - df["unmet"].sum() / df["demand"].sum()) if df["demand"].sum() > 0 else 1.0

    sh = sim["shortages"]
    sh = sh[(pd.to_datetime(sh["date"]).dt.date >= start) & (pd.to_datetime(sh["date"]).dt.date <= end)] if len(sh) else sh
    sh_abc = sh["item_id"].map(abc_by_item) if len(sh) else pd.Series(dtype=str)

    prod = production_orders.copy()
    due = pd.to_datetime(prod["due_date"]).dt.date
    jobs = set(prod.loc[(due >= start) & (due <= end), "order_id"])
    delays = {j: d for j, d in sim["job_delays"].items() if j in jobs and d > 0}

    po = sim["po_lines"].copy()
    od = pd.to_datetime(po["order_date"]).dt.date
    p = po[(od >= start) & (od <= end)].copy()
    p["value"] = p["qty_ordered"] * p["unit_price"]
    p["month"] = pd.to_datetime(p["order_date"]).dt.to_period("M").astype(str)
    rush = p[p["rush"]]

    # excess: items holding more than a year of their own consumption, at the window's rate
    daily = it["demand"] / max(1, n_days)
    excess = it[(daily > 0) & (it["avg_phys"] > daily * 30 * EXCESS_MONTHS)]
    ss_value = float((it["ss_units"] * it["cost"]).sum())
    avg_value = float(it["avg_value"].sum())
    demand_value_per_day = float(it["demand_value"].sum()) / max(1, n_days)

    tiers = tier_by_item or {}
    it["tier"] = it["iid"].map(lambda i: tiers.get(i, "standard"))
    sh_tier = sh["item_id"].map(lambda i: tiers.get(i, "standard")) if len(sh) else pd.Series(dtype=str)
    out = {
        "fill_rate_by_tier": {t: fill(it[it["tier"] == t]) for t in ["line", "service", "standard"]},
        "stockout_episodes_by_tier": {t: int((sh_tier == t).sum()) for t in ["line", "service", "standard"]},
        "stockout_causes": sh["cause"].value_counts().to_dict() if len(sh) else {},
        "start": start.isoformat(), "end": end.isoformat(), "days": n_days, "items": int(len(it)),
        "fill_rate": fill(it),
        "fill_rate_by_abc": {a: fill(it[it["abc"] == a]) for a in ["A", "B", "C"]},
        "stockout_episodes": int(len(sh)),
        "stockout_episodes_by_abc": {a: int((sh_abc == a).sum()) for a in ["A", "B", "C"]},
        "stockout_days": int(it["short_days"].sum()),
        "stockout_days_by_abc": {a: int(it.loc[it["abc"] == a, "short_days"].sum()) for a in ["A", "B", "C"]},
        "items_short": int((it["short_days"] > 0).sum()),
        "jobs": int(len(jobs)), "jobs_delayed": int(len(delays)), "job_delay_days": int(sum(delays.values())),
        "job_delay_days_median": float(np.median(list(delays.values()))) if delays else 0.0,
        "rush_lines": int(len(rush)), "rush_freight": float(rush["freight"].sum()),
        "rush_premium": float(rush["premium"].sum()) if "premium" in rush else 0.0,
        "rush_spend": float(rush["freight"].sum() + (rush["premium"].sum() if "premium" in rush else 0.0)),
        "avg_inventory_value": avg_value,
        "end_inventory_value": float(inv_daily[-1]) if n_days else 0.0,
        "inventory_by_abc": {a: float(it.loc[it["abc"] == a, "avg_value"].sum()) for a in ["A", "B", "C"]},
        "consumption_by_abc": {a: float(it.loc[it["abc"] == a, "demand_value"].sum()) for a in ["A", "B", "C"]},
        "inventory_by_month": {str(k): float(v) for k, v in pd.Series(inv_daily).groupby(ndays_month.to_numpy()).mean().items()},
        "days_of_supply": avg_value / demand_value_per_day if demand_value_per_day > 0 else None,
        "safety_stock_value": ss_value,
        "cycle_stock_value": max(0.0, avg_value - ss_value),
        "excess_items": int(len(excess)), "excess_value": float(excess["avg_value"].sum()),
        "order_lines": int(len(p)), "regular_lines": int((~p["rush"]).sum()),
        "purchases": float(p["value"].sum()),
        "purchases_by_month": {k: float(v) for k, v in p.groupby("month")["value"].sum().items()},
        "consumption_value": float(it["demand_value"].sum()),
    }

    # forecast bias by demand pattern, where the points came from a schedule
    if schedule and segment_by_item:
        acc = {}
        for prim_num, entries in schedule.items():
            iid = _iid_of(prim_num, primary)
            if iid is None or iid not in sim["demand"]:
                continue
            seg = segment_by_item.get(iid, "unknown")
            for (d_, fc_h, h_days) in [(e[0], e[3], e[4]) for e in entries if len(e) >= 5]:
                if d_ < start or d_ > end:
                    continue
                i0 = int(np.searchsorted(days.date, d_)); i1 = min(len(days), i0 + int(h_days))
                actual = float(sim["demand"][iid][i0:i1].sum())
                a = acc.setdefault(seg, [0.0, 0.0]); a[0] += float(fc_h) * (i1 - i0) / max(1, h_days); a[1] += actual
        out["forecast_bias_by_segment"] = {k: (v[0] - v[1]) / v[1] if v[1] > 0 else None for k, v in acc.items()}
    return out


def _iid_of(num, primary):
    for iid, n in primary.items():
        if n == num:
            return iid
    return None
