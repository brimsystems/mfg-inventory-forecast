"""Counterfactual replay: the same replenishment rule on corrected masters.

The as-is replay runs the ERP's reorder logic against the books as they were.
This module builds the inputs for a second run of the identical logic, over
the identical demand and supplier behaviour, with the masters as the
remediation left them:

  one record per physical item        duplicate members retired to the survivor
  every pull recorded                 the bills of materials complete, so backflush
                                      consumes what production consumes
  conversions maintained              receipts land in stock units
  lead times and reorder points       recomputed the way the remediation did:
                                      demand over the actual lead time plus
                                      safety stock at the ABC service level
  documents closed                    a short receipt is followed up, so no line
                                      stays open with a balance that never arrives

What differs between the two runs is therefore only the data the rule saw. The
comparison gives the working capital, purchases, expedites, shortages and job
delays attributable to the errors together, measured in the same units on the
same days. It is a simulation, and the report labels it as one.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .. import config as C
from .purchase_orders import _actual_lead

Z_BY_ABC = {"A": 2.054, "B": 1.645, "C": 1.282}     # 98 / 95 / 90 percent service


def corrected_inputs(events, item_master, item_meta, dup_map, drift_rate, drift_supplier_id, abc_by_item, rng):
    primary = {}
    for num, m in item_meta.items():
        if not m["dead"]:
            iid = m["item_id"]
            primary[iid] = dup_map[iid]["primary"] if iid in dup_map else primary.get(iid, num)

    # every event recorded, against the surviving number
    ev = events.copy()
    ev["item_number"] = ev["item_id"].map(primary)
    ev["recorded"] = True

    # one record per item, with the conversion maintained
    meta_c = {}
    for num, m in item_meta.items():
        mm = dict(m)
        if not m["dead"]:
            mm["dead"] = num != primary[m["item_id"]]
            mm["uom_conv"] = 1
        meta_c[num] = mm

    # parameters recomputed as the remediation did, from the last twelve months
    # of actual usage over the actual lead time
    last = ev[ev["date"] >= C.MODEL_SPAN_END - timedelta(days=365)]
    monthly = (last.groupby("item_id")["qty"].sum() / 12.0).to_dict()
    im = item_master.copy().set_index("item_number")
    base_lead = im["master_lead_time_days"].to_dict()
    for iid, num in primary.items():
        L = _actual_lead(meta_c[num], int(base_lead[num]), C.MODEL_SPAN_END,
                         drift_rate.get(num, 6.0), drift_supplier_id, rng)
        dlt = float(monthly.get(iid, 0.0)) * L / 30.0
        ss = Z_BY_ABC[abc_by_item.get(iid, "C")] * np.sqrt(max(1.0, dlt))
        im.at[num, "master_lead_time_days"] = int(L)
        im.at[num, "reorder_point"] = int(round(dlt + ss))
        im.at[num, "safety_stock"] = int(round(ss))
    return ev, im.reset_index(), meta_c, base_lead, primary


def replay_metrics(sim, cost_by_item, primary, production_orders, year):
    """What a replay did in one calendar year, in the units the shop runs on."""
    days = sim["days"]
    N = len(days)
    in_year = np.asarray(days.year == year)
    value = np.zeros(N)
    per_item = {}
    for iid, s in sim["physical"].items():
        v = np.clip(s, 0, None) * float(cost_by_item.get(iid, 0.0))
        value += v
        if primary.get(iid):
            per_item[primary[iid]] = float(v[in_year].mean())

    po = sim["po_lines"].copy()
    po["year"] = pd.to_datetime(po["order_date"]).dt.year
    p = po[po["year"] == year]
    rush = p[p["rush"]]

    sh = sim["shortages"].copy()
    if len(sh):
        sh = sh[pd.to_datetime(sh["date"]).dt.year == year]

    prod = production_orders.copy()
    prod["year"] = pd.to_datetime(prod["due_date"]).dt.year
    jobs = set(prod.loc[prod["year"] == year, "order_id"])
    delays = {j: d for j, d in sim["job_delays"].items() if j in jobs and d > 0}

    return {
        "inventory_value_avg": float(value[in_year].mean()),
        "inventory_value_end": float(value[-1]),
        "purchases": float((p["qty_ordered"] * p["unit_price"]).sum()),
        "po_lines": int(len(p)),
        "rush_lines": int(len(rush)),
        "rush_freight": float(rush["freight"].sum()),
        "rush_premium": float(rush["premium"].sum()),
        "shortage_episodes": int(len(sh)),
        "shortage_items": int(sh["item_id"].nunique()) if len(sh) else 0,
        "shortages_by_cause": {k: int(v) for k, v in sh["cause"].value_counts().items()} if len(sh) else {},
        "jobs_in_year": int(len(jobs)),
        "jobs_delayed": int(len(delays)),
        "job_delay_days": int(sum(delays.values())),
        "inventory_by_item": per_item,
    }
