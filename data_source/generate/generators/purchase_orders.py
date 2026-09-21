"""Purchase orders, with defect D2 (stale lead times) and the purchasing side of
D5 (fragmented supplier) and D3 (box vs each).

Replenishment orders are placed as consumption accumulates, so the cadence and
quantities track real demand. For items on the drift supplier the recorded
master lead time stays at its creation value while the actual receipt-minus-order
gap ramps upward over the final months, which is what miscalibrates their reorder
points and drives their stockouts.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .. import config as C


def _actual_lead(is_d2: bool, master_lead: int, order_date: date, rng) -> int:
    if not is_d2:
        return max(2, int(round(master_lead + rng.normal(0, 1.5))))
    drift_start = C.END_DATE - timedelta(days=C.D2_DRIFT_MONTHS * 30)
    if order_date <= drift_start:
        base = C.D2_DRIFT_START_DAYS
    else:
        frac = (order_date - drift_start).days / (C.D2_DRIFT_MONTHS * 30)
        base = C.D2_DRIFT_START_DAYS + frac * (C.D2_DRIFT_END_DAYS - C.D2_DRIFT_START_DAYS)
    return max(2, int(round(base + rng.normal(0, 2.0))))


def build_purchase_orders(plan, demand, item_master, item_meta, dup_map,
                          supplier_truth, d5, rng):
    monthly = demand.pivot(index="item_id", columns="month", values="demand_units").fillna(0)
    monthly = monthly.sort_index(axis=1)
    months = list(monthly.columns)
    cost_by_item = plan.set_index("item_id")["unit_cost"].to_dict()

    # recorded numbers by canonical item (single record or duplicate members)
    recorded = {}
    for num, meta in item_meta.items():
        recorded.setdefault(meta["item_id"], []).append(num)

    d3_items = _d3_lookup(item_meta)
    rows = []
    po_seq = 0
    for iid, series in monthly.iterrows():
        vals = series.to_numpy()
        avg = vals[-12:].mean()
        if avg <= 0:
            continue
        batch = max(1, int(round(avg * 2.0)))          # ~two months of cover
        nums = recorded[iid]
        weights = dup_map[iid]["weights"] if iid in dup_map else [1.0]
        acc = 0.0
        for mi, q in enumerate(vals):
            acc += q
            while acc >= batch:
                acc -= batch
                order_date = months[mi] + timedelta(days=int(rng.integers(1, 26)))
                if order_date > C.END_DATE:
                    break
                num = nums[int(rng.choice(len(nums), p=weights))] if len(nums) > 1 else nums[0]
                meta = item_meta[num]
                is_d2 = meta["is_d2"]
                master_lead = int(item_master.loc[item_master["item_number"] == num,
                                                  "master_lead_time_days"].iloc[0])
                alead = _actual_lead(is_d2, master_lead, order_date, rng)
                promised = order_date + timedelta(days=master_lead)
                received = order_date + timedelta(days=alead)
                sup = meta["supplier_id"]
                # D5: some of the fragmented vendor's POs book to its second id.
                if sup == d5["ids"][0] and rng.random() < C.D5_SECOND_ID_SHARE:
                    sup = d5["ids"][1]
                # D3: purchased by the box, so quantity and price are per box.
                box = d3_items.get(num)
                if box:
                    qty = max(1, int(round(batch / box)))
                    price = round(cost_by_item[iid] * box * rng.uniform(0.95, 1.15), 2)
                else:
                    qty = batch
                    price = round(cost_by_item[iid] * rng.uniform(0.9, 1.2), 2)
                recv_qty = qty if rng.random() < 0.93 else int(qty * rng.uniform(0.6, 1.0))
                po_seq += 1
                rows.append({
                    "po_id":            f"PO-{po_seq:06d}",
                    "line":             1,
                    "item_number":      num,
                    "supplier_id":      sup,
                    "order_date":       order_date.isoformat(),
                    "promised_date":    promised.isoformat(),
                    "received_date":    received.isoformat() if received <= C.END_DATE else None,
                    "quantity_ordered": qty,
                    "quantity_received":recv_qty if received <= C.END_DATE else 0,
                    "unit_price":       price,
                })
    return pd.DataFrame(rows)


def _d3_lookup(item_meta) -> dict:
    return {num: meta["box_size"] for num, meta in item_meta.items() if meta.get("box_size")}
