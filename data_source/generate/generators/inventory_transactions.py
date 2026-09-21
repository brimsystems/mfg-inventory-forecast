"""Inventory transaction ledger: issues, receipts, adjustments and returns.

Issues are the demand signal. Each month's consumption for a physical item is
split into a few work-order issues and routed to one of that item's recorded
numbers (so duplicate records split the history, defect D1). Issues are booked in
each; receipts follow the purchase orders and, for box-bought items, are booked
in boxes with no conversion, which is defect D3 in the ledger.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .. import config as C


def build_inventory_transactions(demand, dup_map, item_meta, purchase_orders, rng):
    monthly = demand.pivot(index="item_id", columns="month", values="demand_units").fillna(0)
    monthly = monthly.sort_index(axis=1)
    months = list(monthly.columns)

    recorded = {}
    for num, meta in item_meta.items():
        recorded.setdefault(meta["item_id"], []).append(num)

    rows = []
    seq = 0

    def add(item_number, tdate, ttype, qty, uom, wo):
        nonlocal seq
        seq += 1
        rows.append({
            "transaction_id":  f"TX-{seq:07d}",
            "item_number":     item_number,
            "transaction_date":tdate.isoformat(),
            "type":            ttype,
            "quantity":        int(qty),
            "uom":             uom,
            "work_order_id":   wo,
            "location":        rng.choice(C.LOCATIONS),
        })

    # Issues from demand.
    for iid, series in monthly.iterrows():
        nums = recorded[iid]
        weights = dup_map[iid]["weights"] if iid in dup_map else [1.0]
        for mi, q in enumerate(series.to_numpy()):
            if q <= 0:
                continue
            n_lines = int(min(4, max(1, round(q / 8))))
            splits = _split_int(int(q), n_lines, rng)
            month0 = months[mi]
            for s in splits:
                if s <= 0:
                    continue
                day = month0 + timedelta(days=int(rng.integers(0, 27)))
                if day.weekday() >= 5:
                    day += timedelta(days=7 - day.weekday())
                num = nums[int(rng.choice(len(nums), p=weights))] if len(nums) > 1 else nums[0]
                wo = f"WO-{rng.integers(100000, 999999)}"
                add(num, day, "issue", s, "EA", wo)
                if rng.random() < C.RETURN_RATE:
                    add(num, day + timedelta(days=2), "return", max(1, int(s * 0.2)), "EA", wo)
                if rng.random() < C.ADJUSTMENT_RATE:
                    add(num, day + timedelta(days=1), "adjustment",
                        int(rng.choice([-2, -1, 1, 2])), "EA", None)

    # Receipts from purchase orders (booked in the purchase UOM).
    d3 = {num: meta["box_size"] for num, meta in item_meta.items() if meta.get("box_size")}
    po = purchase_orders[purchase_orders["received_date"].notna()]
    for r in po.itertuples(index=False):
        uom = "BOX" if r.item_number in d3 else "EA"
        add(r.item_number, pd.to_datetime(r.received_date).date(), "receipt",
            r.quantity_received, uom, None)

    return pd.DataFrame(rows)


def _split_int(total: int, parts: int, rng) -> list:
    if parts <= 1:
        return [total]
    props = rng.dirichlet(np.full(parts, 2.0))
    raw = np.floor(props * total).astype(int)
    raw[0] += total - int(raw.sum())          # push the remainder onto the first line
    return raw.tolist()
