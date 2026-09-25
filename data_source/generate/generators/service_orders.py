"""Service-parts orders: replacement parts sold to the installed base.

The service channel of each item's demand is realized as service-order lines, a
few lines to an order, shipped from stock. Shipments drive ISSUE transactions in
the ledger. Lines are routed to one of the item's recorded numbers, so duplicate
records split the service history the same way production consumption does.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .. import config as C


def build_service_orders(service_demand, item_meta, dup_map, plan, rng):
    """service_demand: DataFrame item_id, month, qty (the service channel)."""
    cost_by_item = plan.set_index("item_id")["unit_cost"].to_dict()
    usage_by_item = {int(i): C.CLASS_USAGE_MULT.get(c, 1.0) for i, c in zip(plan["item_id"], plan["item_class"])}
    recorded = {}
    for num, meta in item_meta.items():
        if not meta["dead"]:
            recorded.setdefault(meta["item_id"], []).append(num)

    # Build a pool of (item_id, date, qty) shipment lines.
    lines = []
    for r in service_demand.itertuples(index=False):
        iid, q = int(r.item_id), int(r.qty)
        if q <= 0 or iid not in recorded:
            continue
        # a customer orders fasteners by the box, not one line per six screws
        n_lines = int(min(3, max(1, round(q / (6 * usage_by_item.get(iid, 1.0))))))
        for s in _split_int(q, n_lines, rng):
            if s <= 0:
                continue
            day = r.month + timedelta(days=int(rng.integers(0, 27)))
            if day.weekday() >= 5:
                day += timedelta(days=7 - day.weekday())
            if day > C.END_DATE:
                continue
            lines.append((iid, day, s))

    rng.shuffle(lines)
    rows = []
    order_seq = 0
    i = 0
    while i < len(lines):
        order_seq += 1
        k = int(rng.integers(1, 4))               # 1-3 lines per order
        oid = f"SO-{order_seq:06d}"
        cust = f"CUST-{int(rng.integers(1, 200)):03d}"
        for line_no in range(1, k + 1):
            if i >= len(lines):
                break
            iid, day, s = lines[i]; i += 1
            nums = recorded[iid]
            weights = dup_map[iid]["weights"] if iid in dup_map else None
            num = nums[int(rng.choice(len(nums), p=weights))] if len(nums) > 1 else nums[0]
            order_date = day - timedelta(days=int(rng.integers(1, 6)))
            rows.append({
                "order_id":   oid,
                "line":       line_no,
                "item_number":num,
                "customer_id":cust,
                "order_date": max(order_date, C.START_DATE).isoformat(),
                "ship_date":  day.isoformat(),
                "qty":        int(s),
                "unit_price": round(float(cost_by_item.get(iid, 1.0)) * rng.uniform(1.3, 2.2), 2),
            })
    return pd.DataFrame(rows)


def _split_int(total, parts, rng):
    if parts <= 1:
        return [total]
    props = rng.dirichlet(np.full(parts, 2.0))
    raw = np.floor(props * total).astype(int)
    raw[0] += total - int(raw.sum())
    return raw.tolist()
