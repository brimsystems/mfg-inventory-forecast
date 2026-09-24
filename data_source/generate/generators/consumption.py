"""Physical consumption events: what actually leaves the shelf, and whether the
ERP recorded it.

Three channels. Backflush fires when a job is reported complete and explodes the
product's bill of materials: the TRUE bill gives the physical consumption, the
RECORDED bill gives what the ERP subtracts, so a component omitted from the
recorded bill is consumed but never recorded (M3 -> T1). Service orders issue on
shipment. Manual pulls cover non-job usage; on the BOM-omitted items a share of
them are never entered (T1). Every event carries the recorded item number it
posts to (so duplicate records split the history) and a flag for whether the
ERP saw it.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .. import config as C
from .bom import effective_component_map


def build_consumption_events(production_orders, bom_true, bom_recorded, service_orders,
                             manual_demand, item_meta, dup_map, omitted_item_ids, rng, fix_from=None):
    """fix_from: the date from which the bills are complete, so backflush and manual
    pulls on the omitted components are recorded (the remediation's BOM corrections).
    A world that never fixes them passes END_DATE. The random stream is consumed
    identically either way, so two builds differ only in the recorded flag."""
    fix_from = C.REMEDIATION_END if fix_from is None else fix_from
    true_map = effective_component_map(bom_true)
    rec_map = effective_component_map(bom_recorded)

    recorded = {}
    for num, meta in item_meta.items():
        if not meta["dead"]:
            recorded.setdefault(meta["item_id"], []).append(num)

    def route(iid):
        nums = recorded[iid]
        if len(nums) == 1:
            return nums[0]
        return nums[int(rng.choice(len(nums), p=dup_map[iid]["weights"]))]

    rows = []

    # ── Backflush: physical from the true bill, recorded from the recorded bill
    done = production_orders[production_orders["qty_completed"] > 0]
    for r in done.itertuples(index=False):
        cd = pd.to_datetime(r.completed_date).date()
        tmap = true_map.get(r.product_number, {})
        rmap = tmap if cd > fix_from else rec_map.get(r.product_number, {})
        for c, qper in tmap.items():
            if c not in recorded:
                continue
            phys = qper * int(r.qty_completed)
            rec_q = rmap.get(c, 0) * int(r.qty_completed)
            num = route(c)
            if rec_q > 0:
                rows.append((num, c, cd, rec_q, "BACKFLUSH", r.order_id, True))
            if phys - rec_q > 0:
                rows.append((num, c, cd, phys - rec_q, "BACKFLUSH", r.order_id, False))

    # ── Service issues: recorded on shipment ─────────────────────────────────
    num_to_iid = {n: m["item_id"] for n, m in item_meta.items()}
    for r in service_orders.itertuples(index=False):
        rows.append((r.item_number, num_to_iid.get(r.item_number), pd.to_datetime(r.ship_date).date(),
                     int(r.qty), "SERVICE", r.order_id, True))

    # ── Manual pulls: a share on the omitted items are never entered (T1) ────
    omitted = set(int(i) for i in omitted_item_ids)
    for r in manual_demand.itertuples(index=False):
        iid, q = int(r.item_id), int(r.qty)
        if q <= 0 or iid not in recorded:
            continue
        n_lines = int(min(4, max(1, round(q / 7))))
        for s in _split_int(q, n_lines, rng):
            if s <= 0:
                continue
            day = r.month + timedelta(days=int(rng.integers(0, 27)))
            if day.weekday() >= 5:
                day += timedelta(days=7 - day.weekday())
            if day > C.END_DATE:
                continue
            num = route(iid)
            # the bills are completed in remediation, after which the pull is recorded
            u = rng.random()
            unrecorded = (iid in omitted) and day <= fix_from and (u < C.T1_UNRECORDED_SHARE)
            rows.append((num, iid, day, s, "MANUAL", f"WO-{int(rng.integers(100000, 999999))}", not unrecorded))

    ev = pd.DataFrame(rows, columns=["item_number", "item_id", "date", "qty", "channel", "job_id", "recorded"])
    return ev.sort_values(["date", "item_number"]).reset_index(drop=True)


def _split_int(total, parts, rng):
    if total <= 0:
        return [0]
    if parts <= 1:
        return [total]
    props = rng.dirichlet(np.full(parts, 2.0))
    raw = np.floor(props * total).astype(int)
    raw[0] += total - int(raw.sum())
    return raw.tolist()
