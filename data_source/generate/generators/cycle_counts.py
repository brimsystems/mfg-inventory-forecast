"""Cycle counts and the annual physical inventory.

Before remediation the shop runs one annual physical inventory per year, and the
variance between system and counted quantity is large for items with phantom
inventory (BOM omissions, UOM faults, chronic adjustments). During the
remediation period a cycle-count program runs weekly, unreliable items first, and
its variance narrows as balances are corrected.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .. import config as C


def build_cycle_counts(item_master, item_meta, on_hand, unreliable_nums, rng):
    live = item_master[item_master["status"] == "ACTIVE"].copy()
    live = live[live["item_number"].isin([n for n, m in item_meta.items() if not m["dead"]])]
    rows = []
    seq = 0

    def variance(num, base_var):
        v = base_var
        if num in unreliable_nums:
            v += rng.uniform(0.10, 0.35)
        conv = item_meta.get(num, {}).get("uom_conv", 1)
        if conv and conv > 1:
            v += rng.uniform(0.05, 0.20)
        return v

    # Annual physical inventory each January of the history.
    for yr in range(C.START_DATE.year, C.MODEL_SPAN_END.year + 1):
        count_date = pd.Timestamp(year=yr, month=1, day=int(rng.integers(8, 20))).date()
        if count_date < C.START_DATE or count_date > C.END_DATE:
            continue
        counted = live.sample(frac=C.CYCLE_COUNT_ANNUAL_COVERAGE, random_state=yr)
        for r in counted.itertuples(index=False):
            num = r.item_number
            sysq = int(on_hand.get(num, rng.integers(0, 50)))
            v = variance(num, C.__dict__.get("D4_VARIANCE_BASE", 0.04) if hasattr(C, "D4_VARIANCE_BASE") else 0.04)
            cq = max(0, int(round(sysq * (1 - rng.normal(0.0, v)))))
            seq += 1
            rows.append({"count_id": f"CC-{seq:06d}", "item_number": num,
                         "count_date": count_date.isoformat(), "system_qty": sysq,
                         "counted_qty": cq, "counter_id": rng.choice(C.OFFICE_USERS),
                         "program": "ANNUAL"})

    # Remediation cycle counts: weekly from week 2, unreliable items first.
    weeks = C.remediation_weeks()
    unrel_live = [n for n in unreliable_nums if n in set(live["item_number"])]
    rng.shuffle(unrel_live)
    per_week = max(1, len(unrel_live) // max(1, (C.REMEDIATION_WEEKS - 1)))
    idx = 0
    for w, monday in weeks:
        if w < 2:
            continue
        batch = unrel_live[idx: idx + per_week]; idx += per_week
        for num in batch:
            sysq = int(on_hand.get(num, rng.integers(0, 50)))
            # counts correct balances: post-count variance is small
            cq = max(0, int(round(sysq * (1 - rng.normal(0.0, 0.03)))))
            seq += 1
            rows.append({"count_id": f"CC-{seq:06d}", "item_number": num,
                         "count_date": (monday + timedelta(days=int(rng.integers(0, 5)))).isoformat(),
                         "system_qty": sysq, "counted_qty": cq,
                         "counter_id": rng.choice(C.OFFICE_USERS), "program": "CYCLE"})
    return pd.DataFrame(rows)
