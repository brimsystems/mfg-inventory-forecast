"""Cycle counts, with defect D4 (phantom inventory).

Counted quantity diverges from system quantity for a share of items, and the
divergence grows with time since the last count. Items that also carry a
unit-of-measure fault (D3) or a duplicate record (D1) are over-represented among
the phantom items, because those are exactly the records whose on-hand the system
cannot keep straight.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .. import config as C

COUNTERS = [f"CNT-{i:02d}" for i in range(1, 7)]
_CADENCE_DAYS = {"A": 60, "B": 120, "C": 240}


def build_cycle_counts(item_master, item_meta, annual_by_item, abc_by_item,
                       dup_map, defects, rng):
    d3_items = {n for n, m in item_meta.items() if m.get("box_size")}
    d1_members = {n for c in dup_map.values() for n in c["records"]}

    rows = []
    seq = 0
    for r in item_master.itertuples(index=False):
        num = r.item_number
        meta = item_meta[num]
        iid = meta["item_id"]
        if rng.random() > C.CYCLE_COUNT_ANNUAL_COVERAGE:
            continue
        abc = abc_by_item.get(iid, "C")
        weight = dict(zip(dup_map[iid]["records"], dup_map[iid]["weights"]))[num] if iid in dup_map else 1.0
        avg_month = max(0.5, annual_by_item[iid] / 12.0 * weight)

        # phantom flag, biased toward UOM and duplicate records
        p = 0.08 + (0.55 if num in d3_items else 0.0) + (0.35 if num in d1_members else 0.0)
        phantom = rng.random() < min(0.95, p)

        cadence = _CADENCE_DAYS[abc]
        d = C.START_DATE + timedelta(days=int(rng.integers(0, cadence)))
        last = C.START_DATE
        while d <= C.END_DATE:
            months_since = (d - last).days / 30.0
            system_qty = max(0, int(round(avg_month * rng.uniform(0.5, 2.5))))
            if num in d3_items:
                system_qty = int(system_qty * rng.uniform(3, 8))    # box/each inflation
            sd = C.D4_VARIANCE_BASE + C.D4_VARIANCE_PER_MONTH * months_since
            if phantom:
                gap = abs(rng.normal(0, sd * 2.5))
                counted = max(0, int(round(system_qty * (1 - gap))))
            else:
                counted = max(0, int(round(system_qty * (1 + rng.normal(0, sd * 0.4)))))
            seq += 1
            rows.append({
                "count_id":        f"CC-{seq:06d}",
                "item_number":     num,
                "count_date":      d.isoformat(),
                "system_quantity": system_qty,
                "counted_quantity":counted,
                "counter_id":      rng.choice(COUNTERS),
            })
            last = d
            d += timedelta(days=int(cadence * rng.uniform(0.8, 1.3)))
    return pd.DataFrame(rows)
