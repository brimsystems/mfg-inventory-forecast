"""Cycle counts and the annual physical inventory, as measurements of the shelf.

A count records what the ERP believed (the ledger balance on the count date,
summed the way the ERP sums it, whatever unit each row was keyed in) against
what was actually there (physical on-hand, with a small counting error), and
posts the difference as a balance correction. Before remediation the shop ran
one annual physical (reason code COUNT). During the remediation period a
cycle-count program runs weekly, unreliable items first (reason code CYCLE).

The counts are keyed in the stock unit. On a UOM-mismatch item the ledger
balance mixes spools and feet, so the count corrects a nonsense number to a
real one every time, which is one of the ways that error surfaces.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .. import config as C

SIGN = {"RECEIPT": 1, "RETURN": 1, "ISSUE": -1, "BACKFLUSH": -1, "SCRAP": -1, "ADJUST": 1}


def build_cycle_counts(sim, ledger, item_meta, dup_map, unreliable_nums, abc_by_item, rng):
    days = sim["days"]
    day_idx = {d.date(): i for i, d in enumerate(days)}

    # ledger balance per item through any date, as the ERP sums it
    lg = ledger[["item_number", "txn_date", "type", "qty"]].copy()
    lg["s"] = lg["qty"] * lg["type"].map(SIGN).fillna(0)
    lg = lg.sort_values("txn_date")
    per = {n: (g["txn_date"].to_numpy(), np.cumsum(g["s"].to_numpy())) for n, g in lg.groupby("item_number")}
    posted = defaultdict(float)     # count corrections already posted, per item

    def system_qty(num, d):
        base = 0.0
        if num in per:
            dates, cum = per[num]
            k = int(np.searchsorted(dates, d, side="right"))
            base = float(cum[k - 1]) if k else 0.0
        return int(round(base + posted[num]))

    def weight_of(num):
        iid = item_meta[num]["item_id"]
        if iid in dup_map:
            return dup_map[iid]["weights"][dup_map[iid]["records"].index(num)]
        return 1.0

    pending = [(r.date, r.item_number, int(r.counted_qty), "ANNUAL") for r in sim["counts"].itertuples(index=False)]

    retired = {n for cl in dup_map.values() for n in cl["records"] if n != cl["primary"]}
    unrel = [n for n in unreliable_nums if n in item_meta and not item_meta[n]["dead"] and n not in retired]
    rng.shuffle(unrel)
    rest = [n for n, m in item_meta.items() if not m["dead"] and n not in retired and n not in set(unrel)]
    rng.shuffle(rest)
    unrel = unrel + rest          # a baseline count of every live item, unreliable ones first
    weeks = [w for w in C.remediation_weeks() if w[0] >= 2]
    per_week = -(-len(unrel) // max(1, len(weeks)))
    i = 0
    for w, monday in weeks:
        batch = unrel[i: i + per_week]; i += per_week
        for num in batch:
            d = monday + timedelta(days=int(rng.integers(0, 5)))
            if d not in day_idx:
                continue
            phys = max(sim["physical"][item_meta[num]["item_id"]][day_idx[d]], 0.0) * weight_of(num)
            counted = max(0, int(round(phys * (1 + rng.normal(0, C.COUNT_NOISE_SD)))))
            pending.append((d.isoformat(), num, counted, "CYCLE"))

    # after remediation the program runs on the ABC schedule: A items every
    # month, B items once a quarter, C items at their annual turn (not yet due).
    # Retired duplicate numbers drop out; the survivor is counted for the pile.
    for num, meta in item_meta.items():
        if meta["dead"] or num in retired:
            continue
        cls = abc_by_item.get(meta["item_id"], "C")
        if cls == "A":
            dates = [date(C.FORWARD_START.year, m, int(rng.integers(2, 27)))
                     for m in range(C.FORWARD_START.month, C.FORWARD_START.month + C.FORWARD_MONTHS)]
        elif cls == "B":
            dates = [C.REMEDIATION_END + timedelta(days=int(rng.integers(7, 85)) + 91 * q)
                     for q in range(-(-C.FORWARD_MONTHS // 3))]
        else:
            dates = []
        for d in dates:
            d = d - timedelta(days=max(0, d.weekday() - 4))     # a working day
            if d not in day_idx or d > C.END_DATE:
                continue
            phys = max(sim["physical"][meta["item_id"]][day_idx[d]], 0.0) * weight_of(num)
            counted = max(0, int(round(phys * (1 + rng.normal(0, C.COUNT_NOISE_SD)))))
            pending.append((d.isoformat(), num, counted, "CYCLE"))

    rows, adj = [], []
    for seq, (d, num, counted, program) in enumerate(sorted(pending), start=1):
        system = system_qty(num, d)
        rows.append({"count_id": f"CC-{seq:06d}", "item_number": num, "count_date": d,
                     "system_qty": system, "counted_qty": counted,
                     "counter_id": rng.choice(C.OFFICE_USERS), "program": program})
        if counted != system:
            adj.append({"item_number": num, "date": d, "qty": counted - system,
                        "reason": "COUNT" if program == "ANNUAL" else "CYCLE"})
            posted[num] += counted - system

    counts = pd.DataFrame(rows)
    adjustments = pd.DataFrame(adj, columns=["item_number", "date", "qty", "reason"])
    return counts, adjustments
