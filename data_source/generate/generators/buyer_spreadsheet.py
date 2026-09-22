"""The purchasing manager's spreadsheet.

She keeps a spreadsheet of the components that stop the assembly line when they
are short: mostly A-class parts with stockout history. It uses her own free-text
naming, and its on-hand and lead-time notes disagree with the ERP on most of the
tracked items. On the disagreements she is usually closer to counted reality.
Returns the recorded spreadsheet and the truth used to score the reconciliation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C


def build_buyer_spreadsheet(item_master, item_meta, abc_by_item, on_hand, plan, rng):
    # Candidate line-stopping components: A-class, non-dead, mechanical/electrical/
    # fastener/hardware parts the assembly line waits on.
    live = item_master[item_master["item_number"].isin(
        [n for n, m in item_meta.items() if not m["dead"]])].copy()
    live["item_id"] = live["item_number"].map({n: m["item_id"] for n, m in item_meta.items()})
    live["abc"] = live["item_id"].map({int(k): v for k, v in abc_by_item.items()})
    pool = live[live["abc"] == "A"]
    if len(pool) < C.BUYER_SPREADSHEET_ITEMS:
        pool = live
    chosen = pool.drop_duplicates("item_id").sample(
        n=min(C.BUYER_SPREADSHEET_ITEMS, len(pool.drop_duplicates("item_id"))),
        random_state=C.RANDOM_SEED)

    rows, truth = [], []
    for r in chosen.itertuples(index=False):
        num = r.item_number
        erp_oh = int(on_hand.get(num, rng.integers(0, 60)))
        disagree = rng.random() < C.BUYER_DISAGREE_SHARE
        buyer_right = disagree and (rng.random() < C.BUYER_RIGHT_SHARE)
        if disagree:
            # her note differs from the ERP; when she is right it is closer to truth
            delta = int(rng.integers(3, 25)) * rng.choice([-1, 1])
            buyer_oh = max(0, erp_oh + delta)
        else:
            buyer_oh = erp_oh
        item_ref = _buyer_name(r.description, rng)
        rows.append({
            "item_ref": item_ref,
            "on_hand_note": f"~{buyer_oh} on hand" + (" (runs short)" if rng.random() < 0.4 else ""),
            "lead_time_note": rng.choice(["~6 wks", "~8 wks", "3-4 wks", "long lead",
                                          "expedite avail", "2 wks stock"]),
            "reorder_note": rng.choice(["reorder 50", "min 20", "keep 2 boxes", "order early",
                                        "call rep", ""]),
            "preferred_supplier_note": rng.choice(["primary vendor", "backup only", "cheapest",
                                                   "fast ship", ""]),
            "last_updated": C.AS_OF_DATE.isoformat(),
        })
        truth.append({"item_ref": item_ref, "item_number": num, "erp_on_hand": erp_oh,
                      "buyer_on_hand": buyer_oh, "disagree": bool(disagree),
                      "buyer_closer": bool(buyer_right)})
    return pd.DataFrame(rows), truth


def _buyer_name(desc, rng):
    """The purchasing manager's shorthand: lowercase, abbreviated, partial."""
    s = str(desc).lower()
    reps = {"gearmotor": "gearmtr", "bearing": "brg", "sensor": "sns", "enclosure": "encl",
            "fitting": "fitg", "reducer": "redcr", "contactor": "cont"}
    for k, v in reps.items():
        s = s.replace(k, v)
    toks = s.split()
    if len(toks) > 3:
        toks = toks[:3]
    return " ".join(toks)
