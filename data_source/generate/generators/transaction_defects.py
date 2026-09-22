"""Transaction-level defects T1-T7 (addendum).

These are injected as a post-processing pass over the clean purchase orders and
inventory ledger, so the master-level generators (D1-D6) are untouched. Every
injected defect is recorded in a truth dictionary keyed by transaction or PO id,
so detection precision and recall can be scored and the corrected consumption
mart can be rebuilt.

  T1 free-text / non-stock lines   consumption diverted to generic codes
  T2 quantity keying errors        x10 and box-as-each, weighted to D3 items
  T3 issues posted to wrong item   within a material family or duplicate cluster
  T4 unrecorded consumption        usage escaping as chronic negative adjustments
  T5 receipt-date batching         postings clustered on Mondays / month-end
  T6 purchase orders never closed  phantom on-order quantity
  T7 duplicate postings            the same line posted twice
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .. import config as C

_ABBREV = {"round": "rd", "square": "sq", "flat": "fl", "carbide": "carb",
           "sheet": "sht", "plate": "plt", "coolant": "cool", "gloves": "glv"}
_ONEOFF = ["shop supplies", "misc hardware", "freight charge", "tooling repair",
           "calibration service", "safety supplies", "packaging lot", "rental item",
           "consumable assortment", "office supply"]


def _freetext(desc: str, rng) -> str:
    s = str(desc).lower()
    for k, v in _ABBREV.items():
        if rng.random() < 0.7:
            s = s.replace(k, v)
    s = " ".join(s.split())
    if rng.random() < 0.20 and len(s) > 6:          # occasional dropped character
        i = int(rng.integers(1, len(s) - 1))
        s = s[:i] + s[i + 1:]
    return s.strip()


def apply(item_master, plan, item_meta, dup_map, po, tx, rng):
    tx = tx.copy().reset_index(drop=True)
    po = po.copy().reset_index(drop=True)
    for col in ("description", "entered_code"):
        tx[col] = None
        po[col] = None
    tx["unit_price"] = np.nan
    tx["defect"] = None
    po["po_status"] = "CLOSED"

    desc_by_num = item_master.set_index("item_number")["description"].to_dict()
    cost_by_num = item_master.set_index("item_number")["standard_cost"].to_dict()
    canon_by_num = {n: m["item_id"] for n, m in item_meta.items()}
    fam_by_canon = plan.set_index("item_id")["family"].to_dict()
    class_by_num = item_master.set_index("item_number")["item_class"].to_dict()
    d3_nums = {n for n, m in item_meta.items() if m.get("box_size")}
    box_by_num = {n: m["box_size"] for n, m in item_meta.items() if m.get("box_size")}

    truth = {t: [] for t in ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]}
    seq = [int(tx["transaction_id"].str[3:].astype(int).max()) + 1]

    def new_tx_id():
        i = seq[0]; seq[0] += 1
        return f"TX-9{i:06d}"

    issue_vol = tx[tx["type"] == "issue"].groupby("item_number")["quantity"].sum()

    # ── T1 free-text / non-stock ─────────────────────────────────────────────
    # Concentrate on higher-consumption items, including some of the largest, so
    # recovering their hidden demand is a material forecast improvement.
    top = issue_vol.sort_values(ascending=False).head(int(C.T1_AFFECTED_ITEMS * 0.6)).index.tolist()
    others = [n for n in issue_vol.index if n not in top]
    rng.shuffle(others)
    affected = set(top + others[:C.T1_AFFECTED_ITEMS - len(top)])

    new_rows = []
    diverted = 0
    for num in affected:
        idx = tx.index[(tx["item_number"] == num) & (tx["type"] == "issue")]
        k = int(round(len(idx) * C.T1_DIVERTED_SHARE))
        if k == 0:
            continue
        pick = rng.choice(idx, size=k, replace=False)
        for i in pick:
            code = rng.choice(C.GENERIC_ITEM_CODES)
            truth["t1"].append({"transaction_id": tx.at[i, "transaction_id"],
                                 "true_item_number": num, "is_oneoff": False})
            tx.at[i, "entered_code"] = code
            tx.at[i, "description"] = _freetext(desc_by_num.get(num, ""), rng)
            tx.at[i, "unit_price"] = round(float(cost_by_num.get(num, 1) or 1) * rng.uniform(0.9, 1.2), 2)
            tx.at[i, "item_number"] = code
            tx.at[i, "defect"] = "T1"
        diverted += k
    # genuine one-off free-text lines (true negatives)
    for _ in range(int(diverted * C.T1_ONEOFF_RATIO)):
        code = rng.choice(C.GENERIC_ITEM_CODES)
        tid = new_tx_id()
        truth["t1"].append({"transaction_id": tid, "true_item_number": None, "is_oneoff": True})
        d = rng.integers(0, (C.END_DATE - C.START_DATE).days)
        new_rows.append({"transaction_id": tid, "item_number": code,
                         "transaction_date": (C.START_DATE + timedelta(days=int(d))).isoformat(),
                         "type": "issue", "quantity": int(rng.integers(1, 40)), "uom": "EA",
                         "work_order_id": f"WO-{rng.integers(100000,999999)}", "location": rng.choice(C.LOCATIONS),
                         "description": rng.choice(_ONEOFF), "entered_code": code,
                         "unit_price": round(rng.uniform(5, 400), 2), "defect": "T1"})

    # ── T4 unrecorded consumption -> negative adjustments ────────────────────
    hw_pool = [n for n in issue_vol.index if class_by_num.get(n) in ("Hardware", "Consumables", "Fasteners")]
    rng.shuffle(hw_pool)
    t4_items = hw_pool[:C.T4_ITEMS]
    for num in t4_items:
        truth["t4"].append({"item_number": num})
        idx = tx.index[(tx["item_number"] == num) & (tx["type"] == "issue")]
        k = int(round(len(idx) * C.T4_UNRECORDED_SHARE))
        if k:
            drop = rng.choice(idx, size=k, replace=False)     # usage that never gets issued
            tx = tx.drop(index=drop)
        # chronic negative write-off adjustments at counts
        for m in C.month_starts():
            if rng.random() < C.T4_MONTHLY_ADJ_PROB:
                tid = new_tx_id()
                new_rows.append({"transaction_id": tid, "item_number": num,
                                 "transaction_date": (m + timedelta(days=int(rng.integers(20, 27)))).isoformat(),
                                 "type": "adjustment", "quantity": -int(rng.integers(2, 12)), "uom": "EA",
                                 "work_order_id": None, "location": rng.choice(C.LOCATIONS),
                                 "description": rng.choice(["", "count variance", "shrink", "adj"]),
                                 "entered_code": None, "unit_price": np.nan, "defect": "T4"})

    if new_rows:
        tx = pd.concat([tx, pd.DataFrame(new_rows)], ignore_index=True)

    # ── T2 quantity keying errors (weight to D3) ─────────────────────────────
    cand = tx.index[tx["type"].isin(["issue", "receipt"]) & (~tx["defect"].notna())]
    w = np.array([C.T2_D3_WEIGHT if tx.at[i, "item_number"] in d3_nums else 1.0 for i in cand])
    n2 = int(len(tx) * C.T2_TXN_SHARE)
    pick = rng.choice(cand, size=min(n2, len(cand)), replace=False, p=w / w.sum())
    for i in pick:
        q = tx.at[i, "quantity"]
        if q <= 0:
            continue
        if tx.at[i, "item_number"] in box_by_num and rng.random() < 0.5:
            factor = box_by_num[tx.at[i, "item_number"]]; kind = "uom"
        else:
            factor = 10; kind = "magnitude"
        truth["t2"].append({"transaction_id": tx.at[i, "transaction_id"], "true_quantity": int(q),
                            "recorded_quantity": int(q * factor), "kind": kind})
        tx.at[i, "quantity"] = int(q * factor)
        tx.at[i, "defect"] = "T2"

    # ── T3 issues posted to the wrong item (same family or dup cluster) ───────
    fam_members = {}
    for n, iid in canon_by_num.items():
        fam = fam_by_canon.get(iid)
        if fam:
            fam_members.setdefault(fam, []).append(n)
    class_members = {}
    for n in canon_by_num:
        class_members.setdefault(class_by_num.get(n), []).append(n)
    dup_siblings = {}
    for c in dup_map.values():
        for n in c["records"]:
            dup_siblings[n] = [m for m in c["records"] if m != n]
    cand = tx.index[(tx["type"] == "issue") & (~tx["defect"].notna())]
    n3 = int((tx["type"] == "issue").sum() * C.T3_ISSUE_SHARE)
    pick = rng.choice(cand, size=min(n3, len(cand)), replace=False)
    for i in pick:
        num = tx.at[i, "item_number"]
        iid = canon_by_num.get(num)
        fam = fam_by_canon.get(iid)
        pool = [m for m in fam_members.get(fam, []) if m != num] if fam else []
        pool += dup_siblings.get(num, [])
        if not pool:                                   # fall back to a same-class neighbour
            pool = [m for m in class_members.get(class_by_num.get(num), []) if m != num]
        if not pool:
            continue
        wrong = rng.choice(pool)
        truth["t3"].append({"transaction_id": tx.at[i, "transaction_id"],
                            "recorded_item_number": wrong, "true_item_number": num})
        tx.at[i, "item_number"] = wrong
        tx.at[i, "defect"] = "T3"

    # ── T7 duplicate postings ────────────────────────────────────────────────
    cand = tx.index[tx["type"].isin(["issue", "receipt"])]
    n7 = int(len(tx) * C.T7_DUP_SHARE)
    dups = []
    for i in rng.choice(cand, size=min(n7, len(cand)), replace=False):
        row = tx.loc[i].to_dict()
        tid = new_tx_id()
        row["transaction_id"] = tid
        row["defect"] = "T7"
        d = pd.to_datetime(row["transaction_date"]) + timedelta(days=int(rng.choice([0, 0, 1, 3])))
        row["transaction_date"] = d.date().isoformat()
        truth["t7"].append({"transaction_id": tid, "original_transaction_id": tx.at[i, "transaction_id"]})
        dups.append(row)
    if dups:
        tx = pd.concat([tx, pd.DataFrame(dups)], ignore_index=True)

    # ── T5 receipt-date batching ─────────────────────────────────────────────
    rec = po.index[po["received_date"].notna()]
    n5 = int(len(rec) * C.T5_BATCH_SHARE)
    for i in rng.choice(rec, size=min(n5, len(rec)), replace=False):
        rd = pd.to_datetime(po.at[i, "received_date"])
        # snap to the next Monday within the window, else displace 1-4 days late
        nd = None
        for extra in range(1, C.T5_MAX_DISPLACEMENT + 1):
            if (rd + timedelta(days=extra)).weekday() == 0:
                nd = rd + timedelta(days=extra); break
        if nd is None:
            nd = rd + timedelta(days=int(rng.integers(1, C.T5_MAX_DISPLACEMENT + 1)))
        truth["t5"].append({"po_id": po.at[i, "po_id"], "true_received_date": rd.date().isoformat(),
                            "recorded_received_date": nd.date().isoformat()})
        po.at[i, "received_date"] = nd.date().isoformat()

    # ── T6 purchase orders never closed ──────────────────────────────────────
    old = po.index[(pd.to_datetime(po["order_date"]) < pd.Timestamp(C.END_DATE) - pd.Timedelta(days=90))]
    n6 = int(len(old) * C.T6_OPEN_SHARE)
    for i in rng.choice(old, size=min(n6, len(old)), replace=False):
        q = po.at[i, "quantity_ordered"]
        po.at[i, "quantity_received"] = int(q * rng.uniform(0.3, 0.7))
        po.at[i, "received_date"] = None
        po.at[i, "po_status"] = "OPEN"
        truth["t6"].append({"po_id": po.at[i, "po_id"], "quantity_ordered": int(q),
                            "quantity_received": int(po.at[i, "quantity_received"])})

    tx = tx.drop(columns=["defect"]).reset_index(drop=True)
    return po, tx, truth
