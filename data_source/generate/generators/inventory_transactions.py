"""Inventory transaction ledger: backflush, issues, receipts, adjustments,
returns and scrap, with the transaction-level defects.

Consumption is recorded three ways: BACKFLUSH from completed production orders
(quantity = BOM qty_per times completed quantity, so the consistency the audit
checks holds by construction), ISSUE from service-parts shipments, and manual
ISSUE for non-job pulls. Receipts follow the purchase orders. The defects:

  T1  unrecorded consumption   BOM-omitted usage escapes; chronic negative
                               adjustments follow the annual count
  T2  adjustments as catch-all a large share of quantity moved flows through
                               ADJUST with blank or generic reason codes
  T6  wrong references         manual issues posted to a similar item / wrong job
  T7  quantity and unit errors order-of-magnitude and box/each keying errors
  T8  duplicate postings       the same transaction posted twice within minutes
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .. import config as C


def build_inventory_transactions(production_orders, prod_map, item_master, item_meta,
                                 dup_map, service_orders, manual_demand, purchase_orders,
                                 omitted_backflush, plan, rng):
    class_by_item = plan.set_index("item_id")["item_class"].to_dict()
    fam_by_item = plan.set_index("item_id")["family"].to_dict()
    recorded = {}
    for num, meta in item_meta.items():
        if not meta["dead"]:
            recorded.setdefault(meta["item_id"], []).append(num)
    primary_num = {iid: (dup_map[iid]["primary"] if iid in dup_map else nums[0])
                   for iid, nums in recorded.items()}
    conv_by_num = {n: m["uom_conv"] for n, m in item_meta.items()}
    stock_uom_by_num = {n: m["stock_uom"] for n, m in item_meta.items()}

    rows = []
    seq = [0]
    truth = {t: [] for t in ["t1", "t2", "t6", "t7", "t8"]}

    def user_for(kind):
        if kind in ("floor", "recv"):
            return rng.choice(C.SHARED_LOGINS) if rng.random() < C.SHARED_LOGIN_SHARE \
                else rng.choice(C.OFFICE_USERS)
        return rng.choice(C.OFFICE_USERS)

    def add(item_number, tdate, ttype, qty, uom, job, reason, kind, defect=None):
        seq[0] += 1
        rows.append({
            "txn_id":     f"TX-{seq[0]:07d}",
            "item_number":item_number,
            "txn_date":   tdate.isoformat() if hasattr(tdate, "isoformat") else tdate,
            "txn_time":   f"{int(rng.integers(6,18)):02d}:{int(rng.integers(0,60)):02d}",
            "type":       ttype,
            "qty":        int(qty),
            "uom":        uom,
            "job_id":     job,
            "reason_code":reason,
            "location":   rng.choice(C.LOCATIONS),
            "user_id":    user_for(kind),
            "_defect":    defect,
        })

    # ── BACKFLUSH from completed production orders ──────────────────────────
    done = production_orders[production_orders["status"] == "COMPLETED"]
    for r in done.itertuples(index=False):
        comps = prod_map.get(r.product_number, {})
        cd = pd.to_datetime(r.completed_date).date()
        for c, qper in comps.items():
            if c not in primary_num:
                continue
            nums = recorded[c]
            weights = dup_map[c]["weights"] if c in dup_map else None
            num = nums[int(rng.choice(len(nums), p=weights))] if len(nums) > 1 else nums[0]
            add(num, cd, "BACKFLUSH", qper * int(r.qty_completed),
                stock_uom_by_num.get(num, "EA"), r.order_id, "BACKFLUSH", "floor")

    # ── ISSUE from service-parts shipments ──────────────────────────────────
    for r in service_orders.itertuples(index=False):
        add(r.item_number, pd.to_datetime(r.ship_date).date(), "ISSUE", r.qty,
            stock_uom_by_num.get(r.item_number, "EA"), r.order_id, "SERVICE", "floor")

    # ── Manual ISSUE from the manual channel, with T1 unrecorded + T6 wrong ─
    omitted_items = set(int(i) for i in omitted_backflush["item_id"].unique()) if len(omitted_backflush) else set()
    manual = manual_demand.copy()
    for r in manual.itertuples(index=False):
        iid, q = int(r.item_id), int(r.qty)
        if q <= 0 or iid not in recorded:
            continue
        # T1: on omitted / affected items, a share of manual usage is never recorded
        drop = 0
        if iid in omitted_items and rng.random() < 0.8:
            drop = int(round(q * C.T1_UNRECORDED_SHARE))
        q_rec = max(0, q - drop)
        n_lines = int(min(4, max(1, round(q_rec / 7))))
        for s in _split_int(q_rec, n_lines, rng):
            if s <= 0:
                continue
            day = r.month + timedelta(days=int(rng.integers(0, 27)))
            if day.weekday() >= 5:
                day += timedelta(days=7 - day.weekday())
            if day > C.END_DATE:
                continue
            nums = recorded[iid]
            weights = dup_map[iid]["weights"] if iid in dup_map else None
            num = nums[int(rng.choice(len(nums), p=weights))] if len(nums) > 1 else nums[0]
            # T6 wrong reference: post to a similar item (family / dup sibling / class)
            if rng.random() < C.T6_ISSUE_SHARE:
                wrong = _wrong_item(iid, num, dup_map, fam_by_item, class_by_item, recorded, primary_num, rng)
                if wrong:
                    truth["t6"].append({"recorded_item_number": wrong, "true_item_number": num})
                    add(wrong, day, "ISSUE", s, stock_uom_by_num.get(wrong, "EA"),
                        f"WO-{int(rng.integers(100000,999999))}", "MANUAL", "floor", "T6")
                    continue
            add(num, day, "ISSUE", s, stock_uom_by_num.get(num, "EA"),
                f"WO-{int(rng.integers(100000,999999))}", "MANUAL", "floor")

    # ── T1 chronic negative adjustments on omitted items ────────────────────
    for r in omitted_backflush.groupby("item_id", as_index=False)["qty"].sum().itertuples(index=False):
        iid = int(r.item_id)
        if iid not in primary_num:
            continue
        num = primary_num[iid]
        truth["t1"].append({"item_number": num, "item_id": iid, "annual_unrecorded": int(r.qty)})
        for m in C.month_starts():
            if rng.random() < C.T1_MONTHLY_ADJ_PROB:
                mag = max(1, int(abs(rng.normal(r.qty / 24.0, r.qty / 40.0 + 1))))
                reason = rng.choice(C.GENERIC_REASON_CODES) if rng.random() < C.T2_BLANK_REASON_SHARE \
                    else rng.choice(C.SPECIFIC_REASON_CODES)
                add(num, m + timedelta(days=int(rng.integers(20, 27))), "ADJUST", -mag,
                    stock_uom_by_num.get(num, "EA"), None, reason, "floor")

    # ── RECEIPT from purchase orders (booked in the purchase UOM) ───────────
    po = purchase_orders[purchase_orders["received_date"].notna() &
                         (purchase_orders["item_number"].isin(item_master["item_number"]))]
    for r in po.itertuples(index=False):
        conv = conv_by_num.get(r.item_number)
        uom = "BOX" if (conv and conv > 1) else stock_uom_by_num.get(r.item_number, "EA")
        add(r.item_number, pd.to_datetime(r.received_date).date(), "RECEIPT", r.qty_received,
            uom, r.po_id, "PO", "recv")

    tx = pd.DataFrame(rows)

    # ── T2 adjustments as catch-all: a bounded number of events whose total
    # quantity reaches the target share of quantity moved. ──────────────────
    qty_moved = tx.loc[tx["type"].isin(["ISSUE", "BACKFLUSH", "RECEIPT"]), "qty"].abs().sum()
    cur_adj = tx.loc[tx["type"] == "ADJUST", "qty"].abs().sum()
    target_adj = C.T2_ADJ_SHARE_OF_QTY * qty_moved
    remaining = max(0.0, target_adj - cur_adj)
    live_nums = [n for n, m in item_meta.items() if not m["dead"]]
    months = C.month_starts()
    n_events = min(len(live_nums) * 8, 20000)
    extra = []
    if remaining > 0 and n_events > 0 and live_nums:
        weights = rng.dirichlet(np.full(n_events, 0.6))    # skewed: a few big write-offs
        for w in weights:
            num = rng.choice(live_nums)
            mag = max(1, int(round(w * remaining)))
            seq[0] += 1
            m = months[int(rng.integers(0, len(months)))]
            reason = rng.choice(C.GENERIC_REASON_CODES) if rng.random() < C.T2_BLANK_REASON_SHARE \
                else rng.choice(C.SPECIFIC_REASON_CODES)
            sign = -1 if rng.random() < 0.7 else 1
            tid = f"TX-{seq[0]:07d}"
            extra.append({"txn_id": tid, "item_number": num,
                          "txn_date": (m + timedelta(days=int(rng.integers(0, 27)))).isoformat(),
                          "txn_time": f"{int(rng.integers(6,18)):02d}:{int(rng.integers(0,60)):02d}",
                          "type": "ADJUST", "qty": sign * mag, "uom": stock_uom_by_num.get(num, "EA"),
                          "job_id": None, "reason_code": reason, "location": rng.choice(C.LOCATIONS),
                          "user_id": rng.choice(C.SHARED_LOGINS), "_defect": "T2"})
            truth["t2"].append({"txn_id": tid, "item_number": num})
    if extra:
        tx = pd.concat([tx, pd.DataFrame(extra)], ignore_index=True)

    # ── T7 quantity and unit errors (weight to M5/UOM items) ────────────────
    cand = tx.index[tx["type"].isin(["ISSUE", "RECEIPT"]) & tx["_defect"].isna()]
    w = np.array([C.T7_M5_WEIGHT if (conv_by_num.get(tx.at[i, "item_number"]) or 1) > 1 else 1.0
                  for i in cand])
    n7 = int(len(tx) * C.T7_TXN_SHARE)
    if len(cand):
        for i in rng.choice(cand, size=min(n7, len(cand)), replace=False, p=w / w.sum()):
            q = tx.at[i, "qty"]
            if q <= 0:
                continue
            conv = conv_by_num.get(tx.at[i, "item_number"])
            factor = int(conv) if (conv and conv > 1 and rng.random() < 0.5) else 10
            truth["t7"].append({"txn_id": tx.at[i, "txn_id"], "true_qty": int(q),
                               "recorded_qty": int(q * factor)})
            tx.at[i, "qty"] = int(q * factor)
            tx.at[i, "_defect"] = "T7"

    # ── T8 duplicate postings ───────────────────────────────────────────────
    cand = tx.index[tx["type"].isin(["ISSUE", "RECEIPT", "BACKFLUSH"])]
    n8 = int(len(tx) * C.T8_DUP_SHARE)
    dups = []
    for i in rng.choice(cand, size=min(n8, len(cand)), replace=False):
        row = tx.loc[i].to_dict()
        seq[0] += 1
        row["txn_id"] = f"TX-{seq[0]:07d}"
        row["_defect"] = "T8"
        truth["t8"].append({"txn_id": row["txn_id"], "original_txn_id": tx.at[i, "txn_id"]})
        dups.append(row)
    if dups:
        tx = pd.concat([tx, pd.DataFrame(dups)], ignore_index=True)

    tx = tx.drop(columns=["_defect"]).reset_index(drop=True)
    return tx, truth


def _wrong_item(iid, num, dup_map, fam_by_item, class_by_item, recorded, primary_num, rng):
    sibs = [m for m in dup_map[iid]["records"] if m != num] if iid in dup_map else []
    if sibs:
        return rng.choice(sibs)
    fam = fam_by_item.get(iid)
    cls = class_by_item.get(iid)
    pool = [primary_num[j] for j in recorded
            if j != iid and (fam and fam_by_item.get(j) == fam or class_by_item.get(j) == cls)]
    return rng.choice(pool) if pool else None


def _split_int(total, parts, rng):
    if total <= 0:
        return [0]
    if parts <= 1:
        return [total]
    props = rng.dirichlet(np.full(parts, 2.0))
    raw = np.floor(props * total).astype(int)
    raw[0] += total - int(raw.sum())
    return raw.tolist()
