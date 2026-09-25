"""Inventory transaction ledger, assembled from the consumption events, the
purchase orders and the replay's adjustment postings, with the transaction-level
errors the ERP would carry.

  BACKFLUSH  recorded backflush per completed job (a delayed job posts late)
  ISSUE      service shipments and manual pulls the ERP saw
  RECEIPT    purchase-order receipts on the recorded (batched) posting date,
             booked in the purchase unit
  ADJUST     the annual count corrections and the chronic write-offs posted in
             the replay, the cycle-count corrections of the remediation program,
             and the catch-all adjustments (T2)

Errors applied here: T2 adjustments as a catch-all, T6 wrong references,
T7 quantity and unit errors, T8 duplicate postings. The pulls the ERP never saw
are not here by definition; they are logged in the truth (T1).
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .. import config as C


def build_ledger(events, purchase_orders, sim_adjustments, job_delays,
                 item_master, item_meta, dup_map, plan, omitted_item_ids, rng):
    class_by_item = plan.set_index("item_id")["item_class"].to_dict()
    fam_by_item = plan.set_index("item_id")["family"].to_dict()
    recorded = {}
    for num, meta in item_meta.items():
        if not meta["dead"]:
            recorded.setdefault(meta["item_id"], []).append(num)
    primary_num = {iid: (dup_map[iid]["primary"] if iid in dup_map else nums[0]) for iid, nums in recorded.items()}
    conv_by_num = {n: (m["uom_conv"] or 1) for n, m in item_meta.items()}
    uom_by_num = {n: m["stock_uom"] for n, m in item_meta.items()}
    puom_by_num = {n: m["purchase_uom"] for n, m in item_meta.items()}
    cls_by_num = {n: m["cls"] for n, m in item_meta.items()}

    rows, seq = [], [0]
    truth = {t: [] for t in ["t1", "t1_events", "t2", "t6", "t7", "t8"]}

    def user_for(kind, tdate):
        if kind in ("floor", "recv"):
            if tdate > C.LOGIN_DATE.isoformat():        # individual logins issued
                return rng.choice(C.FLOOR_USERS)
            return rng.choice(C.SHARED_LOGINS) if rng.random() < C.SHARED_LOGIN_SHARE else rng.choice(C.OFFICE_USERS)
        return rng.choice(C.OFFICE_USERS)

    def reason_for(tdate):
        # a reason code is required from the control date; before it, most are generic
        if tdate > C.REASON_CODE_DATE.isoformat():
            return rng.choice(C.SPECIFIC_REASON_CODES)
        return rng.choice(C.GENERIC_REASON_CODES) if rng.random() < C.T2_BLANK_REASON_SHARE \
            else rng.choice(C.SPECIFIC_REASON_CODES)

    def add(item_number, tdate, ttype, qty, uom, job, reason, kind, defect=None):
        seq[0] += 1
        tdate = tdate if isinstance(tdate, str) else tdate.isoformat()
        rows.append({"txn_id": f"TX-{seq[0]:07d}", "item_number": item_number,
                     "txn_date": tdate,
                     "txn_time": f"{int(rng.integers(6, 18)):02d}:{int(rng.integers(0, 60)):02d}",
                     "type": ttype, "qty": int(qty), "uom": uom, "job_id": job, "reason_code": reason,
                     "location": rng.choice(C.LOCATIONS), "user_id": user_for(kind, tdate), "_defect": defect})
        return rows[-1]["txn_id"]

    # ── consumption the ERP recorded ─────────────────────────────────────────
    # from the forward start the duplicate records are retired, so every posting
    # lands on the surviving number
    events = events.copy()
    after = pd.to_datetime(events["date"]).dt.date >= C.FORWARD_START
    events.loc[after, "item_number"] = events.loc[after, "item_id"].map(lambda i: primary_num.get(int(i)))
    events = events.dropna(subset=["item_number"])
    for r in events.itertuples(index=False):
        if not r.recorded:
            truth["t1_events"].append({"item_number": r.item_number, "item_id": int(r.item_id),
                                       "date": r.date.isoformat(), "qty": int(r.qty), "channel": r.channel})
            continue
        d = r.date
        if r.channel == "BACKFLUSH":
            dly = job_delays.get(r.job_id, 0)
            if dly:
                d = min(d + timedelta(days=int(dly)), C.END_DATE)
            add(r.item_number, d, "BACKFLUSH", r.qty, uom_by_num.get(r.item_number, "EA"), r.job_id, "BACKFLUSH", "floor")
        elif r.channel == "SERVICE":
            add(r.item_number, d, "ISSUE", r.qty, uom_by_num.get(r.item_number, "EA"), r.job_id, "SERVICE", "floor")
        else:
            # T6: a manual pull posted against a similar item
            # individual logins and the negative-balance block halve the mispostings
            if rng.random() < C.T6_ISSUE_SHARE * (0.5 if d.isoformat() > C.LOGIN_DATE.isoformat() else 1.0):
                wrong = _wrong_item(int(r.item_id), r.item_number, dup_map, fam_by_item, class_by_item, recorded, primary_num, rng)
                if wrong:
                    tid = add(wrong, d, "ISSUE", r.qty, uom_by_num.get(wrong, "EA"), r.job_id, "MANUAL", "floor", "T6")
                    truth["t6"].append({"txn_id": tid, "recorded_item_number": wrong, "true_item_number": r.item_number})
                    continue
            add(r.item_number, d, "ISSUE", r.qty, uom_by_num.get(r.item_number, "EA"), r.job_id, "MANUAL", "floor")

    # T1 summary per item: the usage that never got recorded
    if truth["t1_events"]:
        agg = {}
        for e in truth["t1_events"]:
            key = (e["item_number"], e["item_id"])
            agg[key] = agg.get(key, 0) + e["qty"]
        for (num, iid), q in agg.items():
            truth["t1"].append({"item_number": primary_num.get(iid, num), "item_id": iid, "annual_unrecorded": int(q)})

    # ── receipts on the recorded posting date, in the purchase unit ─────────
    # a receipt on a generic item code (T3) posts to expense, never to a balance
    po = purchase_orders[purchase_orders["received_date"].notna() & (purchase_orders["qty_received"] > 0)
                         & ~purchase_orders["item_number"].isin(C.GENERIC_ITEM_CODES)]
    for r in po.itertuples(index=False):
        conv = conv_by_num.get(r.item_number, 1)
        uom = puom_by_num.get(r.item_number, "EA") if (conv and conv > 1) else uom_by_num.get(r.item_number, "EA")
        add(r.item_number, str(r.received_date), "RECEIPT", r.qty_received, uom, r.po_id, "PO", "recv")

    # ── the floor's own corrections, posted in the replay ──────────────────
    for r in sim_adjustments.itertuples(index=False):
        add(r.item_number, str(r.date), "ADJUST", r.qty, uom_by_num.get(r.item_number, "EA"), None,
            reason_for(str(r.date)), "floor")

    tx = pd.DataFrame(rows)

    # ── T2 adjustments as a catch-all, to the target share of quantity moved ─
    moved = tx.loc[tx["type"].isin(["ISSUE", "BACKFLUSH", "RECEIPT"]), "qty"].abs().sum()
    cur = tx.loc[tx["type"] == "ADJUST", "qty"].abs().sum()
    remaining = max(0.0, C.T2_ADJ_SHARE_OF_QTY * moved - cur)
    live_nums = [n for n, m in item_meta.items() if not m["dead"]]
    cheap = ("Hardware", "Fasteners", "Fittings", "Consumables")
    omit_pool = [primary_num[int(i)] for i in set(int(x) for x in omitted_item_ids) if int(i) in primary_num]
    other_pool = [n for n in live_nums if cls_by_num.get(n) in cheap and n not in omit_pool]
    rng.shuffle(other_pool); other_pool = other_pool[:60]
    vol = (tx[tx["type"].isin(["ISSUE", "BACKFLUSH"])].assign(q=lambda d: d["qty"].abs())
           .groupby("item_number")["q"].sum() / 36.0)
    # catch-all adjustments end with the remediation: a reason code is required from then on
    months = [m for m in C.month_starts() if m <= C.REMEDIATION_END]
    produced, events_n, extra = 0.0, 0, []
    avg_mag = remaining / max(1, len(live_nums) * 6)
    while produced < remaining and events_n < 30000 and live_nums:
        events_n += 1
        neg = rng.random() < 0.62
        # write-offs concentrate on the cheap items the floor pulls by hand, the
        # omitted components among them; the rest scatter across the master
        u = rng.random()
        if neg and u < 0.10 and omit_pool:
            num = rng.choice(omit_pool)
        elif neg and u < 0.80 and other_pool:
            num = rng.choice(other_pool)
        else:
            num = rng.choice(live_nums)
        cap = max(2, int(3 * vol.get(num, 1.0)))
        mag = max(1, min(cap, int(rng.exponential(max(avg_mag, 1.0))) + 1))
        produced += mag
        seq[0] += 1
        m = months[int(rng.integers(0, len(months)))]
        tdate = (m + timedelta(days=int(rng.integers(0, 27)))).isoformat()
        reason = reason_for(tdate)
        tid = f"TX-{seq[0]:07d}"
        extra.append({"txn_id": tid, "item_number": num,
                      "txn_date": tdate,
                      "txn_time": f"{int(rng.integers(6, 18)):02d}:{int(rng.integers(0, 60)):02d}",
                      "type": "ADJUST", "qty": (-mag if neg else mag), "uom": uom_by_num.get(num, "EA"),
                      "job_id": None, "reason_code": reason, "location": rng.choice(C.LOCATIONS),
                      "user_id": user_for("floor", tdate), "_defect": "T2"})
        truth["t2"].append({"txn_id": tid, "item_number": num})
    if extra:
        tx = pd.concat([tx, pd.DataFrame(extra)], ignore_index=True)

    # ── T7 quantity and unit errors (an extra zero on a small quantity, or a
    #    box keyed as eaches), weighted to the UOM-mismatch items ─────────────
    cand = tx.index[tx["type"].isin(["ISSUE", "RECEIPT"]) & tx["_defect"].isna() & (tx["qty"] > 0)]
    w = np.array([C.T7_M5_WEIGHT if (conv_by_num.get(tx.at[i, "item_number"]) or 1) > 1 else 1.0 for i in cand])
    n7 = int(len(tx) * C.T7_TXN_SHARE)
    if len(cand):
        for i in rng.choice(cand, size=min(n7, len(cand)), replace=False, p=w / w.sum()):
            if tx.at[i, "txn_date"] > C.LOGIN_DATE.isoformat() and rng.random() < 0.5:
                continue
            q = int(tx.at[i, "qty"])
            conv = conv_by_num.get(tx.at[i, "item_number"])
            if conv and conv > 1 and rng.random() < 0.5:
                factor = int(conv)
            elif q <= 9:
                factor = 10
            else:
                continue
            truth["t7"].append({"txn_id": tx.at[i, "txn_id"], "true_qty": q, "recorded_qty": q * factor})
            tx.at[i, "qty"] = q * factor
            tx.at[i, "_defect"] = "T7"

    # ── T8 duplicate postings ───────────────────────────────────────────────
    cand = tx.index[tx["type"].isin(["ISSUE", "RECEIPT", "BACKFLUSH"])]
    dups = []
    for i in rng.choice(cand, size=min(int(len(tx) * C.T8_DUP_SHARE), len(cand)), replace=False):
        if tx.at[i, "txn_date"] > C.LOGIN_DATE.isoformat() and rng.random() < 0.5:
            continue
        row = tx.loc[i].to_dict()
        seq[0] += 1
        row["txn_id"] = f"TX-{seq[0]:07d}"; row["_defect"] = "T8"
        truth["t8"].append({"txn_id": row["txn_id"], "original_txn_id": tx.at[i, "txn_id"]})
        dups.append(row)
    if dups:
        tx = pd.concat([tx, pd.DataFrame(dups)], ignore_index=True)

    tx = tx.drop(columns=["_defect"]).sort_values(["txn_date", "txn_id"]).reset_index(drop=True)
    return tx, truth


def post_count_adjustments(tx, count_adjustments, item_meta, rng):
    """Append the count program's balance corrections (annual COUNT, remediation CYCLE)."""
    if count_adjustments.empty:
        return tx
    uom_by_num = {n: m["stock_uom"] for n, m in item_meta.items()}
    seq = int(tx["txn_id"].str[3:].astype(int).max())
    rows = []
    for r in count_adjustments.itertuples(index=False):
        seq += 1
        rows.append({"txn_id": f"TX-{seq:07d}", "item_number": r.item_number, "txn_date": str(r.date),
                     "txn_time": f"{int(rng.integers(8, 17)):02d}:{int(rng.integers(0, 60)):02d}",
                     "type": "ADJUST", "qty": int(r.qty), "uom": uom_by_num.get(r.item_number, "EA"),
                     "job_id": None, "reason_code": r.reason, "location": rng.choice(C.LOCATIONS),
                     "user_id": rng.choice(C.OFFICE_USERS)})
    out = pd.concat([tx, pd.DataFrame(rows)], ignore_index=True)
    return out.sort_values(["txn_date", "txn_id"]).reset_index(drop=True)


def _wrong_item(iid, num, dup_map, fam_by_item, class_by_item, recorded, primary_num, rng):
    sibs = [m for m in dup_map[iid]["records"] if m != num] if iid in dup_map else []
    if sibs:
        return rng.choice(sibs)
    fam, cls = fam_by_item.get(iid), class_by_item.get(iid)
    pool = [primary_num[j] for j in recorded
            if j != iid and ((fam and fam_by_item.get(j) == fam) or class_by_item.get(j) == cls)]
    return rng.choice(pool) if pool else None
