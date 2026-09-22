"""Purchase orders: multi-line replenishment orders grouped by supplier, with the
purchasing-side defects.

Replenishment buy events accumulate as consumption is drawn down, then group into
orders by supplier and order week: distributor and hardware suppliers carry many
lines per order, motors and drives one to a few. The defects planted here:

  M2  stale lead times   recorded master lead stays; actual receipt lead drifts
                         up generally and sharply for one supplier
  M5  UOM mismatch       box/spool/length buys booked in the purchase UOM
  M6  supplier fragments  some orders booked to an alias supplier id
  T3  free-text lines    generic codes with typed descriptions (some stocked)
  T4  batched receipts   received dates snapped to Mondays / month-end
  T5  open documents     partial receipts left open past 90 days
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .. import config as C

_ABBREV = {"washer": "wshr", "bearing": "brg", "fitting": "fitg", "sensor": "sens",
           "bracket": "brkt", "coupling": "cplg", "enclosure": "encl"}
_ONEOFF = ["shop supplies", "freight charge", "crating", "rush machining",
           "custom weldment", "prototype part", "field service kit", "rental",
           "calibration service", "special order casting"]


def _actual_lead(meta, master_lead, order_date, drift_rate, drift_supplier_id, rng):
    if meta["is_sharp"] and meta["supplier_id"] == drift_supplier_id:
        ramp_start = C.MODEL_SPAN_END - timedelta(days=C.M2_DRIFT_SUPPLIER_MONTHS * 30)
        end_ratio = rng.uniform(*C.M2_DRIFT_SUPPLIER_RATIO)
        target = master_lead * end_ratio
        if order_date <= ramp_start:
            base = C.M2_DRIFT_SUPPLIER_START
        else:
            frac = min(1.0, (order_date - ramp_start).days / (C.M2_DRIFT_SUPPLIER_MONTHS * 30))
            base = C.M2_DRIFT_SUPPLIER_START + frac * (target - C.M2_DRIFT_SUPPLIER_START)
        return max(2, int(round(base + rng.normal(0, 2.0))))
    # Every item's actual lead drifts upward over the history while the recorded
    # master value stays put; flagged items (M2) drift harder.
    frac = (order_date - C.START_DATE).days / max(1, (C.MODEL_SPAN_END - C.START_DATE).days)
    return max(2, int(round(master_lead + drift_rate * frac + rng.normal(0, 1.5))))


def build_purchase_orders(total_consumption, item_master, item_meta, dup_map,
                          suppliers, sup_frag, drift_supplier_id, plan, rng):
    monthly = total_consumption.pivot(index="item_id", columns="month", values="qty").fillna(0)
    monthly = monthly.sort_index(axis=1)
    months = list(monthly.columns)
    cost_by_item = plan.set_index("item_id")["unit_cost"].to_dict()
    class_by_item = plan.set_index("item_id")["item_class"].to_dict()
    lead_by_num = item_master.set_index("item_number")["master_lead_time_days"].to_dict()

    recorded = {}
    for num, meta in item_meta.items():
        if not meta["dead"]:
            recorded.setdefault(meta["item_id"], []).append(num)

    # Per-item end-of-history lead-time drift (days): a general upward creep for
    # every item, plus a harder drift for the flagged M2 items.
    drift_rate = {}
    for num, meta in item_meta.items():
        if meta["dead"]:
            continue
        drift_rate[num] = rng.uniform(1, 8) + (rng.uniform(12, 26) if meta["is_drift"] else 0.0)

    # ── Phase 1: buy events per item ────────────────────────────────────────
    buys = []   # (order_date, item_number, supplier_id, qty_each, item_id)
    for iid, series in monthly.iterrows():
        if iid not in recorded:
            continue
        vals = series.to_numpy()
        avg = vals[-12:].mean()
        if avg <= 0:
            continue
        batch = max(1, int(round(avg * 2.0)))       # ~two months cover
        nums = recorded[iid]
        weights = dup_map[iid]["weights"] if iid in dup_map else None
        acc = 0.0
        for mi, q in enumerate(vals):
            acc += q
            while acc >= batch:
                acc -= batch
                order_date = months[mi] + timedelta(days=int(rng.integers(1, 26)))
                if order_date > C.END_DATE:
                    break
                num = nums[int(rng.choice(len(nums), p=weights))] if len(nums) > 1 else nums[0]
                sup = item_meta[num]["supplier_id"]
                if sup in sup_frag["alias_ids"] and rng.random() < 0.4:
                    aliases = sup_frag["alias_ids"][sup]
                    if aliases:
                        sup = rng.choice(aliases)
                buys.append((order_date, num, sup, batch, iid))

    # ── Phase 2: group buys into multi-line POs by supplier and order week ──
    buys.sort(key=lambda b: (b[2], b[0]))
    rows, truth = [], {"t3": [], "t4": [], "t5": []}
    po_seq = 0
    i = 0
    stock_nums = set(item_master["item_number"])
    desc_by_num = item_master.set_index("item_number")["description"].to_dict()
    while i < len(buys):
        order_date, _, sup, _, _ = buys[i]
        stype = suppliers.loc[suppliers["supplier_id"] == sup, "supplier_type"]
        stype = stype.iloc[0] if len(stype) else "Distributor"
        many = stype in ("Fasteners", "Distributor", "Material")
        max_lines = int(rng.integers(10, 21)) if many else int(rng.integers(1, 4))
        po_seq += 1
        po_id = f"PO-{po_seq:06d}"
        line_no = 0
        # gather buys for the same supplier within a 5-day window
        window_end = order_date + timedelta(days=5)
        while i < len(buys) and buys[i][2] == sup and buys[i][0] <= window_end and line_no < max_lines:
            od, num, s, qty_each, iid = buys[i]; i += 1
            line_no += 1
            meta = item_meta[num]
            master_lead = int(lead_by_num.get(num, 21))
            alead = _actual_lead(meta, master_lead, od, drift_rate.get(num, 6.0), drift_supplier_id, rng)
            promised = od + timedelta(days=master_lead)
            received = od + timedelta(days=alead)
            conv = meta["uom_conv"]
            if conv and conv > 1:                    # M5: booked in purchase UOM
                qord = max(1, int(round(qty_each / conv)))
                price = round(cost_by_item[iid] * conv * rng.uniform(0.95, 1.15), 2)
            else:
                qord = qty_each
                price = round(cost_by_item[iid] * rng.uniform(0.9, 1.2), 2)
            recv_ok = received <= C.END_DATE
            qrec = qord if rng.random() < 0.93 else int(qord * rng.uniform(0.6, 1.0))
            rows.append({
                "po_id": po_id, "line": line_no, "item_number": num,
                "description_text": None, "supplier_id": s,
                "order_date": od.isoformat(), "promised_date": promised.isoformat(),
                "received_date": received.isoformat() if recv_ok else None,
                "qty_ordered": qord, "qty_received": (qrec if recv_ok else 0),
                "unit_price": price, "status": "CLOSED" if recv_ok else "OPEN",
            })

    po = pd.DataFrame(rows)

    # ── T3 free-text / non-stock PO lines ───────────────────────────────────
    n_lines = len(po)
    n_ft = int(round(n_lines * C.T3_FREETEXT_SHARE))
    ft_rows = []
    # a share correspond to stocked items (the defect); the rest are ETO one-offs
    stocked_targets = po.sample(n=int(n_ft * C.T3_STOCKED_RATIO), random_state=C.RANDOM_SEED)
    for r in stocked_targets.itertuples(index=False):
        code = rng.choice(C.GENERIC_ITEM_CODES)
        po_seq += 1
        ft_rows.append({
            "po_id": f"PO-{po_seq:06d}", "line": 1, "item_number": code,
            "description_text": _freetext(desc_by_num.get(r.item_number, ""), rng),
            "supplier_id": r.supplier_id, "order_date": r.order_date,
            "promised_date": r.promised_date, "received_date": r.received_date,
            "qty_ordered": r.qty_ordered, "qty_received": r.qty_received,
            "unit_price": r.unit_price, "status": r.status,
        })
        truth["t3"].append({"po_id": f"PO-{po_seq:06d}", "true_item_number": r.item_number,
                            "is_stocked": True})
    for _ in range(n_ft - len(stocked_targets)):
        code = rng.choice(C.GENERIC_ITEM_CODES)
        po_seq += 1
        d = C.START_DATE + timedelta(days=int(rng.integers(0, (C.END_DATE - C.START_DATE).days)))
        lead = int(rng.integers(7, 60))
        ft_rows.append({
            "po_id": f"PO-{po_seq:06d}", "line": 1, "item_number": code,
            "description_text": rng.choice(_ONEOFF),
            "supplier_id": rng.choice(suppliers["supplier_id"]),
            "order_date": d.isoformat(), "promised_date": (d + timedelta(days=lead)).isoformat(),
            "received_date": (d + timedelta(days=lead)).isoformat(),
            "qty_ordered": int(rng.integers(1, 20)), "qty_received": int(rng.integers(1, 20)),
            "unit_price": round(rng.uniform(20, 2000), 2), "status": "CLOSED",
        })
        truth["t3"].append({"po_id": f"PO-{po_seq:06d}", "true_item_number": None,
                            "is_stocked": False})
    if ft_rows:
        po = pd.concat([po, pd.DataFrame(ft_rows)], ignore_index=True)

    # ── T4 receipt-date batching (snap to Monday / +1-5 days) ───────────────
    rec_idx = po.index[po["received_date"].notna()]
    n4 = int(len(rec_idx) * C.T4_BATCH_SHARE)
    for idx in rng.choice(rec_idx, size=min(n4, len(rec_idx)), replace=False):
        rd = pd.to_datetime(po.at[idx, "received_date"]).date()
        nd = None
        for extra in range(1, C.T4_MAX_DISPLACEMENT + 1):
            if (rd + timedelta(days=extra)).weekday() == 0:
                nd = rd + timedelta(days=extra); break
        if nd is None:
            nd = rd + timedelta(days=int(rng.integers(1, C.T4_MAX_DISPLACEMENT + 1)))
        if nd > C.END_DATE:
            nd = rd
        truth["t4"].append({"po_id": po.at[idx, "po_id"], "line": int(po.at[idx, "line"]),
                            "true_received_date": rd.isoformat(), "recorded_received_date": nd.isoformat()})
        po.at[idx, "received_date"] = nd.isoformat()

    # ── T5 open documents (partial receipt, never closed, >90 days) ─────────
    old = po.index[(pd.to_datetime(po["order_date"]).dt.date < C.END_DATE - timedelta(days=90)) &
                   (po["received_date"].notna())]
    n5 = int(len(old) * C.T5_OPEN_PO_SHARE)
    for idx in rng.choice(old, size=min(n5, len(old)), replace=False):
        q = po.at[idx, "qty_ordered"]
        po.at[idx, "qty_received"] = int(q * rng.uniform(0.3, 0.7))
        po.at[idx, "received_date"] = None
        po.at[idx, "status"] = "OPEN"
        truth["t5"].append({"po_id": po.at[idx, "po_id"], "line": int(po.at[idx, "line"]),
                            "qty_ordered": int(q), "qty_received": int(po.at[idx, "qty_received"])})

    return po, truth


def assemble_purchase_orders(po_lines, suppliers, item_master, rng):
    """Turn the replay's buy lines into the purchase-order file: regular lines
    grouped into multi-line orders by supplier and order week, rush lines as
    single-line orders carrying their premium and freight. Then the purchasing
    side defects: T3 free-text lines, T4 batched receipt dates, and T5 lines
    left open (the replay decided which balances never arrived)."""
    stype_by_sup = suppliers.set_index("supplier_id")["supplier_type"].to_dict()
    desc_by_num = item_master.set_index("item_number")["description"].to_dict()
    lines = po_lines.copy()
    lines["od"] = pd.to_datetime(lines["order_date"]).dt.date
    reg = lines[~lines["rush"]].sort_values(["supplier_id", "od", "line_seq"])
    rush = lines[lines["rush"]].sort_values(["od", "line_seq"])

    rows, truth = [], {"t3": [], "t4": [], "t5": []}
    po_seq = 0

    def emit(r, po_id, line_no):
        arrived = r.arrival_date is not None and not (isinstance(r.arrival_date, float) and np.isnan(r.arrival_date))
        received = int(r.qty_received) if arrived else 0
        status = "CLOSED" if received >= int(r.qty_ordered) else "OPEN"
        rows.append({
            "po_id": po_id, "line": line_no, "item_number": r.item_number, "description_text": None,
            "supplier_id": r.supplier_id, "order_date": r.order_date, "promised_date": r.promised_date,
            "received_date": r.arrival_date if (arrived and received > 0) else None,
            "qty_ordered": int(r.qty_ordered), "qty_received": received, "unit_price": float(r.unit_price),
            "status": status, "rush": bool(r.rush), "freight": float(r.freight),
        })
        if bool(r.never_closed):
            truth["t5"].append({"po_id": po_id, "line": line_no, "qty_ordered": int(r.qty_ordered),
                                "qty_received": received})

    i, recs = 0, list(reg.itertuples(index=False))
    while i < len(recs):
        r0 = recs[i]
        many = stype_by_sup.get(r0.supplier_id) in ("Fasteners", "Distributor", "Material")
        max_lines = int(rng.integers(10, 21)) if many else int(rng.integers(1, 4))
        po_seq += 1
        po_id = f"PO-{po_seq:06d}"
        window_end = r0.od + timedelta(days=5)
        line_no = 0
        while i < len(recs) and recs[i].supplier_id == r0.supplier_id and recs[i].od <= window_end and line_no < max_lines:
            line_no += 1
            emit(recs[i], po_id, line_no)
            i += 1
    for r in rush.itertuples(index=False):
        po_seq += 1
        emit(r, f"PO-{po_seq:06d}", 1)

    po = pd.DataFrame(rows)

    # ── T3 free-text / non-stock lines ──────────────────────────────────────
    n_ft = int(round(len(po) * C.T3_FREETEXT_SHARE))
    ft_rows = []
    # a free-text buy of a stocked item is a small one-off, typed in for a job
    # and used on it: the spend is real, the item's history never sees it
    stocked = po[~po["rush"] & po["received_date"].notna()].sample(n=int(n_ft * C.T3_STOCKED_RATIO),
                                                                    random_state=C.RANDOM_SEED)
    for r in stocked.itertuples(index=False):
        code = rng.choice(C.GENERIC_ITEM_CODES)
        po_seq += 1
        pid = f"PO-{po_seq:06d}"
        q = int(rng.integers(1, 6))
        ft_rows.append({"po_id": pid, "line": 1, "item_number": code,
                        "description_text": _freetext(desc_by_num.get(r.item_number, ""), rng),
                        "supplier_id": r.supplier_id, "order_date": r.order_date, "promised_date": r.promised_date,
                        "received_date": r.received_date, "qty_ordered": q, "qty_received": q,
                        "unit_price": r.unit_price, "status": "CLOSED", "rush": False, "freight": 0.0})
        truth["t3"].append({"po_id": pid, "true_item_number": r.item_number, "is_stocked": True})
    for _ in range(n_ft - len(stocked)):
        code = rng.choice(C.GENERIC_ITEM_CODES)
        po_seq += 1
        pid = f"PO-{po_seq:06d}"
        d = C.START_DATE + timedelta(days=int(rng.integers(0, (C.END_DATE - C.START_DATE).days)))
        lead = int(rng.integers(7, 60))
        q = int(rng.integers(1, 6))
        ft_rows.append({"po_id": pid, "line": 1, "item_number": code, "description_text": rng.choice(_ONEOFF),
                        "supplier_id": rng.choice(suppliers["supplier_id"]), "order_date": d.isoformat(),
                        "promised_date": (d + timedelta(days=lead)).isoformat(),
                        "received_date": (d + timedelta(days=lead)).isoformat(), "qty_ordered": q, "qty_received": q,
                        "unit_price": round(rng.uniform(15, 400), 2), "status": "CLOSED", "rush": False, "freight": 0.0})
        truth["t3"].append({"po_id": pid, "true_item_number": None, "is_stocked": False})
    if ft_rows:
        po = pd.concat([po, pd.DataFrame(ft_rows)], ignore_index=True)

    # ── T4 batched receipt dates: the books post days after physical arrival ─
    rec_idx = po.index[po["received_date"].notna()]
    for idx in rng.choice(rec_idx, size=min(int(len(rec_idx) * C.T4_BATCH_SHARE), len(rec_idx)), replace=False):
        rd = pd.to_datetime(po.at[idx, "received_date"]).date()
        nd = None
        for extra in range(1, C.T4_MAX_DISPLACEMENT + 1):
            if (rd + timedelta(days=extra)).weekday() == 0:
                nd = rd + timedelta(days=extra); break
        if nd is None:
            nd = rd + timedelta(days=int(rng.integers(1, C.T4_MAX_DISPLACEMENT + 1)))
        if nd > C.END_DATE:
            continue
        truth["t4"].append({"po_id": po.at[idx, "po_id"], "line": int(po.at[idx, "line"]),
                            "true_received_date": rd.isoformat(), "recorded_received_date": nd.isoformat()})
        po.at[idx, "received_date"] = nd.isoformat()
    return po, truth


def _freetext(desc, rng):
    s = str(desc).lower()
    for k, v in _ABBREV.items():
        if rng.random() < 0.7:
            s = s.replace(k, v)
    s = " ".join(s.split())
    if rng.random() < 0.2 and len(s) > 6:
        i = int(rng.integers(1, len(s) - 1))
        s = s[:i] + s[i + 1:]
    return s.strip()
