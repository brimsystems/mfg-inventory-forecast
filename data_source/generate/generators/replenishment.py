"""Replenishment replay: the ERP's own reorder points and lead times, run day by
day against the books, with physical stock tracked alongside.

For every recorded item number the ERP places a regular order when book on-hand
plus book on-order falls to the recorded reorder point. Because the books carry
the errors, the errors produce their consequences here:

  stale lead times   the order is placed on the recorded lead time and arrives on
                     the actual one, so it lands late
  phantom on-order   a partially received line whose balance never arrives stays
                     in book on-order and suppresses the next order
  unrecorded usage   book on-hand sits above physical, so the order fires late
  duplicates         each record reorders on its own point against one shared pile
  UOM mismatch       receipts post in boxes while issues post in eaches

Physical stock is shared across duplicate members and can go negative, as the
ERP allows. When physical stock cannot cover a day's consumption, a shortage is
logged with its cause, and any job backflushing that day is delayed until the
next arrival. The purchasing manager compensates on the items she tracks: she
orders on her own (near-actual) lead time and places a rush order, at a price
premium with freight, when inbound will not arrive in time. Rush orders are
also placed on untracked items once a stockout is visible. The annual physical
count and the chronic write-offs on omitted items post here too, so the books
the ERP decides on are the books that reach the ledger.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .. import config as C
from .purchase_orders import _actual_lead

PHANTOM_BOOK_TOL = 0.25     # book above physical by this share counts as "unrecorded" cause


def draw_drift_rates(item_meta, rng):
    """Per-item upward drift of the actual lead time over the history (days)."""
    return {num: rng.uniform(1, 8) + (rng.uniform(12, 26) if meta["is_drift"] else 0.0)
            for num, meta in item_meta.items() if not meta["dead"]}


def simulate(events, item_master, item_meta, dup_map, plan, drift_supplier_id, sup_frag,
             buyer_nums, omitted_item_ids, rng, drift_rate=None, never_closed=None, actual_base=None,
             fix_from=None, forward=None, release_by_job=None):
    """fix_from: the date the remediation's BOM corrections and controls take hold
    (chronic write-offs on omitted items stop). A world that never fixes them
    passes END_DATE.

    forward: None, or the corrected masters the shop runs on from a date:
      {"start": date, "lead": {num: days}, "rop": {num: units}, "ss": {num: units},
       "schedule": {num: [(date, rop, ss), ...]} or None, "never_closed": 0.0}
    On the start date each item's duplicate records merge into the primary in
    stock units, the conversion is maintained, the phantom on-order balances are
    closed, and the reorder point and lead time switch to the corrected values;
    with a schedule, the reorder point then follows it month by month. Every item
    draws from its own random stream, so the history before the start date is
    identical whatever runs after it."""
    days = pd.date_range(C.START_DATE, C.END_DATE, freq="D")
    fix_from = C.REMEDIATION_END if fix_from is None else fix_from
    N = len(days)
    ZERO = np.zeros(N)
    day_idx = {d.date(): i for i, d in enumerate(days)}
    cost_by_item = plan.set_index("item_id")["unit_cost"].to_dict()
    rop_by_num = item_master.set_index("item_number")["reorder_point"].to_dict()
    lead_by_num = item_master.set_index("item_number")["master_lead_time_days"].to_dict()
    # the actual lead time drifts from the lead time the item was set up with;
    # a replay on corrected masters still has to draw it from that base
    actual_base = lead_by_num if actual_base is None else actual_base
    omitted = set(int(i) for i in omitted_item_ids)
    never_closed = C.PARTIAL_NEVER_CLOSED if never_closed is None else never_closed

    # the suppliers' actual lead times are a fact of the world, not of the
    # books, so the same drift serves a replay on corrected masters
    if drift_rate is None:
        drift_rate = draw_drift_rates(item_meta, rng)

    recorded = {}
    for num, meta in item_meta.items():
        if not meta["dead"]:
            recorded.setdefault(meta["item_id"], []).append(num)

    # demand arrays: physical per item, recorded per member
    ev = events.copy()
    ev["t"] = ev["date"].map(day_idx)
    ev = ev.dropna(subset=["t"]); ev["t"] = ev["t"].astype(int)
    phys_by_iid = {}
    for iid, g in ev.groupby("item_id"):
        a = np.zeros(N); np.add.at(a, g["t"].to_numpy(), g["qty"].to_numpy()); phys_by_iid[int(iid)] = a
    book_by_num = {}
    for num, g in ev[ev["recorded"]].groupby("item_number"):
        a = np.zeros(N); np.add.at(a, g["t"].to_numpy(), g["qty"].to_numpy()); book_by_num[num] = a
    # the ERP's MRP look-ahead sees only recorded demand (scheduled jobs through
    # the recorded bill, service orders), so omitted components stay invisible
    cum_by_num = {n: np.concatenate([[0.0], np.cumsum(a)]) for n, a in book_by_num.items()}
    # once the records are merged the look-ahead sees the whole part's recorded demand
    cum_by_iid = {}
    for iid, g in ev[ev["recorded"]].groupby("item_id"):
        a = np.zeros(N); np.add.at(a, g["t"].to_numpy(), g["qty"].to_numpy())
        cum_by_iid[int(iid)] = np.concatenate([[0.0], np.cumsum(a)])
    fwd_t = day_idx.get(forward["start"]) if forward else None
    # the look-ahead sees only what the ERP genuinely knows ahead of time: the
    # component requirements of jobs already released to the floor, which it
    # nets against stock (MRP). A job is invisible before its release date;
    # manual pulls and service orders are not known until they happen.
    bf = ev[ev["recorded"] & (ev["channel"] == "BACKFLUSH")].copy()
    rel = release_by_job or {}
    bf["r"] = bf["job_id"].map(lambda j: day_idx.get(rel.get(j), 0) if rel.get(j) in day_idx else 0).astype(int)
    known_num, known_iid = {}, {}
    for key, store in (("item_number", known_num), ("item_id", known_iid)):
        for k, g in bf.groupby(key):
            g = g.sort_values("t")
            store[int(k) if key == "item_id" else k] = (g["t"].to_numpy(), g["r"].to_numpy(), g["qty"].to_numpy(float))

    def ahead(n, t, L, iid=None):
        merged = fwd_t is not None and t >= fwd_t
        arr = known_iid.get(iid) if merged else known_num.get(n)
        if arr is None:
            return 0.0
        c, r, q = arr
        lo = int(np.searchsorted(c, t, side="right")); hi = int(np.searchsorted(c, t + L, side="right"))
        if hi <= lo:
            return 0.0
        return float(q[lo:hi][r[lo:hi] <= t].sum())

    jobs_by_iid_day = {}
    for r in ev[ev["channel"] == "BACKFLUSH"].itertuples(index=False):
        jobs_by_iid_day.setdefault((int(r.item_id), int(r.t)), []).append(r.job_id)

    # annual physical: a January count day each year
    count_days = []
    for yr in range(C.START_DATE.year, C.MODEL_SPAN_END.year + 1):
        d = date(yr, 1, int(rng.integers(8, 20)))
        if C.START_DATE <= d <= C.END_DATE:
            count_days.append(day_idx[d])
    count_days = set(count_days)
    month_adj_days = set()
    for m in C.month_starts():
        d = m + timedelta(days=int(rng.integers(20, 27)))
        if d in day_idx:
            month_adj_days.add(day_idx[d])

    po_lines, adjustments, counts, shortages, rushes = [], [], [], [], []
    job_delays = {}
    physical_series, book_series, unmet_series, short_day_series = {}, {}, {}, {}
    line_seq = 0
    outer_rng = rng

    for iid, nums in recorded.items():
        # each item draws from its own stream, so what runs after the forward
        # start cannot change the history before it
        rng = np.random.default_rng([C.RANDOM_SEED, 7, int(iid)])
        nc = never_closed                      # this item's share of short receipts left open
        phys_d = phys_by_iid.get(iid, np.zeros(N))
        # usage the shop could know on the day: the trailing year of consumption
        # (the first quarter of the history serves as a burn-in)
        cum_u = np.concatenate([[0.0], np.cumsum(phys_d)])

        def usage(t_):
            if t_ < 90:
                return max(cum_u[90] / 90.0, 0.02)
            lo_ = max(0, t_ - 365)
            return max((cum_u[t_] - cum_u[lo_]) / (t_ - lo_), 0.02)
        avg_daily = usage(0)
        cost = float(cost_by_item.get(iid, 1.0))
        meta0 = item_meta[nums[0]]
        cls = meta0["cls"]
        conv = int(meta0["uom_conv"] or 1)
        tracked = any(n in buyer_nums for n in nums)
        weights = dup_map[iid]["weights"] if iid in dup_map else [1.0]

        rop = {n: (0.0 if pd.isna(rop_by_num.get(n)) else float(rop_by_num.get(n))) for n in nums}
        mlead = {n: int(lead_by_num.get(n, 21)) for n in nums}
        blead = {n: int(actual_base.get(n, 21)) for n in nums}
        # the buyer's own reorder level on tracked items, from her lead-time note
        if tracked:
            ref = _actual_lead(meta0, blead[nums[0]], C.MODEL_SPAN_END, drift_rate.get(nums[0], 6.0),
                               drift_supplier_id, rng)
            buyer_look = int(round(ref * C.BUYER_LEAD_MULT))
        q_each = max(1, int(round(avg_daily * C.ORDER_COVER_DAYS)))
        q_order = max(1, int(round(q_each / conv))) if conv > 1 else q_each

        physical = C.INITIAL_STOCK_COVER * max(sum(rop.values()), q_each)
        book = {n: physical * w / (conv if conv > 1 else 1) for n, w in zip(nums, weights)}
        open_po = []          # dicts: num, arrival, qty_book, phantom, rush, promised
        last_order = {n: -10**6 for n in nums}
        backlog = 0.0         # physical demand deferred until material arrives
        waiting = []          # (job, day it ran short), resolved when the material lands
        in_episode = False    # currently short
        corrected_this_episode = False
        learned = False       # after a surprise shortage the floor watches the bin by sight
        suppressed_since = {n: None for n in nums}
        pseries = np.zeros(N); bseries = {n: np.zeros(N) for n in nums}
        useries = np.zeros(N); sdays = np.zeros(N, dtype=bool)
        prim = dup_map[iid]["primary"] if iid in dup_map else nums[0]
        sched = (forward or {}).get("schedule") or {}
        sched_by_t = {}
        for (d_, r_, s_) in sched.get(prim, []):
            if d_ in day_idx:
                sched_by_t[day_idx[d_]] = (float(r_), float(s_))

        for t in range(N):
            # ── the corrected masters take over on the forward start date ───
            if fwd_t is not None and t == fwd_t:
                total_book = sum(book.values()) * (conv if conv > 1 else 1)
                for po in open_po:
                    po["num"] = prim
                    if po.get("phantom", 0) > 0:
                        po["phantom"] = 0                  # closed in remediation
                nums = [prim]; weights = [1.0]
                book = {prim: total_book}
                conv = 1; q_order = q_each
                rop = {prim: float(forward["rop"].get(prim, rop.get(prim, 0.0)))}
                mlead = {prim: int(forward["lead"].get(prim, mlead.get(prim, 21)))}
                blead = {prim: blead.get(prim, 21)}
                nc = forward.get("never_closed", nc)
                book = {prim: max(physical, 0.0)}      # the baseline count restated the balance
                suppressed_since = {prim: None}
                last_order = {prim: max(last_order.values())}
                bseries = {prim: bseries.get(prim, np.zeros(N))}
                learned = False
                tracked = False                            # the spreadsheet is retired
            if t % 7 == 0 or t == fwd_t:
                avg_daily = usage(t)
                q_each = max(1, int(round(avg_daily * C.ORDER_COVER_DAYS)))
                q_order = max(1, int(round(q_each / conv))) if conv > 1 else q_each
            if t in sched_by_t:
                rop[prim] = sched_by_t[t][0]
            if fwd_t is not None and t > fwd_t and days[t].day == 1:
                k = (days[t].year - C.FORWARD_START.year) * 12 + days[t].month - C.FORWARD_START.month
                cls_abc = forward.get("abc", {}).get(iid, "C")
                if cls_abc == "A" or (cls_abc == "B" and k % 3 == 0):
                    book[prim] = max(physical, 0.0)
            # ── arrivals ────────────────────────────────────────────────
            for po in open_po:
                if po["arrival"] == t and not po["received"]:
                    po["received"] = True
                    qty = po["qty_book"]
                    if rng.random() < C.PARTIAL_RECEIPT_PROB:
                        got = max(1, int(round(qty * rng.uniform(*C.PARTIAL_RECEIVED_RANGE))))
                        po["qty_recv"] = got
                        if rng.random() < nc:
                            po["phantom"] = qty - got      # balance never arrives, stays on order
                        else:
                            po["balance_arrival"] = t + int(rng.integers(7, 15))
                    else:
                        po["qty_recv"] = qty
                    physical += po["qty_recv"] * (conv if conv > 1 else 1)
                    book[po["num"]] += po["qty_recv"]
                    po["recv_day"] = t
                    if backlog > 0:
                        take = min(backlog, max(physical, 0.0))
                        physical -= take; backlog -= take
                    if waiting and po["qty_recv"] > 0:
                        # the jobs that stood short get the material first and resume
                        for job, t0 in waiting:
                            job_delays[job] = max(job_delays.get(job, 0), t - t0)
                        waiting = []
                elif po.get("balance_arrival") == t:
                    bal = po["qty_book"] - po["qty_recv"]
                    physical += bal * (conv if conv > 1 else 1)
                    book[po["num"]] += bal
                    po["qty_recv"] = po["qty_book"]; po["balance_arrival"] = None
                    if backlog > 0:
                        take = min(backlog, max(physical, 0.0))
                        physical -= take; backlog -= take
                    if waiting and bal > 0:
                        for job, t0 in waiting:
                            job_delays[job] = max(job_delays.get(job, 0), t - t0)
                        waiting = []
            # ── consumption ──────────────────────────────────────────────
            d = phys_d[t]
            if d > 0:
                avail = max(physical, 0.0)
                if avail >= d:
                    physical -= d
                    if in_episode and backlog <= 0:
                        in_episode = False; corrected_this_episode = False
                else:
                    short = d - avail
                    physical -= avail
                    backlog += short
                    useries[t] = short
                    for job in jobs_by_iid_day.get((iid, t), []):
                        waiting.append((job, t))
                    if not in_episode:
                        in_episode = True
                        if not tracked:
                            learned = True
                        cause = _cause(nums, book, physical, open_po, rop, suppressed_since, t, conv)
                        shortages.append({"item_id": iid, "item_number": nums[0], "date": days[t].date().isoformat(),
                                          "short_qty": int(round(short)), "cause": cause,
                                          "jobs": len(jobs_by_iid_day.get((iid, t), [])), "tracked": tracked})
                    # the shelf is empty while the system shows stock: the floor
                    # posts a correction, which is where chronic write-offs come from
                    book_total = sum(book.values()) * (conv if conv > 1 else 1)
                    if not corrected_this_episode and book_total > 0 and rng.random() < 0.6:
                        for n, w in zip(nums, weights):
                            gap = int(round(book[n]))
                            if gap > 0:
                                book[n] = 0.0
                                adjustments.append({"item_number": n, "date": days[t].date().isoformat(),
                                                    "qty": -gap, "kind": "chronic"})
                        corrected_this_episode = True
            # the ERP subtracts recorded issues in eaches, even from a book kept in
            # boxes (M5): that mismatch is the error, so no conversion is applied
            for n in nums:
                book[n] -= book_by_num.get(n, ZERO)[t]
            # ── chronic write-offs on omitted items (T1) ─────────────────
            if (iid in omitted and t in month_adj_days and days[t].date() <= fix_from
                    and rng.random() < C.T1_MONTHLY_ADJ_PROB):
                for n in nums:
                    gap = book[n] - max(physical, 0) * (weights[nums.index(n)])
                    if gap > 1:
                        mag = int(max(1, round(gap * rng.uniform(0.3, 0.8))))
                        book[n] -= mag
                        adjustments.append({"item_number": n, "date": days[t].date().isoformat(), "qty": -mag,
                                            "kind": "chronic"})
            # ── annual physical count ────────────────────────────────────
            if t in count_days:
                # the count is keyed in the stock unit and replaces the book; the
                # ledger posts the correction against its own balance (see cycle_counts)
                for n, w in zip(nums, weights):
                    counted = max(0, int(round(max(physical, 0) * w * (1 + rng.normal(0, C.COUNT_NOISE_SD)))))
                    counts.append({"item_number": n, "date": days[t].date().isoformat(),
                                   "counted_qty": counted, "program": "ANNUAL"})
                    book[n] = float(counted)
            # ── reorder decisions, per recorded number ───────────────────
            for n in nums:
                on_order = sum(po["qty_book"] - po.get("qty_recv", 0) if po["received"] else po["qty_book"]
                               for po in open_po if po["num"] == n and (not po["received"] or po.get("phantom", 0) > 0
                                                                        or po.get("balance_arrival")))
                on_order_real = sum(po["qty_book"] for po in open_po if po["num"] == n and not po["received"])
                if tracked:
                    level = avg_daily * C.BUYER_SAFETY_DAYS
                    look = ahead(n, t, buyer_look, iid)
                elif fwd_t is not None and t >= fwd_t:
                    # on the corrected masters the point covers the demand the ERP
                    # cannot see (forecast unscheduled usage plus safety stock), and
                    # MRP nets the requirements of released jobs on top of it
                    level = rop[n]
                    look = ahead(n, t, mlead[n], iid)
                else:
                    level = rop[n]
                    look = ahead(n, t, mlead[n], iid)
                position = book[n] + on_order - look
                if position > level and book[n] + on_order_real <= level:
                    if suppressed_since[n] is None:
                        suppressed_since[n] = t
                else:
                    suppressed_since[n] = None
                # the buyer can see the pile: no regular order when months of stock sit on the shelf
                shelf_full = max(physical, 0.0) >= avg_daily * C.SHELF_FULL_COVER_DAYS
                nothing_inbound = not any(not po["received"] for po in open_po)
                if conv > 1:
                    # receipts post in spools or boxes and issues in feet or eaches, so
                    # the book on these items is nonsense within weeks and the buyer
                    # orders them by sight, on the master lead time
                    triggered = False
                    by_sight = (n == nums[0] and nothing_inbound
                                and max(physical, 0.0) <= avg_daily * (mlead[n] + C.FLOOR_VISUAL_MIN_DAYS))
                else:
                    triggered = position <= level
                    # after a surprise shortage the floor reorders by sight, whatever the books say
                    by_sight = (learned and n == nums[0] and nothing_inbound
                                and max(physical, 0.0) <= avg_daily * (mlead[n] + C.FLOOR_VISUAL_MIN_DAYS))
                if (triggered or by_sight) and not shelf_full and t - last_order[n] >= C.REORDER_GAP_DAYS:
                    line_seq += 1
                    meta = item_meta[n]
                    alead = _actual_lead(meta, blead[n], days[t].date(), drift_rate.get(n, 6.0), drift_supplier_id, rng)
                    sup = meta["supplier_id"]
                    if sup in sup_frag["alias_ids"] and rng.random() < 0.4 and sup_frag["alias_ids"][sup]:
                        sup = rng.choice(sup_frag["alias_ids"][sup])
                    price = round(cost * (conv if conv > 1 else 1) * rng.uniform(0.9, 1.2), 2)
                    # a planned order covers the net requirement, in standard lots
                    if by_sight:
                        lots = 1
                    else:
                        need_units = max(0.0, level + look - (book[n] + on_order))
                        lots = int(np.ceil(need_units / max(1.0, q_each))) if need_units > 0 else 1
                    qty = q_order * max(1, min(lots, C.MRP_MAX_LOT_MULT))
                    po = {"line_seq": line_seq, "num": n, "item_id": iid, "supplier": sup, "order_t": t,
                          "promised": t + mlead[n], "arrival": t + alead, "qty_book": qty, "price": price,
                          "rush": False, "freight": 0.0, "received": False, "phantom": 0}
                    open_po.append(po); po_lines.append(po); last_order[n] = t
            # ── rush orders ──────────────────────────────────────────────
            pending = [po for po in open_po if not po["received"]]
            has_rush = any(po["rush"] for po in pending)
            if not has_rush:
                horizon = min([po["arrival"] for po in pending] or [t + 14]) - t
                need = phys_d[t + 1: t + 1 + max(1, int(horizon))].sum() + backlog
                will_short = max(physical, 0.0) - need < 0
                new_episode = in_episode and len(shortages) and shortages[-1]["item_id"] == iid \
                    and shortages[-1]["date"] == days[t].date().isoformat()
                if (tracked and will_short and pending) or ((not tracked) and new_episode and rng.random() < C.RUSH_ESCALATION_PROB):
                    n = nums[0]
                    meta = item_meta[n]
                    alead = _actual_lead(meta, blead[n], days[t].date(), drift_rate.get(n, 6.0), drift_supplier_id, rng)
                    rlead = max(2, int(round(alead * C.RUSH_LEAD_FRACTION)))
                    prem = rng.uniform(*C.RUSH_PRICE_PREMIUM)
                    lo, hi = C.RUSH_FREIGHT_BY_CLASS.get(cls, (60, 150))
                    freight = round(float(rng.uniform(lo, hi)), 2)
                    cause = _cause(nums, book, physical, open_po, rop, suppressed_since, t, conv)
                    line_seq += 1
                    gap = need - max(physical, 0.0)
                    rq = max(1, int(np.ceil(max(gap, avg_daily * 14) / (conv if conv > 1 else 1))))
                    po = {"line_seq": line_seq, "num": n, "item_id": iid, "supplier": meta["supplier_id"],
                          "order_t": t, "promised": t + rlead, "arrival": t + rlead, "qty_book": rq,
                          "price": round(cost * (conv if conv > 1 else 1) * (1 + prem), 2),
                          "premium": round(rq * cost * (conv if conv > 1 else 1) * prem, 2),
                          "rush": True, "freight": freight, "received": False, "phantom": 0, "cause": cause}
                    open_po.append(po); po_lines.append(po)
                    rushes.append({"item_id": iid, "item_number": n, "date": days[t].date().isoformat(),
                                   "cause": cause, "freight": freight, "premium": round(prem, 3),
                                   "premium_usd": po["premium"], "tracked": tracked})
            pseries[t] = physical
            sdays[t] = backlog > 0
            for n in nums:
                bseries[n][t] = book[n]

        # material that never came: the job waits to the end of the history
        for job, t0 in waiting:
            job_delays[job] = max(job_delays.get(job, 0), N - 1 - t0)
        physical_series[iid] = pseries
        unmet_series[iid] = useries
        short_day_series[iid] = sdays
        for n in nums:
            book_series[n] = bseries[n]
    rng = outer_rng

    po_df = pd.DataFrame([{
        "line_seq": p["line_seq"], "item_number": p["num"], "item_id": p["item_id"], "supplier_id": p["supplier"],
        "order_date": days[p["order_t"]].date().isoformat(),
        "promised_date": days[min(p["promised"], N - 1)].date().isoformat() if p["promised"] < N else
                         (days[0].date() + timedelta(days=int(p["promised"]))).isoformat(),
        "arrival_date": days[p["arrival"]].date().isoformat() if p["arrival"] < N else None,
        "qty_ordered": int(p["qty_book"]),
        "qty_received": int(p.get("qty_recv", 0)) if p["received"] else 0,
        "unit_price": p["price"], "rush": p["rush"], "freight": p["freight"], "premium": p.get("premium", 0.0),
        "never_closed": bool(p.get("phantom", 0) > 0),
        "cause": p.get("cause"),
    } for p in po_lines])
    return {
        "po_lines": po_df,
        "adjustments": pd.DataFrame(adjustments),
        "counts": pd.DataFrame(counts),
        "shortages": pd.DataFrame(shortages),
        "rushes": pd.DataFrame(rushes),
        "job_delays": job_delays,
        "physical": physical_series,
        "book": book_series,
        "demand": phys_by_iid,
        "unmet": unmet_series,
        "short_days": short_day_series,
        "days": days,
    }


def _cause(nums, book, physical, open_po, rop, suppressed_since, t, conv):
    """Attribute a shortage or rush to the data error most directly behind it."""
    if any(suppressed_since[n] is not None for n in nums):
        return "phantom_on_order"
    late = [po for po in open_po if not po["received"] and po["arrival"] > po["promised"] and t >= po["promised"]]
    if late:
        return "stale_lead_time"
    book_total = sum(book.values()) * (conv if conv > 1 else 1)
    if physical <= 0 and book_total > 0 and (book_total - max(physical, 0)) > PHANTOM_BOOK_TOL * max(book_total, 1):
        return "unrecorded_consumption"
    return "other"
