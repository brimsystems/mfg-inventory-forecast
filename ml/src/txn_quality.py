"""Transaction-level detection and attribution (addendum steps 4-5).

One detector per transaction defect, each working from the recorded data alone and
emitting affected records with evidence and a confidence score. Findings split
into confirmed (high confidence) and probable (needs human review), which is how
a real data-quality engagement reports transactional work. T1 free-text lines are
attributed back to a probable item by reusing the D1 similarity machinery plus
supplier and unit-price evidence. The correction tables written here feed the
fully-cleaned consumption mart.

Precision and recall are scored against the injected truth for validation only.

Run:  python -m ml.src.txn_quality
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .resolution import REPO, normalize_tokens
from difflib import SequenceMatcher

RAW = REPO / "data_source" / "raw"
SEEDS = REPO / "data_pipeline" / "seeds"
TRUTH = REPO / "data_source" / "truth"
OUT = REPO / "ml" / "data" / "data_quality" / "txn"

GENERIC = ["NONSTOCK", "MISC", "SHOPSUPPLY"]
T1_ATTRIBUTE_CONF = 0.60      # attribute at or above this; below is held for review


def _load():
    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv")
    po = pd.read_csv(RAW / "erp" / "purchase_orders.csv")
    im = pd.read_csv(RAW / "erp" / "item_master.csv")
    cw = pd.read_csv(SEEDS / "item_crosswalk.csv")
    truth = json.loads((TRUTH / "txn_defects.json").read_text())
    canon = json.loads((TRUTH / "crosswalks.json").read_text())["item_number_to_canonical_id"]
    x = dict(zip(cw["item_number"], cw["canonical_item_number"]))
    return tx, po, im, x, canon, truth


# ── T1 attribution ──────────────────────────────────────────────────────────
def attribute_t1(tx, im, xwalk, canon):
    free = tx[tx["item_number"].isin(GENERIC) & tx["description"].notna()].copy()
    master = im.copy()
    master["tokens"] = master["description"].map(lambda d: set(normalize_tokens(d)))
    cost = dict(zip(master["item_number"], master["standard_cost"]))
    rows = []
    mrecords = master.to_dict("records")
    for r in free.itertuples(index=False):
        ftok = set(normalize_tokens(r.description))
        best, best_s = None, 0.0
        for m in mrecords:
            jac = len(ftok & m["tokens"]) / max(1, len(ftok | m["tokens"]))
            if jac < 0.3:
                continue
            seq = SequenceMatcher(None, " ".join(sorted(ftok)), " ".join(sorted(m["tokens"]))).ratio()
            price_ok = 0.0
            if not np.isnan(r.unit_price) and m["standard_cost"] and m["standard_cost"] > 0:
                price_ok = 1 - min(1.0, abs(r.unit_price - m["standard_cost"]) / m["standard_cost"])
            s = 0.5 * jac + 0.3 * seq + 0.2 * price_ok
            if s > best_s:
                best_s, best = s, m["item_number"]
        conf = round(best_s, 3)
        rows.append({"transaction_id": r.transaction_id, "quantity": r.quantity,
                     "probable_item": xwalk.get(best, best) if best else None,
                     "confidence": conf, "confirmed": conf >= T1_ATTRIBUTE_CONF})
    att = pd.DataFrame(rows)
    # score vs truth
    t1truth = {t["transaction_id"]: t for t in json.loads((TRUTH / "txn_defects.json").read_text())["t1"]}
    tp = fp = 0; attributable = sum(1 for v in t1truth.values() if not v["is_oneoff"])
    for r in att[att["confirmed"]].itertuples(index=False):
        tt = t1truth.get(r.transaction_id)
        if tt and not tt["is_oneoff"] and xwalk.get(tt["true_item_number"], tt["true_item_number"]) == r.probable_item:
            tp += 1
        else:
            fp += 1
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / attributable if attributable else 0.0
    return att, {"free_lines": len(free), "attributed": int(att["confirmed"].sum()),
                 "held_for_review": int((~att["confirmed"]).sum()), "precision": prec, "recall": rec}


# ── T2 quantity keying errors ───────────────────────────────────────────────
def detect_t2(tx, im):
    box_sizes = [25, 50, 100, 250]
    flags = []
    for num, g in tx[tx["type"].isin(["issue", "receipt"])].groupby("item_number"):
        q = g["quantity"].to_numpy(float)
        if len(q) < 6:
            continue
        med = np.median(q); mad = np.median(np.abs(q - med)) or 1.0
        cv = 1.4826 * mad / med if med > 0 else 9.0   # robust CV, unaffected by the error itself
        if med <= 0:
            continue
        for r in g.itertuples(index=False):
            if r.quantity < med + 4 * mad:                       # must be a real outlier
                continue
            ratio = r.quantity / med
            near10 = 8.5 <= ratio <= 12
            boxmatch = any(abs(ratio - b) <= b * 0.15 for b in box_sizes) and ratio >= 15
            if not (near10 or boxmatch):                          # signature of a keying error
                continue
            factor = 10 if near10 else round(ratio)
            # confirm only where the item's demand is stable enough that a 10x is
            # unambiguously an error; everything else is flagged for review.
            confirmed = bool(near10 and cv < 0.35 and r.quantity >= q.max())
            flags.append({"transaction_id": r.transaction_id, "item_number": num,
                          "raw_qty": int(r.quantity), "median": round(med, 1),
                          "corrected_qty": int(round(r.quantity / factor)),
                          "confidence": 0.85 if confirmed else 0.5, "confirmed": confirmed})
    f = pd.DataFrame(flags)
    truth = {t["transaction_id"] for t in json.loads((TRUTH / "txn_defects.json").read_text())["t2"]}
    if len(f):
        tp = f[f["confirmed"]]["transaction_id"].isin(truth).sum()
        prec = tp / max(1, f["confirmed"].sum()); rec = tp / max(1, len(truth))
    else:
        prec = rec = 0.0
    return f, {"flagged": len(f), "confirmed": int(f["confirmed"].sum()) if len(f) else 0,
               "precision": float(prec), "recall": float(rec)}


# ── T4 unrecorded consumption via chronic adjustments ───────────────────────
def detect_t4(tx):
    adj = tx[tx["type"] == "adjustment"]
    g = adj.groupby("item_number").agg(n=("quantity", "size"), net=("quantity", "sum"))
    flagged = g[(g["net"] < -20) & (g["n"] >= 8)].reset_index()
    flagged["implied_usage"] = -flagged["net"]
    truth = {t["item_number"] for t in json.loads((TRUTH / "txn_defects.json").read_text())["t4"]}
    tp = flagged["item_number"].isin(truth).sum()
    return flagged, {"flagged_items": len(flagged), "precision": float(tp / max(1, len(flagged))),
                     "recall": float(tp / max(1, len(truth)))}


# ── T6 phantom on-order ─────────────────────────────────────────────────────
def detect_t6(po):
    po = po.copy()
    po["order_date"] = pd.to_datetime(po["order_date"])
    med_lead = (po[po["received_date"].notna()]
                .assign(lead=(pd.to_datetime(po["received_date"]) - po["order_date"]).dt.days)
                .groupby("supplier_id")["lead"].median())
    open_lines = po[po["received_date"].isna() | (po.get("po_status") == "OPEN")].copy()
    open_lines["days_open"] = (pd.Timestamp("2026-08-31") - open_lines["order_date"]).dt.days
    open_lines["norm_lead"] = open_lines["supplier_id"].map(med_lead).fillna(30)
    phantom = open_lines[open_lines["days_open"] > open_lines["norm_lead"] + 30].copy()
    phantom["phantom_qty"] = phantom["quantity_ordered"] - phantom["quantity_received"].fillna(0)
    truth = {t["po_id"] for t in json.loads((TRUTH / "txn_defects.json").read_text())["t6"]}
    tp = phantom["po_id"].isin(truth).sum()
    return phantom[["po_id", "supplier_id", "days_open", "phantom_qty"]], {
        "flagged": len(phantom), "precision": float(tp / max(1, len(phantom))),
        "recall": float(tp / max(1, len(truth)))}


# ── T7 duplicate postings ───────────────────────────────────────────────────
def detect_t7(tx):
    t = tx[tx["type"].isin(["issue", "receipt"])].copy()
    t["d"] = pd.to_datetime(t["transaction_date"])
    t["wo"] = t["work_order_id"].fillna("NA")
    t["loc"] = t["location"].fillna("NA")
    t = t.sort_values(["item_number", "type", "quantity", "wo", "loc", "d"])
    rows = []
    # A true duplicate copies the whole line (item, quantity, work order, location);
    # a copy within a day is confirmed, a wider gap is flagged for review.
    for _, g in t.groupby(["item_number", "type", "quantity", "wo", "loc"]):
        ds = g["d"].tolist(); ids = g["transaction_id"].tolist()
        for i in range(1, len(ds)):
            gap = (ds[i] - ds[i - 1]).days
            if gap <= 3:
                rows.append({"transaction_id": ids[i], "gap_days": gap, "confirmed": gap <= 1})
    f = pd.DataFrame(rows)
    truth = {t["transaction_id"] for t in json.loads((TRUTH / "txn_defects.json").read_text())["t7"]}
    conf = f[f["confirmed"]] if len(f) else f
    tp = conf["transaction_id"].isin(truth).sum() if len(conf) else 0
    return f, {"flagged": len(f), "confirmed": int(len(conf)),
               "precision": float(tp / max(1, len(conf))),
               "recall": float(conf["transaction_id"].isin(truth).sum() / max(1, len(truth))) if len(conf) else 0.0}


def run():
    tx, po, im, xwalk, canon, truth = _load()
    OUT.mkdir(parents=True, exist_ok=True)

    t1, s1 = attribute_t1(tx, im, xwalk, canon)
    t2, s2 = detect_t2(tx, im)
    t4, s4 = detect_t4(tx)
    t6, s6 = detect_t6(po)
    t7, s7 = detect_t7(tx)

    t1.to_parquet(OUT / "t1_attribution.parquet", index=False)
    t2.to_parquet(OUT / "t2_corrections.parquet", index=False)
    t4.to_parquet(OUT / "t4_adjustments.parquet", index=False)
    t6.to_parquet(OUT / "t6_phantom.parquet", index=False)
    t7.to_parquet(OUT / "t7_duplicates.parquet", index=False)

    # confirmed vs probable share across all transaction findings
    confirmed = s1["attributed"] + s2["confirmed"] + s4["flagged_items"] + s6["flagged"] + s7["confirmed"]
    total = s1["free_lines"] + s2["flagged"] + s4["flagged_items"] + s6["flagged"] + s7["flagged"]
    summary = {"T1": s1, "T2": s2, "T4": s4, "T6": s6, "T7": s7,
               "confirmed_share": confirmed / max(1, total)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n=== Transaction detection & attribution ===")
    print(f"  T1 attribution   {s1['attributed']} attributed, {s1['held_for_review']} held; "
          f"precision {s1['precision']*100:.0f}%, recall {s1['recall']*100:.0f}%")
    print(f"  T2 keying        {s2['confirmed']} confirmed of {s2['flagged']} flagged; "
          f"precision {s2['precision']*100:.0f}%, recall {s2['recall']*100:.0f}%")
    print(f"  T4 unrecorded    {s4['flagged_items']} items; precision {s4['precision']*100:.0f}%, recall {s4['recall']*100:.0f}%")
    print(f"  T6 phantom PO    {s6['flagged']} lines; precision {s6['precision']*100:.0f}%, recall {s6['recall']*100:.0f}%")
    print(f"  T7 duplicates    {s7['confirmed']} confirmed of {s7['flagged']} flagged; "
          f"precision {s7['precision']*100:.0f}%, recall {s7['recall']*100:.0f}%")
    print(f"  Confirmed share  {summary['confirmed_share']*100:.0f}% (target 55-70%; rest flagged for review)")
    print()


if __name__ == "__main__":
    run()
