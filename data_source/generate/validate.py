"""Validation checkpoint (spec section 4): run every realism check against the
generated dataset and report each value with its expected range. Records the
results to data_source/truth/realism_checks.txt so they are reproducible.

Run:  python -m data_source.generate.validate
"""
from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config as C

RAW = C.RAW_DIR
TRUTH = C.REPO_ROOT / "data_source" / "truth"
_lines = []


def _chk(name, value, lo, hi, fmt="{:.1f}"):
    ok = (lo is None or value >= lo) and (hi is None or value <= hi)
    rng = f"{fmt.format(lo) if lo is not None else '-'}..{fmt.format(hi) if hi is not None else '-'}"
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {name:<52} {fmt.format(value):>10}   expect {rng}"
    _lines.append(line)
    print(line)
    return ok


def run():
    im = pd.read_csv(RAW / "erp" / "item_master.csv", low_memory=False)
    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv", low_memory=False)
    po = pd.read_csv(RAW / "erp" / "purchase_orders.csv", low_memory=False)
    prod = pd.read_csv(RAW / "erp" / "production_orders.csv", low_memory=False)
    svc = pd.read_csv(RAW / "erp" / "service_orders.csv", low_memory=False)
    bom = pd.read_csv(RAW / "erp" / "bill_of_materials.csv", low_memory=False)
    cross = json.loads((TRUTH / "crosswalks.json").read_text())
    po_def = json.loads((TRUTH / "po_defects.json").read_text())

    dead_nums = {m["item_number"] for m in _dead_meta()}
    live = im[~im["item_number"].isin(dead_nums)]

    print("\n=== Volume and shape ===")
    _chk("Item master records", len(im), 2200, 2600, "{:.0f}")
    _chk("Live share of master (%)", len(live) / len(im) * 100, 55, 65)
    _chk("Inventory transactions", len(tx), 180000, 260000, "{:.0f}")
    _chk("PO lines", len(po), 28000, 40000, "{:.0f}")
    _chk("Production orders", len(prod), 5000, 7500, "{:.0f}")
    _chk("Service orders", svc["order_id"].nunique(), 3500, 5500, "{:.0f}")

    # spend concentration (top 20% live items by purchase spend)
    po_live = po[po["item_number"].isin(live["item_number"])].copy()
    po_live["spend"] = po_live["qty_received"].fillna(0) * po_live["unit_price"].fillna(0)
    spend = po_live.groupby("item_number")["spend"].sum().sort_values(ascending=False)
    top20 = int(len(spend) * 0.2)
    _chk("Top 20% live items = % of spend", spend.head(top20).sum() / spend.sum() * 100, 75, 85)

    # consumption concentration
    cons = tx[tx["type"].isin(["ISSUE", "BACKFLUSH"])].copy()
    cost = im.set_index("item_number")["standard_cost"].to_dict()
    cons["val"] = cons["qty"].abs() * cons["item_number"].map(cost).fillna(0)
    cval = cons.groupby("item_number")["val"].sum().sort_values(ascending=False)
    _chk("Top 20% items = % of consumption value",
         cval.head(int(len(cval) * 0.2)).sum() / cval.sum() * 100, 70, 80)

    print("\n=== Pre-remediation defect levels ===")
    _chk("Dead items (% of master)", len(dead_nums) / len(im) * 100, 35, 45)
    dead_rop = sum(1 for m in _dead_meta() if m["keeps_rop"]) / max(1, len(dead_nums)) * 100
    _chk("Dead items still carrying a reorder point (%)", dead_rop, 25, 35)

    # lead-time drift: master vs actual median
    drift = _lead_drift(po, im, live)
    _chk("Live items lead-time diff > 3 days (%)", drift[">3"], 55, 75)
    _chk("Live items lead-time diff > 7 days (%)", drift[">7"], 20, 30)

    # duplicate clusters: physical live items in a cluster / physical live items
    dup = cross["duplicate_clusters"]
    recs_in_clusters = sum(len(v["records"]) for v in dup.values())
    physical_live = len(live) - (recs_in_clusters - len(dup))
    _chk("Live items in duplicate clusters (%)", len(dup) / max(1, physical_live) * 100, 5, 8)
    _chk("Number of duplicate clusters", len(dup), 60, 90, "{:.0f}")

    # UOM mismatch
    n_uom = im["uom_conversion"].isna().sum()   # placeholder; real count below
    m5 = len(cross["m5_items"])
    _chk("UOM-mismatch items", m5, 40, 55, "{:.0f}")

    # missing fields / MISC
    blank = live[live[["standard_cost", "reorder_point", "primary_supplier_id"]].isna().any(axis=1)]
    _chk("Live items with a blocking blank field (%)", len(blank) / len(live) * 100, 10, 15)
    _chk("Live items item_class = MISC (%)", (live["item_class"] == "MISC").mean() * 100, 10, 14)

    # BOM omissions (top-level products, from the omission truth)
    n_products = cross.get("n_products", C.N_PRODUCTS)
    prod_with_omission = len(cross.get("m3_affected_products", []))
    _chk("Products with a BOM omission (%)", prod_with_omission / n_products * 100, 30, 40)

    # adjustments
    moved = tx.loc[tx["type"].isin(["ISSUE", "BACKFLUSH", "RECEIPT"]), "qty"].abs().sum()
    adj = tx.loc[tx["type"] == "ADJUST", "qty"].abs().sum()
    _chk("Adjustment share of quantity moved (%)", adj / moved * 100, 15, 25)
    adj_rows = tx[tx["type"] == "ADJUST"]
    blank_reason = adj_rows["reason_code"].isna() | adj_rows["reason_code"].astype(str).isin(["", "nan", "ADJ", "VAR", "MISC", "COUNT"])
    _chk("Adjustments with blank/generic reason (%)", blank_reason.mean() * 100, 60, 75)

    # free-text PO lines
    ft = po[po["item_number"].isin(C.GENERIC_ITEM_CODES)]
    _chk("Free-text PO lines (%)", len(ft) / len(po) * 100, 10, 14)
    stocked = sum(1 for r in po_def["t3"] if r.get("is_stocked"))
    _chk("Free-text lines matching a stocked item (%)", stocked / max(1, len(po_def["t3"])) * 100, 50, 60)

    # receipts on peak weekday
    rec = tx[tx["type"] == "RECEIPT"].copy()
    rec["wd"] = pd.to_datetime(rec["txn_date"]).dt.weekday
    peak = rec["wd"].value_counts(normalize=True).max() * 100
    _chk("Receipts posted on the peak weekday (%)", peak, 35, 50)

    # open POs > 90 days
    po["od"] = pd.to_datetime(po["order_date"]).dt.date
    old = po[po["od"] < C.END_DATE - timedelta(days=90)]
    open_old = old[old["status"] == "OPEN"]
    _chk("PO lines open > 90 days (%)", len(open_old) / max(1, len(old)) * 100, 5, 8)

    # shared logins on floor / receiving
    floor = tx[tx["type"].isin(["ISSUE", "BACKFLUSH", "RECEIPT"])]
    shared = floor["user_id"].isin(C.SHARED_LOGINS).mean() * 100
    _chk("Floor/receiving txns under shared logins (%)", shared, 70, 85)

    n_fail = sum(1 for l in _lines if "[FAIL]" in l)
    print(f"\n{len(_lines) - n_fail}/{len(_lines)} checks pass, {n_fail} out of range")
    (TRUTH / "realism_checks.txt").write_text("\n".join(_lines) + f"\n\n{len(_lines)-n_fail}/{len(_lines)} pass\n")
    print(f"Recorded -> {TRUTH / 'realism_checks.txt'}")


def _dead_meta():
    # reconstruct dead-item flags by re-reading the master: dead items have the
    # 100000+ number range from the generator.
    im = pd.read_csv(RAW / "erp" / "item_master.csv", low_memory=False)
    out = []
    for n in im["item_number"]:
        parts = str(n).split("-")
        if len(parts) >= 2 and parts[1].isdigit() and int(parts[1]) >= 100000:
            row = im[im["item_number"] == n].iloc[0]
            out.append({"item_number": n, "keeps_rop": not pd.isna(row["reorder_point"])})
    return out


def _dup_scale(dup, live):
    # count of live records in clusters vs live canonical items
    recs = sum(len(v["records"]) for v in dup.values())
    return recs / max(1, len(dup)) / 1.0 * (len(dup) / max(1, len(dup)))  # ~members per item


def _lead_drift(po, im, live):
    p = po[po["received_date"].notna()].copy()
    p["lead"] = (pd.to_datetime(p["received_date"]) - pd.to_datetime(p["order_date"])).dt.days
    med = p.groupby("item_number")["lead"].median()
    master = im.set_index("item_number")["master_lead_time_days"]
    live_nums = set(live["item_number"])
    diff = (med - master).dropna()
    diff = diff[[n in live_nums for n in diff.index]].abs()
    return {">3": (diff > 3).mean() * 100, ">7": (diff > 7).mean() * 100}


def _products_with_omission():
    # compare true vs recorded BOM product coverage via the omissions truth
    try:
        # omissions are not written as a table; infer from true backflush file
        return int(round(C.N_PRODUCTS * C.M3_PRODUCT_SHARE))
    except Exception:
        return 0


if __name__ == "__main__":
    run()
