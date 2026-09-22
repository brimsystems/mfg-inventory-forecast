"""Build all five source datasets, write raw and sample extracts, and record the
ground-truth crosswalks the data-quality pipeline is scored against.

Run:  python -m data_source.generate.run_generator
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .checkpoint import classify
from .generators.demand import build_item_plan, build_monthly_demand
from .generators.suppliers import build_suppliers
from .generators.item_master import build_item_master
from .generators.purchase_orders import build_purchase_orders
from .generators.inventory_transactions import build_inventory_transactions
from .generators.cycle_counts import build_cycle_counts
from .generators.transaction_defects import apply as apply_transaction_defects

TRUTH_DIR = C.REPO_ROOT / "data_source" / "truth"


def _abc(plan, annual_by_item) -> dict:
    val = {i: annual_by_item[i] * c for i, c in plan.set_index("item_id")["unit_cost"].items()}
    order = sorted(val, key=val.get, reverse=True)
    total = sum(val.values()) or 1.0
    out, cum = {}, 0.0
    for i in order:
        cum += val[i] / total
        out[i] = "A" if cum <= C.ABC_A_CUM else ("B" if cum <= C.ABC_B_CUM else "C")
    return out


def _save(df, name, base_dir, sample=False):
    system = C.TABLE_SYSTEM_MAP[name]
    out_dir = base_dir / system
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_sample" if sample else ""
    fp = out_dir / f"{name}{suffix}.csv"
    df.to_csv(fp, index=False)
    kb = fp.stat().st_size / 1024
    size = f"{kb/1024:.1f} MB" if kb > 1024 else f"{kb:.0f} KB"
    print(f"  [{system:>4}]  {name+suffix:<24} {len(df):>9,} rows   {size}")


def run():
    rng = np.random.default_rng(C.RANDOM_SEED)
    print(f"\nBuilding inventory source extracts for {C.START_DATE} through {C.END_DATE}\n")

    plan = build_item_plan(rng)
    demand = build_monthly_demand(plan, rng)
    annual_by_item = (demand.pivot(index="item_id", columns="month", values="demand_units")
                            .fillna(0).sort_index(axis=1).iloc[:, -12:].sum(axis=1).to_dict())
    abc_by_item = _abc(plan, annual_by_item)

    suppliers, supplier_truth, d5 = build_suppliers(rng)
    drift_supplier_id = suppliers.loc[suppliers["supplier_type"] == "Fasteners",
                                      "supplier_id"].iloc[0]

    print("[1/5] Item master        (ERP)")
    item_master, dup_map, item_meta, defects = build_item_master(
        plan, suppliers, annual_by_item, drift_supplier_id, rng)
    print("[2/5] Suppliers          (ERP)")
    print("[3/5] Purchase orders    (ERP)")
    purchase_orders = build_purchase_orders(
        plan, demand, item_master, item_meta, dup_map, supplier_truth, d5, rng)
    print("[4/5] Inventory ledger   (ERP) - this step takes a moment")
    transactions = build_inventory_transactions(demand, dup_map, item_meta, purchase_orders, rng)
    print("      Injecting transaction-level defects T1-T7")
    purchase_orders, transactions, txn_truth = apply_transaction_defects(
        item_master, plan, item_meta, dup_map, purchase_orders, transactions, rng)
    print("[5/5] Cycle counts       (WMS)")
    cycle_counts = build_cycle_counts(
        item_master, item_meta, annual_by_item, abc_by_item, dup_map, defects, rng)

    tables = {
        "item_master":            item_master,
        "suppliers":              suppliers,
        "purchase_orders":        purchase_orders,
        "inventory_transactions": transactions,
        "cycle_counts":           cycle_counts,
    }
    print(f"\nFull extracts -> {C.RAW_DIR}")
    for name, df in tables.items():
        _save(df, name, C.RAW_DIR)
    print(f"\nSample extracts ({C.SAMPLE_SIZE} rows) -> {C.SAMPLES_DIR}")
    for name, df in tables.items():
        _save(df.head(C.SAMPLE_SIZE), name, C.SAMPLES_DIR, sample=True)

    _write_truth(plan, dup_map, item_meta, supplier_truth, d5, defects, abc_by_item, drift_supplier_id)
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    (TRUTH_DIR / "txn_defects.json").write_text(json.dumps(txn_truth, indent=2, default=str))
    print(f"Transaction-defect truth -> {TRUTH_DIR / 'txn_defects.json'}")
    _summary(plan, demand, item_master, purchase_orders, transactions, cycle_counts,
             dup_map, defects, drift_supplier_id, d5)
    _txn_summary(txn_truth, purchase_orders, transactions)


def _write_truth(plan, dup_map, item_meta, supplier_truth, d5, defects, abc, drift_supplier_id):
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    item_number_to_id = {n: int(m["item_id"]) for n, m in item_meta.items()}
    dup_truth = {str(k): {"records": v["records"], "primary": v["primary"]}
                 for k, v in dup_map.items()}
    box = {n: int(m["box_size"]) for n, m in item_meta.items() if m.get("box_size")}
    payload = {
        "item_number_to_canonical_id": item_number_to_id,
        "duplicate_clusters":          dup_truth,
        "supplier_id_to_canonical":    supplier_truth,
        "d5_supplier":                 d5,
        "d2_drift_supplier":           drift_supplier_id,
        "d2_items":                    sorted(int(i) for i in defects["d2_items"]),
        "d3_box_sizes":                box,
        "d6_items":                    sorted(int(i) for i in defects["d6_items"]),
        "abc_by_item":                 {str(k): v for k, v in abc.items()},
    }
    (TRUTH_DIR / "crosswalks.json").write_text(json.dumps(payload, indent=2))
    print(f"\nGround-truth crosswalks -> {TRUTH_DIR / 'crosswalks.json'}")


def _summary(plan, demand, item_master, po, tx, cc, dup_map, defects, drift_supplier_id, d5):
    print("\n" + "=" * 72)
    print("DATA & DEFECT SUMMARY")
    print("=" * 72)

    wide = demand.pivot(index="item_id", columns="month", values="demand_units").fillna(0).sort_index(axis=1)
    cls = pd.Series({i: classify(wide.loc[i].to_numpy(dtype=float)) for i in wide.index})
    print("\nDemand-segment mix (classified)")
    for seg in C.SEGMENTS:
        print(f"  {seg:<13} {(cls == seg).mean()*100:5.1f}%")

    print("\nPlanted defects")
    n_records = sum(len(v["records"]) for v in dup_map.values())
    print(f"  D1 duplicate clusters      {len(dup_map)} clusters, {n_records} recorded numbers")
    print(f"  D2 stale lead times        {len(defects['d2_items'])} items on {drift_supplier_id} "
          f"({C.D2_DRIFT_START_DAYS}->{C.D2_DRIFT_END_DAYS} days)")
    print(f"  D3 unit-of-measure         {len(defects['d3_items'])} hardware items (box vs each)")
    cc_var = (cc["counted_quantity"] - cc["system_quantity"]).abs() / cc["system_quantity"].replace(0, np.nan)
    print(f"  D4 phantom inventory       {(cc_var > 0.10).mean()*100:.0f}% of counts vary >10%")
    print(f"  D5 supplier records        vendor {d5['canonical']} under {len(d5['spellings'])} "
          f"spellings / {len(set(d5['ids']))} ids")
    miss = item_master[["standard_cost", "current_reorder_point", "primary_supplier_id"]].isna().any(axis=1)
    print(f"  D6 missing fields          {miss.mean()*100:.0f}% of item records")

    # D2 confirmation: actual receipt lead time on the drift supplier, early vs late.
    po2 = po[po["received_date"].notna()].copy()
    po2["order_date"] = pd.to_datetime(po2["order_date"])
    po2["lead"] = (pd.to_datetime(po2["received_date"]) - po2["order_date"]).dt.days
    dpo = po2[po2["supplier_id"] == drift_supplier_id]
    early = dpo[dpo["order_date"] < pd.Timestamp(C.END_DATE) - pd.Timedelta(days=C.D2_DRIFT_MONTHS*30)]
    late = dpo[dpo["order_date"] >= pd.Timestamp(C.END_DATE) - pd.Timedelta(days=270)]
    if len(early) and len(late):
        print(f"\nD2 drift check (drift supplier actual lead time)")
        print(f"  early window median {early['lead'].median():.0f} d   "
              f"recent window median {late['lead'].median():.0f} d")

    print(f"\nRow counts   item_master {len(item_master):,}   POs {len(po):,}"
          f"   transactions {len(tx):,}   cycle_counts {len(cc):,}")
    print("=" * 72 + "\n")


def _txn_summary(txn, po, tx):
    print("\n" + "=" * 72)
    print("TRANSACTION-LEVEL DEFECT SUMMARY (T1-T7)")
    print("=" * 72)
    n_tx, n_po = len(tx), len(po)
    t1_div = sum(1 for r in txn["t1"] if not r["is_oneoff"])
    t1_one = sum(1 for r in txn["t1"] if r["is_oneoff"])
    print(f"  T1 free-text lines     {len(txn['t1'])}  ({t1_div} attributable, {t1_one} genuine one-offs)"
          f"  = {len(txn['t1'])/n_tx*100:.1f}% of ledger lines")
    print(f"  T2 keying errors       {len(txn['t2'])}  = {len(txn['t2'])/n_tx*100:.2f}% of lines")
    print(f"  T3 wrong-item issues   {len(txn['t3'])}  = {len(txn['t3'])/n_tx*100:.2f}% of lines")
    print(f"  T4 unrecorded items    {len(txn['t4'])}  items with chronic write-offs")
    print(f"  T5 batched receipts    {len(txn['t5'])}  = {len(txn['t5'])/max(1,po['received_date'].notna().sum())*100:.0f}% of receipts")
    print(f"  T6 never-closed POs    {len(txn['t6'])}  = {len(txn['t6'])/n_po*100:.1f}% of PO lines")
    print(f"  T7 duplicate postings  {len(txn['t7'])}  = {len(txn['t7'])/n_tx*100:.2f}% of lines")
    # T5 lead-time displacement
    if txn["t5"]:
        disp = [(pd.to_datetime(r["recorded_received_date"]) - pd.to_datetime(r["true_received_date"])).days
                for r in txn["t5"]]
        print(f"  T5 median displacement {np.median(disp):.0f} days (biases computed lead time upward)")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    run()
