"""Build all nine source datasets, write raw and sample extracts, and record the
ground-truth crosswalks the data-quality pipeline is scored against.

Run:  python -m data_source.generate.run_generator
"""
from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config as C
from .checkpoint import classify
from .generators.demand import build_item_plan, build_monthly_demand
from .generators.suppliers import build_suppliers
from .generators.bom import (build_catalog, build_bom, apply_m3_omissions,
                             build_product_builds, explode_builds, effective_component_map)
from .generators.item_master import build_item_master
from .generators.production_orders import build_production_orders
from .generators.service_orders import build_service_orders
from .generators.purchase_orders import build_purchase_orders
from .generators.inventory_transactions import build_inventory_transactions
from .generators.cycle_counts import build_cycle_counts
from .generators.buyer_spreadsheet import build_buyer_spreadsheet

TRUTH_DIR = C.REPO_ROOT / "data_source" / "truth"


def _abc(plan, annual_by_item) -> dict:
    cost = plan.set_index("item_id")["unit_cost"].to_dict()
    val = {i: annual_by_item.get(i, 0.0) * cost.get(i, 0.0) for i in cost}
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
    fp = out_dir / f"{name}{'_sample' if sample else ''}.csv"
    df.to_csv(fp, index=False)
    kb = fp.stat().st_size / 1024
    size = f"{kb/1024:.1f} MB" if kb > 1024 else f"{kb:.0f} KB"
    print(f"  [{system:>10}]  {name:<22} {len(df):>9,} rows   {size}")


def _channel_split(direct, service_items):
    """Partition the item-level engine demand into a small service channel (for
    service-relevant items only) and a manual channel (the remainder)."""
    d = direct.rename(columns={"demand_units": "qty"}).copy()
    is_svc = d["item_id"].isin(service_items)
    d["service"] = np.where(is_svc, (d["qty"] * C.SERVICE_SCALE).round(), 0).astype(int)
    d["manual"] = (d["qty"] - d["service"]).clip(lower=0)
    service = d.loc[d["service"] > 0, ["item_id", "month", "service"]].rename(columns={"service": "qty"})
    manual = d[["item_id", "month", "manual"]].rename(columns={"manual": "qty"})
    return service, manual


def _on_hand_from_tx(tx, item_meta):
    conv = {n: (m["uom_conv"] or 1) for n, m in item_meta.items()}
    t = tx.copy()
    sign = {"RECEIPT": 1, "RETURN": 1, "ISSUE": -1, "BACKFLUSH": -1, "SCRAP": -1, "ADJUST": 1}
    t["signed"] = t.apply(lambda r: r["qty"] * sign.get(r["type"], 0)
                          * (conv.get(r["item_number"], 1) if r["type"] == "RECEIPT" and r["uom"] == "BOX" else 1),
                          axis=1)
    oh = t.groupby("item_number")["signed"].sum()
    return {n: max(0, int(v)) for n, v in oh.items()}


def run():
    rng = np.random.default_rng(C.RANDOM_SEED)
    print(f"\nBuilding equipment-builder source extracts for {C.START_DATE} through {C.END_DATE}\n")

    # ── Demand and consumption structure ────────────────────────────────────
    plan = build_item_plan(rng)
    direct = build_monthly_demand(plan, rng)          # item-level service + manual channels
    service_items = set(rng.choice(plan["item_id"].to_numpy(),
                                   size=int(round(len(plan) * C.SERVICE_ITEM_SHARE)), replace=False))
    service_demand, manual_demand = _channel_split(direct, service_items)

    products, subs = build_catalog(rng)
    bom_true, prod_subs, sub_components = build_bom(products, subs, plan, rng)
    bom_recorded, omissions, omit_items = apply_m3_omissions(bom_true, plan, rng)
    builds = build_product_builds(products, rng)
    bf_recorded = explode_builds(builds, bom_recorded)
    bf_true = explode_builds(builds, bom_true)
    prod_map = effective_component_map(bom_recorded)

    # unrecorded (omitted) backflush = true minus recorded, positive part
    merged = bf_true.merge(bf_recorded, on=["item_id", "month"], how="left",
                           suffixes=("_true", "_rec")).fillna({"qty_rec": 0})
    merged["qty"] = (merged["qty_true"] - merged["qty_rec"]).clip(lower=0).astype(int)
    omitted_backflush = merged[merged["qty"] > 0][["item_id", "month", "qty"]]

    # total recorded consumption drives replenishment and ABC
    total = (pd.concat([direct.rename(columns={"demand_units": "qty"})[["item_id", "month", "qty"]],
                        bf_recorded]).groupby(["item_id", "month"], as_index=False)["qty"].sum())
    wide = total.pivot(index="item_id", columns="month", values="qty").fillna(0).sort_index(axis=1)
    annual_by_item = wide.iloc[:, -12:].sum(axis=1).to_dict()
    abc_by_item = _abc(plan, annual_by_item)

    # ── Masters ─────────────────────────────────────────────────────────────
    print("[1/9] Suppliers          (ERP)")
    suppliers, supplier_truth, sup_frag, drift_supplier_id = build_suppliers(rng)
    print("[2/9] Item master        (ERP)")
    item_master, dup_map, item_meta, defects = build_item_master(
        plan, suppliers, annual_by_item, drift_supplier_id, sup_frag, rng)

    # ── BOM dataset (recorded, with omissions) ──────────────────────────────
    print("[3/9] Bill of materials  (ERP)")
    bom_ds = _bom_dataset(bom_recorded, item_meta, dup_map)

    # ── Orders and ledger ───────────────────────────────────────────────────
    print("[4/9] Production orders  (ERP)")
    production_orders = build_production_orders(builds, products, rng)
    print("[5/9] Service orders     (ERP)")
    service_orders = build_service_orders(service_demand, item_meta, dup_map, plan, rng)
    print("[6/9] Purchase orders    (ERP)")
    purchase_orders, po_truth = build_purchase_orders(
        total, item_master, item_meta, dup_map, suppliers, sup_frag, drift_supplier_id, plan, rng)
    print("[7/9] Inventory ledger   (ERP) - this step takes a moment")
    transactions, tx_truth = build_inventory_transactions(
        production_orders, prod_map, item_master, item_meta, dup_map, service_orders,
        manual_demand, purchase_orders, omitted_backflush, plan, rng)
    transactions = _add_stray_dead_txns(transactions, defects, rng)

    on_hand = _on_hand_from_tx(transactions, item_meta)
    unreliable = _unreliable_nums(item_meta, defects, omit_items, dup_map)

    print("[8/9] Cycle counts       (WMS)")
    cycle_counts = build_cycle_counts(item_master, item_meta, on_hand, unreliable, rng)
    print("[9/9] Buyer spreadsheet  (Purchasing)")
    buyer_spreadsheet, buyer_truth = build_buyer_spreadsheet(
        item_master, item_meta, abc_by_item, on_hand, plan, rng)

    tables = {
        "item_master":            item_master,
        "supplier_master":        suppliers,
        "bill_of_materials":      bom_ds,
        "production_orders":      production_orders,
        "service_orders":         service_orders,
        "purchase_orders":        purchase_orders,
        "inventory_transactions": transactions,
        "cycle_counts":           cycle_counts,
        "buyer_spreadsheet":      buyer_spreadsheet,
    }
    print(f"\nFull extracts -> {C.RAW_DIR}")
    for name, df in tables.items():
        _save(df, name, C.RAW_DIR)
    print(f"\nSample extracts ({C.SAMPLE_SIZE} rows) -> {C.SAMPLES_DIR}")
    for name, df in tables.items():
        _save(df.head(C.SAMPLE_SIZE), name, C.SAMPLES_DIR, sample=True)

    # top-level products affected by a BOM omission (direct or via a subassembly)
    sub_omitted = set(omissions.loc[omissions["parent_type"] == "subassembly", "parent"])
    affected_products = set(omissions.loc[omissions["parent_type"] == "product", "parent"])
    for p, slist in prod_subs.items():
        if any(s in sub_omitted for s in slist):
            affected_products.add(p)

    _write_truth(plan, dup_map, item_meta, supplier_truth, sup_frag, defects, abc_by_item,
                 drift_supplier_id, omit_items, omissions, po_truth, tx_truth, buyer_truth, bf_true,
                 sorted(affected_products), C.N_PRODUCTS)
    _summary(plan, total, item_master, purchase_orders, transactions, cycle_counts,
             production_orders, service_orders, dup_map, defects, abc_by_item)


def _bom_dataset(bom_recorded, item_meta, dup_map):
    """Render the recorded BOM with component item numbers (primary record)."""
    primary = {}
    for num, m in item_meta.items():
        if not m["dead"]:
            iid = m["item_id"]
            if iid in dup_map:
                primary[iid] = dup_map[iid]["primary"]
            else:
                primary.setdefault(iid, num)
    rows = []
    for r in bom_recorded.itertuples(index=False):
        if r.component_type == "purchased":
            comp = primary.get(int(r.component_item))
            if comp is None:
                continue
        else:
            comp = r.component_item
        rows.append({"product_number": r.parent, "component_item": comp,
                     "qty_per": r.qty_per, "uom": "EA",
                     "effective_date": C.START_DATE.isoformat()})
    return pd.DataFrame(rows)


def _add_stray_dead_txns(tx, defects, rng):
    rows = []
    seqbase = int(tx["txn_id"].str[3:].astype(int).max()) + 1
    for meta in defects["dead_meta"]:
        if meta["stray"]:
            d = C.MODEL_SPAN_END - timedelta(days=int(rng.integers(30, 300)))
            rows.append({"txn_id": f"TX-{seqbase:07d}", "item_number": meta["item_number"],
                         "txn_date": d.isoformat(), "txn_time": "09:00", "type": "ISSUE",
                         "qty": int(rng.integers(1, 4)), "uom": "EA", "job_id": None,
                         "reason_code": "MANUAL", "location": "CRIB",
                         "user_id": rng.choice(C.SHARED_LOGINS)})
            seqbase += 1
    if rows:
        tx = pd.concat([tx, pd.DataFrame(rows)], ignore_index=True)
    return tx


def _unreliable_nums(item_meta, defects, omit_items, dup_map):
    nums = set()
    omit = set(int(i) for i in omit_items)
    for num, m in item_meta.items():
        if m["dead"]:
            continue
        if m["item_id"] in omit or (m["uom_conv"] and m["uom_conv"] > 1):
            nums.add(num)
    for c in dup_map.values():
        nums.update(c["records"])
    return nums


def _write_truth(plan, dup_map, item_meta, supplier_truth, sup_frag, defects, abc,
                 drift_supplier_id, omit_items, omissions, po_truth, tx_truth, buyer_truth, bf_true,
                 affected_products, n_products):
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    dup_truth = {str(k): {"records": v["records"], "primary": v["primary"]}
                 for k, v in dup_map.items()}
    payload = {
        "duplicate_clusters":       dup_truth,
        "supplier_id_to_canonical": supplier_truth,
        "supplier_fragments":       sup_frag["records"],
        "m2_drift_supplier":        drift_supplier_id,
        "m2_drift_items":           sorted(int(i) for i in defects["m2_drift_items"]),
        "m2_sharp_items":           sorted(int(i) for i in defects["m2_sharp_items"]),
        "m3_omitted_items":         [int(i) for i in omit_items],
        "m3_affected_products":     affected_products,
        "n_products":               n_products,
        "m5_items":                 sorted(int(i) for i in defects["m5_items"]),
        "m7_blank_items":           sorted(int(i) for i in defects["m7_blank"]),
        "abc_by_item":              {str(k): v for k, v in abc.items()},
    }
    (TRUTH_DIR / "crosswalks.json").write_text(json.dumps(payload, indent=2))
    (TRUTH_DIR / "po_defects.json").write_text(json.dumps(po_truth, indent=2, default=str))
    (TRUTH_DIR / "txn_defects.json").write_text(json.dumps(tx_truth, indent=2, default=str))
    (TRUTH_DIR / "buyer_reconciliation.json").write_text(json.dumps(buyer_truth, indent=2, default=str))
    bf_true.to_csv(TRUTH_DIR / "true_backflush.csv", index=False)
    print(f"\nGround-truth -> {TRUTH_DIR}")


def _summary(plan, total, item_master, po, tx, cc, prod, svc, dup_map, defects, abc):
    print("\n" + "=" * 72)
    print("DATA & DEFECT SUMMARY")
    print("=" * 72)
    wide = total.pivot(index="item_id", columns="month", values="qty").fillna(0).sort_index(axis=1)
    cls = pd.Series({i: classify(wide.loc[i].to_numpy(dtype=float)) for i in wide.index})
    print("\nDemand-segment mix (classified)")
    for seg in C.SEGMENTS:
        print(f"  {seg:<13} {(cls == seg).mean()*100:5.1f}%")
    n_dead = sum(1 for m in defects["dead_meta"])
    print(f"\nItem master {len(item_master):,} records; dead {n_dead:,} "
          f"({n_dead/len(item_master)*100:.0f}%)  live {len(item_master)-n_dead:,}")
    print(f"Rows  POs {len(po):,}  transactions {len(tx):,}  production {len(prod):,}  "
          f"service {len(svc):,}  cycle_counts {len(cc):,}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    run()
