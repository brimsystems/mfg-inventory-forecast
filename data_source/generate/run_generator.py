"""Build all nine source datasets, write raw and sample extracts, and record the
ground truth the data-quality pipeline is scored against.

The build runs in the order the shop's records are produced: demand and the
product structure, the masters, the jobs and service orders, the physical
consumption those create, then a day-by-day replay of the ERP's own
replenishment against its books (which is where the purchase orders, the rush
buys, the shortages and the count corrections come from), and finally the
ledger, the counts and the buyer's spreadsheet.

Run:  python -m data_source.generate.run_generator
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config as C
from .checkpoint import classify
from .generators.demand import build_item_plan, build_monthly_demand
from .generators.suppliers import build_suppliers
from .generators.bom import (build_catalog, build_bom, apply_m3_omissions,
                             build_product_builds, explode_builds)
from .generators.item_master import build_item_master
from .generators.production_orders import build_production_orders, apply_delays
from .generators.service_orders import build_service_orders
from .generators.consumption import build_consumption_events
from .generators.replenishment import simulate, draw_drift_rates
from .generators.counterfactual import corrected_inputs, replay_metrics
from .generators.forward_metrics import window_metrics
from .generators.purchase_orders import assemble_purchase_orders
from .generators.inventory_transactions import build_ledger, post_count_adjustments
from .generators.cycle_counts import build_cycle_counts
from .generators.buyer_spreadsheet import build_buyer_spreadsheet

TRUTH_DIR = C.REPO_ROOT / "data_source" / "truth"


def _abc(plan, annual_by_item):
    cost = plan.set_index("item_id")["unit_cost"].to_dict()
    val = {i: annual_by_item.get(i, 0.0) * cost.get(i, 0.0) for i in cost}
    order = sorted(val, key=val.get, reverse=True)
    total = sum(val.values()) or 1.0
    out, cum = {}, 0.0
    for i in order:
        cum += val[i] / total
        out[i] = "A" if cum <= C.ABC_A_CUM else ("B" if cum <= C.ABC_B_CUM else "C")
    return out, val


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
    d = direct.rename(columns={"demand_units": "qty"}).copy()
    is_svc = d["item_id"].isin(service_items)
    d["service"] = np.where(is_svc, (d["qty"] * C.SERVICE_SCALE).round(), 0).astype(int)
    d["manual"] = (d["qty"] - d["service"]).clip(lower=0)
    service = d.loc[d["service"] > 0, ["item_id", "month", "service"]].rename(columns={"service": "qty"})
    manual = d[["item_id", "month", "manual"]].rename(columns={"manual": "qty"})
    return service, manual


def _on_hand_from_tx(tx, item_meta):
    conv = {n: (m["uom_conv"] or 1) for n, m in item_meta.items()}
    sign = {"RECEIPT": 1, "RETURN": 1, "ISSUE": -1, "BACKFLUSH": -1, "SCRAP": -1, "ADJUST": 1}
    t = tx.copy()
    t["signed"] = t["qty"] * t["type"].map(sign).fillna(0)
    stock_uom = {n: m["stock_uom"] for n, m in item_meta.items()}
    box = (t["type"] == "RECEIPT") & (t["uom"] != t["item_number"].map(stock_uom))
    t.loc[box, "signed"] = t.loc[box, "signed"] * t.loc[box, "item_number"].map(conv).fillna(1)
    oh = t.groupby("item_number")["signed"].sum()
    return {n: max(0, int(v)) for n, v in oh.items()}


def run():
    rng = np.random.default_rng(C.RANDOM_SEED)
    print(f"\nBuilding equipment-builder source extracts for {C.START_DATE} through {C.END_DATE}\n")

    # ── Demand and product structure ────────────────────────────────────────
    plan = build_item_plan(rng)
    direct = build_monthly_demand(plan, rng)
    service_items = set(rng.choice(plan["item_id"].to_numpy(),
                                   size=int(round(len(plan) * C.SERVICE_ITEM_SHARE)), replace=False))
    service_demand, manual_demand = _channel_split(direct, service_items)

    products, subs = build_catalog(rng)
    bom_true, prod_subs, sub_components = build_bom(products, subs, plan, rng)
    bom_recorded, omissions, omit_items = apply_m3_omissions(bom_true, plan, rng)
    builds = build_product_builds(products, rng)
    bf_recorded = explode_builds(builds, bom_recorded)
    bf_true = explode_builds(builds, bom_true)

    total = (pd.concat([direct.rename(columns={"demand_units": "qty"})[["item_id", "month", "qty"]],
                        bf_recorded]).groupby(["item_id", "month"], as_index=False)["qty"].sum())
    wide = total.pivot(index="item_id", columns="month", values="qty").fillna(0).sort_index(axis=1)
    annual_by_item = wide.iloc[:, -12:].sum(axis=1).to_dict()

    # ── Masters ─────────────────────────────────────────────────────────────
    print("[1/9] Suppliers          (ERP)")
    suppliers, supplier_truth, sup_frag, drift_supplier_id = build_suppliers(rng)
    print("[2/9] Item master        (ERP)")
    item_master, dup_map, item_meta, defects = build_item_master(
        plan, suppliers, annual_by_item, drift_supplier_id, sup_frag, rng)
    abc_by_item, value_by_item = _abc(plan, annual_by_item)   # after the M5 per-unit costing
    print("[3/9] Bill of materials  (ERP)")
    bom_ds = _bom_dataset(bom_recorded, item_meta, dup_map)

    # ── Jobs, service orders and physical consumption ───────────────────────
    print("[4/9] Production orders  (ERP)")
    production_orders = build_production_orders(builds, products, rng)
    print("[5/9] Service orders     (ERP)")
    service_orders = build_service_orders(service_demand, item_meta, dup_map, plan, rng)
    events = build_consumption_events(production_orders, bom_true, bom_recorded, service_orders,
                                      manual_demand, item_meta, dup_map, omit_items,
                                      np.random.default_rng(C.RANDOM_SEED + 21))
    # the same events with the bills never fixed, for the counterfactual that runs
    # the old masters through the forward window
    events_dirty = build_consumption_events(production_orders, bom_true, bom_recorded, service_orders,
                                            manual_demand, item_meta, dup_map, omit_items,
                                            np.random.default_rng(C.RANDOM_SEED + 21), fix_from=C.END_DATE)

    # the purchasing manager tracks the highest-value A items by hand
    primary = {}
    for num, m in item_meta.items():
        if not m["dead"]:
            iid = m["item_id"]
            primary[iid] = dup_map[iid]["primary"] if iid in dup_map else primary.get(iid, num)
    top = sorted((i for i in value_by_item if abc_by_item.get(i) == "A" and i in primary),
                 key=lambda i: value_by_item[i], reverse=True)[:C.BUYER_SPREADSHEET_ITEMS]
    buyer_nums = {primary[i] for i in top}

    # ── Replenishment replay: purchase orders, rushes, shortages, counts ────
    print("[6/9] Purchase orders    (ERP) - replaying replenishment, this takes a minute")
    drift_rate = draw_drift_rates(item_meta, rng)
    cost_by_item = plan.set_index("item_id")["unit_cost"].to_dict()

    # the corrected masters the remediation loaded, which the shop runs on from
    # the forward start; the demand model's monthly reorder points if it has them
    ev_c, im_c, meta_c, base_lead, _ = corrected_inputs(events, item_master, item_meta, dup_map, drift_rate,
                                                        drift_supplier_id, abc_by_item, rng)
    imc = im_c.set_index("item_number")
    forward = {"start": C.FORWARD_START, "never_closed": 0.0,
               "lead": imc["master_lead_time_days"].to_dict(),
               "rop": imc["reorder_point"].fillna(0).to_dict(),
               "ss": imc["safety_stock"].fillna(0).to_dict(), "schedule": None,
               "abc": {int(k): v for k, v in abc_by_item.items()}}
    schedule_path = C.REPO_ROOT / "ml" / "data" / "policy" / "rop_schedule.json"
    schedule = None
    if schedule_path.exists():
        sched = json.loads(schedule_path.read_text())
        schedule = {num: [(date.fromisoformat(e[0]), e[1], e[2], e[3], e[4]) for e in entries]
                    for num, entries in sched["items"].items()}
        forward["schedule"] = {n: [(d_, r_, s_) for (d_, r_, s_, *_x) in e] for n, e in schedule.items()}
        print(f"      Forward window: the demand model's reorder points ({len(schedule):,} items, {len(sched['months'])} months)")
    else:
        print("      Forward window: the remediation's recomputed reorder points (no model schedule yet)")
    sim = simulate(events, item_master, item_meta, dup_map, plan, drift_supplier_id, sup_frag,
                   buyer_nums, omit_items, rng, drift_rate=drift_rate,
                   forward=forward)
    production_orders = apply_delays(production_orders, sim["job_delays"])
    purchase_orders, po_truth = assemble_purchase_orders(sim["po_lines"], suppliers, item_master, rng)

    # the same rule on the corrected masters over the whole history (the audit's 2025 counterfactual)
    print("      Counterfactual replay on corrected masters")
    sim_c = simulate(ev_c, im_c, meta_c, {}, plan, drift_supplier_id, sup_frag, buyer_nums, [],
                     np.random.default_rng(C.RANDOM_SEED + 11), drift_rate=drift_rate, never_closed=0.0,
                     actual_base=base_lead)
    counterfactual = {
        "year": C.MODEL_SPAN_END.year,
        "as_is": replay_metrics(sim, cost_by_item, primary, production_orders, C.MODEL_SPAN_END.year),
        "corrected": replay_metrics(sim_c, cost_by_item, primary, production_orders, C.MODEL_SPAN_END.year),
    }

    # the forward window three ways over the same demand: as the shop ran it, with
    # nothing fixed, and with the corrected masters but no model
    print("      Forward-window counterfactuals (nothing fixed; corrected masters without the model)")
    sim_dirty = simulate(events_dirty, item_master, item_meta, dup_map, plan, drift_supplier_id, sup_frag,
                         buyer_nums, omit_items, np.random.default_rng(C.RANDOM_SEED + 12), drift_rate=drift_rate,
                         fix_from=C.END_DATE)
    fwd_rule = dict(forward); fwd_rule["schedule"] = None
    sim_rule = simulate(events, item_master, item_meta, dup_map, plan, drift_supplier_id, sup_frag,
                        buyer_nums, omit_items, np.random.default_rng(C.RANDOM_SEED + 13), drift_rate=drift_rate,
                        forward=fwd_rule)
    prod_for_windows = apply_delays(production_orders.drop(columns=["hold_reason", "delay_days"]), sim["job_delays"])
    forward_results = _forward_results(sim, sim_dirty, sim_rule, schedule, item_master, imc, cost_by_item,
                                       abc_by_item, primary, prod_for_windows, total, wide)

    # ── Counts, ledger, spreadsheet ─────────────────────────────────────────
    print("[7/9] Inventory ledger   (ERP)")
    transactions, tx_truth = build_ledger(events, purchase_orders, sim["adjustments"],
                                          sim["job_delays"], item_master, item_meta, dup_map, plan,
                                          omit_items, rng)
    transactions = _add_stray_dead_txns(transactions, defects, rng)
    # the count program reads the ledger balance and posts what it finds
    print("[8/9] Cycle counts       (WMS)")
    unreliable = _unreliable_nums(item_meta, defects, omit_items, dup_map)
    cycle_counts, count_adj = build_cycle_counts(sim, transactions, item_meta, dup_map, unreliable, abc_by_item, rng)
    transactions = post_count_adjustments(transactions, count_adj, item_meta, rng)
    on_hand = _on_hand_from_tx(transactions, item_meta)
    print("[9/9] Buyer spreadsheet  (Purchasing)")
    buyer_spreadsheet, buyer_truth = build_buyer_spreadsheet(
        item_master, item_meta, abc_by_item, on_hand, plan, rng, preselected=buyer_nums)

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

    sub_omitted = set(omissions.loc[omissions["parent_type"] == "subassembly", "parent"])
    affected_products = set(omissions.loc[omissions["parent_type"] == "product", "parent"])
    for p, slist in prod_subs.items():
        if any(s in sub_omitted for s in slist):
            affected_products.add(p)
    _write_truth(dup_map, item_meta, supplier_truth, sup_frag, defects, abc_by_item, drift_supplier_id,
                 omit_items, po_truth, tx_truth, buyer_truth, bf_true, sorted(affected_products),
                 C.N_PRODUCTS, sim, primary, sorted(buyer_nums))
    (TRUTH_DIR / "counterfactual.json").write_text(json.dumps(counterfactual, indent=2))
    (TRUTH_DIR / "forward_results.json").write_text(json.dumps(forward_results, indent=2, default=float))
    _print_forward(forward_results)
    a, c = counterfactual["as_is"], counterfactual["corrected"]
    print(f"Counterfactual {counterfactual['year']}: inventory ${a['inventory_value_avg']:,.0f} -> "
          f"${c['inventory_value_avg']:,.0f}  | purchases ${a['purchases']:,.0f} -> ${c['purchases']:,.0f}  | "
          f"rush freight ${a['rush_freight']:,.0f} -> ${c['rush_freight']:,.0f}  | shortages "
          f"{a['shortage_episodes']:,} -> {c['shortage_episodes']:,}  | jobs delayed {a['jobs_delayed']:,} -> "
          f"{c['jobs_delayed']:,}")
    _summary(total, item_master, purchase_orders, transactions, cycle_counts, production_orders,
             service_orders, defects, sim)


def _half(year, half):
    return (date(year, 1, 1), date(year, 6, 30)) if half == 1 else (date(year, 7, 1), date(year, 12, 31))


def _forward_results(sim, sim_dirty, sim_rule, schedule, item_master, imc, cost_by_item, abc_by_item,
                     primary, production_orders, total, wide):
    """The metric set over 1H25, 2H25 and the forward window, the latter three ways."""
    iid_of = {n: i for i, n in primary.items()}
    seg_by_iid = {int(i): classify(wide.loc[i].to_numpy(dtype=float)) for i in wide.index}
    imx = item_master.set_index("item_number")
    ss_dirty = {}
    for n, i in iid_of.items():
        v = imx["safety_stock"].get(n, 0.0)
        ss_dirty[i] = 0.0 if pd.isna(v) else float(v)
    ss_rule = {i: float(imc["safety_stock"].get(n, 0.0) or 0.0) for n, i in iid_of.items() if n in imc.index}

    def ss_model(start, end):
        if not schedule:
            return ss_rule
        out = {}
        for n, entries in schedule.items():
            i = iid_of.get(n)
            if i is None:
                continue
            vals = [e[2] for e in entries if start <= e[0] <= end]
            out[i] = float(np.mean(vals)) if vals else ss_rule.get(i, 0.0)
        return out

    y = C.MODEL_SPAN_END.year
    windows = {"1H25": _half(y, 1), "2H25": _half(y, 2)}
    f0 = C.FORWARD_START
    fw = {"1H26": (f0, C.END_DATE)}
    m3 = date(f0.year + (f0.month + 2) // 12, (f0.month + 2) % 12 + 1, 1)
    fw["1H26_months_1_3"] = (f0, m3 - timedelta(days=1))
    fw["1H26_months_4_6"] = (m3, C.END_DATE)

    res = {"as_recorded": {}, "forward": {}, "schedule_used": bool(schedule)}
    for k, (a, b) in windows.items():
        res["as_recorded"][k] = window_metrics(sim, a, b, cost_by_item, abc_by_item, primary, production_orders,
                                               ss_by_item=ss_dirty)
    for k, (a, b) in fw.items():
        res["forward"][k] = {
            "model" if schedule else "clean_rule": window_metrics(sim, a, b, cost_by_item, abc_by_item, primary,
                                                                  production_orders, ss_by_item=ss_model(a, b),
                                                                  schedule=schedule, segment_by_item=seg_by_iid),
            "dirty": window_metrics(sim_dirty, a, b, cost_by_item, abc_by_item, primary, production_orders,
                                    ss_by_item=ss_dirty),
        }
        if schedule:
            res["forward"][k]["clean_rule"] = window_metrics(sim_rule, a, b, cost_by_item, abc_by_item, primary,
                                                             production_orders, ss_by_item=ss_rule)
    return res


def _print_forward(fr):
    M = lambda x: f"${x:,.0f}"
    print("\nForward window and the two halves of last year")
    for k, m in fr["as_recorded"].items():
        print(f"  {k:<18} fill {m['fill_rate']*100:5.1f}% | stockouts {m['stockout_episodes']:4d} ({m['stockout_days']:5d} days) | "
              f"jobs held {m['jobs_delayed']:4d} | rush {m['rush_lines']:4d} {M(m['rush_spend']):>9} | inventory {M(m['avg_inventory_value']):>11} "
              f"({(m['days_of_supply'] or 0):.0f} days) | lines {m['order_lines']:5d} | purchases {M(m['purchases'])}")
    for k, variants in fr["forward"].items():
        for v, m in variants.items():
            print(f"  {k:<18} {v:<11} fill {m['fill_rate']*100:5.1f}% | stockouts {m['stockout_episodes']:4d} ({m['stockout_days']:5d} days) | "
                  f"jobs held {m['jobs_delayed']:4d} | rush {m['rush_lines']:4d} {M(m['rush_spend']):>9} | inventory {M(m['avg_inventory_value']):>11} "
                  f"({(m['days_of_supply'] or 0):.0f} days) | lines {m['order_lines']:5d} | purchases {M(m['purchases'])}")


def _bom_dataset(bom_recorded, item_meta, dup_map):
    primary = {}
    for num, m in item_meta.items():
        if not m["dead"]:
            iid = m["item_id"]
            primary[iid] = dup_map[iid]["primary"] if iid in dup_map else primary.get(iid, num)
    rows = []
    for r in bom_recorded.itertuples(index=False):
        comp = primary.get(int(r.component_item)) if r.component_type == "purchased" else r.component_item
        if comp is None:
            continue
        rows.append({"product_number": r.parent, "component_item": comp, "qty_per": r.qty_per,
                     "uom": "EA", "effective_date": C.START_DATE.isoformat()})
    return pd.DataFrame(rows)


def _add_stray_dead_txns(tx, defects, rng):
    rows, seqbase = [], int(tx["txn_id"].str[3:].astype(int).max()) + 1
    for meta in defects["dead_meta"]:
        if meta["stray"]:
            d = C.MODEL_SPAN_END - timedelta(days=int(rng.integers(30, 300)))
            rows.append({"txn_id": f"TX-{seqbase:07d}", "item_number": meta["item_number"],
                         "txn_date": d.isoformat(), "txn_time": "09:00", "type": "ISSUE",
                         "qty": int(rng.integers(1, 4)), "uom": "EA", "job_id": None, "reason_code": "MANUAL",
                         "location": "CRIB", "user_id": rng.choice(C.SHARED_LOGINS)})
            seqbase += 1
    return pd.concat([tx, pd.DataFrame(rows)], ignore_index=True) if rows else tx


def _unreliable_nums(item_meta, defects, omit_items, dup_map):
    nums, omit = set(), set(int(i) for i in omit_items)
    for num, m in item_meta.items():
        if not m["dead"] and (m["item_id"] in omit or (m["uom_conv"] and m["uom_conv"] > 1)):
            nums.add(num)
    for c in dup_map.values():
        nums.update(c["records"])
    return nums


def _write_truth(dup_map, item_meta, supplier_truth, sup_frag, defects, abc, drift_supplier_id, omit_items,
                 po_truth, tx_truth, buyer_truth, bf_true, affected_products, n_products, sim, primary, buyer_nums):
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "duplicate_clusters":       {str(k): {"records": v["records"], "primary": v["primary"]} for k, v in dup_map.items()},
        "supplier_id_to_canonical": supplier_truth,
        "supplier_fragments":       sup_frag["records"],
        "m2_drift_supplier":        drift_supplier_id,
        "m2_drift_items":           sorted(int(i) for i in defects["m2_drift_items"]),
        "m2_sharp_items":           sorted(int(i) for i in defects["m2_sharp_items"]),
        "m3_omitted_items":         [int(i) for i in omit_items],
        "m3_affected_products":     affected_products,
        "n_products":               n_products,
        "m5_items":                 sorted(int(i) for i in defects["m5_items"]),
        "m5_conversions":           {n: int(m["uom_conv"]) for n, m in item_meta.items() if m["uom_conv"] and m["uom_conv"] > 1},
        "m7_blank_items":           sorted(int(i) for i in defects["m7_blank"]),
        "abc_by_item":              {str(k): v for k, v in abc.items()},
        "buyer_tracked_items":      buyer_nums,
    }
    (TRUTH_DIR / "crosswalks.json").write_text(json.dumps(payload, indent=2))
    (TRUTH_DIR / "po_defects.json").write_text(json.dumps(po_truth, indent=2, default=str))
    (TRUTH_DIR / "txn_defects.json").write_text(json.dumps({k: v for k, v in tx_truth.items() if k != "t1_events"},
                                                           indent=2, default=str))
    pd.DataFrame(tx_truth["t1_events"]).to_csv(TRUTH_DIR / "unrecorded_events.csv", index=False)
    (TRUTH_DIR / "buyer_reconciliation.json").write_text(json.dumps(buyer_truth, indent=2, default=str))
    bf_true.to_csv(TRUTH_DIR / "true_backflush.csv", index=False)
    sim["shortages"].to_csv(TRUTH_DIR / "shortages.csv", index=False)
    sim["rushes"].to_csv(TRUTH_DIR / "rushes.csv", index=False)
    (TRUTH_DIR / "job_delays.json").write_text(json.dumps(sim["job_delays"], indent=2))
    phys_end = {primary[iid]: float(series[-1]) for iid, series in sim["physical"].items() if primary.get(iid)}
    (TRUTH_DIR / "physical_on_hand_end.json").write_text(json.dumps(phys_end, indent=2))
    print(f"\nGround-truth -> {TRUTH_DIR}")


def _summary(total, item_master, po, tx, cc, prod, svc, defects, sim):
    print("\n" + "=" * 72)
    print("DATA & ERROR SUMMARY")
    print("=" * 72)
    wide = total.pivot(index="item_id", columns="month", values="qty").fillna(0).sort_index(axis=1)
    cls = pd.Series({i: classify(wide.loc[i].to_numpy(dtype=float)) for i in wide.index})
    print("\nDemand-segment mix (classified)")
    for seg in C.SEGMENTS:
        print(f"  {seg:<13} {(cls == seg).mean()*100:5.1f}%")
    n_dead = len(defects["dead_meta"])
    print(f"\nItem master {len(item_master):,} records; dead {n_dead:,} ({n_dead/len(item_master)*100:.0f}%)")
    print(f"Rows  POs {len(po):,}  transactions {len(tx):,}  production {len(prod):,}  "
          f"service {len(svc):,}  cycle_counts {len(cc):,}")
    rush = po[po["rush"]]
    n_sh = len(sim["shortages"])
    print(f"\nReplay  rush lines {len(rush):,} ({len(rush)/max(1,len(po))*100:.1f}% of lines), "
          f"freight ${rush['freight'].sum():,.0f}  | shortages {n_sh:,} on "
          f"{sim['shortages']['item_id'].nunique() if n_sh else 0} items  | jobs delayed "
          f"{sum(1 for v in sim['job_delays'].values() if v > 0):,}")
    if n_sh:
        print("  shortage causes:", sim["shortages"]["cause"].value_counts().to_dict())
    if len(sim["rushes"]):
        print("  rush causes:    ", sim["rushes"]["cause"].value_counts().to_dict())
    print("=" * 72 + "\n")


if __name__ == "__main__":
    run()
