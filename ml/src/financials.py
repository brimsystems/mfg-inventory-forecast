"""Financial impact of the data errors, measured from the records.

Every figure here is computed from the ERP extracts, the remediation reference
tables and the ground truth, for the last full calendar year unless stated. Each
is tagged with its basis:

  measured    read directly from the records (a rush line's freight, a count
              correction's value, an open line's balance)
  simulated   from the counterfactual replay: the same reorder rule on the
              corrected masters over the same demand (working capital,
              shortages and delays avoided)
  estimated   needs an assumption that is stated next to it

and with what kind of money it is:

  incurred      cash the shop actually spent because of the error (expedite
                freight and price premiums); stopping it is a recurring saving
  working       inventory held above or below what the same rule would hold on
  capital       corrected data; the net is the cash release, and it is small
  misstatement  the books were wrong and were later corrected (write-offs at
                the count, product cost understated); no cash comes back
  exposure      a wrong number that costs nothing until someone acts on it (a
                reorder point on a dead item); reported with how often it was
                acted on

Run:  python -m ml.src.financials
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from .resolution import REPO

RAW = REPO / "data_source" / "raw"
REM = RAW / "remediation"
TRUTH = REPO / "data_source" / "truth"
OUT = REPO / "ml" / "data" / "financials"

YEAR = 2025
AS_OF = date(2026, 3, 31)
PHANTOM_DAYS = 90                      # an open line older than this is treated as never arriving
GENERIC_REASONS = {"", "nan", "ADJ", "VAR", "MISC"}     # a COUNT reason is traceable, to the count
GENERIC_ITEM_CODES = {"NONSTOCK", "MISC", "SHOPSUPPLY"}
SIGN = {"RECEIPT": 1, "RETURN": 1, "ISSUE": -1, "BACKFLUSH": -1, "SCRAP": -1, "ADJUST": 1}
ATTRIBUTABLE_CAUSES = ["stale_lead_time", "unrecorded_consumption", "phantom_on_order"]


def _year(s):
    return pd.to_datetime(s).dt.year == YEAR


def run():
    im = pd.read_csv(RAW / "erp" / "item_master.csv", low_memory=False)
    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv", low_memory=False)
    po = pd.read_csv(RAW / "erp" / "purchase_orders.csv", low_memory=False)
    cc = pd.read_csv(RAW / "wms" / "cycle_counts.csv")
    prod = pd.read_csv(RAW / "erp" / "production_orders.csv", low_memory=False)
    cross = json.loads((TRUTH / "crosswalks.json").read_text())
    txd = json.loads((TRUTH / "txn_defects.json").read_text())
    pod = json.loads((TRUTH / "po_defects.json").read_text())
    cf = json.loads((TRUTH / "counterfactual.json").read_text())
    rushes = pd.read_csv(TRUTH / "rushes.csv")
    shortages = pd.read_csv(TRUTH / "shortages.csv")
    unrec = pd.read_csv(TRUTH / "unrecorded_events.csv")
    lead = pd.read_csv(REM / "lead_time_computation.csv")
    params = pd.read_csv(REM / "parameter_recommendations.csv")
    dead_disp = pd.read_csv(REM / "dead_item_dispositions.csv")
    ftattr = pd.read_csv(REM / "free_text_attribution.csv")

    # ── item groups and costs ────────────────────────────────────────────────
    dead = set(dead_disp["item_number"])
    live = im[~im["item_number"].isin(dead)].copy()
    class_median = im.groupby("item_class")["standard_cost"].median()
    cost = im.set_index("item_number")["standard_cost"]
    blank_cost = set(cost[cost.isna()].index) & set(live["item_number"])
    cost = cost.fillna(im.set_index("item_number")["item_class"].map(class_median)).fillna(cost.median())

    member_of = {r: c["primary"] for c in cross["duplicate_clusters"].values() for r in c["records"]}
    dup_members = set(member_of)
    conv = cross["m5_conversions"]
    t1_items = {r["item_number"] for r in txd["t1"]}
    ld = lead.set_index("item_number")
    stale_lead = set(ld.index[(ld["median_actual"] - ld["master_lead_time"]).abs() > 3])
    old = params["old_reorder_point"].fillna(0)
    delta = (params["new_reorder_point"] - old).abs()
    stale_rop = set(params.loc[(delta >= 5) & (delta >= 0.30 * old.clip(lower=1)), "item_number"])
    t6_ids = {r["txn_id"]: r for r in txd["t6"]}
    t7_ids = {r["txn_id"]: r for r in txd["t7"]}
    t8_ids = {r["txn_id"] for r in txd["t8"]}
    t6_items = {r["recorded_item_number"] for r in txd["t6"]} | {r["true_item_number"] for r in txd["t6"]}
    t7_items = set(tx.loc[tx["txn_id"].isin(t7_ids), "item_number"])
    t8_items = set(tx.loc[tx["txn_id"].isin(t8_ids), "item_number"])
    t3_items = {r["true_item_number"] for r in pod["t3"] if r.get("is_stocked")}

    def group(n):
        # one group per item, the master error first, so a dollar lands once
        if n in dup_members: return "Duplicate Item Records"
        if n in conv: return "UOM Mismatch"
        if n in t1_items: return "Unrecorded Consumption"
        if n in stale_rop: return "Stale Reorder Points"
        if n in stale_lead: return "Stale Lead Times"
        if n in t7_items: return "Quantity and Unit Errors"
        if n in t6_items: return "Wrong References"
        if n in t8_items: return "Duplicate Postings"
        return "No master error"

    out = {"year": YEAR, "as_of": AS_OF.isoformat()}

    # ── incurred: expedites, by the cause the replay recorded ───────────────
    r = rushes[_year(rushes["date"])].copy()
    r["total"] = r["freight"] + r["premium_usd"]
    by_cause = r.groupby("cause").agg(lines=("total", "size"), freight=("freight", "sum"),
                                      premium=("premium_usd", "sum"), total=("total", "sum"))
    attributable = by_cause.reindex(ATTRIBUTABLE_CAUSES).fillna(0)
    po_rush = po[po["rush"] & _year(po["order_date"])]
    out["expedites"] = {
        "basis": "measured",
        "rush_lines": int(len(po_rush)),
        "freight": float(po_rush["freight"].sum()),
        "premium": float(r["premium_usd"].sum()),
        "total": float(r["total"].sum()),
        "by_cause": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in by_cause.to_dict("index").items()},
        "attributable_lines": int(attributable["lines"].sum()),
        "attributable_total": float(attributable["total"].sum()),
        "attributable_freight": float(attributable["freight"].sum()),
        "attributable_premium": float(attributable["premium"].sum()),
        "counterfactual_total": float(cf["corrected"]["rush_freight"] + cf["corrected"]["rush_premium"]),
    }

    # ── shortages and delayed jobs, as counts and days ──────────────────────
    s = shortages[_year(shortages["date"])]
    jobs = prod[_year(prod["due_date"])]
    delayed = jobs[jobs["delay_days"] > 0]
    out["shortages"] = {
        "basis": "measured",
        "episodes": int(len(s)),
        "items": int(s["item_number"].nunique()),
        "by_cause": {k: int(v) for k, v in s["cause"].value_counts().items()},
        "jobs": int(len(jobs)),
        "jobs_delayed": int(len(delayed)),
        "delay_days": int(delayed["delay_days"].sum()),
        "delay_days_median": float(delayed["delay_days"].median()) if len(delayed) else 0.0,
        "counterfactual": {k: cf["corrected"][k] for k in ["shortage_episodes", "shortage_items", "jobs_delayed", "job_delay_days"]},
    }

    # ── working capital: the same rule on corrected masters ─────────────────
    a, c = cf["as_is"]["inventory_by_item"], cf["corrected"]["inventory_by_item"]
    rows = []
    for n, v in a.items():
        d = v - c.get(n, 0.0)
        rows.append({"item": n, "group": group(n), "excess": max(d, 0.0), "shortfall": max(-d, 0.0)})
    wc = pd.DataFrame(rows)
    by_group = wc.groupby("group").agg(items=("item", "size"), excess=("excess", "sum"), shortfall=("shortfall", "sum"))
    out["working_capital"] = {
        "basis": "simulated",
        "as_is_avg": cf["as_is"]["inventory_value_avg"],
        "corrected_avg": cf["corrected"]["inventory_value_avg"],
        "excess": float(wc["excess"].sum()),
        "shortfall": float(wc["shortfall"].sum()),
        "net": float(wc["excess"].sum() - wc["shortfall"].sum()),
        "items_excess": int((wc["excess"] > 0).sum()),
        "items_shortfall": int((wc["shortfall"] > 0).sum()),
        "by_group": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in by_group.to_dict("index").items()},
        "purchases_as_is": cf["as_is"]["purchases"],
        "purchases_corrected": cf["corrected"]["purchases"],
    }

    # ── misstatement: count corrections, by the error behind the item ───────
    t = tx.copy()
    t["value"] = t["qty"] * t["item_number"].map(cost)
    adj = t[(t["type"] == "ADJUST") & _year(t["txn_date"]) & ~t["item_number"].isin(dead)].copy()
    adj["group"] = adj["item_number"].map(group)
    adj["generic"] = adj["reason_code"].isna() | adj["reason_code"].astype(str).isin(GENERIC_REASONS)
    g = adj.groupby("group")["value"]
    out["adjustments"] = {
        "basis": "measured",
        "rows": int(len(adj)),
        "write_off": float(-adj.loc[adj["qty"] < 0, "value"].sum()),
        "write_up": float(adj.loc[adj["qty"] > 0, "value"].sum()),
        "by_group": {k: {"rows": int(len(v)), "write_off": float(-v[v < 0].sum()), "write_up": float(v[v > 0].sum())}
                     for k, v in g},
        "generic_rows": int(adj["generic"].sum()),
        "generic_abs_value": float(adj.loc[adj["generic"], "value"].abs().sum()),
        "count_rows": int(adj["reason_code"].isin(["COUNT", "CYCLE"]).sum()),
    }
    cyc = cc[cc["program"] == "CYCLE"].copy()
    cyc["delta"] = (cyc["counted_qty"] - cyc["system_qty"]) * cyc["item_number"].map(cost)
    out["remediation_counts"] = {
        "basis": "measured",
        "counts": int(len(cyc)),
        "write_down": float(-cyc.loc[cyc["delta"] < 0, "delta"].sum()),
        "write_up": float(cyc.loc[cyc["delta"] > 0, "delta"].sum()),
        "net": float(cyc["delta"].sum()),
    }

    # ── misstatement: consumption never charged (BOM omissions, T1) ─────────
    u = unrec[_year(unrec["date"])].copy()
    u["value"] = u["qty"] * u["item_number"].map(cost)
    out["unrecorded"] = {
        "basis": "measured",
        "events": int(len(u)),
        "items": int(u["item_number"].nunique()),
        "value": float(u["value"].sum()),
        "backflush_value": float(u.loc[u["channel"] == "BACKFLUSH", "value"].sum()),
        "other_value": float(u.loc[u["channel"] != "BACKFLUSH", "value"].sum()),
        "products_affected": len(cross.get("m3_affected_products", [])),
        "n_products": cross.get("n_products", 80),
    }

    # ── phantom on-order and open documents ─────────────────────────────────
    op = po[po["status"] == "OPEN"].copy()
    op["age"] = (pd.Timestamp(AS_OF) - pd.to_datetime(op["order_date"])).dt.days
    op["balance"] = (op["qty_ordered"] - op["qty_received"]).clip(lower=0) * op["unit_price"]
    stale_open = op[op["age"] > PHANTOM_DAYS]
    open_jobs = prod[(prod["status"] == "OPEN") & (pd.to_datetime(prod["due_date"]) < pd.Timestamp(AS_OF))]
    out["phantom_on_order"] = {
        "basis": "measured",
        "open_lines": int(len(op)),
        "open_value": float(op["balance"].sum()),
        "stale_lines": int(len(stale_open)),
        "stale_value": float(stale_open["balance"].sum()),
        "never_closed_lines": len(pod["t5"]),
        "open_jobs_past_due": int(len(open_jobs)),
        "shortages_caused": int(s["cause"].eq("phantom_on_order").sum()),
    }

    # ── exposures: wrong numbers that cost nothing until acted on ───────────
    dead_im = im[im["item_number"].isin(dead) & im["reorder_point"].notna()]
    dead_lines = po[po["item_number"].isin(dead)]
    out["dead_items"] = {
        "basis": "measured",
        "with_reorder_point": int(len(dead_im)),
        "suggestion_value": float((dead_im["reorder_point"] * dead_im["item_number"].map(cost)).sum()),
        "po_lines_ever": int(len(dead_lines)),
        "po_value_ever": float((dead_lines["qty_ordered"] * dead_lines["unit_price"]).sum()),
    }

    # duplicate members: a purchase on one number while a sibling held the stock
    lg = t[t["item_number"].isin(dup_members)].sort_values("txn_date")
    lg["s"] = lg["qty"] * lg["type"].map(SIGN).fillna(0)
    bal = {n: (g_["txn_date"].to_numpy(), np.cumsum(g_["s"].to_numpy())) for n, g_ in lg.groupby("item_number")}

    def balance(n, d):
        if n not in bal:
            return 0.0
        dates, cum = bal[n]
        k = int(np.searchsorted(dates, d, side="right"))
        return float(cum[k - 1]) if k else 0.0

    siblings = {}
    for cl in cross["duplicate_clusters"].values():
        for n in cl["records"]:
            siblings[n] = [m for m in cl["records"] if m != n]
    dp = po[po["item_number"].isin(dup_members) & _year(po["order_date"]) & ~po["rush"]]
    hits, hit_value = 0, 0.0
    for r_ in dp.itertuples(index=False):
        held = sum(balance(m, r_.order_date) for m in siblings[r_.item_number])
        if held >= r_.qty_ordered:
            hits += 1
            hit_value += r_.qty_ordered * r_.unit_price
    out["duplicates"] = {
        "basis": "measured",
        "clusters": len(cross["duplicate_clusters"]),
        "records": len(dup_members),
        "po_lines": int(len(dp)),
        "lines_while_sibling_held_stock": hits,
        "value_while_sibling_held_stock": float(hit_value),
        "count_churn": out["adjustments"]["by_group"].get("Duplicate Item Records", {}),
    }

    ft = po[po["item_number"].isin(GENERIC_ITEM_CODES) & _year(po["order_date"])]
    confirmed = ftattr[ftattr["confirmation"] == "confirmed"]
    ft_conf = ft[ft["po_id"].isin(confirmed["po_id"])]
    out["free_text"] = {
        "basis": "measured",
        "lines": int(len(ft)),
        "spend": float((ft["qty_ordered"] * ft["unit_price"]).sum()),
        "confirmed_lines": int(len(ft_conf)),
        "confirmed_spend": float((ft_conf["qty_ordered"] * ft_conf["unit_price"]).sum()),
        "stocked_items": len(t3_items),
        "count_churn": out["adjustments"]["by_group"].get("Free-Text Purchases", {}),
    }

    canon = cross["supplier_id_to_canonical"]
    aliases = {a for f in cross["supplier_fragments"] for a in f["aliases"]}
    py = po[_year(po["order_date"]) & po["received_date"].notna()].copy()
    py["value"] = py["qty_received"] * py["unit_price"]
    py["canon"] = py["supplier_id"].map(canon).fillna(py["supplier_id"])
    frag_canon = {f["canonical"] for f in cross["supplier_fragments"]}
    out["suppliers"] = {
        "basis": "measured",
        "vendors": len(frag_canon),
        "records": len(frag_canon) + len(aliases),
        "alias_spend": float(py.loc[py["supplier_id"].isin(aliases), "value"].sum()),
        "vendor_spend": float(py.loc[py["canon"].isin(frag_canon), "value"].sum()),
        "purchases": float(py["value"].sum()),
    }

    t["s"] = t["qty"] * t["type"].map(SIGN).fillna(0)
    raw_book = t.groupby("item_number")["s"].sum()
    blank_units = raw_book.reindex(sorted(blank_cost)).fillna(0).clip(lower=0)
    out["blank_fields"] = {
        "basis": "estimated",
        "assumption": "on-hand under a blank-cost item is valued at the median standard cost of its class",
        "blank_cost_items": len(blank_cost),
        "blank_cost_units": float(blank_units.sum()),
        "blank_cost_value": float((blank_units * blank_units.index.map(cost)).sum()),
        "blank_rop_items": int(live["reorder_point"].isna().sum()),
        "blank_supplier_items": int(live["primary_supplier_id"].isna().sum()),
    }

    m5_bal = raw_book.reindex(list(conv)).fillna(0)
    out["uom"] = {
        "basis": "measured",
        "items": len(conv),
        "negative_or_zero_balances": int((m5_bal <= 0).sum()),
        "count_churn": out["adjustments"]["by_group"].get("UOM Mismatch", {}),
    }

    ty = t[_year(t["txn_date"])]
    t6 = ty[ty["txn_id"].isin(t6_ids)]
    t7 = ty[ty["txn_id"].isin(t7_ids)].copy()
    t7["true"] = t7["txn_id"].map(lambda i: t7_ids[i]["true_qty"])
    t8 = ty[ty["txn_id"].isin(t8_ids)]
    out["postings"] = {
        "basis": "measured",
        "wrong_reference_rows": int(len(t6)),
        "wrong_reference_value": float(t6["value"].sum()),
        "keying_rows": int(len(t7)),
        "keying_value": float(((t7["qty"] - t7["true"]) * t7["item_number"].map(cost)).sum()),
        "duplicate_rows": int(len(t8)),
        "duplicate_value": float(t8["value"].sum()),
    }
    t4 = pd.DataFrame(pod["t4"])
    t4["lag"] = (pd.to_datetime(t4["recorded_received_date"]) - pd.to_datetime(t4["true_received_date"])).dt.days
    t4y = t4[pd.to_datetime(t4["true_received_date"]).dt.year == YEAR]
    out["batched"] = {"basis": "measured", "lines": int(len(t4y)), "mean_lag_days": float(t4y["lag"].mean()),
                      "max_lag_days": int(t4y["lag"].max())}

    # ── headline ────────────────────────────────────────────────────────────
    out["headline"] = {
        "recurring_saving": out["expedites"]["attributable_total"],
        "recurring_saving_basis": "measured",
        "working_capital_net": out["working_capital"]["net"],
        "working_capital_basis": "simulated",
        "shortage_episodes": [out["shortages"]["episodes"], cf["corrected"]["shortage_episodes"]],
        "jobs_delayed": [out["shortages"]["jobs_delayed"], cf["corrected"]["jobs_delayed"]],
        "delay_days": [out["shortages"]["delay_days"], cf["corrected"]["job_delay_days"]],
        "misstatement_corrected": out["remediation_counts"]["net"],
        "product_cost_understated": out["unrecorded"]["backflush_value"],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "financials.json").write_text(json.dumps(out, indent=2, default=float))

    M = lambda x: f"${x:,.0f}"
    e, w, sh = out["expedites"], out["working_capital"], out["shortages"]
    print(f"\n=== Financial impact, CY{YEAR} ===")
    print(f"  Expedites (measured): {e['rush_lines']} rush lines, freight {M(e['freight'])} + premium {M(e['premium'])}; "
          f"attributable to errors {M(e['attributable_total'])} on {e['attributable_lines']} lines")
    for k, v in e["by_cause"].items():
        print(f"      {k:<24} {int(v['lines']):4d} lines  {M(v['total'])}")
    print(f"  Shortages (measured): {sh['episodes']} episodes on {sh['items']} items; jobs delayed {sh['jobs_delayed']} of "
          f"{sh['jobs']} ({sh['delay_days']} days) -> corrected replay {sh['counterfactual']}")
    print(f"  Working capital (simulated): as-is {M(w['as_is_avg'])} vs corrected {M(w['corrected_avg'])}; "
          f"excess {M(w['excess'])} on {w['items_excess']} items, shortfall {M(w['shortfall'])} on {w['items_shortfall']}; net {M(w['net'])}")
    for k, v in w["by_group"].items():
        print(f"      {k:<28} excess {M(v['excess']):>10}  shortfall {M(v['shortfall']):>10}  ({int(v['items'])} items)")
    ad = out["adjustments"]
    print(f"  Count corrections (measured): write-off {M(ad['write_off'])}, write-up {M(ad['write_up'])} on {ad['rows']} rows")
    for k, v in ad["by_group"].items():
        print(f"      {k:<28} off {M(v['write_off']):>10}  up {M(v['write_up']):>10}  ({v['rows']} rows)")
    print(f"  Remediation counts: write-down {M(out['remediation_counts']['write_down'])}, write-up "
          f"{M(out['remediation_counts']['write_up'])}, net {M(out['remediation_counts']['net'])}")
    un = out["unrecorded"]
    print(f"  Never charged (measured): {M(un['value'])} on {un['items']} items; product cost understated {M(un['backflush_value'])}")
    ph = out["phantom_on_order"]
    print(f"  Phantom on-order: {ph['stale_lines']} lines older than {PHANTOM_DAYS} days, {M(ph['stale_value'])}; "
          f"{ph['shortages_caused']} shortage episodes")
    dd, du, ftx, su = out["dead_items"], out["duplicates"], out["free_text"], out["suppliers"]
    print(f"  Dead items: {dd['with_reorder_point']} carry a reorder point worth {M(dd['suggestion_value'])} of suggestions; "
          f"PO lines ever placed on them {dd['po_lines_ever']}")
    print(f"  Duplicates: {du['lines_while_sibling_held_stock']} of {du['po_lines']} PO lines placed while a sibling held the stock, {M(du['value_while_sibling_held_stock'])}")
    print(f"  Free text: {ftx['lines']} lines, {M(ftx['spend'])}; confirmed stocked {ftx['confirmed_lines']} lines {M(ftx['confirmed_spend'])}")
    print(f"  Suppliers: alias spend {M(su['alias_spend'])} of {M(su['vendor_spend'])} for the {su['vendors']} vendors")
    bf, uo, ps, bt = out["blank_fields"], out["uom"], out["postings"], out["batched"]
    print(f"  Blank cost: {bf['blank_cost_items']} items, {bf['blank_cost_units']:.0f} units, ~{M(bf['blank_cost_value'])} (estimated)")
    print(f"  UOM: {uo['items']} items, {uo['negative_or_zero_balances']} at zero or negative book")
    print(f"  Postings: wrong ref {ps['wrong_reference_rows']} rows {M(ps['wrong_reference_value'])}; keying {ps['keying_rows']} rows "
          f"{M(ps['keying_value'])}; duplicates {ps['duplicate_rows']} rows {M(ps['duplicate_value'])}; batched {bt['lines']} lines, "
          f"mean lag {bt['mean_lag_days']:.1f} days")
    print(f"\n  -> {OUT / 'financials.json'}\n")


if __name__ == "__main__":
    run()
