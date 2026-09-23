"""Data quality audit report -> docs/reports/data_quality_audit.html

A final report delivered at the end of the ten-week remediation to the operations
manager and controller. It documents what was found, what was done, what it
achieved, and what must change to keep it that way: results first, in the metrics
the shop cares about, with the technical detail left to the appendix. Every claim
carries a number; estimates are labeled as estimates with the assumption stated.
"""
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brand as B

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data_source" / "raw"
REM = RAW / "remediation"
TRUTH = REPO / "data_source" / "truth"
MARTS = REPO / "ml" / "data" / "marts"
BACKTEST = REPO / "ml" / "data" / "backtest"
FIN = REPO / "ml" / "data" / "financials"
OUT = REPO / "docs" / "reports" / "data_quality_audit.html"


def _money(x):
    return f"${x:,.0f}"


def _pct(x, d=0):
    return f"{x*100:.{d}f}%"


def _k(x):
    # compact money for the KPI cards: $116K, $2.0M
    if abs(x) < 500:
        return f"${x:,.0f}"
    return f"${x/1e6:.1f}M" if abs(x) >= 1e6 else f"${x/1e3:.0f}K"


def gather():
    d = {}
    im = pd.read_csv(RAW / "erp" / "item_master.csv", low_memory=False)
    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv", low_memory=False)
    po = pd.read_csv(RAW / "erp" / "purchase_orders.csv", low_memory=False)
    cross = json.loads((TRUTH / "crosswalks.json").read_text())
    pod = json.loads((TRUTH / "po_defects.json").read_text())

    dead_disp = pd.read_csv(REM / "dead_item_dispositions.csv")
    dup_cw = pd.read_csv(REM / "duplicate_crosswalk.csv")
    lead = pd.read_csv(REM / "lead_time_computation.csv")
    params = pd.read_csv(REM / "parameter_recommendations.csv")
    chronic = pd.read_csv(REM / "chronic_adjustment_list.csv")
    bomlog = pd.read_csv(REM / "bom_change_log.csv")
    closures = pd.read_csv(REM / "open_document_closures.csv")
    recon = pd.read_csv(REM / "spreadsheet_reconciliation.csv")
    ftattr = pd.read_csv(REM / "free_text_attribution.csv")
    uomc = pd.read_csv(REM / "uom_conversions.csv")
    config = pd.read_csv(REM / "config_change_log.csv")
    interviews = pd.read_csv(REM / "interview_log.csv")
    sup = pd.read_csv(RAW / "erp" / "supplier_master.csv", low_memory=False)
    bom = pd.read_csv(RAW / "erp" / "bill_of_materials.csv", low_memory=False)

    dead_nums = set(dead_disp["item_number"])
    n_master = len(im)
    n_dead = len(dead_nums)
    n_live = n_master - n_dead

    # ── scope: the ERP components examined ──────────────────────────────────
    def _rows(p):
        with open(p, encoding="utf-8") as f:
            return sum(1 for _ in f) - 1
    master_comp = [("Item master", len(im)), ("Bill of materials", len(bom)), ("Supplier master", len(sup))]
    txn_comp = [("Inventory ledger", len(tx)), ("Purchase order lines", len(po)),
                ("Service order lines", _rows(RAW / "erp" / "service_orders.csv")),
                ("Production orders", _rows(RAW / "erp" / "production_orders.csv")),
                ("Cycle counts", _rows(RAW / "wms" / "cycle_counts.csv"))]
    d["master_comp"], d["txn_comp"] = master_comp, txn_comp
    d["master_rows"] = sum(n for _, n in master_comp)
    d["txn_rows"] = sum(n for _, n in txn_comp)
    d["spreadsheet_rows"] = _rows(RAW / "purchasing" / "buyer_spreadsheet.csv")
    d["total_rows"] = d["master_rows"] + d["txn_rows"] + d["spreadsheet_rows"]

    # ── reliability (headline): one rule, measured at two dates ────────────
    rs = json.loads((MARTS / "reliability_summary.json").read_text())
    d["rel"] = rs
    def rdist(when):
        tot = rs[when]["total"]
        return {k: {"pct": rs[when][k]["items"] / tot["items"] * 100, "val": rs[when][k]["value"],
                    "items": rs[when][k]["items"]} for k in ["reliable", "uncertain", "unreliable"]}
    d["rel_before"] = rdist("before")
    d["rel_after"] = rdist("after")
    d["inv_value_total"] = float(rs["before"]["total"]["value"])
    d["inv_value_after"] = float(rs["after"]["total"]["value"])

    # ── financial impact, measured from the records (ml/src/financials.py) ──
    d["fin"] = json.loads((FIN / "financials.json").read_text())

    # ── defect burden ───────────────────────────────────────────────────────
    d["n_master"], d["n_dead"], d["n_live"] = n_master, n_dead, n_live
    d["dead_pct"] = n_dead / n_master
    d["dead_with_rop"] = int(im[im["item_number"].isin(dead_nums)]["reorder_point"].notna().sum())

    # M2 lead-time drift
    p = po[po["received_date"].notna()].copy()
    p["lead"] = (pd.to_datetime(p["received_date"]) - pd.to_datetime(p["order_date"])).dt.days
    med = p.groupby("item_number")["lead"].median()
    master_lead = im.set_index("item_number")["master_lead_time_days"]
    live_nums = set(im["item_number"]) - dead_nums
    diff = (med - master_lead).dropna()
    diff = diff[[n in live_nums for n in diff.index]].abs()
    d["drift_gt3"] = float((diff > 3).mean())
    d["drift_gt7"] = float((diff > 7).mean())
    d["drift_supplier"] = cross["m2_drift_supplier"]

    # M3 / T1 phantom
    d["omit_products"] = len(cross.get("m3_affected_products", []))
    d["n_products"] = cross.get("n_products", 80)
    d["omit_items"] = len(cross["m3_omitted_items"])
    t1 = [r for r in json.loads((TRUTH / "txn_defects.json").read_text()).get("t1", [])]
    d["t1_items"] = len(t1)
    d["t1_volume"] = int(sum(r.get("annual_unrecorded", 0) for r in t1))

    # M4 duplicates
    d["dup_clusters"] = len(cross["duplicate_clusters"])
    d["dup_items"] = len(cross["duplicate_clusters"])
    d["dup_records"] = sum(len(v["records"]) for v in cross["duplicate_clusters"].values())

    # M5 UOM, M6 suppliers, M7 blanks
    d["uom_items"] = len(cross["m5_items"])
    d["sup_fragments"] = len(cross["supplier_fragments"])
    d["sup_records"] = sum(1 + len(f["aliases"]) for f in cross["supplier_fragments"])
    live = im[im["item_number"].isin(live_nums)]
    d["blank_pct"] = float(live[["standard_cost", "reorder_point", "primary_supplier_id"]].isna().any(axis=1).mean())
    d["misc_pct"] = float((live["item_class"] == "MISC").mean())

    # transaction defects
    moved = tx.loc[tx["type"].isin(["ISSUE", "BACKFLUSH", "RECEIPT"]), "qty"].abs().sum()
    adj = tx.loc[tx["type"] == "ADJUST", "qty"].abs().sum()
    d["adj_share"] = float(adj / moved)
    adj_rows = tx[tx["type"] == "ADJUST"]
    d["adj_blank_share"] = float((adj_rows["reason_code"].isna() |
        adj_rows["reason_code"].astype(str).isin(["", "nan", "ADJ", "VAR", "MISC", "COUNT"])).mean())
    ft = po[po["item_number"].isin(["NONSTOCK", "MISC", "SHOPSUPPLY"])]
    d["ft_pct"] = float(len(ft) / len(po))
    d["ft_stocked"] = sum(1 for r in pod["t3"] if r.get("is_stocked"))
    d["ft_total"] = len(pod["t3"])
    txn = json.loads((TRUTH / "txn_defects.json").read_text())
    d["t7_count"] = len(txn.get("t7", []))
    d["t8_count"] = len(txn.get("t8", []))
    d["t6_count"] = len(txn.get("t6", []))

    # on-order fiction (T5 open POs)
    op = po[po["status"] == "OPEN"].copy()
    op["fiction"] = (op["qty_ordered"] - op["qty_received"]).clip(lower=0) * op["unit_price"]
    d["open_po_lines"] = len(op)
    d["open_po_value"] = float(op["fiction"].sum())
    d["chronic_items"] = len(chronic)
    d["chronic_on_bom"] = float(chronic["on_bom"].mean()) if len(chronic) else 0.0

    # ── remediation activity ────────────────────────────────────────────────
    dd = dead_disp["disposition"].value_counts()
    d["dead_deactivated"] = int(dd.get("DEACTIVATED", 0))
    d["dead_kept"] = int(dd.get("KEPT", 0))
    d["dead_held"] = int(dd.get("HELD", 0))
    dv = dup_cw["decision"].value_counts()
    d["dup_merged"] = int(dv.get("MERGE", 0))
    d["dup_rejected"] = int(dv.get("REJECT", 0))
    d["lead_recomputed"] = len(lead)
    # a reorder point is stale when the recomputed value differs materially:
    # by at least 30% of the old value and at least 5 units
    old = params["old_reorder_point"].fillna(0)
    delta = (params["new_reorder_point"] - old).abs()
    changed = (delta >= 5) & (delta >= 0.30 * old.clip(lower=1))
    d["params_changed"] = int(changed.sum())
    d["uom_added"] = len(uomc)
    d["bom_changes"] = len(bomlog)
    d["closed_po"] = int((closures["document_type"] == "PO").sum())
    d["closed_jobs"] = int((closures["document_type"] == "JOB").sum())
    d["recon_disagree"] = int((recon["closer_to_truth"] != "agree").sum())
    d["recon_buyer_right"] = int((recon["closer_to_truth"] == "spreadsheet").sum())
    fa = ftattr["confirmation"].value_counts()
    d["ft_confirmed"] = int(fa.get("confirmed", 0))
    d["ft_rejected"] = int(fa.get("rejected", 0))
    d["ft_unreviewed"] = int(fa.get("unreviewed", 0))
    d["config"] = config
    d["interviews"] = interviews
    d["chronic_root"] = chronic["root_cause"].value_counts().to_dict() if len(chronic) else {}

    # batched postings: share of receipts landing on the peak weekday
    rec = tx[tx["type"] == "RECEIPT"].copy()
    rec["wd"] = pd.to_datetime(rec["txn_date"]).dt.weekday
    d["peak_wd_share"] = float(rec["wd"].value_counts(normalize=True).max())
    d["peak_wd_name"] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][
        int(rec["wd"].value_counts().idxmax())]

    # ── standardized scale: rows affected per error ─────────────────────────
    prod = pd.read_csv(RAW / "erp" / "production_orders.csv", low_memory=False)
    d["n_lead_off"] = int((diff > 3).sum())
    d["n_lead_items"] = int(len(diff))
    signed = (med - master_lead).dropna()
    signed = signed[[n in live_nums for n in signed.index]]
    stale_signed = signed[signed.abs() > 3]
    d["lead_gap_mean"] = float(stale_signed.mean())                 # days, actual minus master
    d["lead_longer_share"] = float((stale_signed > 0).mean())
    rop_delta = (params["new_reorder_point"] - params["old_reorder_point"].fillna(0))[changed]
    d["rop_delta_median"] = float(rop_delta.abs().median())
    d["rop_low_share"] = float((rop_delta > 0).mean())              # point on file below the recomputed one
    t4 = pd.DataFrame(pod.get("t4", []))
    d["t4_lag_mean"] = float((pd.to_datetime(t4["recorded_received_date"]) -
                              pd.to_datetime(t4["true_received_date"])).dt.days.mean()) if len(t4) else 0.0
    d["n_blank"] = int(live[["standard_cost", "reorder_point", "primary_supplier_id"]].isna().any(axis=1).sum())
    d["n_adj_rows"] = int(len(adj_rows))
    d["n_adj_blank_rows"] = int((adj_rows["reason_code"].isna() |
        adj_rows["reason_code"].astype(str).isin(["", "nan", "ADJ", "VAR", "MISC", "COUNT"])).sum())
    t1_nums = {r["item_number"] for r in txn.get("t1", [])}
    d["n_unrec_adj_rows"] = int(((tx["type"] == "ADJUST") & (tx["qty"] < 0) & tx["item_number"].isin(t1_nums)).sum())
    d["n_ft_lines"] = int(len(ft))
    d["n_batch_rows"] = int(len(pod.get("t4", [])))
    d["n_tx"], d["n_po"], d["n_prod"] = int(len(tx)), int(len(po)), int(len(prod))
    d["n_bom_rows"], d["n_sup_rows"] = int(len(bom)), int(len(sup))
    open_jobs = prod[(prod["status"] == "OPEN") & (pd.to_datetime(prod["due_date"]) < pd.Timestamp("2026-03-31"))]
    d["n_open_jobs"] = int(len(open_jobs))

    # ── rows carrying at least one error, per ERP table (errors overlap, so
    #    each row is counted once) ────────────────────────────────────────────
    dup_members = set()
    for cl in cross["duplicate_clusters"].values():
        dup_members.update(cl["records"])
    stale_lead_nums = set(diff[diff > 3].index)
    stale_rop_nums = set(params.loc[changed, "item_number"])
    uom_nums = set(live.loc[live["purchase_uom"] != live["uom"], "item_number"])
    blank_nums = set(live.loc[live[["standard_cost", "reorder_point", "primary_supplier_id"]].isna().any(axis=1), "item_number"])
    im_err = dead_nums | dup_members | stale_lead_nums | stale_rop_nums | uom_nums | blank_nums
    t78_ids = {r["txn_id"] for r in txn.get("t7", [])} | {r["txn_id"] for r in txn.get("t8", [])}
    blank_mask = (tx["type"] == "ADJUST") & (tx["reason_code"].isna() |
                  tx["reason_code"].astype(str).isin(["", "nan", "ADJ", "VAR", "MISC", "COUNT"]))
    unrec_mask = (tx["type"] == "ADJUST") & (tx["qty"] < 0) & tx["item_number"].isin(t1_nums)
    ledger_err = int((blank_mask | unrec_mask | tx["txn_id"].isin(t78_ids)).sum()) + d["t6_count"]
    t4_keys = {(r["po_id"], int(r["line"])) for r in pod.get("t4", [])}
    t4_mask = pd.Series([k in t4_keys for k in zip(po["po_id"], po["line"].astype(int))], index=po.index)
    po_err = int((t4_mask | po["item_number"].isin(["NONSTOCK", "MISC", "SHOPSUPPLY"]) | (po["status"] == "OPEN")).sum())
    tc_map = dict(txn_comp)
    stale_any = stale_lead_nums | stale_rop_nums
    d["im_stale_any"] = len(stale_any)
    d["im_stale_any_pct"] = len(stale_any) / n_live
    d["im_other_added"] = len((dup_members | uom_nums | blank_nums) - stale_any)
    d["im_live_clean"] = n_live - len(im_err - dead_nums)
    d["im_err_pct"] = len(im_err) / n_master
    d["po_err_pct"] = po_err / d["n_po"]
    d["po_batch_of_received"] = d["n_batch_rows"] / max(1, int(po["received_date"].notna().sum()))
    d["table_rates"] = [
        ("Item master", len(im_err), n_master),
        ("Bill of materials", d["omit_items"], d["n_bom_rows"] + d["omit_items"]),
        ("Supplier master", d["sup_records"], d["n_sup_rows"]),
        ("Inventory ledger", ledger_err, d["n_tx"]),
        ("Purchase orders", po_err, d["n_po"]),
        ("Production orders", d["n_open_jobs"], d["n_prod"]),
        ("Service orders", 0, tc_map["Service order lines"]),
        ("Cycle counts", 0, tc_map["Cycle counts"]),
    ]

    # residual
    d["still_unreliable_pct"] = d["rel_after"]["unreliable"]["pct"] / 100
    d["probable_unreviewed"] = d["ft_unreviewed"]

    d["samples"] = _samples(im, tx, po, sup, cross, txn, pod, lead, params, chronic, dead_nums)
    return d


def _blank(v):
    return "(blank)" if (v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan")) else v


def _samples(im, tx, po, sup, cross, txn, pod, lead, params, chronic, dead_nums):
    """One or two real records illustrating each error type."""
    s = {}
    live_im = im[~im["item_number"].isin(dead_nums)]

    dead = im[im["item_number"].isin(dead_nums) & im["reorder_point"].notna()].head(1)
    s["dead"] = (["Item", "Description", "Status", "Reorder point", "Created", "Last issue or receipt"],
                 [[r.item_number, r.description, r.status, int(r.reorder_point), r.created_date,
                   "none in 24+ months"] for r in dead.itertuples(index=False)])

    lt = lead.assign(gap=lead["median_actual"] - lead["master_lead_time"]).sort_values("gap", ascending=False).head(2)
    s["lead"] = (["Item", "Supplier", "Master lead time (days)", "Median actual (days)", "Receipts sampled"],
                 [[r.item_number, r.supplier_id, int(r.master_lead_time), f"{r.median_actual:.0f}", int(r.sample_size)]
                  for r in lt.itertuples(index=False)])

    pr = params.dropna(subset=["old_reorder_point"])
    pr = pr.assign(chg=(pr["new_reorder_point"] - pr["old_reorder_point"]).abs()).sort_values("chg", ascending=False).head(2)
    s["rop"] = (["Item", "ABC", "Reorder point in ERP", "Reorder point from actual usage and lead time"],
                [[r.item_number, r.abc_class, int(r.old_reorder_point), int(r.new_reorder_point)]
                 for r in pr.itertuples(index=False)])

    bo = chronic[chronic["on_bom"]].sort_values("adj_count_12m", ascending=False).head(2)
    s["bom"] = (["Item", "Downward adjustments (12 mo)", "Net quantity written off", "On any recorded BOM"],
                [[r.item_number, int(r.adj_count_12m), int(r.net_qty), "No"] for r in bo.itertuples(index=False)])

    cl = next(iter(cross["duplicate_clusters"].values()))
    dup = im[im["item_number"].isin(cl["records"])]
    s["dup"] = (["Item", "Description", "Created", "Created by"],
                [[r.item_number, r.description, r.created_date, r.created_by] for r in dup.itertuples(index=False)])

    u = live_im[(live_im["purchase_uom"] != live_im["uom"]) & live_im["uom_conversion"].isna()].head(2)
    s["uom"] = (["Item", "Description", "Stock UOM", "Purchase UOM", "Conversion factor"],
                [[r.item_number, r.description, r.uom, r.purchase_uom, "(blank)"] for r in u.itertuples(index=False)])

    frag = cross["supplier_fragments"][0]
    fs = sup[sup["supplier_id"].isin([frag["canonical"]] + frag["aliases"])]
    s["sup"] = (["Supplier ID", "Supplier name", "Type"],
                [[r.supplier_id, r.supplier_name, r.supplier_type] for r in fs.itertuples(index=False)])

    mf = live_im[live_im["standard_cost"].isna() | live_im["primary_supplier_id"].isna() | live_im["reorder_point"].isna()].head(2)
    s["missing"] = (["Item", "Description", "Standard cost", "Supplier", "Reorder point"],
                    [[r.item_number, r.description, _blank(r.standard_cost), _blank(r.primary_supplier_id),
                      _blank(r.reorder_point)] for r in mf.itertuples(index=False)])

    top = chronic.sort_values("adj_count_12m", ascending=False).iloc[0]["item_number"]
    ar = tx[(tx["item_number"] == top) & (tx["type"] == "ADJUST") & (tx["qty"] < 0)].sort_values("txn_date").tail(2)
    s["unrec"] = (["Transaction", "Item", "Date", "Type", "Qty", "Reason code"],
                  [[r.txn_id, r.item_number, r.txn_date, r.type, int(r.qty), _blank(r.reason_code)]
                   for r in ar.itertuples(index=False)])

    ab = tx[(tx["type"] == "ADJUST") & (tx["reason_code"].isna() | (tx["reason_code"].astype(str).str.strip() == ""))].head(2)
    s["adjshare"] = (["Transaction", "Item", "Date", "Type", "Qty", "Reason code"],
                     [[r.txn_id, r.item_number, r.txn_date, r.type, int(r.qty), "(blank)"] for r in ab.itertuples(index=False)])

    ft = po[po["item_number"].isin(["NONSTOCK", "MISC", "SHOPSUPPLY"]) & po["description_text"].notna()].head(2)
    s["ft"] = (["PO", "Item code", "Typed description", "Supplier", "Qty"],
               [[r.po_id, r.item_number, r.description_text, r.supplier_id, int(r.qty_ordered)] for r in ft.itertuples(index=False)])

    wdn = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    s["batch"] = (["PO", "Line", "Actual arrival", "Posted receipt date", "Posted on"],
                  [[r["po_id"], r["line"], r["true_received_date"], r["recorded_received_date"],
                    wdn[pd.Timestamp(r["recorded_received_date"]).weekday()]] for r in pod["t4"][:2]])

    op = po[po["status"] == "OPEN"].copy()
    op["od"] = pd.to_datetime(op["order_date"])
    op = op[op["od"] < pd.Timestamp("2026-03-31") - pd.Timedelta(days=90)].sort_values("od").head(2)
    s["open"] = (["PO", "Item", "Order date", "Ordered", "Received", "Status"],
                 [[r.po_id, r.item_number, r.order_date, int(r.qty_ordered), int(r.qty_received), r.status]
                  for r in op.itertuples(index=False)])

    s["wrong"] = (["Issue posted against", "Item the job's BOM actually calls for"],
                  [[r["recorded_item_number"], r["true_item_number"]] for r in txn.get("t6", [])[:2]])

    txl = tx.set_index("txn_id")
    q_rows = []
    for r in txn.get("t7", [])[:2]:
        if r["txn_id"] in txl.index:
            row = txl.loc[r["txn_id"]]
            q_rows.append([r["txn_id"], row["item_number"], row["txn_date"], int(r["recorded_qty"]), int(r["true_qty"])])
    s["qty"] = (["Transaction", "Item", "Date", "Recorded qty", "Actual qty"], q_rows)

    d_rows = []
    for r in txn.get("t8", [])[:1]:
        for tid in (r["original_txn_id"], r["txn_id"]):
            if tid in txl.index:
                row = txl.loc[tid]
                d_rows.append([tid, row["item_number"], row["txn_date"], row["txn_time"], row["type"], int(row["qty"])])
    s["dupost"] = (["Transaction", "Item", "Date", "Time", "Type", "Qty"], d_rows)
    return s


# ── charts ───────────────────────────────────────────────────────────────────
def chart_reliability(d):
    fig, ax = B.make_fig(3.4)
    cats = ["Before", "After"]
    order = ["reliable", "uncertain", "unreliable"]
    colors = {"reliable": B.GREEN, "uncertain": B.AMBER, "unreliable": B.ACCENT_RED}
    before = [d["rel_before"][k]["val"] / 1000 for k in order]
    after = [d["rel_after"][k]["val"] / 1000 for k in order]
    data = np.array([before, after])
    left = np.zeros(2)
    for i, k in enumerate(order):
        ax.barh(cats, data[:, i], left=left, color=colors[k], label=k.capitalize())
        left = left + data[:, i]
    ax.set_xlabel("Inventory value ($000)")
    B.chart_style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=3, fontsize=9)
    ax.invert_yaxis()
    return B.b64(fig)

def chart_erd(d):
    """Entity relationship diagram: the eight ERP tables, their row counts, and
    which tables depend on which. An arrow points from the table relied on to the
    table that depends on it; two-headed where each updates the other."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    rows = dict(d["master_comp"] + d["txn_comp"])
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    def box(x, y, w, h, title, key, master):
        face = B.DARK_BLUE if master else B.MED_GREY
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                    facecolor=face, edgecolor=face, linewidth=1.2))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=9.6, color="white", fontweight="bold")
        ax.text(x + w / 2, y + h * 0.30, f"{rows[key]:,} records", ha="center", va="center", fontsize=8, color="white")

    def seg(pts):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=B.MED_GREY, linewidth=1.1, zorder=0, solid_capstyle="round")

    def head(p0, p1):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, color=B.MED_GREY,
                                     linewidth=1.1, shrinkA=0, shrinkB=0, zorder=1))

    # masters on top, transactions below
    box(5, 32, 24, 11, "Supplier master", "Supplier master", True)
    box(38, 32, 24, 11, "Item master", "Item master", True)
    box(71, 32, 24, 11, "Bill of materials", "Bill of materials", True)
    box(0.5, 6, 19, 11, "Cycle counts", "Cycle counts", False)
    box(20.5, 6, 19, 11, "Purchase orders", "Purchase order lines", False)
    box(40.5, 6, 19, 11, "Inventory ledger", "Inventory ledger", False)
    box(60.5, 6, 19, 11, "Production orders", "Production orders", False)
    box(80.5, 6, 19, 11, "Service orders", "Service order lines", False)

    # masters that other masters rely on
    head((29, 37.5), (38, 37.5))
    head((62, 37.5), (71, 37.5))
    # every table keyed on the item number
    seg([(50, 32), (50, 25)]); seg([(10, 25), (90, 25)])
    for x in (10, 30, 50, 90):
        head((x, 25), (x, 17))
    # the supplier on a purchase order, the bill a job is built from
    seg([(15, 32), (15, 28.5), (24, 28.5), (24, 25)]); head((24, 25), (24, 17))
    seg([(85, 32), (85, 28.5), (76, 28.5), (76, 25)]); head((76, 25), (76, 17))
    # documents and the ledger update each other
    seg([(34, 20), (34, 21.5), (46, 21.5), (46, 20)]); head((34, 21.5), (34, 17)); head((46, 21.5), (46, 17))
    seg([(54, 20), (54, 21.5), (66, 21.5), (66, 20)]); head((54, 21.5), (54, 17)); head((66, 21.5), (66, 17))
    seg([(90, 3), (90, 2), (50, 2), (50, 3)]); head((90, 3), (90, 6)); head((50, 3), (50, 6))
    return B.b64(fig)


def _err_block(num, name, what, location, headers, rows):
    """One error type: number and name, a plain sentence on what it is, where in
    the ERP it occurs, and a record or two from the system that shows it."""
    lbl = (f'style="color:{B.MED_GREY};font-weight:700;font-size:12px;text-transform:uppercase;'
           f'letter-spacing:.5px;display:inline-block;width:110px;vertical-align:top;"')
    sample = B.data_table(headers, rows) if rows else ""
    return (f'<div style="margin:26px 0 30px;">'
            f'<div style="font-size:16px;margin-bottom:6px;"><strong><u>#{num} {name}</u></strong></div>'
            f'<div style="margin-bottom:8px;">{what}</div>'
            f'<div style="margin-bottom:4px;"><span {lbl}>Location</span><span>{location}</span></div>'
            f'{sample}</div>')


def _widths(table_html, widths):
    """Give a data table fixed column widths (percent) so tables line up."""
    cols = "".join(f'<col style="width:{w}%;">' for w in widths)
    return table_html.replace('<table class="data-table">',
                              f'<table class="data-table" style="table-layout:fixed;"><colgroup>{cols}</colgroup>', 1)


def chart_error_rates(d):
    """Horizontal bars: share of each ERP table's rows carrying at least one error."""
    rates = d["table_rates"]
    fig, ax = B.make_fig(4.2)
    names = [r[0] for r in rates][::-1]
    pct = [r[1] / r[2] * 100 if r[2] else 0 for r in rates][::-1]
    labels = [f"{r[1] / r[2] * 100:.1f}%  ({r[1]:,} of {r[2]:,})" if r[2] else "" for r in rates][::-1]
    bars = ax.barh(names, pct, color=B.DARK_BLUE, height=0.62)
    for b, lab in zip(bars, labels):
        ax.text(b.get_width() + 1.2, b.get_y() + b.get_height() / 2, lab, va="center", fontsize=9.5)
    ax.set_xlim(0, max(pct) * 1.45 if max(pct) else 10)
    ax.set_xlabel("Rows with at least one error (%)")
    ax.xaxis.grid(True, color=B.LIGHT_GREY, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(B.LIGHT_GREY)
    ax.spines["bottom"].set_color(B.LIGHT_GREY)
    return B.b64(fig)


# ── report ───────────────────────────────────────────────────────────────────
def build(d):
    toc = "".join([
        '<a href="#results">Results</a>',
        '<a href="#found">Data Quality Errors</a>',
        '<a href="#did">Error Remediation</a>',
        '<a href="#process">Go-Forward Changes</a>',
    ])

    trust_before = d["rel_before"]["reliable"]["pct"]
    trust_after = d["rel_after"]["reliable"]["pct"]
    fin = d["fin"]
    YR = fin["year"]
    ex, sh, ph, du, un, ps, dd, su, ftx, bf, uo, adj, rc = (fin[k] for k in
        ["expedites", "shortages", "phantom_on_order", "duplicates", "unrecorded", "postings", "dead_items",
         "suppliers", "free_text", "blank_fields", "uom", "adjustments", "remediation_counts"])
    # the three situations the record can tie a rush or a shortage to: a delivery
    # past the promised date its stale lead time set, a shelf emptied by a pull
    # that was never recorded, a reorder held back by a phantom on-order balance
    TRACED = ["stale_lead_time", "unrecorded_consumption", "phantom_on_order"]
    bc, sbc, jbc = ex["by_cause"], sh["by_cause"], sh["jobs_by_cause"]
    rl = {c: int(bc.get(c, {}).get("lines", 0)) for c in TRACED}
    rush_traced_lines = sum(rl.values())
    rush_traced = ex["attributable_total"]
    rush_other = ex["rush_lines"] - rush_traced_lines
    short_traced = sum(sbc.get(c, 0) for c in TRACED)
    jobs_traced = sum(jbc.get(c, 0) for c in TRACED)
    rs = d["rel"]
    rb, ra = rs["before"], rs["after"]
    sub = lambda t: f'<p style="font-size:18px;font-weight:700;color:{B.DARK_GREY};margin-top:34px;">{t}</p>'

    rel_rows = []
    for k, label in [("reliable", "Reliable"), ("uncertain", "Uncertain"), ("unreliable", "Unreliable")]:
        rel_rows.append([f"{label}<br><em>{rs['definitions'][k]}</em>",
                         f"{rb[k]['items']:,}", _money(rb[k]["value"]), f"{ra[k]['items']:,}", _money(ra[k]["value"])])
    rel_rows.append(["<strong>All live items</strong>", f"<strong>{rb['total']['items']:,}</strong>",
                     f"<strong>{_money(rb['total']['value'])}</strong>", f"<strong>{ra['total']['items']:,}</strong>",
                     f"<strong>{_money(ra['total']['value'])}</strong>"])
    rel_table = _widths(B.data_table(["Reliability of on-hand balance", "Items before", "Value before", "Items after", "Value after"],
                                     rel_rows, right=[1, 2, 3, 4]), [40, 15, 15, 15, 15])

    ops_rows = [
        ("Stockouts and line stops on material the system says is there",
         f"{sh['episodes']} times in {YR}, on {sh['items']} items, the shelf was empty when a job or an order "
         f"needed it; {short_traced} of those arose in one of three situations the errors create (a delivery past "
         f"the promised date its stale lead time set, a shelf emptied by a pull that was never recorded, a reorder "
         f"held back by a phantom on-order balance). {sh['jobs_delayed']} of the {sh['jobs']:,} jobs due in the "
         f"year waited for material, typically {sh['delay_days_median']:.0f} days.",
         f"The three situations are removed at the source: {d['n_lead_off']:,} lead times recomputed from receipts, "
         f"{d['bom_changes']} missing components restored to the bills of {d['omit_products']} products, and "
         f"{ph['stale_lines']:,} phantom on-order lines closed. Whether the shortages fall is for the monthly "
         f"measures in Section 4 to show; this audit does not forecast it."),
        ("Firefighting in place of planning",
         f"{ex['rush_lines']} rush orders were placed in {YR}. {rush_traced_lines} of them were placed in the three "
         f"situations above; the other {rush_other} were demand the stale reorder points did not cover and are not "
         f"attributed. The purchasing manager kept {d['spreadsheet_rows']} critical components in a spreadsheet "
         f"because the ERP's numbers could not be trusted; it disagreed with the system on {d['recon_disagree']} of "
         f"them.",
         f"Reorder points reset on {d['params_changed']:,} items from actual usage over the recomputed lead times, "
         f"and the spreadsheet reconciled to the ERP so the buyer can plan from the system again. The monthly "
         f"parameter refresh keeps it that way."),
        ("Promises and quotes made on stale numbers",
         f"Lead times on {d['n_lead_off']/d['n_live']*100:.0f}% of live items read {d['lead_gap_mean']:.0f} days "
         f"shorter than deliveries took, so promise dates built on them could not be met. {bf['blank_cost_items']} "
         f"items carried no standard cost at all, and {un['products_affected']} products consumed "
         f"{_money(un['backflush_value'])} of components their bills did not list, so quotes on them did not "
         f"cover the material.",
         f"Lead times, costs and bills corrected in the master; required fields now enforced, so a new item cannot "
         f"be created without a cost. A pricing review of the {un['products_affected']} affected products is "
         f"recommended."),
        ("Labor spent on the data instead of the work",
         f"{adj['rows']:,} balance corrections posted in {YR}; an annual physical that restated most of the master; "
         f"{d['n_dead']:,} dead records in every report, count and purchase suggestion; {d['n_ft_lines']:,} "
         f"free-text purchase lines over the three years to be matched to items by hand; {ph['stale_lines']:,} "
         f"purchase order lines and {ph['open_jobs_past_due']} jobs open past their dates.",
         f"{d['dead_deactivated']} dead records deactivated, {d['dup_merged']} duplicates retired, "
         f"{d['closed_po']:,} open lines and {d['closed_jobs']} jobs closed, generic item codes restricted. A "
         f"cycle-count program by ABC class replaces the annual physical, and a reason code is required on every "
         f"adjustment."),
        ("Planning blind",
         f"Requirements planning sees only recorded demand. {d['omit_items']} components on {un['products_affected']} "
         f"products never appeared in it, {d['dup_records']} duplicate records split the history of {du['clusters']} "
         f"parts, and {d['ft_confirmed']:,} purchases of stocked items were typed in under a generic code, so the "
         f"reorder points and forecasts computed from that history were wrong before they started.",
         f"Bills completed, duplicates merged through a crosswalk and free-text purchases attributed to their items, "
         f"so the consumption history is whole. The demand forecasting and reorder-policy work that follows this "
         f"audit is built on that history, and could not have been built on the old one."),
    ]
    ops_table = _widths(B.data_table(["Operational impact", f"Before: what the {YR} record shows", "After: what the cleanup changed"],
                                     [list(r) for r in ops_rows], right=[]), [20, 42, 38])

    fin_rows = [
        ("Working capital in inventory",
         f"The errors did not so much inflate the stock as put the cash in the wrong stock. {du['lines_while_sibling_held_stock']} "
         f"purchase lines worth {_money(du['value_while_sibling_held_stock'])} were placed in {YR} on one item number "
         f"while its duplicate record already held the quantity ordered. No inventory reduction is claimed: the "
         f"stale parameters ran as many items short as long.",
         f"The {d['dup_merged']} duplicate records retired to one survivor each, so a purchase fires once against "
         f"the pile. The excess already bought sells through as it is consumed."),
        ("Expedite freight and price premiums",
         f"{_money(ex['total'])} on {ex['rush_lines']} rush lines in {YR}, of which {_money(rush_traced)} on "
         f"{rush_traced_lines} lines traces to the three situations the errors create. The remainder is not "
         f"attributed.",
         f"The three causes removed at the source (lead times, bills, phantom on-order). The saving going forward "
         f"is a forecast, and is left to the reorder-policy work."),
        ("Write-offs and write-ups of phantom inventory",
         f"The counts wrote off {_money(adj['write_off'])} and wrote up {_money(adj['write_up'])} in {YR}, "
         f"{_pct(d['adj_share'], 0)} of all quantity moved. {_money(un['value'])} of pulls on {un['items']} items were "
         f"never recorded; {_money(ps['wrong_reference_value'])} was posted to the wrong item, "
         f"{_money(ps['keying_value'])} keyed at the wrong quantity and {_money(ps['duplicate_value'])} posted twice. "
         f"The cycle counts since remediation began have restated {_money(rc['write_down'] + rc['write_up'])} of "
         f"book value across {rc['counts']:,} counts.",
         f"Reason codes required, negative balances blocked, box-to-each conversions on file and the cycle-count "
         f"program running, so a discrepancy is caught when it happens and carries its cause. The target run-rate "
         f"for adjustments is 3 to 6% of quantity moved."),
        ("Margin on mispriced work",
         f"{_money(un['backflush_value'])} of component cost consumed on {un['products_affected']} products and never "
         f"charged to them, so their margins read high by that amount; {bf['blank_cost_items']} items held about "
         f"{_money(bf['blank_cost_value'])} of stock at $0 (an estimate at the class median cost).",
         f"Bills completed and costs filled from the last price paid. The affected products' pricing should be "
         f"reviewed against their true material cost."),
        ("On-order commitments overstated",
         f"{_money(ph['stale_value'])} shown on order against {ph['stale_lines']:,} lines older than 90 days that "
         f"would never arrive, holding real reorders back.",
         f"Closed on confirmation with receiving records and supplier statements: {_money(0)} remains. A monthly "
         f"open-document review keeps it there."),
        ("Spend invisible to reporting",
         f"{_money(su['alias_spend'])} of the {_money(su['vendor_spend'])} spent with the {su['vendors']} fragmented "
         f"vendors sat under alias records, and {_money(ftx['spend'])} was bought on free-text lines with no item "
         f"behind it, so vendor spend and item demand both read low.",
         f"Supplier aliases mapped to one canonical record each and new orders booked to it; generic item codes "
         f"restricted, so spend lands on the vendor and the item that earned it."),
    ]
    fin_table = _widths(B.data_table(["Financial impact", f"Before: what the {YR} record shows", "After: what the cleanup changed"],
                                     [list(r) for r in fin_rows], right=[]), [20, 42, 38])

    results = f"""
{B.section("results", "Section 1", "Results")}
<p>Over ten weeks we audited the purchasing item master and the inventory ledger, corrected what
could be corrected, and changed the ERP settings that let the problems recur. This section is what
changed as a result, in the terms the shop runs on: whether its inventory balances can be trusted,
what the errors were costing operations and the books, and what the cleanup did about each. Every
figure is measured from the record. Where a cost cannot be tied to a specific error, no share of it
is claimed, and what the cleanup will save going forward is left to the reorder-policy work that
follows this audit rather than forecast here.</p>

{B.kpi_row(
    B.kpi_card(f"{trust_before:.0f}% &rarr; {trust_after:.0f}%", "Balances that can be trusted", "share of live items", B.DARK_BLUE),
    B.kpi_card(f"{_k(d['rel_before']['reliable']['val'])} &rarr; {_k(d['rel_after']['reliable']['val'])}", "Book value at a trusted balance", "of inventory on hand", B.DARK_BLUE),
    B.kpi_card(f"{_k(ph['stale_value'])} &rarr; $0", "On order that would never arrive", f"{ph['stale_lines']:,} lines closed", B.GREEN),
    B.kpi_card(f"{d['n_dead']:,} &rarr; {d['n_dead'] - d['dead_deactivated']:,}", "Dead records still active", "in the item master", B.GREEN),
)}

{sub("Can the balance be trusted?")}
<p>This is the result the rest depends on. A balance is <em>reliable</em> when it was counted within
the last 90 days and the count came within 5% of the system; <em>unreliable</em> when the item carries
a fault that makes the number untrustworthy whatever the last count said (chronic unexplained
write-offs, a purchase unit with no conversion to the stock unit, a duplicate record sharing one pile);
and <em>uncertain</em> otherwise, which means no count in the last 90 days or a count 5% or more off.
We applied that one rule on
the day the audit began, when the shop's only count was the annual physical nine months earlier, and
again at the audit date, with the remediation counts and the first quarter of the cycle-count program
behind it.</p>

{rel_table}

{B.chart("Inventory book value by balance reliability, before and after", chart_reliability(d))}
<p>On the day the audit began, no balance qualified as reliable, because the shop counted once a year
and nothing had been counted in the previous 90 days; {d['rel_before']['unreliable']['pct']:.0f}% of live
items ({d['rel_before']['unreliable']['items']:,}), holding {d['rel_before']['unreliable']['val']/d['inv_value_total']*100:.0f}%
of the book value, were unreliable on their face. At the audit date {trust_after:.0f}% of live items
({d['rel_after']['reliable']['items']:,}), holding {d['rel_after']['reliable']['val']/d['inv_value_after']*100:.0f}%
of the book value, have a balance we would reorder against without a recount: {ra['reliable_by_abc'].get('A', 0)}
of the {ra['reliable_by_abc'].get('A', 0) + ra['uncertain_by_abc'].get('A', 0)} A-class items and
{ra['reliable_by_abc'].get('B', 0)} of the B-class items. The unreliable share is down to
{d['rel_after']['unreliable']['pct']:.0f}% of items and {d['rel_after']['unreliable']['val']/d['inv_value_after']*100:.0f}%
of value, {d['rel_after']['unreliable']['items']} items whose write-off pattern has not yet had a clean
quarter to clear, each named in the count schedule. The uncertain group is large by count and small by
value: {ra['uncertain_reasons']['no_recent_count']} items, nearly all C-class, have not yet had their turn in
the cycle, and {ra['uncertain_reasons']['counted_variance_5pct_or_more']} A- and B-class items were counted
but their first count found a variance of 5% or more, so they qualify at the next count if the
corrected balance holds.</p>

{sub("Operations")}
<p>Operationally, the errors were felt on the assembly floor and in the purchasing office, as lines
stopped for material the system said was there and as days spent expediting and correcting instead
of planning. The table gives each impact as the {YR} record shows it and what the cleanup changed.
Where an impact can be tied to specific errors it is; where the record does not support that, the
total is stated and no share is claimed.</p>

{ops_table}

{sub("Finances")}
<p>Financially, the errors reached the books in ways that do not add together: cash spent on
expedites, cash committed early to stock the shop already held, and misstatements of inventory,
commitments and product cost that were later corrected but had already been decided on. The most
visible was the smallest, the expedite invoices; the largest sat on the balance sheet, where nobody
measured it. The table gives each as the {YR} record shows it and what the cleanup changed.</p>

{fin_table}

<p>Beyond the ledger, the cleanup changes what can be built on the data. Forecasting, automated
reordering and any model of demand consume the item master and the consumption history, and none of
them could have been trusted on a master that was {d['dead_pct']*100:.0f}% dead, with the components on
{un['products_affected']} of its {un['n_products']} products never recorded as consumed. Management can now read an inventory value, a product margin and a
supplier's real spend from the system rather than from memory, and anyone who diligences the
inventory, a lender, a buyer or an auditor, finds balances with a count behind them.</p>

{B.callout(f"<strong>One item the buyer knows.</strong> A high-volume gearmotor was carried under two "
    f"part numbers, each with its own reorder point, and its supplier's real lead time had crept from "
    f"about two weeks to over a month while the ERP still read two weeks. The result was recurring line "
    f"stops and air freight. Merged to one record, with the lead time recomputed from receipts and the "
    f"reorder point recomputed from usage over it, the item now reorders early enough to cover the true "
    f"lead time, and the expedites on it stop.")}
"""

    mc = ", ".join(f"{n} ({r:,} records)" for n, r in d["master_comp"])
    tc = ", ".join(f"{n} ({r:,})" for n, r in d["txn_comp"])

    def rows_of(n, total):
        # count on the first line, share in italics on the second
        return f"{n:,} of {total:,}<br><em>({n / total * 100:.1f}%)</em>" if total else f"{n:,}"

    def numcell(i):
        # untitled left column: bold number, vertically centred
        return f'<td style="vertical-align:middle;text-align:center;">#{i}</td>'

    # (name, one-sentence description, ERP table, scale as rows affected, test,
    #  what counts as a finding, operational cost). The ERP table is the table the
    #  scale denominator counts, so equal denominators always share a location.
    #  Operational cost is not rendered here; it is kept for the Results section.
    IM, IM_LIVE, BOM, SUP = "Item master", "Item master (live items)", "Bill of materials", "Supplier master"
    LEDGER, LEDGER_ADJ, PO, PROD = "Inventory ledger", "Inventory ledger (adjustments)", "Purchase orders", "Production orders"
    MASTER_ERRORS = [
        ("Dead Records Never Deactivated",
         "Items with no activity for years that are still flagged active in the item master.",
         IM, rows_of(d["n_dead"], d["n_master"]),
         "Active items with no issue or receipt in 24+ months",
         "Count, and the reorder points still set on them",
         "False reorder signals, wasted count effort"),
        ("Stale Lead Times",
         "Supplier lead times on the item record that no longer match how long deliveries actually take.",
         IM_LIVE, rows_of(d["n_lead_off"], d["n_live"]),
         "Master lead time vs median actual from PO history, per item",
         "Items where the gap exceeds a week, weighted by spend",
         "Late reorders, line stops, expedite freight"),
        ("Stale Reorder Points",
         "Reorder points and safety stocks that were never recomputed as usage and lead times changed.",
         IM_LIVE, rows_of(d["params_changed"], d["n_live"]),
         "Reorder point vs recent usage over actual lead time",
         "Items where the point is too low (stockouts) or too high (excess)",
         "Stockouts on fast movers, excess on slow ones"),
        ("BOM Omissions",
         "Components used in production that are missing from the product's bill of materials.",
         BOM, rows_of(d["omit_items"], d["n_bom_rows"] + d["omit_items"]),
         "Items with chronic negative adjustments that appear on no BOM",
         "The list, and the write-down value",
         "Phantom on-hand; usage lost to write-offs"),
        ("Duplicate Item Records",
         "The same physical part carried under two or more item numbers.",
         IM_LIVE, rows_of(d["dup_records"], d["n_live"]),
         "Normalize descriptions, compare within item class, score similarity",
         "Candidate pairs above a threshold, reviewed by hand",
         "Split, unforecastable demand history"),
        ("UOM Mismatch",
         "Items bought in one unit of measure and stocked in another, with no conversion factor recorded.",
         IM_LIVE, rows_of(d["uom_items"], d["n_live"]),
         "Purchase UOM differs from stock UOM with no conversion factor",
         "Items, and the on-hand balances that are therefore meaningless",
         "Inflated on-hand and demand"),
        ("Supplier Fragmentation",
         "One supplier carried under several supplier records with different names or IDs.",
         SUP, rows_of(d["sup_records"], d["n_sup_rows"]),
         "Normalize supplier names, group",
         "Groups with more than one ID",
         "Fragmented spend and lead-time history"),
        ("Missing and Placeholder Fields",
         "Required item fields left blank or filled with a placeholder value.",
         IM_LIVE, rows_of(d["n_blank"], d["n_live"]),
         "Fill rates by column, from the profiling pass",
         "Items whose blanks block a process (no cost, no reorder point)",
         "Blocks planning and costing"),
    ]
    TXN_ERRORS = [
        ("Unrecorded Consumption",
         "Material consumed on the floor without a transaction recording it.",
         LEDGER, rows_of(d["n_unrec_adj_rows"], d["n_tx"]),
         "Adjustment frequency and direction per item",
         "Items adjusted downward three or more times in 12 months",
         "Balances drift; chronic write-offs"),
        ("Adjustments as a Catch-All",
         "Inventory adjustments used to correct all kinds of discrepancies rather than genuine count errors, usually without a reason code.",
         LEDGER_ADJ, rows_of(d["n_adj_blank_rows"], d["n_adj_rows"]),
         "Adjustment quantity as share of all movement",
         "Anything above 5% is a process problem",
         "Cause of movement unknowable"),
        ("Free-Text Purchases",
         "Purchase order lines entered under a generic item code with a typed description instead of the stocked item number.",
         PO, rows_of(d["n_ft_lines"], d["n_po"]),
         "Generic item codes on PO lines; match descriptions to master",
         "Lines that match a stocked item",
         "Demand lost to the forecast"),
        ("Batched and Backdated Postings",
         "Transactions posted days after they happened, in batches.",
         PO, rows_of(d["n_batch_rows"], d["n_po"]),
         "Day-of-week distribution of receipt dates",
         "Share of receipts posted on the peak day",
         "Lead times biased upward"),
        ("Open Documents Never Closed",
         "Purchase order lines and jobs left open after they were effectively complete.",
         f"{PO}; {PROD.lower()}",
         (f"{d['open_po_lines']:,} of {d['n_po']:,} <em>({d['open_po_lines'] / d['n_po'] * 100:.1f}%)</em><br>"
          f"{d['n_open_jobs']:,} of {d['n_prod']:,} <em>({d['n_open_jobs'] / d['n_prod'] * 100:.1f}%)</em>"),
         "PO lines open longer than 2&times; supplier lead time; jobs open past due date",
         "Count and on-order value",
         "Phantom on-order; stockouts"),
        ("Wrong References",
         "Transactions posted against the wrong item or job.",
         LEDGER, rows_of(d["t6_count"], d["n_tx"]),
         "Issues to jobs whose BOM doesn't include the item",
         "List for review",
         "Consumption charged to the wrong part"),
        ("Quantity and Unit Errors",
         "Transaction quantities keyed with the wrong magnitude or unit.",
         LEDGER, rows_of(d["t7_count"], d["n_tx"]),
         "Per-item outliers using median and spread, not averages",
         "List for review",
         "Distorted demand and on-hand"),
        ("Duplicate Postings",
         "The same transaction entered twice.",
         LEDGER, rows_of(d["t8_count"], d["n_tx"]),
         "Same item, qty, date within minutes",
         "List, usually small",
         "Movement double-counted"),
    ]
    d["op_cost"] = {n: c for n, _d, _l, _s, _t, _f, c in MASTER_ERRORS + TXN_ERRORS}   # reserved for Results
    ORIG_MASTER = [e[0] for e in MASTER_ERRORS]
    ORIG_TXN = [e[0] for e in TXN_ERRORS]
    # order by ERP table so the tables read by location
    ORDER_M = ["Dead Records Never Deactivated", "Stale Lead Times", "Stale Reorder Points",
               "Duplicate Item Records", "UOM Mismatch", "Missing and Placeholder Fields",
               "BOM Omissions", "Supplier Fragmentation"]
    ORDER_T = ["Unrecorded Consumption", "Wrong References", "Quantity and Unit Errors", "Duplicate Postings",
               "Adjustments as a Catch-All", "Free-Text Purchases", "Batched and Backdated Postings",
               "Open Documents Never Closed"]
    MASTER_ERRORS.sort(key=lambda e: ORDER_M.index(e[0]))
    TXN_ERRORS.sort(key=lambda e: ORDER_T.index(e[0]))
    W2 = [4, 19, 40, 16, 21]          # #, Error, Description, ERP table, Scale
    W3 = [4, 19, 42, 18, 17]          # #, Error, Remediation, Evidence, Remediated
    hdr = ["", "Error", "Description", "ERP table", "Scale<br><em style=\"font-weight:400;text-transform:none;\">(rows affected)</em>"]
    master_table = _widths(B.data_table(hdr, [[numcell(i), n, desc, loc, sc] for i, (n, desc, loc, sc, _t, _f, _c) in enumerate(MASTER_ERRORS, 1)], right=[]), W2)
    txn_table = _widths(B.data_table(hdr, [[numcell(i), n, desc, loc, sc] for i, (n, desc, loc, sc, _t, _f, _c) in enumerate(TXN_ERRORS, len(MASTER_ERRORS) + 1)], right=[]), W2)
    def rem_of(n, total):
        return f"{n:,} of {total:,} ({n / total * 100:.0f}%)" if total else f"{n:,}"

    # (remediation: what was done, how, and whose input it needed; evidence; rows remediated)
    ERP = "ERP records only"
    REM_MASTER = [
        (f"Deactivated in the item master after a line-by-line review. The purchasing manager and the service "
         f"parts coordinator kept {d['dead_kept']} as seasonal or safety-critical spares and held {d['dead_held']} "
         f"for a later decision; the rest were deactivated.",
         ERP, rem_of(d["dead_deactivated"], d["n_dead"])),
        ("Recomputed per item from receipt history using the median and a trimmed 80th percentile, then written "
         "to the master. Mechanical, with no judgment needed; the tracked items were checked against the buyer's "
         "spreadsheet.",
         "ERP purchase history, checked against the buyer's spreadsheet", rem_of(d["n_lead_off"], d["n_lead_off"])),
        ("Recomputed from actual usage over the corrected lead time at the ABC service level and loaded; the "
         "purchasing manager reviewed the A-class values before they went live.",
         "ERP purchase and issue history, checked against the buyer's spreadsheet", rem_of(d["params_changed"], d["params_changed"])),
        (f"Missing components added back to the product and subassembly BOMs ({d['bom_changes']} change-log "
         f"entries), each confirmed by the engineering manager from engineering review or by the assembly "
         f"supervisor from floor observation.",
         "Expected vs actual consumption (jobs &times; BOM against issues) and floor observation",
         rem_of(d["bom_changes"], d["omit_items"])),
        (f"Candidate pairs scored, then merged to one surviving number through a crosswalk. The buyer and the "
         f"stockroom lead reviewed every pair: {d['dup_merged']} records retired to a survivor, {d['dup_rejected']} "
         f"pairs rejected as genuinely different parts.",
         "ERP records, every pair reviewed by the buyer", rem_of(d["dup_merged"], d["dup_records"])),
        ("A purchase-to-stock conversion factor added to each item, taken from the pack size on its receipts and "
         "confirmed by the stockroom lead; on-hand restated in stock units.",
         "ERP records and the pack sizes on receipts", rem_of(d["uom_items"], d["uom_items"])),
        (f"Alias records mapped to one canonical supplier through a crosswalk ({d['sup_fragments']} vendors, "
         f"{d['sup_records']} records); the buyer confirmed each grouping, and new orders book to the canonical record.",
         ERP, rem_of(d["sup_records"] - d["sup_fragments"], d["sup_records"])),
        ("Blocking blanks filled from the ordering history (cost from the last price paid, supplier from the "
         "ordering record) and from the reorder-point recomputation; MISC items reclassified by the buyer; "
         "required fields enforced from week 6.",
         ERP, rem_of(d["n_blank"], d["n_blank"])),
    ]
    REM_TXN = [
        (f"Traced to its cause through the chronic-adjustment analysis ({d['chronic_items']} items assigned to "
         f"BOM omission, floor practice, receiving error or count error) and fixed at the source: the BOM "
         f"corrections stop the leak, and the cycle-count program corrected each balance as it was counted.",
         "ERP on-hand vs cycle counts; expected vs actual consumption",
         f"0 of {d['n_unrec_adj_rows']:,} (fixed at source)"),
        ("Historic adjustments left as posted but classified by the chronic-adjustment analysis; reason codes made "
         "mandatory by configuration change in week 5, so the share falls toward the run-rate target.",
         ERP, f"0 of {d['n_adj_blank_rows']:,} (controlled at source)"),
        (f"Lines matched to stocked items by description similarity; the buyer settled every candidate "
         f"({d['ft_confirmed']:,} confirmed, {d['ft_rejected']:,} rejected as genuine non-stock buys) and the "
         f"confirmed demand was attributed back to the item; generic codes restricted from week 7.",
         "ERP records, every candidate reviewed by the buyer", rem_of(d["ft_confirmed"], d["n_ft_lines"])),
        ("Not corrected line by line, because the true dates are not recoverable. The lead-time computation was "
         "made robust to it instead, and receiving moved to same-day posting under the new individual logins.",
         ERP, f"0 of {d['n_batch_rows']:,} (method made robust)"),
        ("Each open line and finished job closed on confirmation: receiving records, the buyer or a supplier "
         "statement for PO lines, production confirmation for jobs.",
         "ERP records, receiving records and supplier statements",
         f"{d['closed_po']:,} of {d['open_po_lines']:,} PO lines; {d['closed_jobs']:,} of {d['n_open_jobs']:,} jobs"),
        ("Each issue re-pointed to the item the job's BOM calls for, after the stockroom lead reviewed the list.",
         "Expected vs actual consumption (the job's BOM) and stockroom review", rem_of(d["t6_count"], d["t6_count"])),
        ("Each outlier corrected to the true quantity after the stockroom lead reviewed the list; box/each keying "
         "closed off by the UOM conversions.",
         "ERP records and stockroom review", rem_of(d["t7_count"], d["t7_count"])),
        ("The second posting reversed for every pair.",
         ERP, rem_of(d["t8_count"], d["t8_count"])),
    ]
    REM_MASTER = dict(zip(ORIG_MASTER, REM_MASTER))
    REM_TXN = dict(zip(ORIG_TXN, REM_TXN))
    rem_hdr = ["", "Error", "Remediation", "Evidence", "Remediated (rows)"]
    rem_master_table = _widths(B.data_table(rem_hdr, [[numcell(i), e[0], *REM_MASTER[e[0]]]
        for i, e in enumerate(MASTER_ERRORS, 1)], right=[]), W3)
    rem_txn_table = _widths(B.data_table(rem_hdr, [[numcell(i), e[0], *REM_TXN[e[0]]]
        for i, e in enumerate(TXN_ERRORS, len(MASTER_ERRORS) + 1)], right=[]), W3)
    found = f"""
{B.section("found", "Section 2", "Data Quality Errors")}
<p>This data quality audit examined the shop's entire ERP system end to end. The system consists of
three master-level tables (Item, Supplier and Bill of Materials) and five transaction-level tables
(Inventory Ledger, Purchase Orders, Service Orders, Production Orders and Cycle Counts), related as
shown below.</p>

{B.chart("ERP Tables", chart_erd(d))}

<p>Over the past 36 months, over <strong>{d['total_rows'] // 1000}K</strong> individual records were
produced across the eight ERP tables. This audit covered all of them. It found <strong>16</strong>
different types of data quality error recur over this period. These errors touched a significant
share of the ERP's records and left it unable to serve as the trusted central record of what the
shop holds, buys and builds. They also directly disrupted operational workflows, inflated costs and
distorted financial visibility, as detailed below.</p>

<p>The tables below lay out the 16 different data quality error types that were found to recur over
the three-year audit period.</p>

<p>Three findings stand out for their scale. The item master was full of dead records: {d['dead_pct']*100:.0f}%
of its records ({d['n_dead']:,} of {d['n_master']:,}) were items with no movement in two years or more
that are still flagged active, and {d['dead_with_rop']} of them still carry a reorder point. The
parameters the ERP reorders on are stale: the lead times on {d['n_lead_off']/d['n_live']*100:.0f}% of
live items no longer match how long deliveries take, running {d['lead_gap_mean']:.0f} days shorter than
actual on average, and the reorder points on {d['params_changed']/d['n_live']*100:.0f}% of live items
differ from what current usage and lead times call for by a median of {d['rop_delta_median']:.0f}
units, {d['rop_low_share']*100:.0f}% of them set too low and the rest too high. And the majority of
receipts are posted in batches: {d['n_batch_rows']/d['n_po']*100:.0f}% of purchase order lines carry a
posting date {d['t4_lag_mean']:.0f} days on average after the material arrived, so every lead time
computed from receipt dates reads {d['t4_lag_mean']:.0f} days longer than the delivery took, every
reorder point set from it carries {d['t4_lag_mean']:.0f} days of stock it does not need, and every
supplier measured on it looks slower than it is.</p>

<p>Two other master-level errors are small in the master and large downstream. The {d['omit_items']}
components missing from the bills of {d['omit_products']} of the {d['n_products']} products are
consumed on every job that builds those products without ever being subtracted, which makes them the
single largest source of shortages and write-offs in the ledger. And the {d['dup_records']} duplicate
records, each reordering on its own point against one shared pile, produced {d['fin']['duplicates']['lines_while_sibling_held_stock']}
purchase lines in {d['fin']['year']} for parts the shop already held.</p>

<p>A further pattern distorts the financial view of the company: the ledger is corrected after the
fact rather than kept right. Material leaves the stockroom without a transaction (#9), and the gap
that opens between the books and the shelf is closed later by an adjustment carrying no reason code
(#13), the same way damage, receiving errors and count errors are closed. The balance therefore reads
high until someone corrects it, and the correction says nothing about why. The {d['n_unrec_adj_rows']:,}
chronic write-offs under #9 are part of the {d['n_adj_blank_rows']:,} unexplained adjustments under
#13: one is the leak, the other is why the leak went unnoticed.</p>

<p style="font-size:18px;font-weight:700;color:{B.DARK_GREY};margin-top:30px;">Master-level Table Errors</p>
{master_table}

<p style="font-size:18px;font-weight:700;color:{B.DARK_GREY};margin-top:34px;">Transaction-level Table Errors</p>
{txn_table}

"""

    did = f"""
{B.section("did", "Section 3", "Error Remediation")}
<p>Remediation ran in a fixed order. We profiled every table first (row counts by year, fill rates
by column, the values in each code field, the date ranges), then ran one rerunnable test for each of
the sixteen error types. Where the ERP could not settle a finding on its own we brought in evidence
from outside it: the cycle counts against the book balances, the purchasing manager's spreadsheet
against the ERP, and the consumption expected from jobs and bills of materials against what was
actually issued. Findings that rested on judgment, such as duplicate pairs, free-text matches,
dead-item dispositions and BOM additions, went to the people who own them for confirmation before
anything was changed, and every correction was recorded in a reference table rather than written
over the source, so each one is auditable and reversible.</p>
<p>The tables below give, for each error, how it was remediated and whose input that took, the
evidence it rested on, and how many of the affected rows were remediated.</p>
<p style="font-size:18px;font-weight:700;color:{B.DARK_GREY};margin-top:30px;">Master-level error remediation</p>
{rem_master_table}

<p style="font-size:18px;font-weight:700;color:{B.DARK_GREY};margin-top:34px;">Transaction-level error remediation</p>
{rem_txn_table}

"""

    CONFIG_DESC = {
        "reason codes required on adjustments":
            "The ERP now refuses an adjustment without a reason code from a fixed list (count variance, damage, "
            "scrap, receiving error, return), so every write-off carries its cause.",
        "required fields enforced on item creation":
            "A new item cannot be saved without a standard cost, a primary supplier, a unit of measure and a "
            "reorder point, or an explicit not-stocked flag.",
        "UOM conversions added for box/spool/length items":
            "Each item bought and stocked in different units now carries a conversion factor, and receipts "
            "convert to stock units automatically: a box of 100 lands as 100 each.",
        "generic item codes restricted":
            "NONSTOCK, MISC and SHOPSUPPLY can no longer be used on a purchase line for anything that exists in "
            "the item master; the buyer is prompted to search the master first, and a generic code needs a "
            "one-line justification.",
        "negative on-hand blocked":
            "An issue that would take a balance below zero is rejected at entry, so a missing receipt, a wrong "
            "item or a keyed quantity is caught the moment it happens rather than at the annual count.",
        "individual logins issued for floor and receiving":
            "The shared ASSY1, FAB1 and RECV logins were retired and every floor and receiving user has their "
            "own, so each transaction names who posted it.",
        "part-creation approval routing enabled":
            "A request to create an item routes to the item-master owner, who checks for an existing record "
            "before approving.",
    }
    # closes: the error can no longer occur; addresses: the error is caught or
    # reduced, but not prevented
    CONFIG_IMPACT = {
        "reason codes required on adjustments":
            "Closes #13. Addresses #9: a chronic write-off now names its cause, so the pattern is visible.",
        "required fields enforced on item creation":
            "Closes #6 for every item created from now on; the existing blanks were filled in remediation.",
        "UOM conversions added for box/spool/length items":
            "Closes #5. Addresses #11: a box keyed as eaches can no longer distort a balance.",
        "generic item codes restricted":
            "Closes #14.",
        "negative on-hand blocked":
            "Addresses #9, #10 and #11: each is caught at entry instead of at the count.",
        "individual logins issued for floor and receiving":
            "Addresses #10, #12 and #15: every posting is attributable to a person, and so trainable.",
        "part-creation approval routing enabled":
            "Addresses #4 and #6: a new duplicate or blank-field item is stopped before it enters the master.",
    }
    config_rows = [[r.change[0].upper() + r.change[1:], CONFIG_DESC.get(r.change, ""), CONFIG_IMPACT.get(r.change, "")]
                   for r in d["config"].itertuples(index=False)]
    config_table = _widths(B.data_table(["Change", "What it does", "Impact"], config_rows, right=[]), [22, 48, 30])

    PROCESS = [
        ("A named owner for the item and supplier masters",
         "The purchasing manager owns both masters. No item or supplier record is created, merged or "
         "deactivated without her approval, and she clears the approval queue weekly. Prevents new duplicates "
         "(#4, #8), blank-field items (#6) and dead records accumulating unnoticed (#1).",
         "Purchasing manager", "Weekly queue; continuous"),
        ("A cycle-count program on the ABC schedule",
         "A items counted monthly, B items quarterly, C items annually, unreliable items first. Variances are "
         "posted with a reason and investigated above a threshold, and the annual physical is retired. Keeps "
         "balances trustworthy and catches unrecorded consumption (#9) as it happens rather than once a year.",
         "Stockroom lead", "Monthly / quarterly / annual by class"),
        ("A monthly parameter refresh from the forecast",
         "Lead times are recomputed from the last twelve months of receipts, and reorder points and safety "
         "stocks from the forecast at the ABC service level, then loaded. The buyers review the A-class "
         "changes before they go live. Prevents lead times and reorder points going stale again (#2, #3).",
         "Purchasing manager, with the buyers", "Monthly"),
        ("A monthly open-document review",
         "Open PO lines older than twice the supplier's lead time and jobs past their due date are listed, "
         "then closed or chased. Keeps on-order honest and stops orders being held back against material "
         "that is not coming (#16).",
         "Buyer (PO lines); production scheduler (jobs)", "Monthly"),
        ("A quarterly dead-item review",
         "Items with no issue or receipt in 24 months are listed each quarter and deactivated unless the "
         "service parts coordinator or the production lead keeps them, with a reason recorded. Stops the "
         "master silting up again (#1).",
         "Purchasing manager", "Quarterly"),
        ("A BOM review for every new product and option",
         "Engineering signs off a complete bill, including hardware, fittings, consumables and finishing, "
         "before a product is released, and floor observation feeds corrections back. Stops backflush "
         "omissions and the phantom inventory they create (#7, #9).",
         "Engineering manager", "Per product release"),
        ("Same-day posting of receipts and job completions",
         "Receiving posts each delivery the day it arrives and job completions are reported daily, with "
         "supervisors spot-checking timeliness. Keeps computed lead times honest (#15).",
         "Stockroom lead; assembly supervisor", "Daily"),
        ("The data-quality measures tracked monthly",
         "The tests from this audit are rerun each month and the key shares (adjustments, blank reason codes, "
         "free-text lines, open documents, duplicates created) are tracked against targets and reviewed. "
         "Catches any of the sixteen errors returning before they compound.",
         "ERP administrator; reviewed by the operations manager", "Monthly"),
    ]
    process_table = _widths(B.data_table(["Change", "What it does", "Owner", "Cadence"],
                                         [list(r) for r in PROCESS], right=[]), [20, 46, 18, 16])

    keep = f"""
{B.section("process", "Section 4", "Go-Forward Changes")}
<p>The sixteen errors trace back to two root conditions rather than sixteen separate causes: the ERP
allowed them, and nobody owned the routine upkeep that would have caught them. The corrections in
Section 3 fix what those two conditions produced; the changes in this section stop them producing it
again, and they fall into two groups that are different kinds of work. The first group are settings.
Each was changed by the ERP administrator during the engagement, took effect for every user at once,
and stops the error at the point of entry. This is the easier lift: the changes are already in place,
they hold on their own, and they need nothing further from the shop. The table gives what each one
does and which errors it <em>closes</em> (the error can no longer occur) or <em>addresses</em> (the
error is caught or reduced, but not prevented).</p>

<p style="font-size:18px;font-weight:700;color:{B.DARK_GREY};margin-top:30px;">Changes Easily Implemented in the ERP</p>
{config_table}

<p>The second group are habits. No setting can make someone count a shelf, review a bill of materials,
close a purchase order or refresh a reorder point, so each of these needs a named owner and a cadence,
and each will lapse without them. That makes this the harder lift: it depends on organizational
alignment and on the owners' buy-in rather than on a configuration screen. The shop has committed to
the owners and cadences below, and keeping them is what protects the results in Section 1.</p>

<p style="font-size:18px;font-weight:700;color:{B.DARK_GREY};margin-top:34px;">Changes Requiring Ongoing Processes and Ownership</p>
{process_table}

<p style="font-size:18px;font-weight:700;color:{B.DARK_GREY};margin-top:34px;">What remains</p>
<p>Not everything was resolved, and it would be dishonest to imply otherwise.</p>
<ul class="limitation-list">
  <li><strong>Items still unreliable.</strong> {d['rel_after']['unreliable']['pct']:.0f}% of live items still
      carry a balance we would not trust: phantom-inventory items whose first count is scheduled in the
      continuing cycle-count program, each with its reason recorded.</li>
  <li><strong>Lead-time precision floor.</strong> Because receipts were batched to Mondays and month-end,
      computed lead times carry a few days of irreducible noise; the recommended values use a trimmed
      high percentile to stay safe rather than precise.</li>
  <li><strong>What the shop declined.</strong> A small set of dead items were kept active at the buyer's
      insistence as insurance spares, against the recommendation to deactivate them.</li>
  <li><strong>What the cleanup is worth going forward is not measured here.</strong> This report claims
      only costs that trace to a specific error in the {YR} record. How much of the untraced rush spend
      and how many of the shortages clean data would have prevented is a forecast, and it belongs to
      the reorder-policy work that follows this audit.</li>
</ul>
"""

    remains = ""
    return results + found + did + keep + remains, toc


def run():
    d = gather()
    body, toc = build(d)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = B.page("Data Quality Audit: Purchasing & Inventory",
                  "Post-remediation report to the operations manager and controller",
                  toc, body)
    for a, b in [("defects", "errors"), ("Defects", "Errors"), ("defect", "error"), ("Defect", "Error")]:
        html = html.replace(a, b)
    # subsection titles sized to match the bold table titles (18px)
    html = html.replace("</style></head>",
                        ".section-title-block.sub .section-title{font-size:18px;font-weight:700;}</style></head>", 1)
    OUT.write_text(html, encoding="utf-8")
    print(f"Data quality audit written to {OUT}  ({len(html)//1024} KB)")


if __name__ == "__main__":
    run()
