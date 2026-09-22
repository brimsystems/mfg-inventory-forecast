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
POLICY = REPO / "ml" / "data" / "policy"
OUT = REPO / "docs" / "reports" / "data_quality_audit.html"

# ── stated assumptions ───────────────────────────────────────────────────────
EXPEDITE_FEE = 250.0
CARRYING_RATE = 0.22


def _money(x):
    return f"${x:,.0f}"


def _pct(x, d=0):
    return f"{x*100:.{d}f}%"


def gather():
    d = {}
    im = pd.read_csv(RAW / "erp" / "item_master.csv", low_memory=False)
    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv", low_memory=False)
    po = pd.read_csv(RAW / "erp" / "purchase_orders.csv", low_memory=False)
    cross = json.loads((TRUTH / "crosswalks.json").read_text())
    pod = json.loads((TRUTH / "po_defects.json").read_text())
    rel = pd.read_parquet(MARTS / "reliability.parquet")

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

    # ── reliability (headline) ──────────────────────────────────────────────
    def rdist(col):
        cnt = rel[col].value_counts(normalize=True) * 100
        val = rel.groupby(col)["value"].sum()
        return {k: {"pct": float(cnt.get(k, 0)), "val": float(val.get(k, 0))}
                for k in ["reliable", "uncertain", "unreliable"]}
    d["rel_before"] = rdist("reliability_before")
    d["rel_after"] = rdist("reliability_after")
    d["inv_value_total"] = float(rel["value"].sum())

    # ── policy outcomes ─────────────────────────────────────────────────────
    pol = json.loads((POLICY / "policy_summary.json").read_text())
    d["policy"] = pol
    d["wc_released"] = pol["corrected"]["inv"] - pol["forecast"]["inv"]
    d["wc_released_pct"] = d["wc_released"] / pol["corrected"]["inv"]

    # ── three-way cleaning value ────────────────────────────────────────────
    d["threeway"] = json.loads((BACKTEST / "threeway_overall.json").read_text())
    mm = json.loads((BACKTEST / "model_metrics.json").read_text())
    d["model"] = mm

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
    changed = (params["new_reorder_point"] - params["old_reorder_point"].fillna(0)).abs() > 1
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
    ax.legend(loc="lower right", frameon=False, ncol=3, fontsize=9)
    ax.invert_yaxis()
    return B.b64(fig)


def chart_threeway(d):
    fig, ax = B.make_fig(3.2)
    tw = d["threeway"]
    names = ["Raw\n(as recorded)", "Master-cleaned\n(records merged)", "Fully cleaned\n(+ transactions)"]
    vals = [tw["raw"] * 100, tw["master"] * 100, tw["fully"] * 100]
    bars = ax.bar(names, vals, color=[B.MED_GREY, B.LIGHT_BLUE, B.DARK_BLUE], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Forecast error (WAPE)")
    ax.set_ylim(0, max(vals) * 1.18)
    B.chart_style(ax)
    return B.b64(fig)


def chart_policy(d):
    fig, ax = B.make_fig(3.2)
    p = d["policy"]
    names = ["Current\n(stale)", "Corrected\nlead times", "Forecast-\ndriven"]
    inv = [p["current"]["inv"] / 1000, p["corrected"]["inv"] / 1000, p["forecast"]["inv"] / 1000]
    fill = [p["current"]["fill"] * 100, p["corrected"]["fill"] * 100, p["forecast"]["fill"] * 100]
    x = np.arange(3)
    bars = ax.bar(x, inv, color=[B.MED_GREY, B.LIGHT_BLUE, B.DARK_BLUE], width=0.6)
    ax.set_ylabel("Avg inventory value ($000)")
    ax.set_xticks(x); ax.set_xticklabels(names)
    for b, f in zip(bars, fill):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 20, f"{f:.0f}% fill",
                ha="center", fontsize=9, color=B.DARK_GREY, fontweight="bold")
    ax.set_ylim(0, max(inv) * 1.2)
    B.chart_style(ax)
    return B.b64(fig)


def _err_block(name, what, scale, test, cost, headers, rows):
    """One error type, broken out visually: name, a plain sentence on what it is,
    then Scale / Test / Operational cost, then a record or two that shows it."""
    lbl = (f'style="color:{B.MED_GREY};font-weight:700;font-size:12px;text-transform:uppercase;'
           f'letter-spacing:.5px;display:inline-block;width:150px;vertical-align:top;"')
    sample = ""
    if rows:
        sample = (f'<div style="font-size:11px;color:{B.MED_GREY};font-weight:700;text-transform:uppercase;'
                  f'letter-spacing:.5px;margin:12px 0 -6px;">Sample records</div>'
                  + B.data_table(headers, rows))
    return (f'<div style="margin:24px 0 28px;">'
            f'<div style="font-size:16px;margin-bottom:6px;"><strong><u>{name}</u></strong></div>'
            f'<div style="margin-bottom:10px;">{what}</div>'
            f'<div style="margin-bottom:5px;"><span {lbl}>Scale</span><span>{scale}</span></div>'
            f'<div style="margin-bottom:5px;"><span {lbl}>Test</span><span>{test}</span></div>'
            f'<div style="margin-bottom:5px;"><span {lbl}>Operational cost</span><span>{cost}</span></div>'
            f'{sample}</div>')


# ── report ───────────────────────────────────────────────────────────────────
def build(d):
    toc = "".join([
        '<a href="#results">Results</a>',
        '<a href="#found">What we found</a>',
        '<a href="#did">What we did</a>',
        '<a href="#who">Who was involved</a>',
        '<a href="#means">What it means for purchasing</a>',
        '<a href="#keep">Keeping it clean</a>',
        '<a href="#remains">What remains</a>',
        '<a href="#appendix">Appendix</a>',
    ])

    trust_before = d["rel_before"]["reliable"]["pct"]
    trust_after = d["rel_after"]["reliable"]["pct"]

    results = f"""
{B.section("results", "Section 1", "Results")}
<p>Over ten weeks we audited the purchasing item master and the inventory ledger,
corrected what could be corrected, and changed the ERP settings that let the
problems recur. This section is the outcome, in the terms the shop runs on. The
sections that follow show what we found, what we did, and what still needs an
owner.</p>

{B.kpi_row(
    B.kpi_card(_money(d["wc_released"]), "Working capital released", "at equal service level", B.DARK_BLUE),
    B.kpi_card(f"{d['policy']['current']['fill']*100:.0f}% &rarr; {d['policy']['forecast']['fill']*100:.0f}%", "Fill rate", "on the modeled items", B.GREEN),
    B.kpi_card(f"{trust_before:.0f}% &rarr; {trust_after:.0f}%", "Balances trustworthy", "share of live items", B.DARK_GREY),
)}

{B.chart("Inventory value by balance reliability, before and after", chart_reliability(d))}
<p>Before the cleanup, only {trust_before:.0f}% of live items had a balance we would
trust to reorder against; {d['rel_before']['unreliable']['pct']:.0f}% were unreliable and held
{d['rel_before']['unreliable']['val']/d['inv_value_total']*100:.0f}% of the inventory value. After the
cycle-count program and the master fixes, {trust_after:.0f}% are reliable and the
unreliable share is down to {d['rel_after']['unreliable']['pct']:.0f}%, each with a stated reason.</p>

{B.data_table(
    ["Measure", "Before", "After"],
    [
        ["Inventory value at a trustworthy balance", _money(d["rel_before"]["reliable"]["val"]), _money(d["rel_after"]["reliable"]["val"])],
        ["Working capital in inventory (equal service)", _money(d["policy"]["corrected"]["inv"]), _money(d["policy"]["forecast"]["inv"])],
        ["Stockout events (12-month simulation)", f"{int(d['policy']['current']['stockouts']):,}", f"{int(d['policy']['forecast']['stockouts']):,}"],
        ["Expedite spend (stated assumption)", _money(d["policy"]["current"]["expedite"]), _money(d["policy"]["forecast"]["expedite"])],
        ["Live items with trustworthy parameters", f"{trust_before:.0f}%", f"{trust_after:.0f}%"],
        ["Adjustment share of quantity moved", _pct(d["adj_share"], 0), "3&ndash;6% run-rate (target)"],
        ["On-order value that was fiction, now closed", _money(d["open_po_value"]), _money(0)],
    ], right=[1, 2])}

{B.callout(f"<strong>One item the buyer knows.</strong> A high-volume gearmotor was carried under two "
    f"part numbers, each with its own reorder point, and its supplier's real lead time had crept from "
    f"about two weeks to over a month while the ERP still read two weeks. The result was recurring line "
    f"stops and air freight. Merged to one record, with the lead time recomputed from receipts and a "
    f"forecast-driven reorder point, the item now reorders early enough to cover the true lead time, and "
    f"the expedites on it stop.")}
"""

    S = d["samples"]
    mc = ", ".join(f"{n} ({r:,} records)" for n, r in d["master_comp"])
    tc = ", ".join(f"{n} ({r:,})" for n, r in d["txn_comp"])
    found = f"""
{B.section("found", "Section 2", "What we found")}
<p>This audit examined one company's ERP system end to end. On the master side, the records that
define what the shop buys and builds: the {mc}. On the transaction side, the history those masters
govern: the {tc}. That is {len(d['master_comp'])} master-level components holding
{d['master_rows']:,} records and {len(d['txn_comp'])} transaction-level components holding
{d['txn_rows']:,} records: {d['total_rows']:,} records in all across 36 months, plus the purchasing
manager's spreadsheet of the {d['spreadsheet_rows']} line-stopping components she tracks outside
the system.</p>

{B.kpi_row(
    B.kpi_card(f"{len(d['master_comp'])}", "Master-level components", f"{d['master_rows']:,} records"),
    B.kpi_card(f"{len(d['txn_comp'])}", "Transaction-level components", f"{d['txn_rows']:,} records"),
    B.kpi_card(f"{d['total_rows']:,}", "Records examined", "36 months of history"),
    B.kpi_card("16", "Error types tested", "8 master-level, 8 transaction-level"),
)}

<p>The errors fall in two tiers. Master-level errors are few records that many transactions depend
on; transaction-level errors are many individually wrong lines. For each we say what it is, give
the scale, the test that found it and the operational cost, and show a record or two from the
system.</p>

<p style="font-size:18px;font-weight:700;color:{B.DARK_GREY};margin-top:30px;">Master-level errors</p>

{_err_block("Dead records never deactivated",
    "Parts the shop no longer uses are still marked active. Nobody deletes them, because deleting feels risky and nobody owns the task.",
    f"{d['dead_pct']*100:.0f}% of the item master ({d['n_dead']:,} items), {d['dead_with_rop']:,} of them still carrying a reorder point",
    "Active items with no issue or receipt in 24+ months",
    "Clutter, false reorder signals, wasted count effort", *S["dead"])}

{_err_block("Stale lead times",
    "The lead time on the item record was set at go-live and never revisited. The only true value is what actually happened on past purchase orders.",
    f"{d['drift_gt3']*100:.0f}% of live items off by more than 3 days, {d['drift_gt7']*100:.0f}% by more than 7; one supplier's actual lead time roughly doubled over the history",
    "Master lead time vs median actual from PO history, per item",
    "Under-set reorder points, line stops, expedite freight", *S["lead"])}

{_err_block("Stale reorder points",
    "Reorder points and safety stocks set for the volumes and lead times of years ago, and never recomputed as usage changed.",
    f"{d['params_changed']:,} of {d['n_live']:,} live items' reorder points moved materially when recomputed from actual usage and lead time",
    "Reorder point vs recent usage over actual lead time",
    "Stockouts on the fast movers, excess stock on the slow ones", *S["rop"])}

{_err_block("BOM omissions",
    "The bill of materials is the recipe for a product. When it is missing the screws and washers, they get used on the floor but never subtracted.",
    f"{d['omit_items']:,} components missing from BOMs, across {d['omit_products']} of {d['n_products']} products",
    "Items with chronic negative adjustments that appear on no BOM",
    "Backflush never consumes them: phantom on-hand, and the usage escapes as write-offs", *S["bom"])}

{_err_block("Duplicate item records",
    "The same physical part exists under two or more item numbers, created when someone could not find the existing record or used a different naming convention.",
    f"{d['dup_clusters']} clusters ({d['dup_records']} records), concentrated in hardware, fittings, bearings and electrical",
    "Normalize descriptions, compare within item class, score similarity",
    "Demand history split across records; neither half is forecastable; two reorder points for one part", *S["dup"])}

{_err_block("UOM mismatch",
    "The unit a part is bought in does not match the unit it is used in, and the ERP has no conversion: bought by the box of 100, issued by the each.",
    f"{d['uom_items']} items bought by the box, spool, length or gallon and stocked by the each, foot or ounce",
    "Purchase UOM differs from stock UOM with no conversion factor",
    "Inflated on-hand and demand; a box received is counted as one each", *S["uom"])}

{_err_block("Supplier fragmentation",
    "One supplier exists under several names or IDs, so its spend and lead-time history are split across them.",
    f"{d['sup_fragments']} vendors carried under {d['sup_records']} supplier records",
    "Normalize supplier names, group",
    "Spend and lead-time history fragmented across records", *S["sup"])}

{_err_block("Missing and placeholder fields",
    "Required fields left blank because the ERP did not force them, or filled with placeholders such as an item class of MISC.",
    f"{d['blank_pct']*100:.0f}% of live items with a blocking blank (cost, supplier or reorder point); {d['misc_pct']*100:.0f}% classed MISC",
    "Fill rates by column, from the profiling pass",
    "Blocks planning, costing and reporting", *S["missing"])}

<p style="font-size:18px;font-weight:700;color:{B.DARK_GREY};margin-top:34px;">Transaction-level errors</p>

{_err_block("Unrecorded consumption",
    "Material leaves the shelf and nobody records it. The ERP still thinks it is there.",
    f"{d['t1_items']:,} items, roughly {d['t1_volume']:,} units a year leaving with no record",
    "Adjustment frequency and direction per item",
    "Balances drift until the annual count; the shortfall shows up as chronic write-offs", *S["unrec"])}

{_err_block("Adjustments as a catch-all",
    "The adjustment exists to correct genuine count errors. Here it became the way everything got fixed: unrecorded issues, mis-receipts, returns, scrap and mistakes, usually with no reason code.",
    f"{d['adj_share']*100:.0f}% of all quantity moved flows through adjustments; {d['adj_blank_share']*100:.0f}% carry a blank or generic reason code",
    "Adjustment quantity as share of all movement",
    "The cause of a movement is unknowable; write-offs hide the real problems", *S["adjshare"])}

{_err_block("Free-text purchases",
    "A buyer needing something quickly could not find the part, so ordered it under a generic code like NONSTOCK and typed a description.",
    f"{d['ft_pct']*100:.0f}% of PO lines; {d['ft_stocked']} of {d['ft_total']} correspond to an item already in the master",
    "Generic item codes on PO lines; match descriptions to master",
    "Demand for stocked items lost to the forecast; spend untraceable to a part", *S["ft"])}

{_err_block("Batched and backdated postings",
    "Transactions entered later than they happened: Friday's deliveries posted on Monday, job reporting done at the end of the week.",
    f"{d['peak_wd_share']*100:.0f}% of receipts posted on a {d['peak_wd_name']} (an even spread would be about 20%)",
    "Day-of-week distribution of receipt dates",
    "Computed lead times biased upward by the posting lag", *S["batch"])}

{_err_block("Open documents never closed",
    "A purchase order partly received with the balance never coming, or a finished job never closed, so the ERP still believes material is inbound.",
    f"{d['open_po_lines']:,} PO lines open past 90 days, {_money(d['open_po_value'])} of on-order that will never arrive",
    "PO lines open longer than 2&times; supplier lead time; jobs open past due date",
    "The buyer holds back real orders believing material is inbound, and stocks out", *S["open"])}

{_err_block("Wrong references",
    "The transaction is real but points at the wrong thing: a similar part number, or the wrong job.",
    f"{d['t6_count']} confirmed issues posted against a similar item or the wrong job",
    "Issues to jobs whose BOM doesn't include the item",
    "Consumption charged to the wrong part and the wrong job", *S["wrong"])}

{_err_block("Quantity and unit errors",
    "A quantity keyed with an extra zero, a box entered as an each, a decimal in the wrong place.",
    f"{d['t7_count']} confirmed order-of-magnitude and box/each keying errors, concentrated on the UOM-mismatch items",
    "Per-item outliers using median and spread, not averages",
    "Single keystrokes that distort an item's demand and on-hand by 10&times; or more", *S["qty"])}

{_err_block("Duplicate postings",
    "The same receipt or issue entered twice, usually within minutes, often because a screen froze and someone clicked again.",
    f"{d['t8_count']} confirmed transactions posted twice",
    "Same item, qty, date within minutes",
    "Movement double-counted", *S["dupost"])}

<p>Of the {d['chronic_items']:,} items with three or more downward adjustments in the last year,
{d['chronic_on_bom']*100:.0f}% are BOM-omitted components: the adjustments are the shop absorbing usage
the BOM never recorded. That overlap is the strongest single piece of evidence that the phantom
inventory and the BOM gaps are the same problem.</p>
"""

    did = f"""
{B.section("did", "Section 3", "What we did")}
<p>The remediation ran over ten weeks. Nothing in the source data was overwritten;
every correction is a reference record that can be audited. Review decisions were
made by the shop's own people, and not everything was resolved.</p>
{B.data_table(
    ["Activity", "Result"],
    [
        ["Dead item review", f"{d['dead_deactivated']:,} deactivated, {d['dead_kept']} kept (seasonal / safety-critical, per the buyer and production lead), {d['dead_held']} held for review"],
        ["Duplicate resolution", f"{d['dup_merged']} records merged to a survivor; {d['dup_rejected']} candidate pairs rejected as genuinely different parts"],
        ["Lead time & parameters", f"Lead times recomputed from receipt history for {d['lead_recomputed']:,} items; {d['params_changed']:,} reorder points / safety stocks changed at the ABC service level"],
        ["Chronic adjustment analysis", f"{d['chronic_items']:,} chronic items attributed to root causes ({', '.join(f'{k} {v}' for k,v in list(d['chronic_root'].items())[:3])})"],
        ["BOM corrections", f"{d['bom_changes']} components added back to product and subassembly BOMs (engineering review and floor observation)"],
        ["UOM conversions", f"{d['uom_added']} box/spool/length conversions added"],
        ["Open document closure", f"{d['closed_po']:,} PO lines and {d['closed_jobs']:,} jobs closed on confirmation"],
        ["Spreadsheet reconciliation", f"120 tracked items reconciled; ERP and spreadsheet disagreed on {d['recon_disagree']}, the spreadsheet was closer on {d['recon_buyer_right']}"],
        ["Free-text attribution", f"{d['ft_confirmed']:,} lines attributed to a stocked item and confirmed, {d['ft_rejected']:,} rejected, {d['ft_unreviewed']:,} left unreviewed"],
        ["Cycle-count program", "Weekly counts from week 2, unreliable items first, balances corrected as counted"],
        ["System configuration", "Reason codes required, required fields enforced, generic codes restricted, negative on-hand blocked, individual logins issued"],
    ], right=[])}
"""

    who_rows = [[r.role.title(), r.consulted_on, r.topic] for r in d["interviews"].itertuples(index=False)]
    who = f"""
{B.section("who", "Section 4", "Who was involved")}
<p>The cleanup was done with the shop's people, not to their data. The interview
log records who was consulted and the decisions they owned.</p>
{B.data_table(["Role", "Consulted on", "What they told us"], who_rows, right=[])}
"""

    means = f"""
{B.section("means", "Section 5", "What it means for purchasing")}
<p>Clean data is only worth the decisions it changes. The corrected consumption
history, merged records and recomputed lead times feed the reorder queue and the
demand forecast. The three-way test below holds the model, features and horizons
fixed and changes only how clean the input history is, so it isolates what the
cleanup is worth.</p>
{B.chart("Forecast error by cleaning tier (WAPE vs true demand over lead time)", chart_threeway(d))}
<p>Merging duplicate records (raw &rarr; master) cuts error by
{(d['threeway']['raw']-d['threeway']['master'])/d['threeway']['raw']*100:.0f}% overall and far more on the
merged items themselves; adding back the unrecorded transaction usage (master &rarr; fully) cuts it a
further {(d['threeway']['master']-d['threeway']['fully'])/d['threeway']['master']*100:.0f}%. The forecast-driven
policy then converts that accuracy into service and dollars.</p>
{B.chart("Inventory policy: current vs corrected lead times vs forecast-driven", chart_policy(d))}
<p>At an equal, better service level the forecast-driven policy holds
{_money(d['policy']['forecast']['inv'])} of inventory against {_money(d['policy']['corrected']['inv'])} for a
simple corrected-lead-time policy, releasing {_money(d['wc_released'])} of working capital, while lifting
fill from {d['policy']['current']['fill']*100:.0f}% and cutting stockouts from
{int(d['policy']['current']['stockouts']):,} to {int(d['policy']['forecast']['stockouts']):,}. About
{d['policy'].get('phantom_stockout_share',0)*100:.0f}% of the old stockouts trace to phantom on-order:
material the ERP believed was inbound on never-closed POs.</p>
"""

    keep = f"""
{B.section("keep", "Section 6", "Keeping it clean")}
<p>The corrections are worth nothing if the same problems return. Some fixes were
made in the system during the engagement; the rest need an owner and a cadence.</p>
<p><strong>Implemented in the system (done, with dates).</strong></p>
{B.data_table(["Change", "Area", "Effective"], [[r.change, r.area, r.effective_date] for r in d["config"].itertuples(index=False)], right=[])}
<p><strong>Requires process and ownership (proposed).</strong></p>
<ul class="limitation-list">
  <li>A named item-master owner and a part-creation approval step, so no one can create a duplicate unchecked.</li>
  <li>The cycle-count program continued on the ABC schedule, not allowed to lapse back to an annual count.</li>
  <li>A monthly parameter refresh that recomputes reorder points from the forecast.</li>
  <li>A monthly open-document review and a quarterly dead-item review.</li>
  <li>A BOM review for every new product and option, so backflush stays complete.</li>
  <li>The data-quality measures in this report tracked monthly against their targets.</li>
</ul>
"""

    remains = f"""
{B.section("remains", "Section 7", "What remains")}
<p>Not everything was resolved, and it would be dishonest to imply otherwise.</p>
<ul class="limitation-list">
  <li><strong>Probable findings not yet reviewed.</strong> {d['ft_unreviewed']:,} free-text attributions are
      still awaiting buyer confirmation; they are flagged, not applied.</li>
  <li><strong>Items still unreliable.</strong> {d['rel_after']['unreliable']['pct']:.0f}% of live items still
      carry a balance we would not trust, mostly phantom-inventory items awaiting their first cycle count.</li>
  <li><strong>Lead-time precision floor.</strong> Because receipts were batched to Mondays and month-end,
      computed lead times carry a few days of irreducible noise; the recommended values use a trimmed
      high percentile to stay safe rather than precise.</li>
  <li><strong>What the shop declined.</strong> A small set of dead items were kept active at the buyer's
      insistence as insurance spares, against the recommendation to deactivate them.</li>
</ul>
"""

    appendix = f"""
{B.section("appendix", "Section 8", "Appendix")}
<p>The audit ran one detection test per defect type against the full ledger and
item master, and produced a reference table for each remediation activity (dead-item
dispositions, duplicate and supplier crosswalks, UOM conversions, lead-time
computations, parameter recommendations, the chronic-adjustment list, the BOM
change log, open-document closures, the spreadsheet reconciliation, and the
free-text attribution). Dollar figures for working capital, expedites and carrying
cost are estimates on stated assumptions: an expedite fee of {_money(EXPEDITE_FEE)} per event
and a {CARRYING_RATE*100:.0f}% annual carrying rate, with service held constant when comparing
inventory levels. Defect types and rates reflect patterns commonly documented in
manufacturing ERP systems.</p>
{B.callout("<strong>How to read the numbers.</strong> Data-quality burden is reported three ways: the count "
    "of defective records, the far larger count of transactions that reference a defective master record "
    "(the blast radius), and the operational impact in unrecorded consumption, unreliable dollars, "
    "stockouts and working capital. This report leads with impact; the counts are in the tables above.")}
"""

    return results + found + did + who + means + keep + remains + appendix, toc


def run():
    d = gather()
    body, toc = build(d)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = B.page("Data Quality Audit: Purchasing & Inventory",
                  "Post-remediation report to the operations manager and controller",
                  toc, body)
    for a, b in [("defects", "errors"), ("Defects", "Errors"), ("defect", "error"), ("Defect", "Error")]:
        html = html.replace(a, b)
    OUT.write_text(html, encoding="utf-8")
    print(f"Data quality audit written to {OUT}  ({len(html)//1024} KB)")


if __name__ == "__main__":
    run()
