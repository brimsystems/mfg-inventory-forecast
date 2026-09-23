"""The ten-week remediation period (spec section 7).

Reads the generated dataset and its ground truth, then produces the reference
artifacts a real cleanup would leave behind: an interview log, dead-item
dispositions, a duplicate crosswalk, a supplier crosswalk, UOM conversions, lead
time computations and parameter recommendations, the chronic-adjustment list, a
BOM change log, open-document closures, the spreadsheet reconciliation, the
free-text attribution, and the configuration change log. Nothing in the source
data is overwritten; every correction is recorded here so it is auditable.

Review decisions carry human judgment: a share of dead items are rescued, some
duplicate candidates are rejected, chronic adjustments get mixed root causes, and
some probable findings are left unreviewed.

Run:  python -m data_source.generate.remediation
"""
from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config as C

RAW = C.RAW_DIR
TRUTH = C.REPO_ROOT / "data_source" / "truth"
OUT = RAW / "remediation"

ROLES = ["purchasing manager", "buyer", "stockroom lead", "assembly supervisor",
         "production scheduler", "service parts coordinator", "controller",
         "engineering manager", "ERP administrator", "operations manager"]


def _week_date(week, rng):
    _, monday = C.remediation_weeks()[week - 1]
    return (monday + timedelta(days=int(rng.integers(0, 5)))).isoformat()


def run():
    rng = np.random.default_rng(C.RANDOM_SEED + 7)
    OUT.mkdir(parents=True, exist_ok=True)
    im = pd.read_csv(RAW / "erp" / "item_master.csv", low_memory=False)
    po = pd.read_csv(RAW / "erp" / "purchase_orders.csv", low_memory=False)
    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv", low_memory=False)
    cross = json.loads((TRUTH / "crosswalks.json").read_text())
    txd = json.loads((TRUTH / "txn_defects.json").read_text())
    pod = json.loads((TRUTH / "po_defects.json").read_text())
    buyer = json.loads((TRUTH / "buyer_reconciliation.json").read_text())
    abc = {int(k): v for k, v in cross["abc_by_item"].items()}

    dead_nums = [n for n in im["item_number"]
                 if len(str(n).split("-")) >= 2 and str(n).split("-")[1].isdigit()
                 and int(str(n).split("-")[1]) >= 100000]

    n = {}
    n["interview_log"] = _interview_log(rng)
    n["dead_item_dispositions"] = _dead_dispositions(im, dead_nums, tx, rng)
    n["duplicate_crosswalk"] = _duplicate_crosswalk(cross, im, rng)
    n["supplier_crosswalk"] = _supplier_crosswalk(cross, im)
    n["uom_conversions"] = _uom_conversions(cross, im, rng)
    n["lead_time_computation"] = _lead_time_computation(po, im, cross, rng)
    n["parameter_recommendations"] = _parameter_recs(im, tx, cross, abc, rng)
    n["chronic_adjustment_list"] = _chronic_adjustments(tx, txd, cross, rng)
    n["bom_change_log"] = _bom_change_log(cross, im, rng)
    n["open_document_closures"] = _closures(po, rng)
    n["spreadsheet_reconciliation"] = _spreadsheet_recon(buyer, rng)
    n["free_text_attribution"] = _free_text_attr(pod, rng)
    n["config_change_log"] = _config_change_log(rng)
    n["posting_corrections"] = _posting_corrections(txd, rng)

    for name, df in n.items():
        fp = OUT / f"{name}.csv"
        df.to_csv(fp, index=False)
        print(f"  [remediation]  {name:<28} {len(df):>6,} rows")
    print(f"\nRemediation reference tables -> {OUT}")


# ── Artifacts ────────────────────────────────────────────────────────────────
def _interview_log(rng):
    topics = {
        "purchasing manager": "reorder points not trusted; keeps own spreadsheet",
        "buyer": "expedites on motors and gearboxes; lead times understated",
        "stockroom lead": "counts done once a year; balances drift",
        "assembly supervisor": "hardware pulled without issuing; BOMs miss fasteners",
        "production scheduler": "line stops waiting on long-lead components",
        "service parts coordinator": "warranty pulls not always recorded",
        "controller": "inventory value uncertain; adjustments large at year end",
        "engineering manager": "BOMs not maintained for options and revisions",
        "ERP administrator": "no required fields; generic codes allowed",
        "operations manager": "wants trustworthy reorder points and less expedite",
    }
    rows = [{"role": r, "interview_date": _week_date(1, rng), "topic": t,
             "consulted_on": rng.choice(["dead items", "duplicates", "lead times",
                                         "BOM corrections", "parameters", "reconciliation"])}
            for r, t in topics.items()]
    return pd.DataFrame(rows)


def _dead_dispositions(im, dead_nums, tx, rng):
    moved = set(tx["item_number"].unique())
    rows = []
    for num in dead_nums:
        recent = num in moved                         # stray transaction
        roll = rng.random()
        if roll < C.M1_RESCUE_SHARE:
            disp, reason, by = "KEPT", rng.choice(
                ["seasonal service part", "safety-critical spare", "long-lead insurance item"]), \
                rng.choice(["service parts coordinator", "production lead"])
        elif recent:
            # the one recent posting was checked and found to be a misposting
            disp, reason, by = "DEACTIVATED", "stray posting confirmed as a misposting; no genuine movement", "buyer"
        else:
            disp, reason, by = "DEACTIVATED", "no movement 24+ months", "purchasing manager"
        rows.append({"item_number": num, "disposition": disp, "reason": reason,
                     "decided_by": by, "decision_date": _week_date(int(rng.integers(1, 3)), rng)})
    return pd.DataFrame(rows)


def _duplicate_crosswalk(cross, im, rng):
    rows = []
    desc = im.set_index("item_number")["description"].to_dict()
    clusters = list(cross["duplicate_clusters"].values())
    n_reject = max(1, int(len(clusters) * 0.06))
    reject_idx = set(rng.choice(len(clusters), size=n_reject, replace=False))
    for ci, cl in enumerate(clusters):
        survivor = cl["primary"]
        # in a rejected cluster one pair is judged a different part and stays
        # separate; any other members of the cluster still merge to the survivor
        rejected = [r for r in cl["records"] if r != survivor][:1] if ci in reject_idx else []
        for r in rejected:
            rows.append({"retired_item_number": r, "survivor_item_number": survivor,
                         "reason": "different part on review (revision/spec)", "reviewer": "buyer",
                         "decision": "REJECT", "decision_date": _week_date(int(rng.integers(2, 4)), rng)})
        for r in cl["records"]:
            if r == survivor or r in rejected:
                continue
            rows.append({"retired_item_number": r, "survivor_item_number": survivor,
                         "reason": "same physical item, alternate number/description",
                         "reviewer": rng.choice(["buyer", "stockroom lead"]),
                         "decision": "MERGE", "decision_date": _week_date(int(rng.integers(2, 4)), rng)})
    return pd.DataFrame(rows)


def _supplier_crosswalk(cross, im):
    rows = []
    for frag in cross["supplier_fragments"]:
        canon = frag["canonical"]
        for a in frag["aliases"]:
            rows.append({"alias_supplier_id": a, "canonical_supplier_id": canon,
                         "reason": "same vendor, alternate name/id"})
    return pd.DataFrame(rows)


def _uom_conversions(cross, im, rng):
    m5 = set(cross["m5_items"])
    imm = im.copy()
    imm["_iid"] = _canonical_iids(im, cross)
    rows = []
    for r in imm[imm["_iid"].isin(m5)].itertuples(index=False):
        pu = r.purchase_uom if isinstance(r.purchase_uom, str) else "BOX"
        su = r.uom if isinstance(r.uom, str) else "EA"
        # the true factor comes from the ground truth; the master itself holds none
        conv = cross.get("m5_conversions", {}).get(r.item_number)
        if conv is None:
            conv = r.uom_conversion if not pd.isna(r.uom_conversion) else int(rng.choice([25, 50, 100]))
        rows.append({"item_number": r.item_number, "purchase_uom": pu, "stock_uom": su,
                     "conversion": int(conv), "added_date": _week_date(int(rng.integers(5, 8)), rng)})
    return pd.DataFrame(rows).drop_duplicates("item_number")


def _lead_time_computation(po, im, cross, rng):
    p = po[po["received_date"].notna()].copy()
    p["lead"] = (pd.to_datetime(p["received_date"]) - pd.to_datetime(p["order_date"])).dt.days
    g = p.groupby("item_number")["lead"]
    comp = pd.DataFrame({"median_actual": g.median(), "p80_actual": g.quantile(0.8),
                         "sample_size": g.count()}).reset_index()
    comp = comp[comp["sample_size"] >= 3]
    master = im.set_index("item_number")["master_lead_time_days"]
    sup = im.set_index("item_number")["primary_supplier_id"]
    comp["master_lead_time"] = comp["item_number"].map(master)
    comp["supplier_id"] = comp["item_number"].map(sup)
    comp["recommended_lead_time"] = comp["p80_actual"].round().astype(int)
    comp["computed_date"] = _week_date(3, rng)
    return comp[["item_number", "supplier_id", "master_lead_time", "median_actual",
                 "p80_actual", "sample_size", "recommended_lead_time", "computed_date"]]


def _parameter_recs(im, tx, cross, abc, rng):
    iid = _canonical_iids(im, cross)
    im2 = im.assign(iidc=iid)
    cons = tx[tx["type"].isin(["ISSUE", "BACKFLUSH"])].groupby("item_number")["qty"].apply(
        lambda s: s.abs().sum())
    months = 36
    z = {"A": 2.054, "B": 1.645, "C": 1.282}       # normal quantiles for 0.98 / 0.95 / 0.90
    rows = []
    for r in im2.itertuples(index=False):
        if r.iidc is None or pd.isna(r.iidc):
            continue
        cls = abc.get(int(r.iidc), "C")
        monthly = cons.get(r.item_number, 0) / months
        if monthly <= 0:
            continue
        lead_m = max(0.3, (r.master_lead_time_days or 21) / 30.0)
        demand_lt = monthly * lead_m
        ss = int(round(z[cls] * np.sqrt(max(1.0, demand_lt))))
        rop = int(round(demand_lt + ss))
        rows.append({"item_number": r.item_number, "abc_class": cls,
                     "service_level": C.SERVICE_LEVEL_BY_ABC[cls],
                     "old_reorder_point": r.reorder_point, "old_safety_stock": r.safety_stock,
                     "new_reorder_point": rop, "new_safety_stock": ss})
    return pd.DataFrame(rows)


def _chronic_adjustments(tx, txd, cross, rng):
    adj = tx[tx["type"] == "ADJUST"].copy()
    adj["txn_date"] = pd.to_datetime(adj["txn_date"])
    cut = pd.Timestamp(C.MODEL_SPAN_END) - pd.Timedelta(days=365)
    # chronic means three or more DOWNWARD adjustments in twelve months: the
    # signature of unrecorded consumption, dominated by BOM-omitted components.
    # a count correction carries its cause; chronic means the floor's own
    # unexplained write-offs
    down = adj[(adj["txn_date"] >= cut) & (adj["qty"] < 0) & ~adj["reason_code"].astype(str).isin(["COUNT", "CYCLE"])]
    g = down.groupby("item_number")["qty"]
    chron = pd.DataFrame({"adj_count_12m": g.count(), "net_qty": g.sum()}).reset_index()
    chron = chron[chron["adj_count_12m"] >= 3]
    t1_items = {r["item_number"] for r in txd.get("t1", [])}
    rows = []
    for r in chron.itertuples(index=False):
        on_bom = r.item_number in t1_items
        if on_bom:
            root = "BOM omission"
        else:
            root = rng.choice(["floor practice", "receiving error", "count error"],
                              p=[0.5, 0.25, 0.25])
        rows.append({"item_number": r.item_number, "adj_count_12m": int(r.adj_count_12m),
                     "net_qty": int(r.net_qty), "implied_monthly": round(abs(r.net_qty) / 12.0, 1),
                     "on_bom": on_bom, "root_cause": root})
    return pd.DataFrame(rows)


def _bom_change_log(cross, im, rng):
    iid_to_primary = _iid_to_primary(im, cross)
    rows = []
    for iid in cross["m3_omitted_items"]:
        comp = iid_to_primary.get(int(iid))
        if comp is None:
            continue
        rows.append({"product_number": f"PRD-{rng.choice(['CO','MI','EN'])}-{int(rng.integers(0,80)):03d}",
                     "component_item": comp, "qty_per": int(rng.integers(1, 8)),
                     "source": rng.choice(["engineering review", "floor observation"]),
                     "change_date": _week_date(int(rng.integers(3, 8)), rng)})
    return pd.DataFrame(rows)


def _closures(po, rng):
    # a line open more than 90 days at the end of the engagement is not coming;
    # younger open lines are genuinely inbound and are left alone
    stale = pd.to_datetime(po["order_date"]) < pd.Timestamp(C.REMEDIATION_END) - pd.Timedelta(days=90)
    op = po[(po["status"] == "OPEN") & stale]
    rows = [{"document_type": "PO", "document_id": r.po_id, "line": r.line,
             "closed_date": _week_date(int(rng.integers(1, 4)), rng),
             "confirmation_source": rng.choice(["receiving records", "buyer confirmation", "supplier statement"])}
            for r in op.itertuples(index=False)]
    # finished jobs that were never closed, closed on production confirmation
    prod = pd.read_csv(RAW / "erp" / "production_orders.csv", low_memory=False)
    stuck = prod[(prod["status"] == "OPEN") & (prod["qty_completed"] > 0)]
    for r in stuck.itertuples(index=False):
        rows.append({"document_type": "JOB", "document_id": r.order_id, "line": 1,
                     "closed_date": _week_date(int(rng.integers(1, 3)), rng),
                     "confirmation_source": "production confirmation"})
    return pd.DataFrame(rows)


def _spreadsheet_recon(buyer, rng):
    rows = []
    for b in buyer:
        if not b["disagree"]:
            action = "no change"
        elif b["buyer_closer"]:
            action = "corrected ERP to counted value"
        else:
            action = "kept ERP; spreadsheet updated"
        rows.append({"item_ref": b["item_ref"], "item_number": b["item_number"],
                     "spreadsheet_on_hand": b["buyer_on_hand"], "erp_on_hand": b["erp_on_hand"],
                     "closer_to_truth": ("spreadsheet" if b["buyer_closer"] else "erp") if b["disagree"] else "agree",
                     "action": action, "reconciled_date": _week_date(int(rng.integers(2, 4)), rng)})
    return pd.DataFrame(rows)


def _free_text_attr(pod, rng):
    rows = []
    for r in pod.get("t3", []):
        if not r.get("is_stocked"):
            conf, confirmation = round(rng.uniform(0.1, 0.4), 2), "rejected"
            probable = None
        else:
            # every candidate was settled by the buyer: high-scoring matches
            # confirmed, the rest rejected under the confidence rule she signed off
            conf = round(rng.uniform(0.55, 0.95), 2)
            confirmation = "confirmed" if (conf >= 0.62 and rng.random() < 0.9) else "rejected"
            probable = r["true_item_number"]
        rows.append({"po_id": r["po_id"], "probable_item_number": probable,
                     "confidence": conf, "confirmation": confirmation})
    return pd.DataFrame(rows)


def _posting_corrections(txd, rng):
    """Line-level corrections to the ledger: issues re-pointed to the item the
    job's BOM calls for, keyed quantities corrected, duplicate postings reversed.
    Each was reviewed by the stockroom lead before it was applied."""
    rows = []
    for r in txd.get("t6", []):
        rows.append({"correction": "re-pointed to correct item", "txn_id": r.get("txn_id", ""),
                     "detail": f"{r['recorded_item_number']} -> {r['true_item_number']}",
                     "reviewed_by": "stockroom lead", "corrected_date": _week_date(int(rng.integers(6, 9)), rng)})
    for r in txd.get("t7", []):
        rows.append({"correction": "quantity corrected", "txn_id": r["txn_id"],
                     "detail": f"{r['recorded_qty']} -> {r['true_qty']}",
                     "reviewed_by": "stockroom lead", "corrected_date": _week_date(int(rng.integers(6, 9)), rng)})
    for r in txd.get("t8", []):
        rows.append({"correction": "duplicate posting reversed", "txn_id": r["txn_id"],
                     "detail": f"duplicate of {r['original_txn_id']}",
                     "reviewed_by": "stockroom lead", "corrected_date": _week_date(int(rng.integers(6, 9)), rng)})
    return pd.DataFrame(rows)


def _config_change_log(rng):
    changes = [
        ("reason codes required on adjustments", "controls", 5),
        ("required fields enforced on item creation", "controls", 6),
        ("UOM conversions added for box/spool/length items", "data", 6),
        ("generic item codes restricted", "controls", 7),
        ("negative on-hand blocked", "controls", 7),
        ("individual logins issued for floor and receiving", "access", 8),
        ("part-creation approval routing enabled", "controls", 8),
    ]
    return pd.DataFrame([{"change": c, "area": a, "effective_date": _week_date(w, rng)}
                         for c, a, w in changes])


# ── helpers ──────────────────────────────────────────────────────────────────
def _parse_iid(num):
    """The canonical item_id is the 5-digit block in every live item number
    (PREFIX-00042 or a duplicate member PREFIX-00042-A). Dead items are 100000+."""
    parts = str(num).split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        v = int(parts[1])
        return v if v < 100000 else None
    return None


def _canonical_iids(im, cross):
    return im["item_number"].map(_parse_iid)


def _iid_to_primary(im, cross):
    out = {}
    for num in im["item_number"]:
        iid = _parse_iid(num)
        if iid is not None and len(str(num).split("-")) == 2:   # base number, no variant
            out[iid] = num
    for iid, cl in cross["duplicate_clusters"].items():
        out.setdefault(int(iid), cl["primary"])
    return out


if __name__ == "__main__":
    run()
