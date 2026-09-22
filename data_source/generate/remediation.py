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
        elif recent or roll < C.M1_RESCUE_SHARE + 0.18:
            disp, reason, by = "HELD", "recent activity; confirm before deactivating", "buyer"
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
        if ci in reject_idx:
            other = [r for r in cl["records"] if r != survivor][:1]
            for r in other:
                rows.append({"retired_item_number": r, "survivor_item_number": survivor,
                             "reason": "different part on review (revision/spec)", "reviewer": "buyer",
                             "decision": "REJECT", "decision_date": _week_date(int(rng.integers(2, 4)), rng)})
            continue
        for r in cl["records"]:
            if r == survivor:
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
    down = adj[(adj["txn_date"] >= cut) & (adj["qty"] < 0)]
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
    op = po[po["status"] == "OPEN"]
    rows = [{"document_type": "PO", "document_id": r.po_id, "line": r.line,
             "closed_date": _week_date(1, rng),
             "confirmation_source": rng.choice(["receiving records", "buyer confirmation", "supplier statement"])}
            for r in op.itertuples(index=False)]
    # a sample of completed jobs left open, closed on confirmation
    for _ in range(int(len(op) * 0.5)):
        rows.append({"document_type": "JOB", "document_id": f"JOB-{int(rng.integers(1,7000)):06d}",
                     "line": 1, "closed_date": _week_date(int(rng.integers(1, 3)), rng),
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
            conf = round(rng.uniform(0.55, 0.95), 2)
            roll = rng.random()
            confirmation = "confirmed" if roll < 0.6 else ("unreviewed" if roll < 0.8 else "rejected")
            probable = r["true_item_number"]
        rows.append({"po_id": r["po_id"], "probable_item_number": probable,
                     "confidence": conf, "confirmation": confirmation})
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
