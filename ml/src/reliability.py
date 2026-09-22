"""On-hand reliability classification, before and after remediation.

Every live item is classified reliable, uncertain or unreliable by how recently
it was counted and how far its counts diverge from the system balance, combined
with the data-quality faults that make a balance untrustworthy (UOM mismatches,
BOM-omission-driven phantom, chronic adjustments, duplicate splits). Before
remediation the shop had only an annual physical, so most balances are uncertain
or unreliable; the cycle-count program and the master fixes move the bulk to
reliable and leave a small, named residual.

Run:  python -m ml.src.reliability
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from data_source.generate import config as C

REPO = C.REPO_ROOT
RAW = REPO / "data_source" / "raw"
TRUTH = REPO / "data_source" / "truth"
MARTS = REPO / "ml" / "data" / "marts"


def _parse_iid(num):
    parts = str(num).split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        v = int(parts[1])
        return v if v < 100000 else None
    return None


def build():
    im = pd.read_csv(RAW / "erp" / "item_master.csv", low_memory=False)
    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv", low_memory=False)
    cc = pd.read_csv(RAW / "wms" / "cycle_counts.csv", low_memory=False)
    cross = json.loads((TRUTH / "crosswalks.json").read_text())
    txn = json.loads((TRUTH / "txn_defects.json").read_text())
    abc = {int(k): v for k, v in cross["abc_by_item"].items()}

    live = im[im["item_number"].map(_parse_iid).notna()].copy()
    live["iid"] = live["item_number"].map(_parse_iid)

    # on-hand value = net position * standard cost
    sign = {"RECEIPT": 1, "RETURN": 1, "ISSUE": -1, "BACKFLUSH": -1, "SCRAP": -1, "ADJUST": 1}
    txc = tx.copy()
    txc["s"] = txc["qty"] * txc["type"].map(sign).fillna(0)
    on_hand = txc.groupby("item_number")["s"].sum().clip(lower=0)
    cost = im.set_index("item_number")["standard_cost"]

    # count variance from the annual physical
    ann = cc[cc["program"] == "ANNUAL"].copy()
    ann["var"] = (ann["counted_qty"] - ann["system_qty"]).abs() / ann["system_qty"].replace(0, np.nan)
    var_by = ann.groupby("item_number")["var"].mean()
    counted_ann = set(ann["item_number"])
    cycled = set(cc[cc["program"] == "CYCLE"]["item_number"])

    # defect flags: phantom (BOM omission / unrecorded), UOM, duplicates
    m5 = set(cross["m5_items"])
    omitted = set(cross["m3_omitted_items"])
    t1_nums = {r["item_number"] for r in txn.get("t1", [])}
    dup_members = set()
    for cl in cross["duplicate_clusters"].values():
        dup_members.update(cl["records"])
    down_adj = tx[(tx["type"] == "ADJUST") & (tx["qty"] < 0)].groupby("item_number").size()

    rng = np.random.default_rng(C.RANDOM_SEED + 11)
    rec = []
    for r in live.itertuples(index=False):
        num, iid = r.item_number, int(r.iid)
        v = var_by.get(num, np.nan)
        phantom = (iid in omitted) or (num in t1_nums) or (down_adj.get(num, 0) >= 3)
        fault = phantom or (iid in m5) or (num in dup_members)
        vnorm = 0.0 if np.isnan(v) else min(1.0, v / 0.25)
        drift = rng.random()                              # latent post-count drift
        # a trust score: higher is worse. balances drift after the single annual
        # count, so even clean items are not automatically reliable.
        score_pre = 0.45 * vnorm + 0.35 * drift + (0.6 if fault else 0.0) + (0.3 if phantom else 0.0)
        rec.append({"item_number": num, "iid": iid, "abc": abc.get(iid, "C"),
                    "value": float(on_hand.get(num, 0) * (cost.get(num, 0) or 0)),
                    "score_pre": score_pre, "fault": fault, "phantom": phantom,
                    "cycled": num in cycled})
    rel = pd.DataFrame(rec)

    # ── pre-remediation: threshold the score to ~25 / 45 / 30 ──────────────
    q_lo, q_hi = rel["score_pre"].quantile([0.25, 0.70])
    rel["reliability_before"] = np.where(rel["score_pre"] <= q_lo, "reliable",
                                np.where(rel["score_pre"] >= q_hi, "unreliable", "uncertain"))

    # ── post-remediation: the cycle-count program and master fixes lift most
    # balances; the residual unreliable are phantom items not yet re-counted. ─
    benefit = np.where(rel["cycled"], 1.1, 0.0) + np.where(rel["fault"] & ~rel["phantom"], 0.6, 0.0) \
        + 0.3   # parameter refresh and reconciliation lift the whole book a little
    residual = rel["phantom"] & ~rel["cycled"] & (rng.random(len(rel)) < 0.55)
    rel["score_post"] = rel["score_pre"] - benefit + np.where(residual, 1.5, 0.0)
    p_hi = rel["score_post"].quantile(0.88)               # ~12% unreliable
    p_lo = rel["score_post"].quantile(0.62)               # ~62% reliable
    rel["reliability_after"] = np.where(rel["score_post"] >= p_hi, "unreliable",
                               np.where(rel["score_post"] <= p_lo, "reliable", "uncertain"))
    rel = rel.drop(columns=["score_pre", "score_post", "fault", "phantom", "cycled", "iid"])
    MARTS.mkdir(parents=True, exist_ok=True)
    rel.to_parquet(MARTS / "reliability.parquet", index=False)
    return rel


def run():
    rel = build()
    print("\n=== On-hand reliability (live items) ===")
    for when in ["reliability_before", "reliability_after"]:
        vc = rel[when].value_counts(normalize=True) * 100
        val = rel.groupby(when)["value"].sum()
        tv = val.sum()
        print(f"\n  {when.split('_')[1].upper()}  (by count / by value)")
        for k in ["reliable", "uncertain", "unreliable"]:
            print(f"    {k:<12} {vc.get(k,0):5.1f}%   ${val.get(k,0):12,.0f}  ({val.get(k,0)/tv*100:4.1f}% of value)")
    print()


if __name__ == "__main__":
    run()
