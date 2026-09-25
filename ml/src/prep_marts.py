"""Bridge the regenerated source data into the modeled marts the forecast reads.

Produces, from the raw extracts and the ground truth:

  data_pipeline/seeds/item_crosswalk.csv        item_number -> canonical (survivor)
  ml/data/marts/item_attributes.parquet         per-item attributes for the model
  ml/data/marts/consumption_{raw,master,fully,true}.parquet

The three consumption tiers differ only in how clean the history is:
  raw     the surviving record's own consumption (duplicate history stays split,
          unrecorded usage stays missing)
  master  duplicate records merged to the canonical item (M4 remediation)
  fully   master plus the confirmed transaction corrections: unrecorded
          consumption (T1) added back, keying errors (T7) deflated, duplicate
          postings (T8) removed
The true-demand series is the common evaluation target.

Run:  python -m ml.src.prep_marts
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_source.generate import config as C
from data_source.generate.checkpoint import classify
from data_source.generate.generators.demand import build_item_plan, build_monthly_demand

REPO = C.REPO_ROOT
RAW = REPO / "data_source" / "raw"
TRUTH = REPO / "data_source" / "truth"
SEEDS = REPO / "data_pipeline" / "seeds"
MARTS = REPO / "ml" / "data" / "marts"


def _month(s):
    return pd.to_datetime(s).values.astype("datetime64[M]")


def _parse_iid(num):
    parts = str(num).split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        v = int(parts[1])
        return v if v < 100000 else None
    return None


def build():
    im = pd.read_csv(RAW / "erp" / "item_master.csv", low_memory=False)
    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv", low_memory=False)
    cross = json.loads((TRUTH / "crosswalks.json").read_text())
    txn = json.loads((TRUTH / "txn_defects.json").read_text())
    abc = {int(k): v for k, v in cross["abc_by_item"].items()}

    # ── Crosswalk: item_number -> canonical survivor number ─────────────────
    dup = cross["duplicate_clusters"]
    member_to_primary = {}
    for cl in dup.values():
        for r in cl["records"]:
            member_to_primary[r] = cl["primary"]
    live_nums = [n for n in im["item_number"] if _parse_iid(n) is not None]
    canon = {n: member_to_primary.get(n, n) for n in live_nums}
    cw = pd.DataFrame({"item_number": list(canon), "canonical_item_number": list(canon.values())})
    SEEDS.mkdir(parents=True, exist_ok=True)
    cw.to_csv(SEEDS / "item_crosswalk.csv", index=False)

    iid_to_survivor = {}
    for n in live_nums:
        iid = _parse_iid(n)
        if iid is not None:
            iid_to_survivor.setdefault(iid, canon[n])
    # prefer the cluster primary where the item is duplicated
    for iid_s, cl in [(_parse_iid(cl["primary"]), cl) for cl in dup.values()]:
        if iid_s is not None:
            iid_to_survivor[iid_s] = cl["primary"]

    # ── True demand: engine direct channel + true backflush ─────────────────
    rng = np.random.default_rng(C.RANDOM_SEED)
    plan = build_item_plan(rng)
    direct = build_monthly_demand(plan, rng)          # item_id, month, demand_units
    bf_true = pd.read_csv(TRUTH / "true_backflush.csv") if (TRUTH / "true_backflush.csv").exists() \
        else pd.DataFrame(columns=["item_id", "month", "qty"])
    direct = direct.rename(columns={"demand_units": "qty"})
    true = pd.concat([direct[["item_id", "month", "qty"]], bf_true[["item_id", "month", "qty"]]])
    true = true.groupby(["item_id", "month"], as_index=False)["qty"].sum()
    true["canonical"] = true["item_id"].map(iid_to_survivor)
    true["month"] = _month(true["month"])
    true = true.dropna(subset=["canonical"]).groupby(["canonical", "month"], as_index=False)["qty"].sum() \
        .rename(columns={"qty": "consumption"})

    # ── Recorded consumption (ISSUE + BACKFLUSH), per number and month ───────
    cons = tx[tx["type"].isin(["ISSUE", "BACKFLUSH"])].copy()
    cons["month"] = _month(cons["txn_date"])
    cons["canonical"] = cons["item_number"].map(canon)
    cons = cons.dropna(subset=["canonical"])

    # RAW: only the survivor record's own consumption
    survivors = set(canon.values())
    raw = (cons[cons["item_number"].isin(survivors)].assign(q=lambda d: d["qty"].abs())
           .groupby(["item_number", "month"], as_index=False)["q"].sum()
           .rename(columns={"item_number": "canonical", "q": "consumption"}))

    # MASTER: all members merged to the canonical item
    master = (cons.assign(q=lambda d: d["qty"].abs())
              .groupby(["canonical", "month"], as_index=False)["q"].sum()
              .rename(columns={"q": "consumption"}))

    # FULLY: master + confirmed transaction corrections
    fully = master.copy()
    tx_lookup = tx.set_index("txn_id")

    # T1 unrecorded add-back, distributed over the item's own monthly shape
    t1 = txn.get("t1", [])
    add_rows = []
    master_by_item = master.groupby("canonical")
    for r in t1:
        c = canon.get(r["item_number"], r["item_number"])
        annual = r.get("annual_unrecorded", 0)
        if c in master_by_item.groups and annual > 0:
            sub = master[master["canonical"] == c]
            shape = sub["consumption"] / max(1.0, sub["consumption"].sum())
            for m, w in zip(sub["month"], shape):
                add_rows.append({"canonical": c, "month": m, "consumption": annual * w})
    if add_rows:
        fully = pd.concat([fully, pd.DataFrame(add_rows)], ignore_index=True)

    # T7 keying errors deflated to the true quantity
    sub_rows = []
    for r in txn.get("t7", []):
        tid = r["txn_id"]
        if tid in tx_lookup.index:
            row = tx_lookup.loc[tid]
            c = canon.get(row["item_number"])
            if c is not None:
                sub_rows.append({"canonical": c, "month": _month(pd.Series([row["txn_date"]]))[0],
                                 "consumption": -(r["recorded_qty"] - r["true_qty"])})
    # T8 duplicate postings removed
    for r in txn.get("t8", []):
        tid = r["txn_id"]
        if tid in tx_lookup.index:
            row = tx_lookup.loc[tid]
            c = canon.get(row["item_number"])
            if c is not None and row["type"] in ("ISSUE", "BACKFLUSH"):
                sub_rows.append({"canonical": c, "month": _month(pd.Series([row["txn_date"]]))[0],
                                 "consumption": -abs(row["qty"])})
    if sub_rows:
        fully = pd.concat([fully, pd.DataFrame(sub_rows)], ignore_index=True)
    fully = fully.groupby(["canonical", "month"], as_index=False)["consumption"].sum()
    fully["consumption"] = fully["consumption"].clip(lower=0)

    # ── the same fully-cleaned series in Monday weeks, for the weekly refresh ─
    def _week(sr):
        d = pd.to_datetime(sr)
        return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.normalize()
    cw_ = cons.assign(week=_week(cons["txn_date"]), q=lambda d: d["qty"].abs())
    wk = cw_.groupby(["canonical", "week"], as_index=False)["q"].sum().rename(columns={"q": "consumption"})
    wadd = []
    for r in t1:
        c = canon.get(r["item_number"], r["item_number"]); annual = r.get("annual_unrecorded", 0)
        sub_w = wk[wk["canonical"] == c]
        if len(sub_w) and annual > 0:
            shape = sub_w["consumption"] / max(1.0, sub_w["consumption"].sum())
            wadd += [{"canonical": c, "week": w_, "consumption": annual * x_} for w_, x_ in zip(sub_w["week"], shape)]
    wsub = []
    for r in txn.get("t7", []) + [{"txn_id": x["txn_id"], "_dup": True} for x in txn.get("t8", [])]:
        tid = r["txn_id"]
        if tid in tx_lookup.index:
            row = tx_lookup.loc[tid]; c = canon.get(row["item_number"])
            if c is None or row["type"] not in ("ISSUE", "BACKFLUSH"):
                continue
            wkey = _week(pd.Series([row["txn_date"]]))[0]
            q_ = -abs(row["qty"]) if r.get("_dup") else -(r["recorded_qty"] - r["true_qty"])
            wsub.append({"canonical": c, "week": wkey, "consumption": q_})
    weekly = pd.concat([wk, pd.DataFrame(wadd), pd.DataFrame(wsub)], ignore_index=True)
    weekly = weekly.groupby(["canonical", "week"], as_index=False)["consumption"].sum()
    weekly["consumption"] = weekly["consumption"].clip(lower=0)

    # ── Item attributes ─────────────────────────────────────────────────────
    lt = _corrected_lead()
    cost = im.set_index("item_number")["standard_cost"].to_dict()
    cls = im.set_index("item_number")["item_class"].to_dict()
    tw = true.pivot(index="canonical", columns="month", values="consumption").fillna(0).sort_index(axis=1)
    attrs_rows = []
    for c in survivors:
        iid = _parse_iid(c)
        series = tw.loc[c].to_numpy(float) if c in tw.index else np.zeros(1)
        attrs_rows.append({
            "canonical_item_number": c,
            "standard_cost": float(cost.get(c, np.nan)) if not pd.isna(cost.get(c, np.nan)) else 1.0,
            "item_class": cls.get(c, "MISC"),
            "segment": classify(series),
            "abc": abc.get(iid, "C"),
            "corrected_lead_days": float(lt.get(c, im.loc[im["item_number"] == c, "master_lead_time_days"].iloc[0]
                                                if (im["item_number"] == c).any() else 21)),
            "annual_consumption": float(series[-12:].sum()),
        })
    attrs = pd.DataFrame(attrs_rows)

    MARTS.mkdir(parents=True, exist_ok=True)
    for name, df in [("raw", raw), ("master", master), ("fully", fully), ("true", true)]:
        df.to_parquet(MARTS / f"consumption_{name}.parquet", index=False)
    # the production consumption series the model trains and scores on is the
    # fully-cleaned tier
    fully.to_parquet(MARTS / "consumption_monthly.parquet", index=False)
    weekly.to_parquet(MARTS / "consumption_weekly.parquet", index=False)
    attrs.to_parquet(MARTS / "item_attributes.parquet", index=False)
    return raw, master, fully, true, attrs


def _corrected_lead():
    fp = RAW / "remediation" / "lead_time_computation.csv"
    if not fp.exists():
        return {}
    lt = pd.read_csv(fp)
    return dict(zip(lt["item_number"], lt["recommended_lead_time"]))


def run():
    raw, master, fully, true, attrs = build()
    tot = lambda d: d["consumption"].sum()
    print("\n=== Cleaning-tier consumption marts ===")
    print(f"  true demand      {tot(true):>14,.0f}")
    print(f"  raw (recorded)   {tot(raw):>14,.0f}   {tot(raw)/tot(true)*100:5.1f}% of true")
    print(f"  master-cleaned   {tot(master):>14,.0f}   {tot(master)/tot(true)*100:5.1f}% of true")
    print(f"  fully-cleaned    {tot(fully):>14,.0f}   {tot(fully)/tot(true)*100:5.1f}% of true")
    print(f"  items {len(attrs):,}   segments {attrs['segment'].value_counts().to_dict()}")
    print()


if __name__ == "__main__":
    run()
