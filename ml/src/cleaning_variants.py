"""Three cleaning-tier consumption marts for the decomposition (addendum step 6).

  raw    consumption as recorded, no cleaning: duplicate records stay split (only
         the surviving number's share is seen), free-text lines are lost, keying
         and duplicate-posting errors are present
  master duplicate records merged through the crosswalk (D1-D6 remediation), but
         no transaction cleaning
  fully  master plus the confirmed transaction corrections: T1 attributions added
         back, T2 confirmed keying errors corrected, T4 adjustment-implied usage
         added, T7 confirmed duplicate postings removed

All three are indexed by canonical item and month. The true demand series is also
written, as the common evaluation target that isolates the value of each tier.

Run:  python -m ml.src.cleaning_variants
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .resolution import REPO
from data_source.generate.config import RANDOM_SEED
from data_source.generate.generators.demand import build_item_plan, build_monthly_demand

RAW = REPO / "data_source" / "raw"
SEEDS = REPO / "data_pipeline" / "seeds"
DQ = REPO / "ml" / "data" / "data_quality" / "txn"
MARTS = REPO / "ml" / "data" / "marts"


def _to_month(s):
    return pd.to_datetime(s).values.astype("datetime64[M]")


def build():
    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv")
    cw = pd.read_csv(SEEDS / "item_crosswalk.csv")
    x = dict(zip(cw["item_number"], cw["canonical_item_number"]))
    survivors = set(cw["canonical_item_number"])

    iss = tx[tx["type"] == "issue"].copy()
    iss["month"] = _to_month(iss["transaction_date"])
    iss["canonical"] = iss["item_number"].map(x)
    real = iss.dropna(subset=["canonical"])          # excludes generic (T1) codes

    # RAW: only the surviving record's own issues are attributed to the item.
    raw = (real[real["item_number"].isin(survivors)]
           .groupby(["item_number", "month"])["quantity"].sum().reset_index()
           .rename(columns={"item_number": "canonical", "quantity": "consumption"}))

    # MASTER: all duplicate records merged to the canonical item.
    master = real.groupby(["canonical", "month"])["quantity"].sum().reset_index() \
        .rename(columns={"quantity": "consumption"})

    # FULLY: master + confirmed transaction corrections.
    fully = master.copy()
    tx_date = dict(zip(tx["transaction_id"], _to_month(tx["transaction_date"])))

    # T1 attributions add recovered volume back to the probable item.
    t1 = pd.read_parquet(DQ / "t1_attribution.parquet")
    t1 = t1[t1["confirmed"] & t1["probable_item"].notna()].copy()
    t1["month"] = t1["transaction_id"].map(tx_date)
    add1 = t1.groupby(["probable_item", "month"])["quantity"].sum().reset_index() \
        .rename(columns={"probable_item": "canonical", "quantity": "consumption"})

    # T2 confirmed corrections reduce inflated quantities.
    t2 = pd.read_parquet(DQ / "t2_corrections.parquet")
    t2 = t2[t2["confirmed"]].copy()
    if len(t2):
        t2["month"] = t2["transaction_id"].map(tx_date)
        t2["canonical"] = t2["item_number"].map(x)
        t2["delta"] = -(t2["raw_qty"] - t2["corrected_qty"])
        sub2 = t2.dropna(subset=["canonical"]).groupby(["canonical", "month"])["delta"].sum().reset_index() \
            .rename(columns={"delta": "consumption"})
    else:
        sub2 = pd.DataFrame(columns=["canonical", "month", "consumption"])

    # T7 confirmed duplicate postings removed.
    t7 = pd.read_parquet(DQ / "t7_duplicates.parquet")
    t7 = t7[t7["confirmed"]] if "confirmed" in t7 else t7
    dup_ids = set(t7["transaction_id"])
    dup = iss[iss["transaction_id"].isin(dup_ids) & iss["canonical"].notna()]
    sub7 = dup.groupby(["canonical", "month"])["quantity"].apply(lambda s: -s.sum()).reset_index() \
        .rename(columns={"quantity": "consumption"})

    fully = pd.concat([fully, add1, sub2, sub7], ignore_index=True)
    fully = fully.groupby(["canonical", "month"], as_index=False)["consumption"].sum()
    fully["consumption"] = fully["consumption"].clip(lower=0)

    # T4 adjustment-implied usage: scale the item's series up proportionally so the
    # recovered volume follows the item's own timing rather than a flat add.
    t4 = pd.read_parquet(DQ / "t4_adjustments.parquet")
    master_tot = master.groupby("canonical")["consumption"].sum()
    factor = {}
    for r in t4.itertuples(index=False):
        c = x.get(r.item_number, r.item_number)
        rec = master_tot.get(c, 0)
        if rec > 0:
            factor[c] = (rec + r.implied_usage) / rec
    if factor:
        fully["consumption"] = fully["consumption"] * fully["canonical"].map(factor).fillna(1.0)

    # TRUE demand (ground truth) on the same monthly grid.
    rng = np.random.default_rng(RANDOM_SEED)
    plan = build_item_plan(rng)
    demand = build_monthly_demand(plan, rng)
    # map item_id -> canonical item_number (survivor) via the crosswalk truth
    import json
    canon_map = json.loads((REPO / "data_source" / "truth" / "crosswalks.json").read_text())["item_number_to_canonical_id"]
    id_to_survivor = {}
    for num, iid in canon_map.items():
        surv = x.get(num, num)
        id_to_survivor[iid] = surv
    demand["canonical"] = demand["item_id"].map(id_to_survivor)
    true = demand.dropna(subset=["canonical"]).groupby(["canonical", "month"])["demand_units"].sum().reset_index() \
        .rename(columns={"demand_units": "consumption"})
    true["month"] = _to_month(true["month"])

    MARTS.mkdir(parents=True, exist_ok=True)
    for name, df in [("raw", raw), ("master", master), ("fully", fully), ("true", true)]:
        df.to_parquet(MARTS / f"consumption_{name}.parquet", index=False)
    return raw, master, fully, true


def run():
    raw, master, fully, true = build()
    tot = lambda d: d["consumption"].sum()
    print("\n=== Cleaning-tier consumption marts ===")
    print(f"  true demand      {tot(true):>12,.0f}")
    print(f"  raw (recorded)   {tot(raw):>12,.0f}   {tot(raw)/tot(true)*100:5.1f}% of true")
    print(f"  master-cleaned   {tot(master):>12,.0f}   {tot(master)/tot(true)*100:5.1f}% of true")
    print(f"  fully-cleaned    {tot(fully):>12,.0f}   {tot(fully)/tot(true)*100:5.1f}% of true")
    print()


if __name__ == "__main__":
    run()
