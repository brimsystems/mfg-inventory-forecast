"""Validation checkpoint for the transaction-level defects (addendum step 3).

Confirms the generator-controllable targets before any modeling:
  - the demand-segment mix is unchanged by the transaction defects
  - T1 attribution recovers 10-18% additional consumption on affected items
  - the recorded ledger understates true demand (so cleaning has room to help)
  - T5 biases computed lead times upward by 1-3 days
  - T6 leaves a material phantom on-order quantity

The forecast-improvement and detection-confidence targets are validated after the
detection, attribution and model steps are built.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .checkpoint import classify
from .generators.demand import build_item_plan, build_monthly_demand

RAW = C.REPO_ROOT / "data_source" / "raw"
TRUTH = C.REPO_ROOT / "data_source" / "truth"


def run():
    rng = np.random.default_rng(C.RANDOM_SEED)
    plan = build_item_plan(rng)
    demand = build_monthly_demand(plan, rng)                     # true canonical demand

    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv")
    po = pd.read_csv(RAW / "erp" / "purchase_orders.csv")
    cw = json.loads((TRUTH / "crosswalks.json").read_text())["item_number_to_canonical_id"]
    txn = json.loads((TRUTH / "txn_defects.json").read_text())

    # 1. Segment mix on true demand (should be unchanged from the main checkpoint).
    wide = demand.pivot(index="item_id", columns="month", values="demand_units").fillna(0).sort_index(axis=1)
    cls = pd.Series({i: classify(wide.loc[i].to_numpy(float)) for i in wide.index})
    print("\n=== 1. Segment mix on true demand (unchanged expected) ===")
    for s in C.SEGMENTS:
        print(f"  {s:<13} {(cls == s).mean()*100:5.1f}%")

    # 2. Recorded vs true consumption. Recorded = non-generic issues mapped to the
    # canonical item; generic (T1) lines and dropped (T4) usage are missing.
    iss = tx[tx["type"] == "issue"].copy()
    iss["canonical"] = iss["item_number"].map(lambda n: cw.get(n))
    recorded = iss.dropna(subset=["canonical"]).groupby("canonical")["quantity"].sum()
    true_by_canon = demand.groupby("item_id")["demand_units"].sum()
    true_by_canon.index = true_by_canon.index.astype(float)
    recorded.index = recorded.index.astype(float)
    total_true, total_rec = true_by_canon.sum(), recorded.sum()
    print("\n=== 2. Recorded ledger vs true demand ===")
    print(f"  true consumption units   {total_true:>12,.0f}")
    print(f"  recorded (issues)        {total_rec:>12,.0f}")
    print(f"  understatement           {(1-total_rec/total_true)*100:5.1f}%  (T1 diverted + T4 unrecorded)")

    # 3. T1 demand recovered on affected items.
    t1 = pd.DataFrame(txn["t1"])
    div = t1[~t1["is_oneoff"]].merge(tx[["transaction_id", "quantity"]], on="transaction_id", how="left")
    div["canonical"] = div["true_item_number"].map(lambda n: cw.get(n))
    div_by_canon = div.groupby("canonical")["quantity"].sum()
    affected = div_by_canon.index.astype(float)
    rec_affected = recorded.reindex(affected).fillna(0).sum()
    div_total = div_by_canon.sum()
    print("\n=== 3. T1 attribution recovery (target 10-18%) ===")
    print(f"  affected items           {len(affected)}")
    print(f"  diverted volume          {div_total:>12,.0f}")
    print(f"  recovery vs recorded     {div_total/rec_affected*100:5.1f}%  additional consumption on affected items")

    # 4. T5 lead-time bias.
    t5 = pd.DataFrame(txn["t5"])
    disp = (pd.to_datetime(t5["recorded_received_date"]) - pd.to_datetime(t5["true_received_date"])).dt.days
    print("\n=== 4. T5 receipt batching (target median 1-3 days) ===")
    print(f"  receipts displaced       {len(t5)}  ({len(t5)/po['received_date'].notna().sum()*100:.0f}% of receipts)")
    print(f"  median displacement      {disp.median():.0f} days")

    # 5. T6 phantom on-order.
    t6 = pd.DataFrame(txn["t6"])
    open_qty = (t6["quantity_ordered"] - t6["quantity_received"]).sum()
    print("\n=== 5. T6 phantom on-order (never-closed POs) ===")
    print(f"  open PO lines            {len(t6)}  ({len(t6)/len(po)*100:.1f}% of PO lines)")
    print(f"  phantom on-order units   {open_qty:>12,.0f}  (inbound that will never arrive)")

    # 6. Baseline forecastability: does the understatement move naive accuracy on
    # the affected items (dirty recorded vs true)?
    aff_ids = set(int(x) for x in affected)
    r_wide = (iss.dropna(subset=["canonical"]).assign(
        month=lambda d: pd.to_datetime(d["transaction_date"]).values.astype("datetime64[M]"))
        .groupby(["canonical", "month"])["quantity"].sum().reset_index())
    print("\n=== 6. Understatement is concentrated, not uniform ===")
    aff_true = true_by_canon.reindex(affected).sum()
    aff_rec = recorded.reindex(affected).fillna(0).sum()
    print(f"  affected items recorded  {aff_rec/aff_true*100:5.1f}% of their true demand")
    print(f"  (a forecast on the recorded series runs low for exactly these items)\n")


if __name__ == "__main__":
    run()
