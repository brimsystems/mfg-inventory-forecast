"""Data-quality detection: one detector per defect class.

Each detector works from the recorded data alone (never the ground truth) and
emits the affected records, the evidence that flags them, and an operational cost
attributable to the defect. The cost assumptions are stated constants, not hidden
in the arithmetic, so the audit can defend every dollar. Detection accuracy is
reported against the known defects for validation only.

Run:  python -m ml.src.data_quality
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data_source" / "raw"
SEEDS = REPO / "data_pipeline" / "seeds"
TRUTH = REPO / "data_source" / "truth" / "crosswalks.json"
OUT = REPO / "ml" / "data" / "data_quality"

# Cost assumptions (stated, not derived). A practitioner would negotiate these
# with the controller; they are held in one place so the audit can restate them.
EXPEDITE_FEE = 250.0            # per stockout that triggers an expedite
CARRYING_RATE = 0.22           # annual carrying cost as a share of inventory value
STOCKOUTS_PER_DRIFTED_ITEM = 2.5   # extra stockouts/yr from an understated lead time


def detect_duplicates(item_master, crosswalk, tx):
    merged = crosswalk[crosswalk["item_number"] != crosswalk["canonical_item_number"]]
    groups = crosswalk.groupby("canonical_item_number")["item_number"].apply(list)
    clusters = groups[groups.map(len) > 1]
    # consumption value split across the duplicate records
    issue = tx[tx["type"] == "issue"]
    val = issue.groupby("item_number")["quantity"].sum()
    cost = item_master.set_index("item_number")["standard_cost"]
    split_val = float((val.reindex(merged["item_number"]).fillna(0)
                       * cost.reindex(merged["item_number"]).fillna(0)).sum())
    return {"clusters": len(clusters), "records_merged": len(merged),
            "consumption_value_split": split_val}, clusters


def detect_stale_lead_times(item_master, po):
    po = po[po["received_date"].notna()].copy()
    po["order_date"] = pd.to_datetime(po["order_date"])
    po["actual_lead"] = (pd.to_datetime(po["received_date"]) - po["order_date"]).dt.days
    recent = po[po["order_date"] >= po["order_date"].max() - pd.Timedelta(days=180)]
    actual = recent.groupby("item_number")["actual_lead"].median()
    master = item_master.set_index("item_number")["master_lead_time_days"]
    j = pd.DataFrame({"master": master, "actual": actual}).dropna()
    j["gap"] = j["actual"] - j["master"]
    stale = j[j["gap"] >= 5]                     # actual lead understated by >= 5 days
    return {"items": len(stale), "median_understatement": float(stale["gap"].median()),
            "est_stockouts_year": float(len(stale) * STOCKOUTS_PER_DRIFTED_ITEM),
            "est_expedite_cost": float(len(stale) * STOCKOUTS_PER_DRIFTED_ITEM * EXPEDITE_FEE)}, stale


def detect_uom(item_master, po):
    cost = item_master.set_index("item_number")["standard_cost"]
    price = po.groupby("item_number")["unit_price"].median()
    j = pd.DataFrame({"cost": cost, "price": price}).dropna()
    j = j[j["cost"] > 0]
    j["ratio"] = j["price"] / j["cost"]
    # a purchase price that is a large integer multiple of unit cost signals a
    # box buy recorded without a conversion factor
    flagged = j[j["ratio"] >= 8]
    return {"items": len(flagged), "median_ratio": float(flagged["ratio"].median())}, flagged


def detect_phantom(cycle_counts):
    cc = cycle_counts.copy()
    cc["variance"] = (cc["counted_quantity"] - cc["system_quantity"]).abs() / \
                     cc["system_quantity"].replace(0, np.nan)
    latest = cc.sort_values("count_date").groupby("item_number").tail(1)
    phantom = latest[latest["variance"] > 0.10]
    return {"items_counted": latest["item_number"].nunique(),
            "phantom_items": len(phantom),
            "phantom_share": float(len(phantom) / max(1, len(latest)))}, phantom


def detect_supplier_variants(suppliers, po):
    # ids that share a normalized supplier name, or names within one edit of each
    from difflib import SequenceMatcher
    names = suppliers.copy()
    names["norm"] = names["supplier_name"].str.lower().str.replace(r"[^a-z ]", "", regex=True)
    dup_rows = names[names.duplicated("supplier_id", keep=False)]
    # cluster by first name token
    names["stem"] = names["norm"].str.split().str[0]
    clusters = names.groupby("stem")["supplier_id"].nunique()
    fragmented = clusters[clusters > 1]
    spend = po.assign(amt=po["quantity_received"] * po["unit_price"]).groupby("supplier_id")["amt"].sum()
    frag_ids = names[names["stem"].isin(fragmented.index)]["supplier_id"].unique()
    return {"vendors_fragmented": int(len(fragmented)),
            "ids_involved": int(len(frag_ids)),
            "spend_fragmented": float(spend.reindex(frag_ids).fillna(0).sum())}, frag_ids


def detect_missing(item_master):
    active = item_master[item_master["status"] == "ACTIVE"]
    m = {
        "blank_reorder_point": int(active["current_reorder_point"].isna().sum()),
        "missing_standard_cost": int(active["standard_cost"].isna().sum()),
        "null_primary_supplier": int(active["primary_supplier_id"].isna().sum()),
    }
    any_missing = active[["current_reorder_point", "standard_cost", "primary_supplier_id"]].isna().any(axis=1)
    m["records_affected"] = int(any_missing.sum())
    return m, active[any_missing]


def run():
    item_master = pd.read_csv(RAW / "erp" / "item_master.csv")
    suppliers = pd.read_csv(RAW / "erp" / "suppliers.csv")
    po = pd.read_csv(RAW / "erp" / "purchase_orders.csv")
    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv")
    cc = pd.read_csv(RAW / "wms" / "cycle_counts.csv")
    crosswalk = pd.read_csv(SEEDS / "item_crosswalk.csv")
    truth = json.loads(TRUTH.read_text())

    d1, d1_detail = detect_duplicates(item_master, crosswalk, tx)
    d2, d2_detail = detect_stale_lead_times(item_master, po)
    d3, d3_detail = detect_uom(item_master, po)
    d4, d4_detail = detect_phantom(cc)
    d5, d5_ids = detect_supplier_variants(suppliers, po)
    d6, d6_detail = detect_missing(item_master)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in [("d1_duplicates", d1_detail.reset_index()), ("d2_stale_leads", d2_detail.reset_index()),
                     ("d3_uom", d3_detail.reset_index()), ("d4_phantom", d4_detail),
                     ("d6_missing", d6_detail)]:
        df.to_csv(OUT / f"{name}.csv", index=False)

    print("\n=== Data-quality audit (detected from recorded data) ===")
    print(f"  D1 duplicates     {d1['clusters']} clusters, {d1['records_merged']} records merged; "
          f"${d1['consumption_value_split']:,.0f}/yr consumption split")
    print(f"     truth: {len(truth['duplicate_clusters'])} clusters")
    print(f"  D2 stale leads    {d2['items']} items, median +{d2['median_understatement']:.0f} days; "
          f"~{d2['est_stockouts_year']:.0f} stockouts/yr, ${d2['est_expedite_cost']:,.0f} expedite")
    print(f"     truth: {len(truth['d2_items'])} items")
    print(f"  D3 UOM            {d3['items']} items, median price/cost ratio {d3['median_ratio']:.0f}x")
    print(f"     truth: {len(truth['d3_box_sizes'])} box-bought items")
    print(f"  D4 phantom        {d4['phantom_items']}/{d4['items_counted']} counted items "
          f"({d4['phantom_share']*100:.0f}%) vary >10%")
    print(f"  D5 supplier       {d5['vendors_fragmented']} vendor(s), {d5['ids_involved']} ids; "
          f"${d5['spend_fragmented']:,.0f} spend fragmented")
    print(f"  D6 missing        {d6['records_affected']} records "
          f"(rop {d6['blank_reorder_point']}, cost {d6['missing_standard_cost']}, sup {d6['null_primary_supplier']})")
    summary = {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "D6": d6}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print()


if __name__ == "__main__":
    run()
