"""Cleaned marts: the modeled tables the forecast and policy read.

Consumption is derived from issue transactions, mapped through the item crosswalk
so duplicate records are merged into one canonical series. Lead times are
recalculated from actual purchase-order receipts rather than the stale master
value. Item attributes carry the demand segment (classified from the cleaned
series) and the ABC class.

Run:  python -m ml.src.marts
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .resolution import REPO
from data_source.generate.checkpoint import classify

RAW = REPO / "data_source" / "raw"
SEEDS = REPO / "data_pipeline" / "seeds"
MARTS = REPO / "ml" / "data" / "marts"

ABC_A_CUM, ABC_B_CUM = 0.80, 0.95


def _abc(value: pd.Series) -> pd.Series:
    order = value.sort_values(ascending=False)
    cum = order.cumsum() / order.sum()
    cls = pd.Series(np.where(cum <= ABC_A_CUM, "A", np.where(cum <= ABC_B_CUM, "B", "C")),
                    index=order.index)
    return cls.reindex(value.index)


def build_marts():
    item_master = pd.read_csv(RAW / "erp" / "item_master.csv")
    po = pd.read_csv(RAW / "erp" / "purchase_orders.csv")
    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv")
    crosswalk = pd.read_csv(SEEDS / "item_crosswalk.csv")
    x = dict(zip(crosswalk["item_number"], crosswalk["canonical_item_number"]))

    # Cleaned consumption from issues, merged to the canonical item.
    iss = tx[tx["type"] == "issue"].copy()
    iss["canonical"] = iss["item_number"].map(x).fillna(iss["item_number"])
    iss["date"] = pd.to_datetime(iss["transaction_date"])
    iss["month"] = iss["date"].values.astype("datetime64[M]")
    iss["week"] = iss["date"].dt.to_period("W").dt.start_time

    monthly = iss.groupby(["canonical", "month"])["quantity"].sum().reset_index()
    weekly = iss.groupby(["canonical", "week"])["quantity"].sum().reset_index()

    # Complete the monthly grid so gaps are explicit zeros.
    items = monthly["canonical"].unique()
    months = pd.period_range(monthly["month"].min(), monthly["month"].max(), freq="M").to_timestamp()
    grid = pd.MultiIndex.from_product([items, months], names=["canonical", "month"])
    monthly = (monthly.set_index(["canonical", "month"]).reindex(grid, fill_value=0)
               .reset_index().rename(columns={"quantity": "consumption"}))

    # Corrected lead time from actual receipts (recent window), per canonical item.
    po = po[po["received_date"].notna()].copy()
    po["canonical"] = po["item_number"].map(x).fillna(po["item_number"])
    po["order_date"] = pd.to_datetime(po["order_date"])
    po["lead"] = (pd.to_datetime(po["received_date"]) - po["order_date"]).dt.days
    recent = po[po["order_date"] >= po["order_date"].max() - pd.Timedelta(days=365)]
    corrected_lead = recent.groupby("canonical")["lead"].median()

    # Item attributes on the surviving record.
    survivors = crosswalk.groupby("canonical_item_number")["item_number"].apply(list)
    attr_rows = []
    cost = item_master.set_index("item_number")["standard_cost"]
    cls_map = item_master.set_index("item_number")["item_class"]
    master_lead = item_master.set_index("item_number")["master_lead_time_days"]
    for canon in monthly["canonical"].unique():
        series = (monthly[monthly["canonical"] == canon].sort_values("month")["consumption"].to_numpy(float))
        annual = series[-12:].sum()
        attr_rows.append({
            "canonical_item_number": canon,
            "item_class":            cls_map.get(canon, "Unknown"),
            "standard_cost":         cost.get(canon, np.nan),
            "master_lead_time_days": master_lead.get(canon, np.nan),
            "corrected_lead_days":   float(corrected_lead.get(canon, master_lead.get(canon, 14))),
            "segment":               classify(series),
            "annual_consumption":    float(annual),
        })
    attrs = pd.DataFrame(attr_rows)
    attrs["annual_value"] = attrs["annual_consumption"] * attrs["standard_cost"].fillna(attrs["standard_cost"].median())
    attrs["abc"] = _abc(attrs.set_index("canonical_item_number")["annual_value"]).values

    # Supplier lead-time performance (recorded id).
    sup_perf = (po.groupby("supplier_id")
                  .agg(pos=("po_id", "count"), median_lead=("lead", "median"),
                       p90_lead=("lead", lambda s: s.quantile(0.90))).reset_index())

    MARTS.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(MARTS / "consumption_monthly.parquet", index=False)
    weekly.to_parquet(MARTS / "consumption_weekly.parquet", index=False)
    attrs.to_parquet(MARTS / "item_attributes.parquet", index=False)
    sup_perf.to_parquet(MARTS / "supplier_performance.parquet", index=False)
    return monthly, weekly, attrs, sup_perf


def run():
    monthly, weekly, attrs, sup_perf = build_marts()
    print("\n=== Cleaned marts ===")
    print(f"  consumption_monthly   {len(monthly):,} rows, {monthly['canonical'].nunique()} items, "
          f"{monthly['month'].nunique()} months")
    print(f"  consumption_weekly    {len(weekly):,} rows")
    print(f"  item_attributes       {len(attrs)} items")
    print("  segment mix (cleaned): " +
          ", ".join(f"{k} {v*100:.0f}%" for k, v in attrs['segment'].value_counts(normalize=True).items()))
    print("  ABC mix:               " +
          ", ".join(f"{k} {v}" for k, v in attrs['abc'].value_counts().sort_index().items()))
    print(f"  supplier_performance  {len(sup_perf)} suppliers")
    print()


if __name__ == "__main__":
    run()
