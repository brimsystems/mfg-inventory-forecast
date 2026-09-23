"""Decision quality: the same forecast through dirty inputs and through clean ones.

Cleaning the data improves the forecast only modestly, because only three of
the sixteen errors touch the demand history a model learns from. The other
thirteen corrupt what the forecast is used for. This test holds the forecast
fixed (the chosen model's backtest predictions, per item) and runs the same
order-up-to policy over the holdout year twice, week by week:

  dirty    reorders are timed on the master lead time (stale), against a book
           balance that drifts away from the shelf by the recorded-minus-true
           consumption gap (unrecorded pulls, mispostings, keying errors), with
           the phantom on-order balances counted as inbound, and with each
           duplicate record holding its own safety stock and reordering alone
  clean    reorders are timed on the recomputed lead time, against the physical
           balance, with only genuine on-order counted, one record per part

Demand, supplier delivery and the forecast are identical in both runs; only the
inputs the decision reads differ. Reported: fill rate, stockout item-weeks,
average inventory value and purchases, each scenario on the same items.

Run:  python -m ml.src.decision
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from .resolution import REPO

MARTS = REPO / "ml" / "data" / "marts"
BACKTEST = REPO / "ml" / "data" / "backtest"
RAW = REPO / "data_source" / "raw"
TRUTH = REPO / "data_source" / "truth"
SEEDS = REPO / "data_pipeline" / "seeds"
OUT = REPO / "ml" / "data" / "policy"

HOLDOUT_START = pd.Timestamp("2025-01-01")
WEEKS = 52
Z_BY_ABC = {"A": 2.05, "B": 1.64, "C": 1.28}      # 98 / 95 / 90 percent service
PHANTOM_DAYS = 90
REVIEW_WEEKS = 1


def _weeks(lead_days):
    return max(1, int(round(lead_days / 7.0)))


def _weekly(monthly, months, weeks_index):
    """Spread each month's quantity evenly over its weeks."""
    out = np.zeros(len(weeks_index))
    per_month = {m: [i for i, w in enumerate(weeks_index) if w.month == m.month and w.year == m.year] for m in months}
    for m, q in zip(months, monthly):
        idx = per_month[m]
        if idx:
            out[idx] = q / len(idx)
    return out


def _simulate(true_d, rec_d, fc_w, sd_w, z, lead_decide, lead_actual, phantom, cost):
    """Order-up-to, weekly. The decision reads the book; the shelf obeys the truth."""
    S = fc_w * lead_decide + z * sd_w * np.sqrt(lead_decide) + fc_w * REVIEW_WEEKS
    physical = S
    gap = 0.0                                # book minus physical
    pipeline = defaultdict(float)
    fill = demand = 0.0
    stockouts = 0
    inv_value = 0.0
    purchases = 0.0
    T = len(true_d)
    for t in range(T):
        physical += pipeline.pop(t, 0.0)
        d = true_d[t]
        filled = min(physical, d)
        physical -= filled
        demand += d; fill += filled
        if d - filled > 1e-9:
            stockouts += 1
        gap += true_d[t] - rec_d[t]          # what the shelf lost that the book did not see
        book = physical + gap
        position = book + sum(pipeline.values()) + phantom
        order = max(0.0, S - position)
        if order > 0:
            pipeline[t + lead_actual] += order
            purchases += order * cost
        inv_value += physical * cost
    return {"demand": demand, "fill": fill, "stockouts": stockouts,
            "inv_value": inv_value / T, "purchases": purchases}


def run():
    true = pd.read_parquet(MARTS / "consumption_true.parquet")
    attrs = pd.read_parquet(MARTS / "item_attributes.parquet").set_index("canonical_item_number")
    bt = pd.read_parquet(BACKTEST / "model_backtest.parquet")
    im = pd.read_csv(RAW / "erp" / "item_master.csv", low_memory=False).set_index("item_number")
    po = pd.read_csv(RAW / "erp" / "purchase_orders.csv", low_memory=False)
    xw = pd.read_csv(SEEDS / "item_crosswalk.csv")
    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv", low_memory=False)
    cross = json.loads((TRUTH / "crosswalks.json").read_text())

    months = pd.date_range(HOLDOUT_START, periods=12, freq="MS")
    weeks = pd.date_range(HOLDOUT_START, periods=WEEKS, freq="7D")
    tw = true[true["month"].isin(months)].pivot(index="canonical", columns="month", values="consumption").reindex(columns=months).fillna(0)

    # recorded consumption per record number, straight from the ledger
    rec = tx[tx["type"].isin(["ISSUE", "BACKFLUSH"]) & (tx["txn_date"] >= "2025-01-01") & (tx["txn_date"] <= "2025-12-31")].copy()
    rec["month"] = pd.to_datetime(rec["txn_date"]).dt.to_period("M").dt.to_timestamp()
    rw = rec.pivot_table(index="item_number", columns="month", values="qty", aggfunc="sum").reindex(columns=months).fillna(0)

    # the same forecast for both runs: the model's predictions over the holdout, per item
    fc_m = bt.groupby("item")["pred"].mean()
    sd_m = (bt["actual"] - bt["pred"]).groupby(bt["item"]).std().fillna(0)

    # dirty inputs: master lead time, phantom on-order at the holdout start, duplicate records
    canon_of = xw.set_index("item_number")["canonical_item_number"].to_dict()
    members = defaultdict(list)
    for cl in cross["duplicate_clusters"].values():
        for n in cl["records"]:
            members[cl["primary"]].append(n)
    po["od"] = pd.to_datetime(po["order_date"]); po["rd"] = pd.to_datetime(po["received_date"])
    # a never-closed line carries the date of its partial receipt, so the test is
    # an open status with a balance still outstanding, on a line old enough to be stale
    ph = po[(po["status"] == "OPEN") & (po["od"] < HOLDOUT_START - pd.Timedelta(days=PHANTOM_DAYS))
            & (po["qty_received"] < po["qty_ordered"])].copy()
    ph["canon"] = ph["item_number"].map(canon_of).fillna(ph["item_number"])
    ph["bal"] = (ph["qty_ordered"] - ph["qty_received"]).clip(lower=0)
    phantom = ph.groupby("canon")["bal"].sum()
    prior = tx[(tx["type"].isin(["ISSUE", "BACKFLUSH"])) & (tx["txn_date"] >= "2024-01-01") & (tx["txn_date"] < "2025-01-01")]
    share = prior.groupby("item_number")["qty"].sum()

    rows, skipped = [], defaultdict(int)
    for item in tw.index:
        if item not in attrs.index:
            skipped["no attributes"] += 1; continue
        a = attrs.loc[item]
        td_m = tw.loc[item].to_numpy(float)
        if td_m.sum() <= 0:
            skipped["no demand in holdout"] += 1; continue
        cost = float(a["standard_cost"]) if not np.isnan(a["standard_cost"]) else float(attrs["standard_cost"].median())
        z = Z_BY_ABC[a["abc"]]
        # forecast per week; fall back to the holdout mean where the backtest has no row
        f_m = float(fc_m[item]) if item in fc_m.index else float(td_m.mean())
        s_m = float(sd_m[item]) if item in sd_m.index else float(td_m.std())
        f_w, s_w = f_m / 4.33, s_m / np.sqrt(4.33)
        lead_actual = _weeks(a["corrected_lead_days"])
        master_lead = im["master_lead_time_days"].get(item, np.nan)
        lead_dirty = _weeks(master_lead if not pd.isna(master_lead) else a["corrected_lead_days"])
        td = _weekly(td_m, months, weeks)

        clean = _simulate(td, td, f_w, s_w, z, lead_actual, lead_actual, 0.0, cost)

        recs = members.get(item, [item])
        if len(recs) > 1:
            w = np.array([max(float(share.get(n, 0.0)), 1.0) for n in recs]); w = w / w.sum()
            parts = []
            for n, wi in zip(recs, w):
                rd = _weekly(rw.loc[n].to_numpy(float) if n in rw.index else np.zeros(12), months, weeks)
                parts.append(_simulate(td * wi, rd, f_w * wi, s_w * np.sqrt(wi), z, lead_dirty, lead_actual,
                                       float(phantom.get(item, 0.0)) * wi, cost))
            dirty = {k: sum(p_[k] for p_ in parts) for k in parts[0]}
        else:
            rd = _weekly(rw.loc[item].to_numpy(float) if item in rw.index else np.zeros(12), months, weeks)
            dirty = _simulate(td, rd, f_w, s_w, z, lead_dirty, lead_actual, float(phantom.get(item, 0.0)), cost)

        rows.append({"item": item, "abc": a["abc"], "cost": cost, "records": len(recs),
                     "lead_dirty": lead_dirty, "lead_actual": lead_actual, "phantom": float(phantom.get(item, 0.0)),
                     "drift": float(td.sum() - (sum(_weekly(rw.loc[n].to_numpy(float), months, weeks).sum() for n in recs if n in rw.index))),
                     **{f"clean_{k}": v for k, v in clean.items()}, **{f"dirty_{k}": v for k, v in dirty.items()}})
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT / "decision_comparison.parquet", index=False)

    summary = {}
    for sc in ["dirty", "clean"]:
        summary[sc] = {
            "fill_rate": float(res[f"{sc}_fill"].sum() / res[f"{sc}_demand"].sum()),
            "stockout_item_weeks": int(res[f"{sc}_stockouts"].sum()),
            "items_with_stockout": int((res[f"{sc}_stockouts"] > 0).sum()),
            "avg_inventory_value": float(res[f"{sc}_inv_value"].sum()),
            "purchases": float(res[f"{sc}_purchases"].sum()),
        }
    summary["items"] = int(len(res)); summary["item_weeks"] = int(len(res) * WEEKS)
    summary["skipped"] = dict(skipped)
    summary["note"] = ("same forecast and same true demand in both runs; dirty = master lead time for timing, "
                       "book drifting by the recorded-minus-true gap, phantom on-order counted as inbound, "
                       "duplicate records reordering separately; deliveries on the recomputed lead time in both")
    (OUT / "decision_summary.json").write_text(json.dumps(summary, indent=2))

    M = lambda x: f"${x:,.0f}"
    print("\n=== Decision quality, same forecast, dirty vs clean inputs (52-week holdout) ===")
    print(f"  items {summary['items']:,}  item-weeks {summary['item_weeks']:,}  skipped {dict(skipped)}")
    for sc in ["dirty", "clean"]:
        s = summary[sc]
        print(f"  {sc:<6} fill rate {s['fill_rate']*100:5.1f}% | stockout item-weeks {s['stockout_item_weeks']:6,} "
              f"({s['items_with_stockout']} items) | avg inventory {M(s['avg_inventory_value']):>12} | purchases {M(s['purchases']):>12}")
    return summary


if __name__ == "__main__":
    run()
