"""On-hand balance reliability, measured before and after remediation.

Every live item is classed by one rule, applied at two dates from the counts
and the ledger as they stood on each:

  reliable     counted within the last 90 days, with the count within 5% of the
               system balance, and none of the faults below
  unreliable   a fault that makes the balance untrustworthy whatever the last
               count said: three or more downward adjustments in the trailing
               window (chronic unrecorded consumption), a purchase unit that
               differs from the stock unit with no conversion on file, or a
               duplicate record sharing one pile
  uncertain    everything else: no count within 90 days, or a last count 5%
               or more off

Before is the day remediation began, when the last count was the annual
physical nine months earlier. After is the audit date, with the remediation
counts and the first quarter of the ABC cycle-count program behind it, the
conversions on file, the duplicate records retired to their survivors and the
bills of materials corrected. Value is the book balance on the date at
standard cost, the class median where the cost is blank.

Run:  python -m ml.src.reliability
"""
from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd

from data_source.generate import config as C

REPO = C.REPO_ROOT
RAW = REPO / "data_source" / "raw"
REM = RAW / "remediation"
TRUTH = REPO / "data_source" / "truth"
MARTS = REPO / "ml" / "data" / "marts"

BEFORE = C.REMEDIATION_START
AFTER = C.AS_OF_DATE
RECENT_DAYS = 90
VAR_RELIABLE = 0.05
CHRONIC_N = 3
SIGN = {"RECEIPT": 1, "RETURN": 1, "ISSUE": -1, "BACKFLUSH": -1, "SCRAP": -1, "ADJUST": 1}
COUNT_REASONS = {"COUNT", "CYCLE"}

DEFINITIONS = {
    "reliable": "counted within 90 days, variance under 5%",
    "uncertain": "no recent count, or variance 5% or more",
    "unreliable": "chronic adjustments, UOM issue, or duplicate",
}


def build():
    im = pd.read_csv(RAW / "erp" / "item_master.csv", low_memory=False)
    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv", low_memory=False)
    cc = pd.read_csv(RAW / "wms" / "cycle_counts.csv", low_memory=False)
    cross = json.loads((TRUTH / "crosswalks.json").read_text())
    dead = set(pd.read_csv(REM / "dead_item_dispositions.csv")["item_number"])
    dup_cw = pd.read_csv(REM / "duplicate_crosswalk.csv")
    uomc = pd.read_csv(REM / "uom_conversions.csv")
    abc = {int(k): v for k, v in cross["abc_by_item"].items()}

    live = im[~im["item_number"].isin(dead)].copy()
    live["iid"] = live["item_number"].map(lambda n: int(str(n).split("-")[1]))
    class_median = im.groupby("item_class")["standard_cost"].median()
    cost = im.set_index("item_number")["standard_cost"]
    cost = cost.fillna(im.set_index("item_number")["item_class"].map(class_median)).fillna(cost.median())

    # ledger balance on a date, as the ERP sums it
    t = tx[["item_number", "txn_date", "type", "qty", "reason_code"]].copy()
    t["s"] = t["qty"] * t["type"].map(SIGN).fillna(0)
    t = t.sort_values("txn_date")
    per = {n: (g["txn_date"].to_numpy(), np.cumsum(g["s"].to_numpy())) for n, g in t.groupby("item_number")}

    def balance(num, d):
        if num not in per:
            return 0.0
        dates, cum = per[num]
        k = int(np.searchsorted(dates, d.isoformat(), side="right"))
        return float(cum[k - 1]) if k else 0.0

    # downward adjustments that were not a count correction, by item and date
    adj = t[(t["type"] == "ADJUST") & (t["qty"] < 0) & ~t["reason_code"].astype(str).isin(COUNT_REASONS)]
    adj_dates = {n: g["txn_date"].to_numpy() for n, g in adj.groupby("item_number")}

    def chronic(num, start, end):
        a = adj_dates.get(num)
        if a is None:
            return False
        return int(((a >= start.isoformat()) & (a <= end.isoformat())).sum()) >= CHRONIC_N

    # the last count on or before a date, and how far it was off
    cc = cc.sort_values("count_date")
    cc["var"] = (cc["counted_qty"] - cc["system_qty"]).abs() / cc["system_qty"].clip(lower=1)
    cc.loc[(cc["system_qty"] <= 0) & (cc["counted_qty"] > 0), "var"] = 1.0
    counts = {n: (g["count_date"].to_numpy(), g["var"].to_numpy()) for n, g in cc.groupby("item_number")}

    def last_count(num, d):
        if num not in counts:
            return None, None
        dates, var = counts[num]
        k = int(np.searchsorted(dates, d.isoformat(), side="right"))
        if not k:
            return None, None
        return pd.Timestamp(dates[k - 1]).date(), float(var[k - 1])

    # faults and what the remediation did about them
    conv_items = set(cross["m5_conversions"])
    conv_added = uomc.set_index("item_number")["added_date"].to_dict()
    dup_members, survivor_of = set(), {}
    for cl in cross["duplicate_clusters"].values():
        dup_members.update(cl["records"])
    merged = dup_cw[dup_cw["decision"] == "MERGE"]
    retired = set(merged["retired_item_number"])
    for r in merged.itertuples(index=False):
        survivor_of[r.retired_item_number] = r.survivor_item_number
    cleared_dups = retired | set(merged["survivor_item_number"])

    def classify(num, ref, after):
        if after and num in cleared_dups:
            dup = False
        else:
            dup = num in dup_members
        uom = num in conv_items and not (after and num in conv_added and conv_added[num] <= ref.isoformat())
        # after remediation the trailing window starts where the counts reset the
        # balances and reason codes became mandatory
        start = max(ref - timedelta(days=365), C.REMEDIATION_END) if after else ref - timedelta(days=365)
        chron = chronic(num, start, ref)
        last, var = last_count(num, ref)
        if dup or uom or chron:
            return "unreliable"
        if last is not None and (ref - last).days <= RECENT_DAYS and var < VAR_RELIABLE:
            return "reliable"
        return "uncertain"

    rec = []
    for r in live.itertuples(index=False):
        num = r.item_number
        value_before = max(balance(num, BEFORE), 0.0) * float(cost[num])
        if num in retired:
            value_after, after_cls = 0.0, None            # the pile now sits under the survivor
        else:
            bal = balance(num, AFTER) + sum(balance(m, AFTER) for m, s_ in survivor_of.items() if s_ == num)
            value_after, after_cls = max(bal, 0.0) * float(cost[num]), classify(num, AFTER, True)
        rec.append({"item_number": num, "abc": abc.get(int(r.iid), "C"),
                    "value": value_before, "value_after": value_after,
                    "reliability_before": classify(num, BEFORE, False),
                    "reliability_after": after_cls, "retired": num in retired})
    rel = pd.DataFrame(rec)
    MARTS.mkdir(parents=True, exist_ok=True)
    rel.to_parquet(MARTS / "reliability.parquet", index=False)

    summary = {"before_date": BEFORE.isoformat(), "after_date": AFTER.isoformat(), "definitions": DEFINITIONS,
               "recent_days": RECENT_DAYS, "before": {}, "after": {}}
    for k in DEFINITIONS:
        b = rel[rel["reliability_before"] == k]
        a = rel[rel["reliability_after"] == k]
        summary["before"][k] = {"items": int(len(b)), "value": float(b["value"].sum())}
        summary["after"][k] = {"items": int(len(a)), "value": float(a["value_after"].sum())}
    # why the uncertain items are uncertain at the audit date
    unc = rel[rel["reliability_after"] == "uncertain"]["item_number"]
    recent = [n for n in unc if last_count(n, AFTER)[0] is not None and (AFTER - last_count(n, AFTER)[0]).days <= RECENT_DAYS]
    summary["after"]["uncertain_reasons"] = {"no_recent_count": int(len(unc) - len(recent)),
                                            "counted_variance_5pct_or_more": int(len(recent))}
    summary["after"]["uncertain_by_abc"] = {k: int(v) for k, v in rel[rel["reliability_after"] == "uncertain"]["abc"].value_counts().items()}
    summary["after"]["reliable_by_abc"] = {k: int(v) for k, v in rel[rel["reliability_after"] == "reliable"]["abc"].value_counts().items()}
    summary["before"]["total"] = {"items": int(len(rel)), "value": float(rel["value"].sum())}
    summary["after"]["total"] = {"items": int((~rel["retired"]).sum()), "value": float(rel["value_after"].sum())}
    (MARTS / "reliability_summary.json").write_text(json.dumps(summary, indent=2))
    return rel, summary


def run():
    rel, summary = build()
    print("\n=== On-hand reliability (live items) ===")
    for when in ["before", "after"]:
        tot = summary[when]["total"]
        print(f"\n  {when.upper()}  ({summary[when + '_date']}; {tot['items']:,} items, ${tot['value']:,.0f})")
        for k in DEFINITIONS:
            v = summary[when][k]
            print(f"    {k:<12} {v['items']:5,} items ({v['items']/tot['items']*100:4.1f}%)   "
                  f"${v['value']:12,.0f}  ({v['value']/tot['value']*100:4.1f}% of value)")
    print()


if __name__ == "__main__":
    run()
