"""Three-way cleaning decomposition (addendum step 7).

The same model, hyperparameters, features and rolling origins are run on three
inputs, differing only in how clean the consumption history is: raw, master-level
cleaned, and fully cleaned. The forecast target for all three is the true demand
over each item's lead time, so the comparison isolates what each tier of cleaning
is worth. Results are reported overall, by segment, and on the item subsets each
tier repairs.

Run:  python -m ml.src.training_3way
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .resolution import REPO
from .features import _row_features, FEATURE_COLS, VAL_ORIGINS

MARTS = REPO / "ml" / "data" / "marts"
BACKTEST = REPO / "ml" / "data" / "backtest"
SEEDS = REPO / "data_pipeline" / "seeds"
TRUTH = REPO / "data_source" / "truth"

PARAMS = dict(n_estimators=450, max_depth=6, learning_rate=0.05, subsample=0.85,
              colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0, random_state=42, n_jobs=-1)


def _wide(name):
    df = pd.read_parquet(MARTS / f"consumption_{name}.parquet")
    return df.pivot(index="canonical", columns="month", values="consumption").fillna(0).sort_index(axis=1)


def _frame(variant_wide, true_wide, attrs, months):
    seg = {s: i for i, s in enumerate(["smooth", "erratic", "lumpy", "intermittent"])}
    abcc = {"A": 0, "B": 1, "C": 2}
    clsc = {c: i for i, c in enumerate(sorted(attrs["item_class"].unique()))}
    mnums = [m.month for m in months]
    rows = []
    for item in variant_wide.index:
        if item not in attrs.index or item not in true_wide.index:
            continue
        s = variant_wide.loc[item].to_numpy(float)
        ts = true_wide.loc[item].reindex(months).fillna(0).to_numpy(float)
        a = attrs.loc[item]
        h = max(1, int(round(a["corrected_lead_days"] / 30.0)))
        T = len(s)
        test_first = T - h - 11
        val_first = test_first - VAL_ORIGINS
        for t in range(12, T - h + 1):
            f = _row_features(s, t, mnums[t])
            # annual level comes from THIS tier's own history (up to the origin), so a
            # dirty tier cannot borrow the clean level and mask the cleaning benefit.
            f.update({"item": item, "origin": t, "target": float(ts[t:t + h].sum()),
                      "std_cost": a["standard_cost"], "corrected_lead": a["corrected_lead_days"],
                      "annual": float(s[max(0, t - 12):t].sum()), "price": a["standard_cost"],
                      "seg_code": seg.get(a["segment"], 0), "abc_code": abcc.get(a["abc"], 2),
                      "cls_code": clsc.get(a["item_class"], 0), "horizon": h,
                      "segment": a["segment"], "abc": a["abc"]})
            f["split"] = "test" if t >= test_first else ("val" if t >= val_first and t + h <= test_first
                                                         else ("train" if t + h <= val_first else "gap"))
            rows.append(f)
    fr = pd.DataFrame(rows)
    fr[FEATURE_COLS] = fr[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    fr[FEATURE_COLS] = fr[FEATURE_COLS].fillna(fr.loc[fr["split"] == "train", FEATURE_COLS].median())
    return fr


def _wape(a, f):
    d = np.asarray(a).sum()
    return np.abs(np.asarray(a) - np.asarray(f)).sum() / d if d else np.nan


def run():
    attrs = pd.read_parquet(MARTS / "item_attributes.parquet").set_index("canonical_item_number")
    true_wide = _wide("true")
    months = list(true_wide.columns)

    cw = pd.read_csv(SEEDS / "item_crosswalk.csv")
    clusters = cw.groupby("canonical_item_number")["item_number"].count()
    dup_items = set(clusters[clusters > 1].index)
    txn = json.loads((TRUTH / "txn_defects.json").read_text())
    x = dict(zip(cw["item_number"], cw["canonical_item_number"]))
    t1_items = {x.get(r["true_item_number"], r["true_item_number"]) for r in txn["t1"] if not r["is_oneoff"]}

    # One production model, trained on the fully-cleaned history, then fed each
    # tier's data at inference. This isolates the value of clean input: the model
    # is held fixed, only the consumption history it reads changes.
    fully_fr = _frame(_wide("fully"), true_wide, attrs, months)
    tr = fully_fr[fully_fr["split"].isin(["train", "val"])]
    model = XGBRegressor(**PARAMS)
    model.fit(tr[FEATURE_COLS], np.log1p(tr["target"]))

    results = {}
    for name in ["raw", "master", "fully"]:
        fr = fully_fr if name == "fully" else _frame(_wide(name), true_wide, attrs, months)
        te = fr[fr["split"] == "test"].copy()
        te["pred"] = np.clip(np.expm1(model.predict(te[FEATURE_COLS])), 0, None)
        results[name] = te

    print("\n=== Three-way cleaning decomposition (WAPE vs true demand over lead time) ===")
    print("  tier            overall | duplicate items | T1 free-text items | other")
    base = None
    for name in ["raw", "master", "fully"]:
        te = results[name]
        overall = _wape(te["target"], te["pred"])
        dup = _wape(te[te["item"].isin(dup_items)]["target"], te[te["item"].isin(dup_items)]["pred"])
        t1 = _wape(te[te["item"].isin(t1_items)]["target"], te[te["item"].isin(t1_items)]["pred"])
        other = _wape(te[~te["item"].isin(dup_items | t1_items)]["target"],
                      te[~te["item"].isin(dup_items | t1_items)]["pred"])
        print(f"  {name:<13} {overall*100:6.1f}% | {dup*100:6.1f}%        | {t1*100:6.1f}%          | {other*100:5.1f}%")
    r = _wape(results["raw"]["target"], results["raw"]["pred"])
    mm = _wape(results["master"]["target"], results["master"]["pred"])
    ff = _wape(results["fully"]["target"], results["fully"]["pred"])
    print(f"\n  master-level cleaning gain (raw -> master):        {(r-mm)/r*100:+.1f}% relative WAPE")
    print(f"  transaction cleaning gain (master -> fully):       {(mm-ff)/mm*100:+.1f}% relative WAPE")
    # T1-specific
    for label, subset in [("T1 items", t1_items)]:
        rm = _wape(results["master"][results["master"]["item"].isin(subset)]["target"],
                   results["master"][results["master"]["item"].isin(subset)]["pred"])
        rf = _wape(results["fully"][results["fully"]["item"].isin(subset)]["target"],
                   results["fully"][results["fully"]["item"].isin(subset)]["pred"])
        print(f"  transaction cleaning gain on {label} (master -> fully): {(rm-rf)/rm*100:+.1f}%")
    BACKTEST.mkdir(parents=True, exist_ok=True)
    for name in ["raw", "master", "fully"]:
        results[name].to_parquet(BACKTEST / f"threeway_{name}.parquet", index=False)
    json.dump({"raw": float(r), "master": float(mm), "fully": float(ff)},
              open(BACKTEST / "threeway_overall.json", "w"), indent=2)
    print()


if __name__ == "__main__":
    run()
