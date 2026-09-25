"""The reorder points the shop runs on in the forward window.

Weekly demand model
-------------------
Features are rolling windows that end on the forecast date (usage over the last
4, 13, 26 and 52 weeks, weeks with demand, weeks since the last demand, recent
trend, the same weeks last year, the calendar) plus the item's attributes. The
target is demand over the item's lead time, in whole weeks, from the forecast
date. Candidates are tuned on validation weeks and scored on a held-out 2025.

The model is retrained on the first forecast date of each month and refreshes
every item's reorder point every Monday:

    forecast   = demand over the lead time, corrected for bias per demand pattern
    buffer     = k(ABC) x sqrt( forecast error^2 + (daily usage x lead-time sd)^2 )
    level      = forecast - scheduled share x forecast over the MRP horizon + buffer
    reorder pt = forecast + buffer                      (what the buyer sees)

The ERP nets the requirements of released jobs (MRP) on top of the level. Jobs
are released about a month before they draw parts, less than most lead times, so
the level leaves out only the scheduled demand MRP can see (the scheduled share
over the visibility horizon) and covers everything else. k is calibrated on the 2025 errors at
each class's service level. The lead-time sd comes from each item's receipts. A
point only moves when it changes by more than 20%.

How much to order comes from the same forecast: an economic lot for each item,
sqrt(2 x annual demand x cost of placing an order line / annual cost of holding
a unit), kept between two weeks and four months of demand. Expensive, fast parts
are bought often in small lots; cheap parts less often in larger ones.

The buffer is set against a fill-rate target (98 / 95 / 90 percent of units by
class), not against every order cycle: the expected units short per cycle,
sigma x G(k), may not exceed (1 - fill rate) x the order quantity, where G is
the loss function of the 2025 standardized errors for the item's class. A large
lot protects most of its own cycle, so cheap parts carried in big lots need
little buffer, and the stock goes to the parts ordered often.

The rule the shop would run without the model is refreshed monthly from the
trailing year's usage over the corrected lead time, with a Poisson demand buffer
and the same lead-time term, netted the same way. It keeps the buyers' lot
size, so the difference between the two is the model.

Writes ml/data/policy/rop_schedule.json (model), rule_schedule.json (rule) and
ml/data/backtest/weekly_backtest.parquet, weekly_metrics.json.
Run:  python -m ml.src.forward_policy
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import optuna
import pandas as pd

from .resolution import REPO
from . import training as tr
from .training import _make, _fit_predict, _space, _wape
from data_source.generate import config as C

MARTS = REPO / "ml" / "data" / "marts"
BACKTEST = REPO / "ml" / "data" / "backtest"
RAW = REPO / "data_source" / "raw"
OUT = REPO / "ml" / "data" / "policy"
Z_BY_ABC = {"A": 2.05, "B": 1.64, "C": 1.28}      # 98 / 95 / 90 percent service
SERVICE = {"A": 0.98, "B": 0.95, "C": 0.90}     # cycle service levels, for reference
# fill-rate targets by criticality, not by value: a missing $2 fitting holds a
# job as surely as a missing motor. Line-critical parts sit on a production bill
# (a shortage holds a job), service-critical parts go out on service orders (a
# shortage delays a customer repair), and the rest are shop supplies and pulls.
FILL_TARGET = {"line": 0.99, "service": 0.985, "standard": 0.985}
K_MAX = 4.0
HYSTERESIS = 0.20
ORDER_LINE_COST = 35.0         # buyer, receiving and payables time per order line ($)
HOLDING_RATE = 0.25            # annual cost of holding a dollar of stock
LOT_DAYS = (14, 120)           # an order covers between two weeks and four months of demand
SEG_CODE = {"smooth": 0, "erratic": 1, "lumpy": 2, "intermittent": 3}
ABC_CODE = {"A": 0, "B": 1, "C": 2}
FEATS = ["s4", "s13", "s26", "s52", "nz13", "nz52", "since_nz", "cv13", "trend", "ly", "mean_all",
         "woy", "month", "std_cost", "lead", "annual", "seg_code", "abc_code", "cls_code", "h"]
N_TRIALS = 12


def _monday(d):
    return d - timedelta(days=d.weekday())


def build_frame(weekly, attrs):
    """One row per item and origin week; features end before the origin, target after."""
    weeks = pd.date_range(weekly["week"].min(), _monday(C.END_DATE), freq="W-MON")
    wide = weekly.pivot(index="canonical", columns="week", values="consumption").reindex(columns=weeks).fillna(0)
    cls_code = {c: i for i, c in enumerate(sorted(attrs["item_class"].unique()))}
    T = len(weeks)
    woy = np.array([w.isocalendar()[1] for w in weeks]); mon = np.array([w.month for w in weeks])
    rows = []
    for item, sr in wide.iterrows():
        if item not in attrs.index:
            continue
        a = attrs.loc[item]
        s = sr.to_numpy(float); cs = np.concatenate([[0.0], np.cumsum(s)]); nz = np.concatenate([[0], np.cumsum(s > 0)])
        h = max(1, int(round(float(a["corrected_lead_days"]) / 7.0)))
        last_nz = -1; since = np.zeros(T)
        for t in range(T):
            since[t] = t - last_nz if last_nz >= 0 else t + 1
            if s[t] > 0:
                last_nz = t
        t_idx = np.arange(52, T)
        def win(k):
            return cs[t_idx] - cs[t_idx - k]
        s4, s13, s26, s52 = win(4), win(13), win(26), win(52)
        sq13 = np.array([s[t - 13:t].std() for t in t_idx])
        ly = np.array([s[t - 52:t - 52 + h].sum() for t in t_idx])
        tgt = np.array([s[t:t + h].sum() if t + h <= T else np.nan for t in t_idx])
        f = pd.DataFrame({"item": item, "t": t_idx, "week": weeks[t_idx], "h": h, "target": tgt,
                          "s4": s4, "s13": s13, "s26": s26, "s52": s52,
                          "nz13": nz[t_idx] - nz[t_idx - 13], "nz52": nz[t_idx] - nz[t_idx - 52],
                          "since_nz": since[t_idx], "cv13": np.where(s13 > 0, sq13 / np.maximum(s13 / 13, 1e-9), 0.0),
                          "trend": s4 / 4 - s13 / 13, "ly": ly, "mean_all": cs[t_idx] / t_idx,
                          "woy": woy[t_idx], "month": mon[t_idx], "std_cost": a["standard_cost"],
                          "lead": a["corrected_lead_days"], "annual": a["annual_consumption"],
                          "seg_code": SEG_CODE.get(a["segment"], 0), "abc_code": ABC_CODE.get(a["abc"], 2),
                          "cls_code": cls_code.get(a["item_class"], 0), "segment": a["segment"], "abc": a["abc"]})
        rows.append(f)
    return pd.concat(rows, ignore_index=True), weeks


def _tune(kind, trn, val):
    def objective(trial):
        p = _space(trial, kind)
        return _wape(val["target"], _fit_predict(_make(kind, p), trn[FEATS], trn["target"], val[FEATS]))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params, study.best_value


def _baselines(frame):
    """Simple methods scaled to the horizon: 4-, 13- and 52-week rates and last year's same weeks."""
    return pd.DataFrame({"ma4": frame["s4"] / 4 * frame["h"], "ma13": frame["s13"] / 13 * frame["h"],
                         "ma52": frame["s52"] / 52 * frame["h"], "snaive": frame["ly"]}, index=frame.index)


def run():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    weekly = pd.read_parquet(MARTS / "consumption_weekly.parquet")
    attrs = pd.read_parquet(MARTS / "item_attributes.parquet").set_index("canonical_item_number")
    po = pd.read_csv(RAW / "erp" / "purchase_orders.csv", low_memory=False)
    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv", low_memory=False)
    xw = pd.read_csv(REPO / "data_pipeline" / "seeds" / "item_crosswalk.csv").set_index("item_number")["canonical_item_number"].to_dict()

    frame, weeks = build_frame(weekly, attrs)
    frame[FEATS] = frame[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0)
    week_idx = {w: i for i, w in enumerate(weeks)}
    t_fwd = week_idx[pd.Timestamp(_monday(C.FORWARD_START))]      # first week of the forward window

    # ── backtest as of the forward start: train / validation / held-out 2025 ─
    t_test0 = week_idx[pd.Timestamp(_monday(date(2025, 1, 1) + timedelta(days=7)))]
    t_val0 = t_test0 - 26
    known = frame["t"] + frame["h"] <= t_fwd
    test = frame[known & (frame["t"] >= t_test0)].copy()
    val = frame[(frame["t"] >= t_val0) & (frame["t"] + frame["h"] <= t_test0)]
    trn = frame[frame["t"] + frame["h"] <= t_val0]
    tr._CAP = float(pd.concat([trn, val])["target"].max()) * 4
    print(f"  weekly frame: train {len(trn):,}  validation {len(val):,}  test {len(test):,}")
    results = {}
    for kind in ["Linear", "RandomForest", "XGBoost"]:
        bp, vw = _tune(kind, trn, val)
        tv = pd.concat([trn, val])
        tw_ = _wape(test["target"], _fit_predict(_make(kind, bp), tv[FEATS], tv["target"], test[FEATS]))
        results[kind] = {"params": bp, "val_wape": vw, "test_wape": tw_}
        print(f"  {kind:<13} validation {vw*100:5.1f}%  test {tw_*100:5.1f}%")
    winner = min(results, key=lambda k_: results[k_]["val_wape"])
    if winner != "XGBoost" and results["XGBoost"]["val_wape"] <= results[winner]["val_wape"] + 0.005:
        winner = "XGBoost"
    params = results[winner]["params"]
    tv = pd.concat([trn, val])
    test["pred"] = _fit_predict(_make(winner, params), tv[FEATS], tv["target"], test[FEATS])
    base = _baselines(test)
    seg_method = {sg: min(base.columns, key=lambda b: _wape(g["target"], base.loc[g.index, b]))
                  for sg, g in test.groupby("segment")}
    test["base"] = [base.at[i, seg_method[sg]] for i, sg in zip(test.index, test["segment"])]

    # bias correction, per-item error and the calibrated safety factor, all from 2025
    bias = (test.groupby("segment")["target"].sum() / test.groupby("segment")["pred"].sum()).to_dict()
    test["pred_c"] = test["pred"] * test["segment"].map(bias)
    resid = (test["target"] - test["pred_c"]).groupby(test["item"]).std()
    seg_resid = (test["target"] - test["pred_c"]).groupby(test["segment"]).std()
    std_err = ((test["target"] - test["pred_c"]) / test["item"].map(resid)).replace([np.inf, -np.inf], np.nan)
    k_abc = {a_: float(np.nanquantile(std_err[test["abc"] == a_], q)) for a_, q in SERVICE.items()}
    # the loss function of the standardized errors, per class: expected units
    # short per cycle, in standard errors, for a buffer of k standard errors
    k_grid = np.round(np.arange(0.0, 6.001, 0.01), 2)
    loss = {}
    for a_ in SERVICE:
        z_ = std_err[test["abc"] == a_].dropna().to_numpy()
        loss[a_] = np.array([np.maximum(z_ - k_, 0).mean() for k_ in k_grid])

    def k_fill(abc, tier, sigma, q):
        # the smallest buffer whose expected shortfall per cycle fits the fill-rate target
        if sigma <= 0:
            return 0.0
        allow = (1.0 - FILL_TARGET[tier]) * q / sigma
        ok = np.nonzero(loss.get(abc, loss["C"]) <= allow)[0]
        k_ = float(k_grid[ok[0]]) if len(ok) else float(k_grid[-1])
        return min(k_, K_MAX)
    seg_rows = []
    for sg in ["smooth", "erratic", "lumpy", "intermittent"]:
        g = test[test["segment"] == sg]
        wm, wb = _wape(g["target"], g["pred"]), _wape(g["target"], g["base"])
        seg_rows.append({"segment": sg, "wape_model": wm, "wape_baseline": wb, "lift": (wb - wm) / wb,
                         "bias": float((g["pred"] - g["target"]).sum() / g["target"].sum()),
                         "baseline_method": seg_method[sg]})
    overall = {"model": _wape(test["target"], test["pred"]), "baseline": _wape(test["target"], test["base"])}
    BACKTEST.mkdir(parents=True, exist_ok=True)
    test.rename(columns={"target": "actual"}).to_parquet(BACKTEST / "weekly_backtest.parquet", index=False)
    json.dump({"winner": winner, "candidates": results, "segments": seg_rows, "overall": overall,
               "bias_correction": bias, "safety_factor": k_abc, "normal_z": Z_BY_ABC},
              open(BACKTEST / "weekly_metrics.json", "w"), indent=2, default=float)
    print(f"  selected {winner}: test WAPE {overall['model']*100:.1f}% vs best simple method {overall['baseline']*100:.1f}%")
    print(f"  bias correction {({k_: round(v, 3) for k_, v in bias.items()})}  k {({k_: round(v, 2) for k_, v in k_abc.items()})}")

    # ── lead-time variability and the unscheduled share, per item, from 2024-25 ─
    p = po[(~po["rush"].astype(bool)) & po["received_date"].notna() & (po["order_date"] >= "2024-01-01")
           & (po["order_date"] < C.FORWARD_START.isoformat())].copy()
    p["canon"] = p["item_number"].map(lambda n: xw.get(n, n))
    p["lt"] = (pd.to_datetime(p["received_date"]) - pd.to_datetime(p["order_date"])).dt.days
    g = p.groupby("canon")["lt"]
    sd_item = g.std(); n_item = g.count()
    cls_of = attrs["item_class"].to_dict()
    sd_cls = p.assign(cls=p["canon"].map(cls_of)).groupby("cls")["lt"].std()
    def lt_sd(item):
        if n_item.get(item, 0) >= 4 and not np.isnan(sd_item.get(item, np.nan)):
            return float(sd_item[item])
        return float(sd_cls.get(cls_of.get(item), p["lt"].std()))
    u = tx[tx["type"].isin(["ISSUE", "BACKFLUSH"]) & (tx["txn_date"] >= "2025-01-01") & (tx["txn_date"] < C.FORWARD_START.isoformat())].copy()
    u["canon"] = u["item_number"].map(lambda n: xw.get(n, n)); u["q"] = u["qty"].abs()
    tot = u.groupby("canon")["q"].sum(); bfq = u[u["type"] == "BACKFLUSH"].groupby("canon")["q"].sum()
    unsched = (1 - bfq.reindex(tot.index).fillna(0) / tot).to_dict()
    # how far ahead MRP sees scheduled demand: release to completion of 2025 jobs
    prod = pd.read_csv(RAW / "erp" / "production_orders.csv")
    pj = prod[prod["completed_date"].notna() & (prod["due_date"] >= "2025-01-01") & (prod["due_date"] < C.FORWARD_START.isoformat())
              & (prod["delay_days"] == 0)]
    V = float((pd.to_datetime(pj["completed_date"]) - pd.to_datetime(pj["release_date"])).dt.days.median())
    # the model also reads the order book: a job is known from the day it is booked
    Bk = float((pd.to_datetime(pj["release_date"]) - pd.to_datetime(pj["booked_date"])).dt.days.median())
    VM = V + Bk
    print(f"  MRP visibility horizon (median release to completion, 2025): {V:.0f} days; with the order book {VM:.0f}")

    # criticality: on a corrected production bill, issued to service orders, or neither
    bom = pd.read_csv(RAW / "erp" / "bill_of_materials.csv")
    restored = pd.read_csv(RAW / "remediation" / "bom_change_log.csv")
    on_bill = set(bom["component_item"].map(lambda n: xw.get(n, n))) | set(restored["component_item"].map(lambda n: xw.get(n, n)))
    svc = tx[tx["job_id"].astype(str).str.startswith("SO-") & (tx["txn_date"] >= "2024-01-01")
             & (tx["txn_date"] < C.FORWARD_START.isoformat())]
    on_service = set(svc["item_number"].map(lambda n: xw.get(n, n)))
    tier_of = {i: ("line" if i in on_bill else "service" if i in on_service else "standard") for i in attrs.index}
    print("  criticality:", pd.Series(tier_of).value_counts().to_dict())

    def level_of(fc, share, L, ss, horizon=None):
        h_ = V if horizon is None else horizon
        return fc - (1.0 - share) * fc * min(L, h_) / max(L, 1.0) + ss

    # ── the forward window: retrain monthly, refresh every Monday ─────────────
    origins = [pd.Timestamp(C.FORWARD_START)] + [w for w in weeks if w > pd.Timestamp(C.FORWARD_START) and w <= pd.Timestamp(C.END_DATE)]
    model_sched, rule_sched = {}, {}
    prev, raw_pts, app_pts = {}, {}, {}
    model, trained_month = None, None
    weekly_idx = weekly.set_index(["canonical", "week"])["consumption"]
    for o in origins:
        t_o = week_idx[pd.Timestamp(_monday(o.date()))]
        if trained_month != (o.year, o.month):
            fit = frame[frame["t"] + frame["h"] <= t_o]
            tr._CAP = float(fit["target"].max()) * 4
            model = _make(winner, params); model.fit(fit[FEATS], np.log1p(fit["target"]))
            trained_month = (o.year, o.month)
            refresh_rule = True
        else:
            refresh_rule = False
        cur = frame[frame["t"] == t_o]
        pred = np.clip(np.expm1(model.predict(cur[FEATS])), 0, tr._CAP)
        for r_, pr in zip(cur.itertuples(index=False), pred):
            item = r_.item; a = attrs.loc[item]
            L = float(a["corrected_lead_days"]); h = int(r_.h)
            pc = float(pr) * bias.get(a["segment"], 1.0)
            d = pc / (7.0 * h); fc_lead = d * L
            sd_fc = float(resid.get(item, np.nan))
            if np.isnan(sd_fc):
                sd_fc = float(seg_resid.get(a["segment"], resid.median()))
            sd_fc *= np.sqrt(L / (7.0 * h)); sL = lt_sd(item)
            share = float(unsched.get(item, 1.0))
            # the scheduled demand MRP already sees carries no forecast error
            sd_fc *= np.sqrt(max(0.05, 1.0 - (1.0 - share) * min(L, VM) / max(L, 1.0)))
            sigma = float(np.sqrt(sd_fc ** 2 + (d * sL) ** 2))
            cost = float(a["standard_cost"]) if pd.notna(a["standard_cost"]) and a["standard_cost"] > 0 else 1.0
            eoq = np.sqrt(2.0 * d * 365.0 * ORDER_LINE_COST / (HOLDING_RATE * cost)) if d > 0 else 1.0
            q_lot = float(max(1.0, np.clip(eoq, d * LOT_DAYS[0], d * LOT_DAYS[1])))
            ss = k_fill(a["abc"], tier_of.get(item, "standard"), sigma, q_lot) * sigma
            share = float(unsched.get(item, 1.0))
            rop_raw = fc_lead + ss
            last = prev.get(item)
            if last is not None and last[5] > 0 and abs(rop_raw - last[5]) / last[5] <= HYSTERESIS:
                entry = [o.date().isoformat(), last[1], last[2], round(pc, 3), 7 * h, last[5], last[6], last[7]]
            else:
                entry = [o.date().isoformat(), round(level_of(fc_lead, share, L, ss, VM), 2), round(ss, 2), round(pc, 3), 7 * h,
                         round(rop_raw, 2), round(share, 3), round(q_lot, 1)]
            prev[item] = entry
            model_sched.setdefault(item, []).append(entry)
            raw_pts.setdefault(item, []).append(rop_raw); app_pts.setdefault(item, []).append(entry[5])
            if refresh_rule:
                s52 = float(r_.s52); d_r = s52 / 364.0; dlt = d_r * L
                ss_r = Z_BY_ABC.get(a["abc"], 1.28) * np.sqrt(max(1.0, dlt) + (d_r * sL) ** 2)
                rule_sched.setdefault(item, []).append([o.date().isoformat(), round(level_of(dlt, share, L, ss_r), 2), round(ss_r, 2),
                                                        round(dlt * 7 * h / max(L, 1), 3), 7 * h, round(dlt + ss_r, 2), round(share, 3)])
        print(f"  {o.date()}  {'retrained, ' if refresh_rule else ''}{len(cur):,} points refreshed")

    def churn(points, thresh=0.20):
        moves = n = 0
        for seq in points.values():
            for a_, b_ in zip(seq[:-1], seq[1:]):
                n += 1
                if a_ > 0 and abs(b_ - a_) / a_ > thresh:
                    moves += 1
        return moves / n if n else 0.0
    def changed(points):
        moves = n = 0
        for seq in points.values():
            for a_, b_ in zip(seq[:-1], seq[1:]):
                n += 1; moves += int(abs(b_ - a_) > 1e-9)
        return moves / n if n else 0.0

    meta = {"origins": [o.date().isoformat() for o in origins], "hysteresis": HYSTERESIS, "model": winner,
            "mrp_visibility_days": V, "order_book_days": Bk, "model_visibility_days": VM,
            "bias_correction": bias, "safety_factor": k_abc, "normal_z": Z_BY_ABC,
            "fill_rate_target": FILL_TARGET, "criticality": tier_of, "order_line_cost": ORDER_LINE_COST, "holding_rate": HOLDING_RATE,
            "lot_days": list(LOT_DAYS), "buyer_lot_days": C.ORDER_COVER_DAYS,
            "churn": {"raw_any_change": changed(raw_pts), "raw_over_20": churn(raw_pts),
                      "applied_any_change": changed(app_pts)},
            "entry_fields": ["date", "level", "safety_stock", "forecast_over_horizon", "horizon_days",
                             "reorder_point", "unscheduled_share", "order_qty"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rop_schedule.json").write_text(json.dumps({**meta, "items": model_sched}, indent=1))
    (OUT / "rule_schedule.json").write_text(json.dumps({**meta, "items": rule_sched}, indent=1))
    c_ = meta["churn"]
    print(f"\n  weekly churn: without the band {c_['raw_any_change']*100:.0f}% of item-weeks change "
          f"({c_['raw_over_20']*100:.0f}% by more than 20%); with it {c_['applied_any_change']*100:.0f}%")


if __name__ == "__main__":
    run()
