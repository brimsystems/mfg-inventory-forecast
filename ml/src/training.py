"""Train, tune and select the global demand model, then evaluate it honestly.

Three candidate regressors are tuned with Optuna on a validation block and
compared: a linear baseline (ridge on standardized features), a random forest,
and a gradient-boosted tree. The winner is chosen by validation WAPE, refit on
train plus validation, and scored on the held-out rolling origins the baselines
used. Lift is measured against each segment's relevant baseline method (chosen by
segment, not per row), so the comparison carries no hindsight.

Run:  python -m ml.src.training
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .resolution import REPO
from .features import build_feature_frame, FEATURE_COLS, _row_features

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

MARTS = REPO / "ml" / "data" / "marts"
BACKTEST = REPO / "ml" / "data" / "backtest"
SEEDS = REPO / "data_pipeline" / "seeds"
RAW = REPO / "data_source" / "raw"
MODELS = REPO / "ml" / "models"
N_TRIALS = 15
BL = ["naive", "snaive", "ma3", "ma6", "ses", "croston"]
_CAP = None          # cap on any forecast, so a linear model cannot extrapolate absurdly


def _wape(a, f):
    d = np.asarray(a).sum()
    return np.abs(np.asarray(a) - np.asarray(f)).sum() / d if d else np.nan


def _fit_predict(model, Xtr, ytr, Xte):
    model.fit(Xtr, np.log1p(ytr))
    return np.clip(np.expm1(model.predict(Xte)), 0, _CAP)


def _make(kind, p):
    if kind == "Linear":
        return make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"]))
    if kind == "RandomForest":
        return RandomForestRegressor(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                     min_samples_leaf=p["min_samples_leaf"], n_jobs=-1, random_state=42)
    return XGBRegressor(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                        learning_rate=p["learning_rate"], subsample=p["subsample"],
                        colsample_bytree=p["colsample_bytree"], min_child_weight=p["min_child_weight"],
                        reg_lambda=p["reg_lambda"], random_state=42, n_jobs=-1)


def _space(trial, kind):
    if kind == "Linear":
        return {"alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True)}
    if kind == "RandomForest":
        return {"n_estimators": trial.suggest_int("n_estimators", 150, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 16),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8)}
    return {"n_estimators": trial.suggest_int("n_estimators", 250, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 4.0)}


def _tune(kind, tr, va):
    Xtr, ytr = tr[FEATURE_COLS], tr["target"]
    Xva, yva = va[FEATURE_COLS], va["target"]

    def objective(trial):
        p = _space(trial, kind)
        pred = _fit_predict(_make(kind, p), Xtr, ytr, Xva)
        return _wape(yva, pred)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params, study.best_value


def _relevant_baseline(base: pd.DataFrame) -> dict:
    """Each segment's best baseline method, chosen by pooled WAPE (not per row)."""
    out = {}
    for seg, g in base.groupby("segment"):
        out[seg] = min(BL, key=lambda b: _wape(g["actual"], g[b]))
    return out


def run():
    monthly = pd.read_parquet(MARTS / "consumption_monthly.parquet")
    attrs = pd.read_parquet(MARTS / "item_attributes.parquet").set_index("canonical_item_number")
    base = pd.read_parquet(BACKTEST / "baseline_backtest.parquet")
    wide = monthly.pivot(index="canonical", columns="month", values="consumption").fillna(0).sort_index(axis=1)
    months = list(wide.columns)

    frame = build_feature_frame(wide, attrs, {}, months)
    # Impute feature NaNs (from missing-field items, D6) with train medians so the
    # linear and forest candidates are comparable with the NaN-native booster.
    frame[FEATURE_COLS] = frame[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    med = frame.loc[frame["split"] == "train", FEATURE_COLS].median()
    frame[FEATURE_COLS] = frame[FEATURE_COLS].fillna(med)
    global _CAP
    _CAP = float(frame.loc[frame["split"].isin(["train", "val"]), "target"].max()) * 4
    tr = frame[frame["split"] == "train"]
    va = frame[frame["split"] == "val"]
    te = frame[frame["split"] == "test"].copy()
    trva = frame[frame["split"].isin(["train", "val"])]
    print(f"\nSamples: train {len(tr):,}  val {len(va):,}  test {len(te):,}")

    # tune and compare candidates
    print("\n=== Candidate comparison (tuned with Optuna) ===")
    print("  candidate       val WAPE   test WAPE")
    results = {}
    for kind in ["Linear", "RandomForest", "XGBoost"]:
        best_p, val_wape = _tune(kind, tr, va)
        test_pred = _fit_predict(_make(kind, best_p), trva[FEATURE_COLS], trva["target"], te[FEATURE_COLS])
        test_wape = _wape(te["target"], test_pred)
        results[kind] = {"params": best_p, "val_wape": val_wape, "test_wape": test_wape}
        print(f"  {kind:<14} {val_wape*100:6.1f}%   {test_wape*100:6.1f}%")

    winner = min(results, key=lambda k: results[k]["val_wape"])
    print(f"\n  Selected: {winner} (lowest validation WAPE)")

    # refit winner on train+val, score test
    model = _make(winner, results[winner]["params"])
    te["pred"] = _fit_predict(model, trva[FEATURE_COLS], trva["target"], te[FEATURE_COLS])
    te = te.rename(columns={"target": "actual"})

    # honest baseline: each segment's relevant method
    seg_method = _relevant_baseline(base)
    te = te.merge(base[["item", "origin"] + BL], on=["item", "origin"], how="left")
    te["base"] = te.apply(lambda r: r[seg_method.get(r["segment"], "snaive")], axis=1)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / "demand_model.joblib")
    te.to_parquet(BACKTEST / "model_backtest.parquet", index=False)

    print("\n=== Selected model vs relevant baseline (test) ===")
    print("  segment       model | baseline | lift | MASE | bias   (baseline method)")
    seg_rows = []
    for seg in ["smooth", "erratic", "lumpy", "intermittent"]:
        g = te[te["segment"] == seg]
        if not len(g):
            continue
        wm, wb = _wape(g["actual"], g["pred"]), _wape(g["actual"], g["base"])
        nmae = np.abs(g["actual"] - g["naive"]).mean()
        mase = np.abs(g["actual"] - g["pred"]).mean() / nmae if nmae else np.nan
        bias = (g["pred"] - g["actual"]).sum() / g["actual"].sum()
        lift = (wb - wm) / wb if wb else np.nan
        seg_rows.append({"segment": seg, "wape_model": wm, "wape_baseline": wb, "lift": lift,
                         "mase": mase, "bias": bias, "baseline_method": seg_method.get(seg)})
        print(f"  {seg:<13} {wm*100:5.1f}% | {wb*100:5.1f}% | {lift*100:+5.1f}% | {mase:.2f} | {bias*100:+4.0f}%   ({seg_method.get(seg)})")
    print("\n  ABC:          model | baseline | lift")
    for a in ["A", "B", "C"]:
        g = te[te["abc"] == a]
        wm, wb = _wape(g["actual"], g["pred"]), _wape(g["actual"], g["base"])
        print(f"    {a}  {wm*100:5.1f}% | {wb*100:5.1f}% | {(wb-wm)/wb*100:+5.1f}%")
    wm, wb = _wape(te["actual"], te["pred"]), _wape(te["actual"], te["base"])
    print(f"\n  Overall: model {wm*100:.1f}%  baseline {wb*100:.1f}%  lift {(wb-wm)/wb*100:+.1f}%")

    comp = _dirty_before_after(model, attrs)
    comp.to_parquet(BACKTEST / "clean_before_after.parquet", index=False)
    print(f"\n  Duplicate cleaning (same model, {len(comp)} items): "
          f"WAPE {comp['before'].median()*100:.0f}% -> {comp['after'].median()*100:.0f}%  "
          f"(median improvement {comp['rel_improve'].median()*100:.0f}%, "
          f"{(comp['rel_improve']<=0.02).mean()*100:.0f}% flat/worse)")

    json.dump({"winner": winner, "candidates": results, "segments": seg_rows,
               "overall": {"model": float(wm), "baseline": float(wb)}},
              open(BACKTEST / "model_metrics.json", "w"), indent=2, default=float)
    print()


def _dirty_before_after(model, attrs):
    tx = pd.read_csv(RAW / "erp" / "inventory_transactions.csv")
    crosswalk = pd.read_csv(SEEDS / "item_crosswalk.csv")
    groups = crosswalk.groupby("canonical_item_number")["item_number"].apply(list)
    clusters = groups[groups.map(len) > 1]
    iss = tx[tx["type"] == "issue"].copy()
    iss["date"] = pd.to_datetime(iss["transaction_date"])
    iss["month"] = iss["date"].values.astype("datetime64[M]")
    months = pd.period_range(iss["month"].min(), iss["month"].max(), freq="M").to_timestamp()
    month_nums = [m.month for m in months]

    def series_for(numbers):
        return (iss[iss["item_number"].isin(numbers)].groupby("month")["quantity"].sum()
                .reindex(months, fill_value=0).to_numpy(float))

    def predict_series(s, a):
        T = len(s)
        h = max(1, int(round(a["corrected_lead_days"] / 30.0)))
        rows, acts = [], []
        for t in range(T - h - 11, T - h + 1):
            if t < 12:
                continue
            f = _row_features(s, t, month_nums[t])
            f.update({"std_cost": a["standard_cost"], "corrected_lead": a["corrected_lead_days"],
                      "annual": a["annual_consumption"], "price": a["standard_cost"],
                      "seg_code": 0, "abc_code": 0, "cls_code": 0, "horizon": h})
            rows.append(f); acts.append(s[t:t + h].sum())
        X = pd.DataFrame(rows)
        for c in FEATURE_COLS:
            if c not in X:
                X[c] = 0
        X = X[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
        return np.array(acts), np.clip(np.expm1(model.predict(X)), 0, _CAP)

    recs = []
    for canon, members in clusters.items():
        if canon not in attrs.index:
            continue
        a = attrs.loc[canon]
        act, after = predict_series(series_for(members), a)
        before = np.zeros_like(after)
        for m in members:
            _, half = predict_series(series_for([m]), a)
            before = before + half
        recs.append({"item": canon, "before": _wape(act, before), "after": _wape(act, after)})
    comp = pd.DataFrame(recs).dropna()
    comp["rel_improve"] = (comp["before"] - comp["after"]) / comp["before"]
    return comp


if __name__ == "__main__":
    run()
