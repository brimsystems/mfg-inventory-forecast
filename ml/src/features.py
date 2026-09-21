"""Feature engineering for the global demand model.

One row per item-origin. Every feature is computed from consumption strictly
before the origin month, and the target is total demand over the item's lead-time
horizon starting at the origin, so no feature can encode the forecast period.
Item attributes, price and calendar structure are added so a single global model
can learn patterns (autocorrelation, quarterly release, seasonality) that the
univariate baselines cannot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 6, 12]
ROLL = [3, 6, 12]
VAL_ORIGINS = 6          # validation origins between train and the test window


def _row_features(hist: np.ndarray, t: int, month_num: int) -> dict:
    past = hist[:t]
    f = {}
    for L in LAGS:
        f[f"lag_{L}"] = past[-L] if len(past) >= L else 0.0
    for w in ROLL:
        seg = past[-w:] if len(past) >= 1 else np.array([0.0])
        f[f"roll_mean_{w}"] = seg.mean()
        f[f"roll_std_{w}"] = seg.std()
        f[f"nz_{w}"] = float((seg > 0).sum())
    nz_idx = np.where(past > 0)[0]
    f["months_since_nz"] = float(len(past) - 1 - nz_idx[-1]) if len(nz_idx) else float(len(past))
    f["mean_all"] = past.mean() if len(past) else 0.0
    f["cv_all"] = (past.std() / past.mean()) if len(past) and past.mean() > 0 else 0.0
    f["trend_3_6"] = f["roll_mean_3"] - f["roll_mean_6"]        # recent direction
    f["ratio_1_6"] = f["lag_1"] / (f["roll_mean_6"] + 1.0)      # spike-vs-normal
    f["month"] = month_num
    f["quarter"] = (month_num - 1) // 3 + 1
    f["is_quarter_start"] = float(month_num in (1, 4, 7, 10))
    return f


def build_feature_frame(monthly_wide: pd.DataFrame, attrs: pd.DataFrame,
                        price_by_item: dict, months: list) -> pd.DataFrame:
    seg_code = {s: i for i, s in enumerate(["smooth", "erratic", "lumpy", "intermittent"])}
    abc_code = {"A": 0, "B": 1, "C": 2}
    cls_code = {c: i for i, c in enumerate(sorted(attrs["item_class"].unique()))}
    month_nums = [m.month for m in months]

    rows = []
    for item, series in monthly_wide.iterrows():
        if item not in attrs.index:
            continue
        s = series.to_numpy(float)
        T = len(s)
        a = attrs.loc[item]
        h = max(1, int(round(a["corrected_lead_days"] / 30.0)))
        test_first = T - h - 11                      # first of the last 12 origins
        val_first = test_first - VAL_ORIGINS         # a validation block before the test window
        for t in range(12, T - h + 1):
            f = _row_features(s, t, month_nums[t])
            f.update({
                "item": item, "origin": t, "horizon": h,
                "target": float(s[t:t + h].sum()),
                "std_cost": a["standard_cost"], "corrected_lead": a["corrected_lead_days"],
                "annual": a["annual_consumption"], "price": price_by_item.get(item, a["standard_cost"]),
                "seg_code": seg_code.get(a["segment"], 0), "abc_code": abc_code.get(a["abc"], 2),
                "cls_code": cls_code.get(a["item_class"], 0),
                "segment": a["segment"], "abc": a["abc"],
            })
            # each split keeps its whole horizon before the next window starts
            if t >= test_first:
                f["split"] = "test"
            elif t >= val_first and t + h <= test_first:
                f["split"] = "val"
            elif t + h <= val_first:
                f["split"] = "train"
            else:
                f["split"] = "gap"
            rows.append(f)
    return pd.DataFrame(rows)


FEATURE_COLS = ([f"lag_{L}" for L in LAGS]
                + [f"roll_mean_{w}" for w in ROLL] + [f"roll_std_{w}" for w in ROLL]
                + [f"nz_{w}" for w in ROLL]
                + ["months_since_nz", "mean_all", "cv_all", "trend_3_6", "ratio_1_6",
                   "month", "quarter", "is_quarter_start",
                   "std_cost", "corrected_lead", "annual", "price", "seg_code", "abc_code", "cls_code",
                   "horizon"])
