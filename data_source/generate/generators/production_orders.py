"""Production orders: product build events split into individual jobs.

Each monthly product build is split into one or more production orders of a few
units each, so the order count lands in the realistic range and each order can
backflush its BOM on completion. Configuration codes stand in for the configured
and engineered-to-order variants (about 40% of orders carry one).
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .. import config as C

CONFIG_CODES = ["STD", "CFG-A", "CFG-B", "CFG-C", "ETO"]


def build_production_orders(builds, products, rng):
    fam_by_prod = products.set_index("product_number")["family"].to_dict()
    rows = []
    seq = 0
    for r in builds.itertuples(index=False):
        remaining = int(r.completed)
        if remaining <= 0:
            continue
        month0 = r.month
        while remaining > 0:
            q = int(min(remaining, rng.integers(1, 4)))
            remaining -= q
            seq += 1
            release = month0 + timedelta(days=int(rng.integers(0, 20)))
            if release > C.END_DATE:
                release = C.END_DATE
            lead_days = int(rng.integers(14, 45))
            due = release + timedelta(days=lead_days)
            # Most jobs complete near due; report completion at week-end (defect T4).
            completed_date = due + timedelta(days=int(rng.integers(-4, 6)))
            done = completed_date <= C.END_DATE
            # snap completion to the Friday of its week (week-end reporting)
            if done:
                completed_date = completed_date + timedelta(days=(4 - completed_date.weekday()) % 7)
                if completed_date > C.END_DATE:
                    completed_date = C.END_DATE
            cfg = "STD" if rng.random() < 0.6 else rng.choice(CONFIG_CODES[1:])
            rows.append({
                "order_id":          f"JOB-{seq:06d}",
                "product_number":    r.product_number,
                "configuration_code":cfg,
                "qty_ordered":       q,
                "qty_completed":     q if done else 0,
                "customer_id":       f"CUST-{int(rng.integers(1, 140)):03d}",
                "release_date":      release.isoformat(),
                "due_date":          due.isoformat(),
                "completed_date":    completed_date.isoformat() if done else None,
                "status":            "COMPLETED" if done else "OPEN",
            })
    df = pd.DataFrame(rows)

    # T5: a share of finished jobs are never closed in the system. The work is
    # reported complete (quantity and date are recorded) but the status stays
    # OPEN. Drawn from an independent stream so the rest of the build is unchanged.
    local = np.random.default_rng(C.RANDOM_SEED + 5)
    finished = df.index[df["qty_completed"] > 0]
    left_open = local.choice(finished, size=int(len(finished) * C.T5_OPEN_JOB_SHARE), replace=False)
    df.loc[left_open, "status"] = "OPEN"
    return df
