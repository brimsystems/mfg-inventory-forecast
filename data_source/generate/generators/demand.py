"""Item plan and demand engine.

The item plan assigns every catalog item its latent attributes: class, material
family, unit cost, base demand level and demand segment. The demand engine turns
that plan into a monthly true-demand series per item, built from learnable
structure (autocorrelation, calendar spikes, seasonality, shared family factors)
multiplied by irreducible noise. The noise is what sets the achievable error
floor, so a forecast cannot reach implausible accuracy.

Everything here works on the *canonical* item: one row per real physical item.
Duplicate records, unit-of-measure faults and the other planted defects are
applied downstream, so this module holds the ground truth the case is measured
against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C


# ── Item plan ───────────────────────────────────────────────────────────────
def _assign_by_share(n: int, shares: dict, rng) -> list:
    labels = list(shares)
    probs = np.array([shares[k] for k in labels], dtype=float)
    probs = probs / probs.sum()
    return list(rng.choice(labels, size=n, p=probs))


def build_item_plan(rng) -> pd.DataFrame:
    """Return one row per canonical item with all latent demand attributes."""
    n = C.N_ITEMS
    item_class = _assign_by_share(n, C.ITEM_CLASS_SHARES, rng)
    segment = _assign_by_share(n, C.SEGMENT_MIX, rng)

    # Material family for the metal classes; others carry no family.
    fam_names = list(C.MATERIAL_FAMILIES)
    material, family = [], []
    for cls in item_class:
        if cls in C.METAL_CLASSES:
            fam = rng.choice(fam_names, p=_family_weights())
            grade = rng.choice(C.MATERIAL_FAMILIES[fam])
            family.append(fam)
            material.append(grade)
        else:
            family.append(None)
            material.append(None)

    unit_cost = np.exp(rng.normal(C.STANDARD_COST_LOG_MEAN, C.STANDARD_COST_LOG_STD, n)).round(2)
    unit_cost = np.clip(unit_cost, 0.05, None)
    base_level = np.exp(rng.normal(C.BASE_DEMAND_LOG_MEAN, C.BASE_DEMAND_LOG_STD, n))

    seasonal = rng.random(n) < C.SEASONAL_ITEM_SHARE
    seasonal_amp = np.where(seasonal, rng.uniform(*C.SEASONAL_AMP_RANGE, n), 0.0)
    seasonal_phase = rng.uniform(0, 12, n)
    lumpy_offset = rng.integers(0, 3, n)          # which month of the quarter spikes
    ar_phi = np.array([C.SEGMENT_PARAMS[s]["ar_phi"] for s in segment])
    noise_sigma = np.array([C.SEGMENT_PARAMS[s]["noise_sigma"] for s in segment])
    active_prob = np.array([C.SEGMENT_PARAMS[s]["active_prob"] for s in segment])
    base_lead = rng.integers(C.BASE_LEAD_TIME_RANGE[0], C.BASE_LEAD_TIME_RANGE[1] + 1, n)

    plan = pd.DataFrame({
        "item_id":       np.arange(n),
        "item_class":    item_class,
        "material":      material,
        "family":        family,
        "segment":       segment,
        "unit_cost":     unit_cost,
        "base_level":    base_level,
        "seasonal_amp":  seasonal_amp,
        "seasonal_phase":seasonal_phase,
        "lumpy_offset":  lumpy_offset,
        "ar_phi":        ar_phi,
        "noise_sigma":   noise_sigma,
        "active_prob":   active_prob,
        "base_lead_days":base_lead,
    })
    return plan


def _family_weights():
    w = np.array([len(v) for v in C.MATERIAL_FAMILIES.values()], dtype=float)
    return w / w.sum()


# ── Demand engine ───────────────────────────────────────────────────────────
def _ar1(n_steps: int, phi: float, rng) -> np.ndarray:
    """Standardized AR(1) series with unit marginal variance."""
    z = np.zeros(n_steps)
    innov = np.sqrt(max(1e-6, 1 - phi * phi))
    z[0] = rng.normal()
    for t in range(1, n_steps):
        z[t] = phi * z[t - 1] + innov * rng.normal()
    return z


# Per-segment amplitude of the persistent (autocorrelated, learnable) component.
_AR_SIGMA = {"smooth": 0.16, "erratic": 0.72, "lumpy": 0.10, "intermittent": 0.10}


def build_monthly_demand(plan: pd.DataFrame, rng) -> pd.DataFrame:
    """Return long-form true monthly demand: item_id, month, demand_units."""
    months = C.month_starts()
    tmon = np.array([m.month for m in months])
    midx = np.arange(len(months))

    # Shared material-family factors: one standardized AR(1) per family, so
    # substitutable grades move together.
    fam_factor = {f: _ar1(len(months), 0.6, rng) for f in C.MATERIAL_FAMILIES}

    records = []
    for row in plan.itertuples(index=False):
        seg = row.segment
        z = _ar1(len(months), row.ar_phi, rng)
        persistent = np.exp(_AR_SIGMA[seg] * z)

        if row.seasonal_amp > 0:
            seasonal = np.exp(row.seasonal_amp * np.sin(2 * np.pi * (midx / 12.0 - row.seasonal_phase / 12.0)))
        else:
            seasonal = np.ones(len(months))

        if row.family is not None:
            fam = np.exp(C.FAMILY_FACTOR_WEIGHT * 0.5 * fam_factor[row.family])
        else:
            fam = np.ones(len(months))

        structure = row.base_level * persistent * seasonal * fam
        noise = np.exp(row.noise_sigma * rng.normal(size=len(months)))

        if seg == "lumpy":
            spike = ((midx - row.lumpy_offset) % 3 == 0)
            level = structure * np.where(spike, C.LUMPY_QUARTER_SPIKE_MULT, C.LUMPY_BASELINE_FRACTION)
            # Off-quarter months are usually empty; spikes always fire.
            occur = spike | (rng.random(len(months)) < 0.35)
            demand = level * noise * occur
        elif seg == "intermittent":
            occur = rng.random(len(months)) < row.active_prob
            demand = structure * noise * occur
        elif seg == "erratic":
            occur = rng.random(len(months)) < row.active_prob
            demand = structure * noise * occur
        else:  # smooth
            demand = structure * noise

        demand = np.rint(np.clip(demand, 0, None)).astype(int)
        for mo, q in zip(months, demand):
            records.append((row.item_id, mo, int(q)))

    return pd.DataFrame.from_records(records, columns=["item_id", "month", "demand_units"])
