"""Products, subassemblies and the multi-level bill of materials, plus the
product-build engine that drives backflush consumption.

Consumption of purchased components is recorded by backflush: when a product or
subassembly is reported complete, the ERP generates BACKFLUSH transactions from
that level's BOM, quantity equal to qty_per times the completed quantity. This
module builds the structure and the product-build series, then explodes builds
into per-component monthly backflush and per-order backflush lines. Because
backflush is computed from completed quantity times qty_per, the consistency the
audit checks holds by construction, before defects are applied.

Defect M3 (BOM omissions) removes component edges from the RECORDED bom so their
backflush never happens; the true bom is retained so the unrecorded consumption
can be added back during remediation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from .demand import _ar1, _assign_by_share

_AR_SIGMA = {"smooth": 0.16, "erratic": 0.72, "lumpy": 0.10, "intermittent": 0.10}


# ── Product and subassembly catalog ─────────────────────────────────────────
def build_catalog(rng):
    """Return products and subassemblies as DataFrames with build attributes."""
    products = []
    for i in range(C.N_PRODUCTS):
        fam = C.PRODUCT_FAMILIES[i % len(C.PRODUCT_FAMILIES)]
        seg = _assign_by_share(1, C.SEGMENT_MIX, rng)[0]
        base = float(np.exp(rng.normal(0.55, 0.7)))      # units built per month, median ~1.7
        products.append({
            "product_number": f"PRD-{fam[:2].upper()}-{i:03d}",
            "family": fam, "segment": seg, "base_build": base,
            "seasonal_amp": float(rng.uniform(*C.SEASONAL_AMP_RANGE)) if rng.random() < C.SEASONAL_ITEM_SHARE else 0.0,
            "seasonal_phase": float(rng.uniform(0, 12)),
            "lumpy_offset": int(rng.integers(0, 3)),
            "ar_phi": C.SEGMENT_PARAMS[seg]["ar_phi"],
        })
    subs = [{"subassembly_number": f"SUB-{i:04d}"} for i in range(C.N_SUBASSEMBLIES)]
    return pd.DataFrame(products), pd.DataFrame(subs)


# ── Bill of materials ───────────────────────────────────────────────────────
def build_bom(products, subs, plan, rng):
    """Assign live components to subassemblies and products, and subassemblies to
    products. Returns the true BOM (long: parent, parent_type, component_item,
    qty_per, level) keyed on canonical item_id for components."""
    # BOM components are drawn from the high-runner (smooth / erratic) items, so
    # lumpy and intermittent items stay off the BOM and keep their segment.
    eligible = plan[plan["segment"].isin(C.BOM_COMPONENT_SEGMENTS)]
    comp_ids = (eligible["item_id"].to_numpy() if len(eligible) else plan["item_id"].to_numpy())
    comp_class = plan.set_index("item_id")["item_class"].to_dict()
    sub_numbers = subs["subassembly_number"].tolist()

    rows = []
    # Subassembly content: purchased components.
    sub_components = {}
    for s in sub_numbers:
        k = int(rng.integers(*C.COMPONENTS_PER_SUBASSY_RANGE))
        comps = rng.choice(comp_ids, size=min(k, len(comp_ids)), replace=False)
        sub_components[s] = list(comps)
        for c in comps:
            rows.append({"parent": s, "parent_type": "subassembly",
                         "component_item": int(c), "component_type": "purchased",
                         "qty_per": int(rng.integers(*C.BOM_QTY_PER_RANGE)), "level": 2})

    # Product content: a few subassemblies plus direct purchased components.
    prod_subs = {}
    for p in products["product_number"]:
        ns = int(rng.integers(*C.SUBASSY_PER_PRODUCT_RANGE))
        chosen_subs = list(rng.choice(sub_numbers, size=ns, replace=False))
        prod_subs[p] = chosen_subs
        for s in chosen_subs:
            rows.append({"parent": p, "parent_type": "product",
                         "component_item": s, "component_type": "subassembly",
                         "qty_per": int(rng.integers(1, 3)), "level": 1})
        nc = int(rng.integers(*C.COMPONENTS_PER_PRODUCT_RANGE))
        comps = rng.choice(comp_ids, size=min(nc, len(comp_ids)), replace=False)
        for c in comps:
            rows.append({"parent": p, "parent_type": "product",
                         "component_item": int(c), "component_type": "purchased",
                         "qty_per": int(rng.integers(*C.BOM_QTY_PER_RANGE)), "level": 1})

    bom_true = pd.DataFrame(rows)
    return bom_true, prod_subs, sub_components


def apply_m3_omissions(bom_true, plan, rng):
    """Remove component edges (hardware, fasteners, fittings, consumables,
    finishing) from the RECORDED bom so backflush never consumes them. The
    omissions are the residue a shop with an annual physical would still carry:
    one to three rows left off the bills of a few recent product releases and
    option changes, and at most one on a shared subassembly. Returns the
    recorded bom, the omission truth and the omitted component items."""
    comp_class = plan.set_index("item_id")["item_class"].to_dict()
    purchased = bom_true[bom_true["component_type"] == "purchased"].copy()
    purchased["cls"] = purchased["component_item"].map(comp_class)
    eligible = purchased[purchased["cls"].isin(C.M3_OMISSION_CLASSES)]

    products = bom_true.loc[bom_true["parent_type"] == "product", "parent"].unique()
    n_prod = max(1, int(round(len(products) * C.M3_PRODUCT_SHARE)))
    omit_idx = []
    for prod in rng.choice(products, size=n_prod, replace=False):
        edges = eligible[(eligible["parent"] == prod) & (eligible["parent_type"] == "product")]
        k = min(len(edges), int(rng.integers(1, C.M3_ROWS_PER_PRODUCT + 1)))
        if k:
            omit_idx.extend(int(i) for i in rng.choice(edges.index.to_numpy(), size=k, replace=False))
    subs = eligible[eligible["parent_type"] == "subassembly"]
    if len(subs) and rng.random() < C.M3_SUBASSEMBLY_PROB:
        omit_idx.append(int(rng.choice(subs.index.to_numpy())))

    bom_recorded = bom_true.drop(index=omit_idx).reset_index(drop=True)
    omissions = bom_true.loc[omit_idx, ["parent", "parent_type", "component_item", "qty_per"]]
    return bom_recorded, omissions.reset_index(drop=True), sorted(set(int(i) for i in omissions["component_item"]))


def effective_component_map(bom_recorded):
    """Collapse the subassembly chain: return {product_number: {component_item:
    effective_qty_per}}, the purchased-component quantity consumed per unit of the
    product, summing direct edges and subassembly edges times their qty_per."""
    # Subassembly -> {component_item: qty_per}
    sub_map = {}
    sub_edges = bom_recorded[(bom_recorded["parent_type"] == "subassembly") &
                             (bom_recorded["component_type"] == "purchased")]
    for r in sub_edges.itertuples(index=False):
        sub_map.setdefault(r.parent, {})
        sub_map[r.parent][int(r.component_item)] = sub_map[r.parent].get(int(r.component_item), 0) + r.qty_per

    prod_map = {}
    prod_edges = bom_recorded[bom_recorded["parent_type"] == "product"]
    for r in prod_edges.itertuples(index=False):
        prod_map.setdefault(r.parent, {})
        if r.component_type == "purchased":
            c = int(r.component_item)
            prod_map[r.parent][c] = prod_map[r.parent].get(c, 0) + r.qty_per
        else:  # subassembly edge: multiply the subassembly's components by qty_per
            for c, q in sub_map.get(r.component_item, {}).items():
                prod_map[r.parent][c] = prod_map[r.parent].get(c, 0) + q * r.qty_per
    return prod_map


# ── Product build engine ────────────────────────────────────────────────────
def build_product_builds(products, rng):
    """Return monthly completed quantity per product (long: product_number,
    month, completed)."""
    months = C.month_starts()
    midx = np.arange(len(months))
    records = []
    for row in products.itertuples(index=False):
        z = _ar1(len(months), row.ar_phi, rng)
        persistent = np.exp(_AR_SIGMA[row.segment] * z)
        if row.seasonal_amp > 0:
            seasonal = np.exp(row.seasonal_amp * np.sin(2 * np.pi * (midx / 12.0 - row.seasonal_phase / 12.0)))
        else:
            seasonal = np.ones(len(months))
        structure = row.base_build * persistent * seasonal
        noise = np.exp(0.30 * rng.normal(size=len(months)))
        if row.segment == "lumpy":
            spike = ((midx - row.lumpy_offset) % 3 == 0)
            level = structure * np.where(spike, C.LUMPY_QUARTER_SPIKE_MULT, C.LUMPY_BASELINE_FRACTION)
            occur = spike | (rng.random(len(months)) < 0.35)
            builds = level * noise * occur
        elif row.segment == "intermittent":
            occur = rng.random(len(months)) < 0.6
            builds = structure * noise * occur
        else:
            builds = structure * noise
        builds = np.rint(np.clip(builds, 0, None)).astype(int)
        for mo, q in zip(months, builds):
            records.append((row.product_number, mo, int(q)))
    return pd.DataFrame.from_records(records, columns=["product_number", "month", "completed"])


def explode_builds(builds, bom_recorded):
    """Explode monthly product builds through the RECORDED bom (via the effective
    component map) to per-component monthly backflush: DataFrame item_id, month,
    qty. This is the recorded-consumption truth for the backflush channel."""
    prod_map = effective_component_map(bom_recorded)
    rows = []
    for r in builds.itertuples(index=False):
        if r.completed <= 0:
            continue
        for c, qper in prod_map.get(r.product_number, {}).items():
            rows.append((c, r.month, qper * r.completed))
    comp_bf = (pd.DataFrame(rows, columns=["item_id", "month", "qty"])
               if rows else pd.DataFrame(columns=["item_id", "month", "qty"]))
    return comp_bf.groupby(["item_id", "month"], as_index=False)["qty"].sum()
