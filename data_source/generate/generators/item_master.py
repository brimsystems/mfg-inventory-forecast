"""Item master: canonical items rendered into recorded ERP rows, with the item
side of the planted defects.

  D1  duplicate records  - one physical item under two or three item numbers with
                           different description conventions and creation dates
  D3  unit-of-measure    - purchased by the box, issued by the each, no factor
  D6  missing fields     - blank reorder point, standard cost, or supplier

The function returns the recorded item_master, the duplicate crosswalk truth
(canonical item -> recorded numbers with demand-split weights), and per-record
metadata the transaction, purchase-order and cycle-count generators need.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .. import config as C

FRACTIONS = [("1/8", .125), ("3/16", .1875), ("1/4", .25), ("5/16", .3125),
             ("3/8", .375), ("1/2", .5), ("5/8", .625), ("3/4", .75), ("1", 1.0)]
THREADS = [20, 28, 18, 16, 13, 11, 24]
HEADS = ["SHCS", "HHCS", "FHCS", "BHCS", "HSHCS"]
FASTENER_MAT = ["SS", "HSS", "GR8", "ZP"]
SHAPES = [("Round", "RD"), ("Square", "SQ"), ("Hex", "HEX"), ("Flat", "FL")]
TOOL_TYPES = ["End Mill", "Drill", "Insert", "Tap", "Reamer"]
CONSUMABLES = ["Coolant Concentrate", "Way Oil", "Shop Towels", "Nitrile Gloves",
               "Layout Dye", "Deburring Wheel", "Cutting Fluid", "Thread Locker",
               "Spindle Oil", "Aluminum Cutting Wax", "Brass Brush", "Saw Blade"]
CONSUMABLE_PACK = ["1GAL", "5GAL", "55GAL", "CASE", "BOX", "DRUM", "PAIL"]
CONSUMABLE_BRAND = ["AC", "BX", "MK", "TR", "NF", "GP"]
SERVICES = ["Plating - Zinc", "Heat Treat - Stress Relieve", "Anodize Type II",
            "Passivation", "Black Oxide", "Heat Treat - Harden & Temper",
            "Plating - Nickel", "Powder Coat", "Chromate Conversion"]
SERVICE_SPEC = ["Clear", "Yellow", "Class 1", "Class 2", "Type II", "Grade A", "RoHS"]
FINISH_TOKENS = {"Fasteners": ["ZP", "PL", "BLK", "SS", "GALV"],
                 "Hardware": ["ZP", "PL", "BLK", "SS", "GALV"],
                 "Bar Stock": ["12FT", "20FT", "RAND"], "Sheet": ["48X96", "48X120"],
                 "Tooling": ["TIALN", "TIN", "UNCOATED"],
                 "Consumables": CONSUMABLE_PACK, "Outside Service": SERVICE_SPEC}

CREATED_MIN = date(2018, 1, 1)


def _canonical_description(row, rng) -> str:
    cls = row.item_class
    if cls in ("Fasteners", "Hardware"):
        frac, _ = FRACTIONS[rng.integers(0, len(FRACTIONS))]
        thr = rng.choice(THREADS)
        ln, _ = FRACTIONS[rng.integers(2, len(FRACTIONS))]
        head = rng.choice(HEADS)
        mat = rng.choice(FASTENER_MAT)
        return f"{frac}-{thr} x {ln} {head} {mat}"
    if cls == "Bar Stock":
        shp, _ = SHAPES[rng.integers(0, len(SHAPES))]
        frac, _ = FRACTIONS[rng.integers(0, len(FRACTIONS))]
        return f"{row.material} {shp} {frac} Bar"
    if cls == "Sheet":
        ga = rng.choice([10, 11, 12, 14, 16, 18])
        return f"{row.material} Sheet {ga}GA 48x96"
    if cls == "Tooling":
        t = rng.choice(TOOL_TYPES)
        frac, _ = FRACTIONS[rng.integers(0, len(FRACTIONS))]
        fl = rng.choice([2, 3, 4])
        return f"{frac} {t} {fl}FL Carbide"
    if cls == "Consumables":
        return f"{rng.choice(CONSUMABLES)} {rng.choice(CONSUMABLE_PACK)} {rng.choice(CONSUMABLE_BRAND)}"
    return f"{rng.choice(SERVICES)} {rng.choice(SERVICE_SPEC)}"


def _make_unique(desc: str, cls: str, seen: set, rng) -> str:
    """Guarantee near-unique descriptions across distinct items; only planted
    duplicates should collide after normalization."""
    if desc not in seen:
        seen.add(desc)
        return desc
    for tok in rng.permutation(FINISH_TOKENS.get(cls, ["A", "B", "C"])):
        cand = f"{desc} {tok}"
        if cand not in seen:
            seen.add(cand)
            return cand
    i = 2
    while f"{desc} {i}" in seen:
        i += 1
    seen.add(f"{desc} {i}")
    return f"{desc} {i}"


def _variant_description(desc: str, cls: str, rng) -> str:
    """A different data-entry convention for the same physical item: decimal for
    fraction, reordered tokens, abbreviations. Normalizes back to the same item.
    """
    out = desc
    for frac, dec in FRACTIONS:
        if frac in out:
            out = out.replace(frac, f"{dec:.4g}".lstrip("0") if dec < 1 else f"{dec:g}")
    for full, ab in SHAPES:
        out = out.replace(full, ab)
    out = out.replace(" x ", "x").replace("  ", " ")
    if cls in ("Fasteners", "Hardware") and rng.random() < 0.5:
        parts = out.split()
        if len(parts) >= 2:                    # move head/material token to front
            out = parts[-2] + " " + " ".join(parts[:-2] + parts[-1:])
    return out.strip()


def _item_number(seq: int, cls: str, variant: int = 0) -> str:
    prefix = {"Bar Stock": "BAR", "Sheet": "SHT", "Fasteners": "FAS", "Hardware": "HDW",
              "Tooling": "TL", "Consumables": "CON", "Outside Service": "SVC"}[cls]
    tail = "" if variant == 0 else f"-{'AB'[variant-1]}" if variant <= 2 else f"-{variant}"
    return f"{prefix}-{seq:05d}{tail}"


def build_item_master(plan: pd.DataFrame, suppliers: pd.DataFrame,
                      annual_by_item: dict, drift_supplier_id: str, rng):
    sup_by_type = {t: suppliers.loc[suppliers["supplier_type"] == t, "supplier_id"].unique().tolist()
                   for t in suppliers["supplier_type"].unique()}
    class_supplier_type = {"Bar Stock": "Material", "Sheet": "Material",
                           "Fasteners": "Fasteners", "Hardware": "Fasteners",
                           "Tooling": "Tooling", "Consumables": "Distributor",
                           "Outside Service": "Outside Service"}

    # D1 clusters: smooth, high-consumption items in the target classes.
    pool = plan[(plan["item_class"].isin(C.D1_CLUSTER_CLASSES)) & (plan["segment"] == "smooth")].copy()
    pool["annual"] = pool["item_id"].map(annual_by_item)
    dup_ids = set(pool.sort_values("annual", ascending=False)
                      .head(C.D1_N_CLUSTERS * 2)
                      .sample(n=C.D1_N_CLUSTERS, random_state=C.RANDOM_SEED)["item_id"])

    # D2 items: assign to the drift supplier (metal/fastener items where a
    # stockout is costly). D3 items: hardware. D6: random active items.
    metal_fast = plan[plan["item_class"].isin(["Bar Stock", "Sheet", "Fasteners", "Hardware"])]
    d2_ids = set(rng.choice(metal_fast["item_id"].to_numpy(), size=min(C.D2_N_ITEMS, len(metal_fast)),
                            replace=False))
    hardware = plan[plan["item_class"].isin(["Fasteners", "Hardware"])]
    d3_ids = set(rng.choice(hardware["item_id"].to_numpy(), size=min(C.D3_N_ITEMS, len(hardware)),
                            replace=False))
    d6_ids = set(rng.choice(plan["item_id"].to_numpy(), size=int(C.D6_MISSING_SHARE * C.N_ITEMS),
                            replace=False))

    rows, item_meta, dup_map = [], {}, {}
    seen_desc = set()
    for row in plan.itertuples(index=False):
        iid = row.item_id
        desc = _make_unique(_canonical_description(row, rng), row.item_class, seen_desc, rng)
        # supplier
        stype = class_supplier_type[row.item_class]
        if iid in d2_ids:
            sup = drift_supplier_id
        else:
            cands = sup_by_type.get(stype) or suppliers["supplier_id"].tolist()
            sup = rng.choice(cands)
        master_lead = int(row.base_lead_days)
        if iid in d2_ids:
            master_lead = C.D2_DRIFT_START_DAYS      # stale at the pre-drift value

        annual = annual_by_item[iid]
        avg_month = annual / 12.0
        # UOM: D3 items purchased by the box but the master still reads each.
        box_size = None
        if iid in d3_ids:
            box_size = int(rng.choice(C.D3_BOX_SIZES))
        uom = "EA"

        # current (stale) policy, deliberately miscalibrated
        dol = avg_month * (master_lead / 30.0)
        err = np.exp(rng.normal(0, C.CURRENT_POLICY_ERROR_STD))
        cur_ss = max(0, round(0.5 * dol * err))
        cur_rop = max(0, round((dol + cur_ss) * err))
        std_cost = float(row.unit_cost)

        # duplicate cluster or single record
        if iid in dup_ids:
            k = int(rng.integers(C.D1_MEMBERS_RANGE[0], C.D1_MEMBERS_RANGE[1] + 1))
            w = rng.dirichlet(np.full(k, 4.0))
            numbers = []
            created0 = _rand_date(CREATED_MIN, C.START_DATE - timedelta(days=400), rng)
            for m in range(k):
                num = _item_number(iid, row.item_class, variant=m + 1)
                d = desc if m == 0 else _variant_description(desc, row.item_class, rng)
                created = created0 if m == 0 else _rand_date(created0, C.START_DATE, rng)
                rows.append(_master_row(num, d, uom, row.item_class, std_cost, cur_rop, cur_ss,
                                        master_lead, sup, created, iid, d6_ids, rng, split=True))
                numbers.append(num)
                item_meta[num] = {"item_id": iid, "uom_true": "EA", "box_size": None,
                                  "is_d2": iid in d2_ids, "supplier_id": sup, "member": m}
            dup_map[iid] = {"records": numbers, "weights": list(w), "primary": numbers[0]}
        else:
            num = _item_number(iid, row.item_class)
            created = _rand_date(CREATED_MIN, C.START_DATE, rng)
            rows.append(_master_row(num, desc, uom, row.item_class, std_cost, cur_rop, cur_ss,
                                    master_lead, sup, created, iid, d6_ids, rng, split=False))
            item_meta[num] = {"item_id": iid, "uom_true": "EA", "box_size": box_size,
                              "is_d2": iid in d2_ids, "supplier_id": sup, "member": 0}

    item_master = pd.DataFrame(rows)
    defects = {"d1_clusters": dup_map, "d2_items": d2_ids, "d3_items": d3_ids, "d6_items": d6_ids}
    return item_master, dup_map, item_meta, defects


def _master_row(num, desc, uom, cls, cost, rop, ss, lead, sup, created, iid, d6_ids, rng, split):
    # D6: blank one field on missing items.
    missing = iid in d6_ids
    field = rng.choice(["rop", "cost", "sup"]) if missing else None
    return {
        "item_number":           num,
        "description":           desc,
        "uom":                   uom,
        "item_class":            cls,
        "standard_cost":         np.nan if field == "cost" else cost,
        "current_reorder_point": np.nan if field == "rop" else rop,
        "current_safety_stock":  ss,
        "master_lead_time_days": lead,
        "primary_supplier_id":   None if field == "sup" else sup,
        "status":                "ACTIVE",
        "created_date":          created.isoformat(),
    }


def _rand_date(lo: date, hi: date, rng) -> date:
    span = (hi - lo).days
    return lo + timedelta(days=int(rng.integers(0, max(1, span))))
