"""Item master: canonical live items rendered into recorded ERP rows, dead
records that were never deactivated, and the item side of the planted defects.

  M1  dead records       - active items with no movement in 24+ months
  M2  stale parameters   - lead time / reorder point / safety stock from go-live
  M4  duplicate records  - one physical item under two to four item numbers
  M5  UOM mismatch       - purchase UOM differs from stock UOM, no conversion
  M7  missing fields      - blank cost / supplier / reorder point; item_class MISC

Returns the recorded item_master, the duplicate crosswalk truth (canonical item
-> recorded numbers with demand-split weights), the per-record metadata the
downstream generators need, and a defects dictionary with the item-id sets.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .. import config as C

CREATED_MIN = date(2015, 1, 1)          # ERP installed ~4 years before cutoff; some pre-date it
GO_LIVE     = date(2021, 6, 1)

# Description vocabularies by class.
STEEL_SHAPES = ["Angle", "Tube", "Channel", "Plate", "Flat Bar", "Beam"]
STEEL_SIZES  = ['1"', '1-1/2"', '2"', '3"', '4"', '6"', '1/4"', '3/8"', '1/2"']
MOTORS = ["Gearmotor", "AC Motor", "Servo Motor", "Gear Reducer", "Bearing",
          "Sprocket", "Roller", "V-Belt", "Timing Belt", "Pillow Block", "Coupling"]
MOTOR_SPEC = ["1HP", "2HP", "5HP", "7.5HP", "10HP", "56C", "145T", "184T", "1750RPM"]
FITTINGS = ["Hyd Fitting", "Pneu Fitting", "Hose Assy", "Quick Coupler", "Manifold",
            "Ball Valve", "Flow Control", "Elbow", "Tee", "Reducer Bushing"]
FITTING_SPEC = ['1/4" NPT', '3/8" NPT', '1/2" NPT', "-6 JIC", "-8 JIC", "SAE-4"]
ELECTRICAL = ["VFD", "Photo Sensor", "Prox Sensor", "Contactor", "Enclosure",
              "Wire", "Cable", "Terminal Block", "Relay", "Power Supply", "Limit Switch"]
ELEC_SPEC = ["24VDC", "480V", "3PH", "12AWG", "14AWG", "16AWG", "NEMA 12", "10A", "25A"]
FRACTIONS = [("1/8", .125), ("3/16", .1875), ("1/4", .25), ("5/16", .3125),
             ("3/8", .375), ("1/2", .5), ("5/8", .625), ("3/4", .75), ("1", 1.0)]
THREADS = [20, 28, 18, 16, 13, 11, 24]
HEADS = ["SHCS", "HHCS", "FHCS", "BHCS", "HSHCS"]
FASTENER_MAT = ["SS", "GR5", "GR8", "ZP"]
HARDWARE = ["Flat Washer", "Lock Washer", "Hex Nut", "Nylock Nut", "Clevis Pin",
            "Cotter Pin", "Spacer", "Standoff", "Cable Tie", "Rivet", "Set Screw"]
CONSUMABLES = ["Weld Wire ER70S", "Flux Core Wire", "Grinding Wheel", "Flap Disc",
               "Cutoff Wheel", "Primer Gray", "Enamel Blue", "Powder Coat Black",
               "Anti-Spatter", "Shielding Gas", "Wire Brush"]
CONSUMABLE_PACK = ["33LB", "44LB", "1GAL", "5GAL", "CASE", "BOX", "EA"]
SERVICES = ["Zinc Plating", "Powder Coat", "Heat Treat", "Anodize", "Passivation",
            "Black Oxide", "Galvanize", "Chromate"]
SERVICE_SPEC = ["Clear", "Yellow", "Class 1", "Type II", "RoHS", "Blue", "Black"]

FINISH_TOKENS = {"Fasteners": ["ZP", "PL", "BLK", "SS", "GALV"],
                 "Hardware": ["ZP", "PL", "BLK", "SS"],
                 "Raw Material": ["A36", "A500", "HR", "CR"],
                 "Mechanical": ["TEFC", "56C", "REV-A"],
                 "Electrical": ["UL", "CE", "REV-B"],
                 "Fittings": ["SS", "BRS", "STL"],
                 "Consumables": CONSUMABLE_PACK, "Outside Service": SERVICE_SPEC}


def _canonical_description(cls, material, rng) -> str:
    if cls == "Raw Material":
        return f"{material} {rng.choice(STEEL_SHAPES)} {rng.choice(STEEL_SIZES)}"
    if cls == "Mechanical":
        return f"{rng.choice(MOTORS)} {rng.choice(MOTOR_SPEC)}"
    if cls == "Fittings":
        return f"{rng.choice(FITTINGS)} {rng.choice(FITTING_SPEC)}"
    if cls == "Electrical":
        return f"{rng.choice(ELECTRICAL)} {rng.choice(ELEC_SPEC)}"
    if cls == "Fasteners":
        frac, _ = FRACTIONS[rng.integers(0, len(FRACTIONS))]
        thr = rng.choice(THREADS)
        ln, _ = FRACTIONS[rng.integers(2, len(FRACTIONS))]
        return f"{frac}-{thr} x {ln} {rng.choice(HEADS)} {rng.choice(FASTENER_MAT)}"
    if cls == "Hardware":
        return f"{rng.choice(HARDWARE)} {FRACTIONS[rng.integers(0, len(FRACTIONS))][0]}"
    if cls == "Consumables":
        return f"{rng.choice(CONSUMABLES)} {rng.choice(CONSUMABLE_PACK)}"
    return f"{rng.choice(SERVICES)} {rng.choice(SERVICE_SPEC)}"


def _make_unique(desc, cls, seen, rng) -> str:
    if desc not in seen:
        seen.add(desc); return desc
    for tok in rng.permutation(FINISH_TOKENS.get(cls, ["A", "B", "C"])):
        cand = f"{desc} {tok}"
        if cand not in seen:
            seen.add(cand); return cand
    i = 2
    while f"{desc} {i}" in seen:
        i += 1
    seen.add(f"{desc} {i}"); return f"{desc} {i}"


def _variant_description(desc, cls, rng) -> str:
    """A different data-entry convention for the same physical item: decimal for
    fraction, reordered tokens, abbreviations. Normalizes back to the same item."""
    out = desc
    for frac, dec in FRACTIONS:
        if frac in out:
            out = out.replace(frac, (f"{dec:.4g}".lstrip("0") if dec < 1 else f"{dec:g}"))
    out = out.replace(" x ", "x").replace('"', "").replace("  ", " ")
    if rng.random() < 0.5:
        parts = out.split()
        if len(parts) >= 2:
            out = parts[-1] + " " + " ".join(parts[:-1])
    return out.strip()


_PREFIX = {"Raw Material": "RM", "Mechanical": "MEC", "Fittings": "FIT",
           "Electrical": "ELE", "Fasteners": "FAS", "Hardware": "HDW",
           "Consumables": "CON", "Outside Service": "SVC"}


def _item_number(seq, cls, variant=0) -> str:
    tail = "" if variant == 0 else (f"-{'ABC'[variant-1]}" if variant <= 3 else f"-{variant}")
    return f"{_PREFIX.get(cls, 'ITM')}-{seq:05d}{tail}"


def _rand_date(lo, hi, rng) -> date:
    span = max(1, (hi - lo).days)
    return lo + timedelta(days=int(rng.integers(0, span)))


def build_item_master(plan, suppliers, annual_by_item, drift_supplier_id, sup_frag, rng):
    sup_by_type = {t: suppliers.loc[suppliers["supplier_type"] == t, "supplier_id"].unique().tolist()
                   for t in suppliers["supplier_type"].unique()}

    # ── Choose which live items carry which defects ─────────────────────────
    live = plan.copy()
    live["annual"] = live["item_id"].map(annual_by_item).fillna(0.0)

    # M4 duplicate clusters: higher-consumption items in the target classes.
    m4_pool = live[live["item_class"].isin(C.M4_CLUSTER_CLASSES)]
    n_m4 = int(round(len(live) * C.M4_LIVE_SHARE))
    m4_ids = set(m4_pool.sort_values("annual", ascending=False)
                       .head(n_m4 * 3).sample(n=min(n_m4, len(m4_pool)), random_state=C.RANDOM_SEED)["item_id"])

    # M2 drift: the sharply-drifting supplier's items plus a general share.
    drift_items = set(rng.choice(live["item_id"].to_numpy(),
                                 size=int(round(len(live) * C.M2_DRIFT_SHARE)), replace=False))
    sharp_pool = live[live["item_class"].isin(["Mechanical", "Electrical"])]["item_id"].to_numpy()
    sharp_items = set(rng.choice(sharp_pool, size=min(60, len(sharp_pool)), replace=False))

    # M5 UOM mismatch: hardware/fasteners/electrical/raw/consumables.
    m5_classes = list(C.M5_UOM_PAIRS)
    m5_pool = live[live["item_class"].isin(m5_classes)]["item_id"].to_numpy()
    m5_ids = set(rng.choice(m5_pool, size=min(C.M5_N_ITEMS, len(m5_pool)), replace=False))

    # M7 missing / MISC.
    m7_blank = set(rng.choice(live["item_id"].to_numpy(),
                              size=int(round(len(live) * C.M7_BLANK_SHARE)), replace=False))
    m7_misc = set(rng.choice(live["item_id"].to_numpy(),
                             size=int(round(len(live) * C.M7_MISC_SHARE)), replace=False))

    rows, item_meta, dup_map = [], {}, {}
    seen_desc = set()
    class_by_id = live.set_index("item_id")["item_class"].to_dict()
    material_by_id = live.set_index("item_id")["material"].to_dict()

    for row in live.itertuples(index=False):
        iid = row.item_id
        desc = _make_unique(_canonical_description(row.item_class, row.material, rng),
                            row.item_class, seen_desc, rng)
        stype = C.CLASS_SUPPLIER_TYPE[row.item_class]
        if iid in drift_items and iid in sharp_items:
            sup = drift_supplier_id
        else:
            cands = sup_by_type.get(stype) or suppliers["supplier_id"].tolist()
            sup = rng.choice(cands)
        # Supplier fragmentation (M6): some items point at a fragmented alias id.
        if sup in sup_frag["alias_ids"] and rng.random() < 0.5:
            sup = rng.choice(sup_frag["alias_ids"][sup])

        master_lead = int(row.base_lead_days)
        annual = annual_by_item.get(iid, 0.0)
        avg_month = annual / 12.0

        # M5 UOM: recorded purchase/stock UOM with no conversion.
        purchase_uom, stock_uom, conv = "EA", "EA", 1
        if iid in m5_ids:
            purchase_uom, stock_uom, conv = _pick_uom(row.item_class, rng)

        # Current (stale) policy, deliberately miscalibrated (M2).
        dol = avg_month * (master_lead / 30.0)
        err = np.exp(rng.normal(0, C.CURRENT_POLICY_ERROR_STD))
        cur_ss = max(0, round(0.5 * dol * err))
        cur_rop = max(0, round((dol + cur_ss) * err))
        std_cost = float(row.unit_cost)
        item_class = "MISC" if iid in m7_misc else row.item_class

        base = dict(uom=stock_uom, purchase_uom=purchase_uom, uom_conversion=(conv if conv != 1 else np.nan),
                    item_class=item_class, cost=std_cost, rop=cur_rop, ss=cur_ss,
                    lead=master_lead, sup=sup, blank=(iid in m7_blank))

        if iid in m4_ids:
            k = int(rng.integers(C.M4_MEMBERS_RANGE[0], C.M4_MEMBERS_RANGE[1] + 1))
            w = rng.dirichlet(np.full(k, 4.0))
            numbers = []
            created0 = _rand_date(CREATED_MIN, GO_LIVE, rng)
            for m in range(k):
                num = _item_number(iid, row.item_class, variant=m + 1)
                d = desc if m == 0 else _variant_description(desc, row.item_class, rng)
                created = created0 if m == 0 else _rand_date(created0, C.START_DATE, rng)
                rows.append(_master_row(num, d, created, iid, base, rng))
                numbers.append(num)
                item_meta[num] = {"item_id": iid, "stock_uom": stock_uom, "purchase_uom": purchase_uom,
                                  "uom_conv": conv, "is_drift": iid in drift_items,
                                  "is_sharp": iid in sharp_items, "supplier_id": sup,
                                  "member": m, "dead": False, "cls": row.item_class}
            dup_map[iid] = {"records": numbers, "weights": list(w), "primary": numbers[0]}
        else:
            num = _item_number(iid, row.item_class)
            created = _rand_date(CREATED_MIN, GO_LIVE, rng)
            rows.append(_master_row(num, desc, created, iid, base, rng))
            item_meta[num] = {"item_id": iid, "stock_uom": stock_uom, "purchase_uom": purchase_uom,
                              "uom_conv": conv, "is_drift": iid in drift_items,
                              "is_sharp": iid in sharp_items, "supplier_id": sup,
                              "member": 0, "dead": False, "cls": row.item_class}

    # ── M1 dead records ─────────────────────────────────────────────────────
    dead_meta = []
    for j in range(C.N_DEAD_ITEMS):
        cls = list(C.ITEM_CLASS_SHARES)[int(rng.integers(0, len(C.ITEM_CLASS_SHARES)))]
        mat = rng.choice(sum(C.MATERIAL_FAMILIES.values(), [])) if cls == "Raw Material" else None
        desc = _make_unique(_canonical_description(cls, mat, rng), cls, seen_desc, rng)
        num = _item_number(100000 + j, cls)
        keeps_rop = rng.random() < C.M1_KEEP_REORDER_SHARE
        cost = float(np.exp(rng.normal(C.STANDARD_COST_LOG_MEAN, C.STANDARD_COST_LOG_STD)) *
                     C.CLASS_COST_MULT[cls])
        sup = rng.choice(sup_by_type.get(C.CLASS_SUPPLIER_TYPE[cls], suppliers["supplier_id"].tolist()))
        created = _rand_date(CREATED_MIN, date(2022, 1, 1), rng)
        rows.append({
            "item_number": num, "description": desc, "uom": "EA", "purchase_uom": "EA",
            "uom_conversion": np.nan, "item_class": cls, "standard_cost": round(cost, 2),
            "reorder_point": (max(1, round(rng.uniform(2, 40))) if keeps_rop else np.nan),
            "safety_stock": (max(0, round(rng.uniform(1, 20))) if keeps_rop else np.nan),
            "master_lead_time_days": int(rng.integers(*C.BASE_LEAD_TIME_RANGE)),
            "primary_supplier_id": sup, "status": "ACTIVE",
            "created_by": rng.choice(C.OFFICE_USERS),
            "created_date": created.isoformat(),
            "last_updated": created.isoformat(),
        })
        dead_meta.append({"item_number": num, "keeps_rop": keeps_rop,
                          "stray": rng.random() < C.M1_STRAY_TXN_SHARE, "cls": cls, "cost": cost})

    item_master = pd.DataFrame(rows)
    defects = {"m4_clusters": dup_map, "m2_drift_items": drift_items, "m2_sharp_items": sharp_items,
               "m5_items": m5_ids, "m7_blank": m7_blank, "m7_misc": m7_misc,
               "dead_meta": dead_meta}
    return item_master, dup_map, item_meta, defects


def _pick_uom(cls, rng):
    pairs = C.M5_UOM_PAIRS.get(cls) or C.M5_UOM_PAIRS["Hardware"]
    p, s, c = pairs[int(rng.integers(0, len(pairs)))]
    return p, s, int(c)


def _master_row(num, desc, created, iid, base, rng):
    field = rng.choice(["rop", "cost", "sup"]) if base["blank"] else None
    return {
        "item_number":           num,
        "description":           desc,
        "uom":                   base["uom"],
        "purchase_uom":          base["purchase_uom"],
        # The ERP holds no conversion between purchase and stock UOM; that
        # absence is the error (M5). The true factor lives in item_meta / truth.
        "uom_conversion":        np.nan,
        "item_class":            base["item_class"],
        "standard_cost":         np.nan if field == "cost" else base["cost"],
        "reorder_point":         np.nan if field == "rop" else base["rop"],
        "safety_stock":          base["ss"],
        "master_lead_time_days": base["lead"],
        "primary_supplier_id":   None if field == "sup" else base["sup"],
        "status":                "ACTIVE",
        "created_by":            rng.choice(C.OFFICE_USERS),
        "created_date":          created.isoformat(),
        "last_updated":          _rand_date(created, C.START_DATE, rng).isoformat(),
    }
