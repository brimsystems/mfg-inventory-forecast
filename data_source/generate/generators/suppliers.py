"""Supplier master, including defect M6 (supplier fragmentation).

Three real vendors are each split across multiple names and IDs (seven records
in all), so their spend and lead-time history fragment in every downstream join.
The generator returns the recorded supplier table, the ground-truth map from each
recorded id to its canonical vendor, a fragmentation summary, and the id of the
supplier whose actual lead time drifts sharply over the history (defect M2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from faker import Faker

from .. import config as C


def _spelling_variants(name: str) -> list:
    base = name.replace(",", "")
    stem = base.split()[0]
    return [base, base.upper(), f"{stem} Inc", f"{stem} Co", f"{stem.upper()} MFG"]


def build_suppliers(rng: np.random.Generator):
    fake = Faker()
    Faker.seed(C.RANDOM_SEED)

    rows, truth = [], {}
    for i in range(1, C.N_SUPPLIERS + 1):
        sid = f"SUP-{i:03d}"
        stype = C.SUPPLIER_TYPES[i % len(C.SUPPLIER_TYPES)]
        rows.append({
            "supplier_id":   sid,
            "supplier_name": fake.company(),
            "supplier_type": stype,
            "payment_terms": rng.choice(C.PAYMENT_TERMS),
            "status":        "ACTIVE",
        })
        truth[sid] = sid

    suppliers = pd.DataFrame(rows)

    # ── M6: fragment three real vendors across seven records ────────────────
    # Vendor A -> 3 records, vendors B and C -> 2 records each.
    frag_plan = [3, 2, 2]
    base_ids = list(suppliers["supplier_id"])
    rng.shuffle(base_ids)
    next_id = C.N_SUPPLIERS + 1
    alias_ids = {}          # primary_id -> [alias ids]
    frag_records = []
    used = 0
    for n_records in frag_plan:
        primary = base_ids[used]; used += 1
        canonical_name = suppliers.loc[suppliers["supplier_id"] == primary, "supplier_name"].iloc[0]
        stype = suppliers.loc[suppliers["supplier_id"] == primary, "supplier_type"].iloc[0]
        spellings = _spelling_variants(canonical_name)
        suppliers.loc[suppliers["supplier_id"] == primary, "supplier_name"] = spellings[0]
        aliases = []
        for k in range(1, n_records):
            aid = f"SUP-{next_id:03d}"; next_id += 1
            suppliers = pd.concat([suppliers, pd.DataFrame([{
                "supplier_id": aid, "supplier_name": spellings[k],
                "supplier_type": stype, "payment_terms": rng.choice(C.PAYMENT_TERMS),
                "status": "ACTIVE"}])], ignore_index=True)
            truth[aid] = primary
            aliases.append(aid)
        alias_ids[primary] = aliases
        frag_records.append({"canonical": primary, "aliases": aliases, "spellings": spellings[:n_records]})

    sup_frag = {"alias_ids": alias_ids, "records": frag_records,
                "fragmented_primaries": list(alias_ids)}

    # ── M2 sharp-drift supplier: a Mechanical vendor (motors, gearboxes) ─────
    mech = suppliers.loc[suppliers["supplier_type"] == "Mechanical", "supplier_id"]
    drift_supplier_id = mech.iloc[0] if len(mech) else suppliers["supplier_id"].iloc[0]

    return suppliers, truth, sup_frag, drift_supplier_id
