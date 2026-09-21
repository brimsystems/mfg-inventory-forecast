"""Supplier master, including defect D5 (inconsistent supplier records).

One real vendor is fragmented across three name spellings and two supplier IDs,
so its spend and lead-time history split apart in every downstream join. The
generator returns both the recorded supplier table and the ground-truth map from
each recorded id to its canonical vendor, which the supplier crosswalk is scored
against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from faker import Faker

from .. import config as C


def _spelling_variants(name: str) -> list:
    """Three plausible data-entry spellings of the same company name."""
    base = name.replace(",", "")
    stem = base.split()[0]
    return [
        base,                                  # "Atlas Metals Inc"
        base.upper(),                          # "ATLAS METALS INC"
        f"{stem} Metals",                      # "Atlas Metals" (truncated)
    ]


def build_suppliers(rng: np.random.Generator):
    fake = Faker()
    Faker.seed(C.RANDOM_SEED)

    rows = []
    truth = {}   # recorded supplier_id -> canonical vendor key
    for i in range(1, C.N_SUPPLIERS + 1):
        sid = f"SUP-{i:03d}"
        name = fake.company()
        stype = rng.choice(C.SUPPLIER_TYPES)
        rows.append({
            "supplier_id":   sid,
            "supplier_name": name,
            "supplier_type": stype,
            "payment_terms": rng.choice(C.PAYMENT_TERMS),
            "status":        "ACTIVE",
        })
        truth[sid] = sid

    suppliers = pd.DataFrame(rows)

    # D5: pick one active Material vendor and fragment it across 3 spellings /
    # 2 ids. The primary id keeps its row; a second id and a duplicate-id row
    # carry the alternate spellings.
    material_ids = suppliers.loc[suppliers["supplier_type"] == "Material", "supplier_id"]
    primary = material_ids.iloc[0] if len(material_ids) else suppliers["supplier_id"].iloc[0]
    canonical_name = suppliers.loc[suppliers["supplier_id"] == primary, "supplier_name"].iloc[0]
    spellings = _spelling_variants(canonical_name)

    second_id = f"SUP-{C.N_SUPPLIERS + 1:03d}"
    suppliers.loc[suppliers["supplier_id"] == primary, "supplier_name"] = spellings[0]
    extra = pd.DataFrame([
        {"supplier_id": second_id, "supplier_name": spellings[1],
         "supplier_type": "Material", "payment_terms": "Net 30", "status": "ACTIVE"},
        {"supplier_id": second_id, "supplier_name": spellings[2],
         "supplier_type": "Material", "payment_terms": "Net 45", "status": "ACTIVE"},
    ])
    suppliers = pd.concat([suppliers, extra], ignore_index=True)

    vendor_key = primary                       # canonical identity for the vendor
    truth[primary] = vendor_key
    truth[second_id] = vendor_key

    d5 = {"canonical": vendor_key, "ids": [primary, second_id], "spellings": spellings}
    return suppliers, truth, d5
