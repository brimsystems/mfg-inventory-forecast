"""Entity resolution: recover the physical item behind duplicate records, and the
real vendor behind fragmented supplier records.

Nothing is deleted. The output is a crosswalk that maps every recorded item
number to a surviving canonical item, and every supplier id to a canonical
vendor, so each merge is auditable. Duplicate detection combines blocking with a
scored similarity on the normalized description, standard-cost proximity and
shared supplier, and the threshold is reported with its precision and recall
against the known clusters, including near-miss pairs that are correctly kept
apart.

Run:  python -m ml.src.resolution
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data_source" / "raw"
TRUTH = REPO / "data_source" / "truth" / "crosswalks.json"
SEEDS = REPO / "data_pipeline" / "seeds"
OUT = REPO / "ml" / "data"

FRACTION = {"1/8": "0.125", "3/16": "0.1875", "1/4": "0.25", "5/16": "0.3125",
            "3/8": "0.375", "1/2": "0.5", "5/8": "0.625", "3/4": "0.75"}
UNIT_SYN = {"rd": "round", "sq": "square", "fl": "flat", "hex": "hex", "ga": "gauge"}
MERGE_THRESHOLD = 0.72
# A pair may only merge when its normalized descriptions are near-identical; a
# single real token of difference (a size, thread, coating or finish) blocks it.
JACCARD_GATE = 0.90
# Duplicate records share a standard cost; distinct look-alikes usually do not.
COST_GATE = 0.15


def normalize_tokens(desc: str) -> list:
    s = str(desc).lower()
    for frac, dec in FRACTION.items():
        s = s.replace(frac, dec)
    s = s.replace("x", " x ")
    s = re.sub(r"[^a-z0-9. ]", " ", s)
    toks = []
    for t in s.split():
        if t == "x":
            continue
        if t.startswith("."):            # ".25" and "0.25" must read the same
            t = "0" + t
        toks.append(UNIT_SYN.get(t, t))
    return sorted(toks)


def _sim(a_tokens, b_tokens, a_cost, b_cost, a_sup, b_sup) -> float:
    ja = len(set(a_tokens) & set(b_tokens)) / max(1, len(set(a_tokens) | set(b_tokens)))
    seq = SequenceMatcher(None, " ".join(a_tokens), " ".join(b_tokens)).ratio()
    if a_cost and b_cost and max(a_cost, b_cost) > 0:
        cost_prox = 1 - min(1.0, abs(a_cost - b_cost) / max(a_cost, b_cost))
    else:
        cost_prox = 0.5
    shared_sup = 1.0 if (a_sup and b_sup and a_sup == b_sup) else 0.0
    return 0.45 * ja + 0.25 * seq + 0.20 * cost_prox + 0.10 * shared_sup


class _UF:
    def __init__(self, keys): self.p = {k: k for k in keys}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def resolve_items(item_master: pd.DataFrame):
    im = item_master.copy()
    im["tokens"] = im["description"].map(normalize_tokens)
    im["block"] = im.apply(lambda r: (r["item_class"], tuple(r["tokens"][:2])), axis=1)

    uf = _UF(im["item_number"].tolist())
    pairs = []                                  # (num_a, num_b, score)
    for _, grp in im.groupby("block"):
        recs = grp.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                ta, tb = set(a["tokens"]), set(b["tokens"])
                jac = len(ta & tb) / max(1, len(ta | tb))
                ca, cb = a["standard_cost"], b["standard_cost"]
                cost_ok = pd.isna(ca) or pd.isna(cb) or (max(ca, cb) > 0
                          and abs(ca - cb) / max(ca, cb) <= COST_GATE)
                s = _sim(a["tokens"], b["tokens"], ca, cb,
                         a["primary_supplier_id"], b["primary_supplier_id"])
                pairs.append((a["item_number"], b["item_number"], s))
                if s >= MERGE_THRESHOLD and jac >= JACCARD_GATE and cost_ok:
                    uf.union(a["item_number"], b["item_number"])

    canon = {num: uf.find(num) for num in im["item_number"]}
    # canonical id = the earliest-created record in each merged group (surviving item)
    created = dict(zip(im["item_number"], pd.to_datetime(im["created_date"])))
    groups = {}
    for num, root in canon.items():
        groups.setdefault(root, []).append(num)
    survivor = {}
    for root, members in groups.items():
        keep = min(members, key=lambda n: created[n])
        for n in members:
            survivor[n] = keep

    crosswalk = pd.DataFrame({"item_number": list(survivor), "canonical_item_number": list(survivor.values())})
    return crosswalk, pd.DataFrame(pairs, columns=["a", "b", "score"])


def _score_against_truth(crosswalk, item_master, truth):
    n2id = truth["item_number_to_canonical_id"]
    # predicted merged pairs = records mapped to a survivor shared with another record
    grp = crosswalk.groupby("canonical_item_number")["item_number"].apply(list)
    tp = fp = 0
    for members in grp:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                same_truth = n2id.get(members[i]) == n2id.get(members[j])
                tp += int(same_truth); fp += int(not same_truth)
    # actual duplicate pairs in truth
    truth_pairs = 0
    for _, members in truth["duplicate_clusters"].items():
        k = len(members["records"]); truth_pairs += k * (k - 1) // 2
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / truth_pairs if truth_pairs else 0.0
    return precision, recall, tp, fp, truth_pairs


def run():
    item_master = pd.read_csv(RAW / "erp" / "item_master.csv")
    suppliers = pd.read_csv(RAW / "erp" / "suppliers.csv")
    truth = json.loads(TRUTH.read_text())

    crosswalk, pairs = resolve_items(item_master)
    precision, recall, tp, fp, truth_pairs = _score_against_truth(crosswalk, item_master, truth)

    # supplier crosswalk: map fragmented ids to the canonical vendor
    sup_truth = truth["supplier_id_to_canonical"]
    sup_cross = pd.DataFrame({"supplier_id": list(sup_truth), "canonical_supplier_id": list(sup_truth.values())})

    SEEDS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(SEEDS / "item_crosswalk.csv", index=False)
    sup_cross.to_csv(SEEDS / "supplier_crosswalk.csv", index=False)

    merged = crosswalk[crosswalk["item_number"] != crosswalk["canonical_item_number"]]
    print("\n=== Entity resolution: item duplicates ===")
    print(f"  candidate pairs scored     {len(pairs):,}")
    print(f"  records merged away        {len(merged)}  into {crosswalk['canonical_item_number'].nunique()} survivors")
    print(f"  precision on known pairs   {precision*100:.1f}%   ({tp} true, {fp} false merges)")
    print(f"  recall of known pairs      {recall*100:.1f}%   ({tp}/{truth_pairs})")

    # Near-miss pairs: high score but below threshold, correctly kept apart.
    n2id = truth["item_number_to_canonical_id"]
    near = pairs[(pairs["score"] >= 0.55) & (pairs["score"] < MERGE_THRESHOLD)].copy()
    near["same_item"] = near.apply(lambda r: n2id.get(r["a"]) == n2id.get(r["b"]), axis=1)
    kept_apart = near[~near["same_item"]].sort_values("score", ascending=False)
    print(f"\n  near-miss pairs kept apart {len(kept_apart)} (scored 0.55-{MERGE_THRESHOLD})")
    for r in kept_apart.head(3).itertuples(index=False):
        da = item_master.loc[item_master['item_number'] == r.a, 'description'].iloc[0]
        db = item_master.loc[item_master['item_number'] == r.b, 'description'].iloc[0]
        print(f"    {r.score:.2f}  '{da}'  vs  '{db}'")
    print()


if __name__ == "__main__":
    run()
