"""Primary deliverable: the ERP purchasing reorder queue with the demand model embedded.

An ERP buyer's screen, as of the last day of the forward window. Every stocked
item carries the model's reorder point for the month (the corrected forecast of
demand over the item's lead time plus safety stock), compared with its on-hand
and genuinely open on-order. Items at or below the point are ORDER NOW, items
within two weeks of it are ORDER SOON, and each line flags where the cleanup
changed the item (merged record, corrected lead time, corrected bill, restored
history) so the buyer sees why a number moved. Writes docs/index.html.

Run:  python -m ml.reports.generate_reorder_queue
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data_source" / "raw"
REM = RAW / "remediation"
OUT = REPO / "docs" / "index.html"

import sys
sys.path.insert(0, str(REPO))
from data_source.generate import config as C  # noqa: E402

AS_OF = C.AS_OF_DATE
ORDER_COVER_DAYS = C.ORDER_COVER_DAYS
SOON_DAYS = 14
NROWS = 22
SIGN = {"RECEIPT": 1, "ISSUE": -1, "BACKFLUSH": -1, "ADJUST": 1}

BRAND = "#1F3A5F"; NAVBG = "#264A73"; RED = "#C62828"; AMBER = "#E8920A"; GREEN = "#2E7D32"
BLUE = "#381FA1"; BG = "#F3F5F7"; LINE = "#D9DEE5"; MUTED = "#5F6B7A"; ROWRED = "#FDF1F1"; ROWAMB = "#FFFAEF"


def build_queue():
    sched = json.loads((REPO / "ml" / "data" / "policy" / "rop_schedule.json").read_text())
    attrs = pd.read_parquet(REPO / "ml" / "data" / "marts" / "item_attributes.parquet").set_index("canonical_item_number")
    im = pd.read_csv(RAW / "erp" / "item_master.csv", low_memory=False).set_index("item_number")
    sup = pd.read_csv(RAW / "erp" / "supplier_master.csv").set_index("supplier_id")
    po = pd.read_csv(RAW / "erp" / "purchase_orders.csv", low_memory=False)
    xw = pd.read_csv(REPO / "data_pipeline" / "seeds" / "item_crosswalk.csv").set_index("item_number")["canonical_item_number"].to_dict()
    lead = pd.read_csv(REM / "lead_time_computation.csv")
    bom = pd.read_csv(REM / "bom_change_log.csv")
    ft = pd.read_csv(REM / "free_text_attribution.csv")
    sup_xw = pd.read_csv(REM / "supplier_crosswalk.csv") if (REM / "supplier_crosswalk.csv").exists() else None

    canon = lambda n: xw.get(n, n)
    # suppliers under their canonical record, as the crosswalk left them
    sup_canon = json.loads((REPO / "data_source" / "truth" / "crosswalks.json").read_text())["supplier_id_to_canonical"]
    # on hand: the balance the ERP carries after the merge and the cycle counts,
    # one record per part (the counts restate each surviving record to the shelf)
    phys = json.loads((REPO / "data_source" / "truth" / "physical_on_hand_end.json").read_text())
    on_hand = pd.Series({canon(n): max(0.0, float(v)) for n, v in phys.items()})
    # on order: lines still open that are genuinely inbound (the stale ones were closed)
    op = po[(po["status"] == "OPEN") & (po["qty_ordered"] > po["qty_received"])
            & (pd.to_datetime(po["order_date"]) >= pd.Timestamp(AS_OF) - pd.Timedelta(days=90))]
    on_order = (op["qty_ordered"] - op["qty_received"]).groupby(op["item_number"].map(canon)).sum()

    # component requirements of released jobs not yet complete, within each item's lead time
    prod = pd.read_csv(RAW / "erp" / "production_orders.csv")
    open_jobs = prod[(pd.to_datetime(prod["release_date"]) <= pd.Timestamp(AS_OF))
                     & ((prod["completed_date"].isna()) | (pd.to_datetime(prod["completed_date"]) > pd.Timestamp(AS_OF)))]
    bom = pd.read_csv(RAW / "erp" / "bill_of_materials.csv")
    req = open_jobs.merge(bom, on="product_number")
    req["qty"] = req["qty_ordered"] * req["qty_per"]
    known_jobs = req.groupby(req["component_item"].map(canon))["qty"].sum()
    # a record left with no primary supplier falls back to the one it is bought from most
    po_sup = po.dropna(subset=["supplier_id"]).assign(c=po["item_number"].map(canon)).groupby("c")["supplier_id"]         .agg(lambda x: x.map(lambda v: sup_canon.get(v, v)).mode().iloc[0])
    merged = {canon(n) for n, c in xw.items() if n != c}
    stale = lead[(lead["median_actual"] - lead["master_lead_time"]).abs() > 3]["item_number"].map(canon)
    lead_fixed = set(stale)
    bom_fixed = set(bom["component_item"].map(canon))
    restored = set(ft.loc[ft["confirmation"] == "confirmed", "probable_item_number"].dropna().map(canon))

    rows = []
    for item, entries in sched["items"].items():
        if item not in attrs.index:
            continue
        e = entries[-1]
        d_, level, ss, pred, h_days, rop = e[0], e[1], e[2], e[3], e[4], e[5]
        a = attrs.loc[item]
        lt = float(a["corrected_lead_days"])
        daily = float(pred) / max(1.0, float(h_days))
        # the point the ERP triggers on covers the demand it cannot already see:
        # the forecast over the lead time less what released jobs have allocated
        fc_lead = float(level) - float(ss)
        oh = float(on_hand.get(item, 0.0)); oo = float(on_order.get(item, 0.0))
        position = oh + oo
        cover = position / daily if daily > 0 else np.inf
        # the ERP nets released jobs against stock (MRP): the item is due when stock
        # on hand and on order, less what released jobs will draw within the lead
        # time, falls to the level set for the demand it cannot see
        need = float(known_jobs.get(item, 0.0))
        position = oh + oo - need
        if position <= level:
            status = "ORDER NOW"
        elif position <= level + daily * SOON_DAYS:
            status = "ORDER SOON"
        else:
            status = "OK"
        suggested = max(0.0, level + daily * ORDER_COVER_DAYS - position) if status != "OK" else 0.0
        sid = sup_canon.get(im["primary_supplier_id"].get(item), im["primary_supplier_id"].get(item))
        if not isinstance(sid, str):
            sid = po_sup.get(item)
        sname = sup["supplier_name"].get(sid, "") if isinstance(sid, str) else ""
        flags = [f for f, on in (("Merged record", item in merged), ("Lead time corrected", item in lead_fixed),
                                 ("BOM corrected", item in bom_fixed), ("History restored", item in restored)) if on]
        rows.append({"item": item, "desc": im["description"].get(item, ""), "abc": a["abc"], "supplier": sname,
                     "on_hand": oh, "on_order": oo, "alloc": need, "avail": position, "lead": lt, "fc_lead": fc_lead,
                     "ss": float(ss), "rop": float(level),
                     "suggested": suggested, "status": status, "cover": cover, "cost": float(a["standard_cost"]),
                     "flags": flags})
    q = pd.DataFrame(rows)
    q["rank"] = q["status"].map({"ORDER NOW": 0, "ORDER SOON": 1, "OK": 2})
    q["gap"] = (q["avail"] - q["rop"]) / q["rop"].clip(lower=1)
    return q.sort_values(["rank", "abc", "gap"]).reset_index(drop=True)


def _fmt(x):
    return f"{x:,.0f}"


def render(q):
    counts = q["status"].value_counts().to_dict()
    now, soon, ok = counts.get("ORDER NOW", 0), counts.get("ORDER SOON", 0), counts.get("OK", 0)
    show = pd.concat([q[q["status"] == "ORDER NOW"].head(NROWS - 6), q[q["status"] == "ORDER SOON"].head(4),
                      q[q["status"] == "OK"].head(2)])
    badge = {"ORDER NOW": RED, "ORDER SOON": AMBER, "OK": GREEN}
    rowbg = {"ORDER NOW": ROWRED, "ORDER SOON": ROWAMB, "OK": "#FFFFFF"}
    trs = []
    for r in show.itertuples(index=False):
        tags = "".join(f'<span class="tag">{f}</span>' for f in r.flags)
        trs.append(
            f'<tr style="background:{rowbg[r.status]};">'
            f'<td class="mono">{r.item}</td><td>{r.desc}</td><td class="c">{r.abc}</td><td>{r.supplier}</td>'
            f'<td class="r">{_fmt(r.on_hand)}</td><td class="r">{_fmt(r.alloc)}</td><td class="r">{_fmt(r.on_order)}</td>'
            f'<td class="r">{_fmt(r.avail)}</td><td class="r">{r.lead:.0f}</td>'
            f'<td class="r">{_fmt(r.fc_lead)}</td><td class="r">{_fmt(r.ss)}</td><td class="r"><b>{_fmt(r.rop)}</b></td>'
            f'<td class="r"><b>{_fmt(r.suggested) if r.suggested else "&ndash;"}</b></td>'
            f'<td><span class="badge" style="background:{badge[r.status]};">&bull; {r.status}</span></td>'
            f'<td>{tags}</td></tr>')
    nav = "".join(f'<a class="{"on" if n == "Purchasing" else ""}">{n}</a>'
                  for n in ["Dashboard", "Work Orders", "Scheduling", "Inventory", "Purchasing", "Receiving", "Reports", "Admin"])
    day = pd.Timestamp(AS_OF).strftime("%A, %B %d, %Y")
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reorder Queue | Enterprise Resource Planning</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:"Segoe UI", Tahoma, Arial, sans-serif; font-size:12.5px; color:#1F2933; background:#fff; }}
  .top {{ background:{BRAND}; color:#fff; display:flex; justify-content:space-between; align-items:center; padding:9px 16px; }}
  .top .app {{ font-size:17px; font-weight:700; }}
  .top .who {{ font-size:12px; color:#D6E2F0; }}
  .nav {{ background:{NAVBG}; display:flex; }}
  .nav a {{ color:#E6EEF7; padding:9px 16px; font-size:12.5px; }}
  .nav a.on {{ background:#fff; color:{BRAND}; font-weight:700; }}
  .crumb {{ padding:7px 16px; color:{MUTED}; border-bottom:1px solid {LINE}; background:{BG}; }}
  .crumb span {{ color:#2458A6; }}
  .head {{ display:flex; justify-content:space-between; align-items:flex-start; padding:12px 16px 6px; }}
  .head h1 {{ font-size:15px; }}
  .head .sub {{ color:{MUTED}; margin-top:2px; }}
  .summary {{ border:1px solid {LINE}; border-radius:3px; padding:8px 14px; font-size:12px; color:{MUTED}; }}
  .summary b {{ margin:0 3px 0 10px; }}
  .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-left:10px; }}
  .bar {{ display:flex; gap:6px; align-items:center; padding:6px 16px; }}
  .btn {{ border:1px solid #AEB8C4; background:#F7F9FB; padding:3px 10px; border-radius:2px; font-size:12px; }}
  .btn.dim {{ color:#9AA5B1; }}
  .sep {{ width:10px; }}
  .filters {{ display:flex; gap:12px; align-items:center; padding:6px 16px 8px; color:{MUTED}; }}
  .sel {{ border:1px solid #AEB8C4; padding:2px 8px; color:#1F2933; background:#fff; }}
  table {{ border-collapse:collapse; width:calc(100% - 32px); margin:0 16px 16px; }}
  th {{ background:#E9EDF2; text-align:left; padding:6px 7px; border:1px solid {LINE}; font-weight:600; white-space:nowrap; }}
  td {{ padding:5px 7px; border:1px solid {LINE}; white-space:nowrap; }}
  td.r, th.r {{ text-align:right; }} td.c {{ text-align:center; }}
  .mono {{ font-family:Consolas, monospace; }}
  .badge {{ color:#fff; font-weight:700; font-size:11px; padding:2px 7px; border-radius:3px; }}
  .tag {{ display:inline-block; border:1px solid #9FB3CC; color:#2458A6; background:#EEF4FB; border-radius:10px;
          padding:0 7px; margin-right:4px; font-size:11px; }}
  .ml {{ background:{BLUE}; color:#fff; font-size:10px; font-weight:700; padding:1px 4px; border-radius:2px; margin-left:4px; }}
</style></head><body>
<div class="top"><div class="app">Enterprise Resource Planning</div>
  <div class="who">K. Brooks (Buyer) &nbsp;&nbsp; {day}</div></div>
<div class="nav">{nav}</div>
<div class="crumb"><span>Purchasing</span> &rsaquo; <span>Reorder</span> &rsaquo; Suggested Orders</div>
<div class="head"><div><h1>Reorder Queue, Suggested Orders</h1>
  <div class="sub">All stocked items against this week's reorder points &middot; {pd.Timestamp(AS_OF).strftime('%m/%d/%Y')}</div></div>
  <div class="summary">REORDER SUMMARY <span class="dot" style="background:{RED}"></span><b>{now}</b>Order now
  <span class="dot" style="background:{AMBER}"></span><b>{soon}</b>Order soon
  <span class="dot" style="background:{GREEN}"></span><b>{ok:,}</b>OK</div></div>
<div class="bar"><span class="btn">+ Create PO</span><span class="btn dim">Edit</span><span class="btn">Release to PO</span>
  <span class="btn">Hold</span><span class="sep"></span><span class="btn">Export</span></div>
<div class="filters">Supplier: <span class="sel">All &#9662;</span> Class: <span class="sel">All &#9662;</span>
  Action: <span class="sel">Order now + soon &#9662;</span> <span class="sel" style="width:220px;">&#128269; Search items...</span></div>
<table><thead><tr><th>Item #</th><th>Description</th><th>ABC</th><th>Supplier</th><th class="r">On hand</th>
<th class="r">Allocated</th><th class="r">On order</th><th class="r">Available</th><th class="r">Lead (days)</th><th class="r">Forecast over lead</th><th class="r">Safety stock</th>
<th class="r">Reorder point</th><th class="r">Suggested qty</th><th>Action <span class="ml">ML</span></th><th>Changed by cleanup</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table>
</body></html>"""
    OUT.write_text(html, encoding="utf-8")
    return now, soon, ok


def run():
    q = build_queue()
    now, soon, ok = render(q)
    print(f"Reorder queue written to {OUT}: {now} order now, {soon} order soon, {ok:,} OK")


if __name__ == "__main__":
    run()
