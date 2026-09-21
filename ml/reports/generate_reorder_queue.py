"""Primary deliverable: the ERP purchasing reorder queue with the model embedded.

A JobBOSS-style buyer's screen. Items due for reorder are ranked by urgency, each
with the forecast demand over its lead time, current on-hand, the actual lead time
in use, safety stock, a suggested order, and a short reason. Items whose record
was merged or whose lead time was corrected are flagged so the buyer sees what
changed. Writes docs/index.html.

Run:  python -m ml.reports.generate_reorder_queue
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REC = REPO / "ml" / "data" / "scoring" / "reorder_recommendations.parquet"
METRICS = REPO / "ml" / "data" / "backtest" / "model_metrics.json"
OUT = REPO / "docs" / "index.html"
AS_OF = "August 31, 2026"

BRAND = "#322B4B"; RED = "#CC0000"; AMBER = "#E8920A"; GREEN = "#00A84C"
BLUE = "#381FA1"; LIGHT = "#54C0E8"; BG = "#F3F5F7"; LINE = "#E3E6EB"; MUTED = "#6B7280"
NROWS = 16

NAV = ["Dashboard", "Purchase Orders", "Receiving", "Reorder Queue", "Items", "Suppliers", "Reports"]


def _priority_badge(p):
    color = {"REORDER": RED, "SOON": AMBER, "OK": GREEN}[p]
    return f'<span class="badge" style="background:{color};">{p}</span>'


def _flag_tags(r):
    tags = ""
    if r["flag_lead_corrected"]:
        tags += '<span class="tag tag-lead">lead corrected</span>'
    if r["flag_merged"]:
        tags += '<span class="tag tag-merge">record merged</span>'
    return tags


def _row(r):
    lead = f'{r["corrected_lead_days"]}d'
    if r["flag_lead_corrected"]:
        lead += f' <span class="was">was {r["master_lead_time_days"]}d</span>'
    order = f'{r["suggested_qty"]:,} <span class="unit">ea</span>' if r["suggested_qty"] > 0 else "&mdash;"
    when = "now" if r["priority"] == "REORDER" else ("this week" if r["priority"] == "SOON" else "&mdash;")
    return f"""<tr>
      <td><div class="item">{r['item_number']}</div>
          <div class="desc">{r['description']}</div>
          <div class="reason">{r['reason']}{_flag_tags(r)}</div></td>
      <td class="num">{int(r['on_hand']):,}</td>
      <td class="num">{r['forecast_ltd']:.0f} <span class="unit">/ {r['corrected_lead_days']}d</span></td>
      <td class="num">{lead}</td>
      <td class="num">{int(r['safety_stock']):,}</td>
      <td class="num">{order}<div class="when">{when}</div></td>
      <td>{_priority_badge(r['priority'])}</td>
    </tr>"""


def build():
    rec = pd.read_parquet(REC)
    metrics = json.loads(METRICS.read_text()) if METRICS.exists() else {}
    wape = metrics.get("overall", {}).get("model")
    wape_txt = f"test WAPE {wape*100:.0f}%" if wape else "test WAPE"

    n_reorder = int((rec["priority"] == "REORDER").sum())
    n_soon = int((rec["priority"] == "SOON").sum())
    n_ok = int((rec["priority"] == "OK").sum())
    order_value = float((rec.loc[rec["priority"] == "REORDER", "suggested_qty"]
                         * rec.loc[rec["priority"] == "REORDER", "unit_cost"]).sum())
    n_flag = int((rec["flag_merged"] | rec["flag_lead_corrected"]).sum())

    due = rec[rec["priority"] == "REORDER"].sort_values("cover_days").head(NROWS)
    body_rows = "".join(_row(r) for _, r in due.iterrows())
    nav_html = "".join(
        f'<div class="nav-item{" active" if n == "Reorder Queue" else ""}">{n}</div>' for n in NAV)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reorder Queue &middot; JobBOSS</title>
<style>
  *,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:{BG};
    color:#1a1a1a; font-size:14px; display:flex; min-height:100vh; }}
  .sidebar {{ width:210px; background:{BRAND}; color:#cdd3de; flex-shrink:0; display:flex;
    flex-direction:column; }}
  .brand {{ padding:20px 22px; font-size:17px; font-weight:800; color:#fff; letter-spacing:-.3px; }}
  .brand span {{ color:{LIGHT}; font-weight:600; font-size:12px; }}
  .nav-item {{ padding:11px 22px; font-size:13.5px; cursor:default; border-left:3px solid transparent; }}
  .nav-item.active {{ background:rgba(255,255,255,.08); color:#fff; border-left-color:{LIGHT}; font-weight:600; }}
  .sidebar-foot {{ margin-top:auto; padding:18px 22px; font-size:12px; color:#8b93a6; border-top:1px solid rgba(255,255,255,.08); }}
  .main {{ flex:1; min-width:0; }}
  .topbar {{ background:#fff; border-bottom:1px solid {LINE}; padding:16px 32px; display:flex;
    align-items:center; justify-content:space-between; }}
  .topbar h1 {{ font-size:19px; font-weight:700; color:{BRAND}; }}
  .topbar .sub {{ font-size:12.5px; color:{MUTED}; margin-top:2px; }}
  .newpo {{ background:{BLUE}; color:#fff; padding:9px 16px; border-radius:6px; font-size:13px; font-weight:600; }}
  .wrap {{ padding:24px 32px 40px; }}
  .kpis {{ display:flex; gap:16px; margin-bottom:22px; flex-wrap:wrap; }}
  .kpi {{ flex:1; min-width:170px; background:#fff; border:1px solid {LINE}; border-radius:8px; padding:16px 20px; }}
  .kpi .v {{ font-size:26px; font-weight:800; line-height:1; }}
  .kpi .l {{ font-size:11px; text-transform:uppercase; letter-spacing:.6px; color:{MUTED}; font-weight:700; margin-top:8px; }}
  .panel {{ background:#fff; border:1px solid {LINE}; border-radius:8px; overflow:hidden; }}
  .panel-head {{ padding:15px 20px; border-bottom:1px solid {LINE}; display:flex; align-items:center; justify-content:space-between; }}
  .panel-head h2 {{ font-size:15px; font-weight:700; color:{BRAND}; }}
  .panel-head .filter {{ font-size:12px; color:{MUTED}; border:1px solid {LINE}; border-radius:6px; padding:5px 10px; }}
  table {{ width:100%; border-collapse:collapse; }}
  thead th {{ text-align:left; font-size:10.5px; text-transform:uppercase; letter-spacing:.6px; color:{MUTED};
    font-weight:700; padding:11px 16px; border-bottom:1px solid {LINE}; }}
  thead th.num, td.num {{ text-align:right; }}
  tbody td {{ padding:12px 16px; border-bottom:1px solid #F1F3F6; vertical-align:top; }}
  tbody tr:last-child td {{ border-bottom:none; }}
  .item {{ font-weight:700; color:#111; font-size:13.5px; }}
  .desc {{ color:#4b5563; font-size:12.5px; margin-top:1px; }}
  .reason {{ color:{MUTED}; font-size:11.5px; margin-top:5px; }}
  .num {{ font-variant-numeric:tabular-nums; font-size:13.5px; color:#111; }}
  .unit {{ color:{MUTED}; font-size:11px; }}
  .was {{ color:{RED}; font-size:11px; }}
  .when {{ font-size:11px; color:{MUTED}; margin-top:2px; }}
  .badge {{ display:inline-block; color:#fff; font-size:10.5px; font-weight:800; letter-spacing:.4px;
    padding:3px 9px; border-radius:20px; }}
  .tag {{ display:inline-block; font-size:10px; font-weight:700; padding:1px 7px; border-radius:20px; margin-left:6px; }}
  .tag-lead {{ background:#FCEBD2; color:#8a5a06; }}
  .tag-merge {{ background:#E7E1F5; color:#4a3aa0; }}
  .foot {{ padding:16px 32px 30px; font-size:12px; color:{MUTED}; }}
</style></head>
<body>
  <div class="sidebar">
    <div class="brand">JobBOSS<span> / Purchasing</span></div>
    {nav_html}
    <div class="sidebar-foot">Buyer<br>M. Alvarez</div>
  </div>
  <div class="main">
    <div class="topbar">
      <div><h1>Reorder Queue</h1><div class="sub">Items ranked by urgency &middot; forecast demand over lead time &middot; as of {AS_OF}</div></div>
      <div class="newpo">+ New Purchase Order</div>
    </div>
    <div class="wrap">
      <div class="kpis">
        <div class="kpi"><div class="v" style="color:{RED};">{n_reorder}</div><div class="l">Reorder now</div></div>
        <div class="kpi"><div class="v" style="color:{AMBER};">{n_soon}</div><div class="l">Due soon</div></div>
        <div class="kpi"><div class="v" style="color:{GREEN};">{n_ok}</div><div class="l">OK</div></div>
        <div class="kpi"><div class="v" style="color:{BRAND};">${order_value/1000:,.0f}K</div><div class="l">Suggested order value</div></div>
        <div class="kpi"><div class="v" style="color:{BLUE};">{n_flag}</div><div class="l">Items with data fixes</div></div>
      </div>
      <div class="panel">
        <div class="panel-head"><h2>Items due for reorder</h2><div class="filter">All classes &#9662;</div></div>
        <table>
          <thead><tr>
            <th>Item</th><th class="num">On hand</th><th class="num">Forecast demand</th>
            <th class="num">Lead time</th><th class="num">Safety</th><th class="num">Suggested order</th><th>Priority</th>
          </tr></thead>
          <tbody>{body_rows}</tbody>
        </table>
      </div>
      <div class="foot">{len(rec):,} active items &middot; {n_reorder} due, {n_soon} soon &middot;
        {n_flag} items carry a resolved duplicate record or a corrected lead time &middot;
        Forecasts by BRIM Demand Model (xgboost, {wape_txt})</div>
    </div>
  </div>
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}  ({len(due)} rows shown, {n_reorder} due, {n_flag} flagged)")


if __name__ == "__main__":
    build()
