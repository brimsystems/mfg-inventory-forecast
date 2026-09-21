"""KPI dashboard: the running instrument panel for inventory and supply.

A buyer-and-ops-manager scorecard built on the clean marts. Header scorecard
tiles carry the headline service and inventory numbers; compact charts below
show the monthly and weekly consumption value trend, service by ABC class,
supplier lead-time performance, and the demand mix. Clean current state only,
no cleaning material. Writes docs/reports/dashboard.html.

Run from the repo root:
  PYTHONIOENCODING=utf-8 "../mfg-oee-maintenance/.venv/Scripts/python.exe" -m ml.reports.generate_dashboard
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from . import brand as B

# -- Paths --------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
MARTS = REPO / "ml" / "data" / "marts"
POLICY = REPO / "ml" / "data" / "policy"
OUT = REPO / "docs" / "reports" / "dashboard.html"

# -- Load clean marts ---------------------------------------------------------
ia = pd.read_parquet(MARTS / "item_attributes.parquet")
cm = pd.read_parquet(MARTS / "consumption_monthly.parquet")
cw = pd.read_parquet(MARTS / "consumption_weekly.parquet")
sp = pd.read_parquet(MARTS / "supplier_performance.parquet")
pc = pd.read_parquet(POLICY / "policy_comparison.parquet")
ps = json.load(open(POLICY / "policy_summary.json"))

COST = ia.set_index("canonical_item_number")["standard_cost"]
cm["month"] = pd.to_datetime(cm["month"])
cw["week"] = pd.to_datetime(cw["week"])
cm["value"] = cm["consumption"] * cm["canonical"].map(COST)
cw["value"] = cw["quantity"] * cw["canonical"].map(COST)

WINDOW = f"{cm['month'].min():%b %Y} to {cm['month'].max():%b %Y}"
AS_OF = f"{cm['month'].max():%B %Y}"

# -- Headline figures (current clean state) -----------------------------------
INV_VALUE = float(ps["current"]["inv"])                 # on-hand valuation
FILL = float(ps["current"]["fill"])                     # current fill rate
STOCKOUTS = int(pc["current_stockouts"].sum())          # annual stockout events
EXPEDITES = int(pc["current_expedites"].sum())          # annual expedite events
EXPEDITE_COST = float(ps["current"]["expedite"])        # annual expedite spend
ANNUAL_CONS_VALUE = float(ia["annual_value"].sum())     # annual consumption value
TURNS = ANNUAL_CONS_VALUE / INV_VALUE                   # basis stated on the tile
LEAD_MEDIAN = float(np.average(sp["median_lead"], weights=sp["pos"]))  # order-weighted
LEAD_P90 = float(np.average(sp["p90_lead"], weights=sp["pos"]))
FORECAST_FILL = float(ps["forecast"]["fill"])
FORECAST_STOCKOUTS = int(ps["forecast"]["stockouts"])

N_ITEMS = len(ia)
N_SUPPLIERS = len(sp)


# -- Chart helpers ------------------------------------------------------------
def _thousands(v, _):
    return f"${v/1000:,.0f}K"


def service_band(v):
    return B.GREEN if v >= 0.95 else B.AMBER if v >= 0.90 else B.ACCENT_RED


def chart_consumption_monthly():
    """Consumption value by month, trailing 24 months."""
    d = cm.groupby("month")["value"].sum().sort_index().tail(24)
    fig, ax = B.make_fig(h=3.4)
    ax.plot(d.index, d.values, color=B.DARK_BLUE, lw=2.0, marker="o", ms=3.5,
            label="Monthly consumption value")
    avg = d.tail(12).mean()
    ax.axhline(avg, color=B.MED_GREY, lw=1.3, ls="--",
               label=f"Trailing 12-month average (${avg/1000:,.0f}K)")
    ax.set_ylim(0, d.max() * 1.15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_thousands))
    ax.set_xticks(d.index[::3])
    ax.set_xticklabels([f"{pd.Timestamp(x):%b'%y}" for x in d.index[::3]])
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    B.chart_style(ax)
    return B.b64(fig)


def chart_consumption_weekly():
    """Consumption value by week, most recent 13 weeks."""
    d = cw.groupby("week")["value"].sum().sort_index().tail(13)
    fig, ax = B.make_fig(h=3.4)
    ax.bar(range(len(d)), d.values, color=B.LIGHT_BLUE, width=0.68,
           label="Weekly consumption value")
    avg = d.mean()
    ax.axhline(avg, color=B.DARK_GREY, lw=1.3, ls="--",
               label=f"13-week average (${avg/1000:,.0f}K)")
    ax.set_ylim(0, d.max() * 1.18)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_thousands))
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([f"{pd.Timestamp(x):%m/%d}" for x in d.index], rotation=45, ha="right")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False)
    B.chart_style(ax)
    return B.b64(fig)


def chart_fill_by_abc():
    """Cost-weighted fill rate by ABC class, current state."""
    g = pc.groupby("abc").apply(
        lambda x: np.average(x["current_fill"], weights=x["cost"] + 1e-9),
        include_groups=False)
    order = ["A", "B", "C"]
    vals = np.array([g[a] * 100 for a in order])
    counts = ia["abc"].value_counts()
    fig, ax = B.make_fig(h=3.4)
    x = np.arange(len(order))
    bars = ax.bar(x, vals, color=[service_band(v / 100) for v in vals], width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.0f}%",
                ha="center", va="bottom", fontweight="bold", color=B.TEXT)
    ax.axhline(95, color=B.MED_GREY, lw=1.2, ls="--", label="95% service target")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n({counts[a]} items)" for a in order])
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    B.chart_style(ax)
    return B.b64(fig)


def chart_supplier_lead():
    """Median vs p90 lead time for the ten highest-volume suppliers."""
    top = sp.sort_values("pos", ascending=False).head(10).copy()
    y = np.arange(len(top))
    fig, ax = B.make_fig(h=3.4)
    h = 0.38
    ax.barh(y - h / 2, top["median_lead"], height=h, color=B.DARK_BLUE, label="Median lead")
    ax.barh(y + h / 2, top["p90_lead"], height=h, color=B.LIGHT_BLUE, label="90th percentile lead")
    ax.set_yticks(y)
    ax.set_yticklabels(top["supplier_id"])
    ax.invert_yaxis()
    ax.set_xlabel("Lead time (days)")
    for yy, (m, p) in enumerate(zip(top["median_lead"], top["p90_lead"])):
        ax.text(p + 1, yy + h / 2, f"{p:.0f}", va="center", fontsize=9, color=B.TEXT)
        ax.text(m + 1, yy - h / 2, f"{m:.0f}", va="center", fontsize=9, color=B.TEXT)
    ax.set_xlim(0, top["p90_lead"].max() * 1.15)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    B.chart_style(ax)
    ax.xaxis.grid(True, color=B.LIGHT_GREY); ax.yaxis.grid(False)
    return B.b64(fig)


def chart_class_share():
    """Share of annual consumption value by item class."""
    d = ia.groupby("item_class")["annual_value"].sum().sort_values()
    share = d / d.sum() * 100
    fig, ax = B.make_fig(h=3.4)
    ax.barh(range(len(d)), share.values, color=B.DARK_BLUE, height=0.66)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.index)
    for i, v in enumerate(share.values):
        ax.text(v + 0.4, i, f"{v:.0f}%", va="center", color=B.TEXT)
    ax.set_xlim(0, share.max() * 1.16)
    ax.set_xlabel("Share of annual consumption value")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    B.chart_style(ax)
    ax.xaxis.grid(True, color=B.LIGHT_GREY); ax.yaxis.grid(False)
    return B.b64(fig)


def chart_segment_share():
    """Share of items by demand pattern segment."""
    order = ["smooth", "erratic", "lumpy", "intermittent"]
    counts = ia["segment"].value_counts()
    vals = np.array([counts.get(s, 0) for s in order], dtype=float)
    share = vals / vals.sum() * 100
    colors = [B.DARK_BLUE, B.LIGHT_BLUE, B.AMBER, B.MED_GREY]
    fig, ax = B.make_fig(h=3.4)
    bars = ax.bar(range(len(order)), share, color=colors, width=0.62)
    for bar, v, n in zip(bars, share, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.0f}%",
                ha="center", va="bottom", fontweight="bold", color=B.TEXT)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([s.title() for s in order])
    ax.set_ylim(0, share.max() * 1.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("Share of active items")
    B.chart_style(ax)
    return B.b64(fig)


charts = {
    "consumption_monthly": chart_consumption_monthly(),
    "consumption_weekly": chart_consumption_weekly(),
    "fill_abc": chart_fill_by_abc(),
    "supplier_lead": chart_supplier_lead(),
    "class_share": chart_class_share(),
    "segment_share": chart_segment_share(),
}


def img(k):
    return (f'<img src="data:image/png;base64,{charts[k]}" '
            f'style="width:100%;height:auto;display:block;">')


def cell(title, key):
    return f'<div class="chart-cell"><div class="chart-title">{title}</div>{img(key)}</div>'


# -- Scorecard tiles ----------------------------------------------------------
def tile(value, label, sub, color):
    return (f'<div class="tile"><div class="tile-v" style="color:{color};">{value}</div>'
            f'<div class="tile-l">{label}</div><div class="tile-s">{sub}</div></div>')


fill_color = service_band(FILL)
lead_color = B.AMBER if LEAD_MEDIAN >= 21 else B.GREEN

tiles = [
    tile(f"${INV_VALUE/1e6:.2f}M", "Inventory Value",
         f"on-hand valuation across {N_ITEMS} active items", B.DARK_GREY),
    tile(f"{TURNS:.1f}x", "Inventory Turns",
         "annual consumption value / average inventory value", B.DARK_BLUE),
    tile(f"{FILL*100:.0f}%", "Fill Rate",
         f"line-level service, vs {FORECAST_FILL*100:.0f}% under forecast policy", fill_color),
    tile(f"{STOCKOUTS:,}", "Stockout Events",
         f"per year, vs {FORECAST_STOCKOUTS:,} under forecast policy", B.ACCENT_RED),
    tile(f"{EXPEDITES:,}", "Expedite Frequency",
         f"per year, about ${EXPEDITE_COST/1000:,.0f}K at $250 / event", B.AMBER),
    tile(f"{LEAD_MEDIAN:.0f} days", "Supplier Lead Time",
         f"order-weighted median, p90 {LEAD_P90:.0f} days", lead_color),
]

N_TILES = len(tiles)
N_CHARTS = len(charts)

# -- Page ---------------------------------------------------------------------
CSS = f"""
  *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background:#fff; color:{B.TEXT}; font-size:15px; }}
  .header {{ background:{B.DARK_GREY}; color:#fff; padding:14px 0; }}
  .header .wrap {{ max-width:1400px; margin:0 auto; padding:0 48px;
    display:flex; align-items:baseline; justify-content:space-between; }}
  .header h1 {{ font-size:21px; font-weight:700; letter-spacing:-0.3px; }}
  .header .asof {{ font-size:13px; color:{B.LIGHT_GREY}; }}
  .doc {{ max-width:1400px; margin:0 auto; padding:16px 48px 56px; }}
  .section-band {{ font-size:17px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase;
    color:{B.TEXT}; border-bottom:2px solid {B.DARK_GREY}; padding:12px 0 6px; margin:30px 0 16px; }}
  .section-band.first {{ margin-top:10px; padding-top:0; }}
  .tiles {{ display:grid; grid-template-columns:repeat(6, minmax(0,1fr)); gap:16px; }}
  .tile {{ background:{B.BG_GREY}; border-radius:8px; padding:18px 20px 16px;
    border-bottom:4px solid {B.DARK_GREY}; }}
  .tile-v {{ font-size:30px; font-weight:700; line-height:1.05; margin-bottom:6px; }}
  .tile-l {{ font-size:12px; color:{B.DARK_GREY}; font-weight:700; text-transform:uppercase;
    letter-spacing:.5px; }}
  .tile-s {{ font-size:12px; color:{B.MED_GREY}; margin-top:5px; line-height:1.4; }}
  .chart-grid {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr);
    gap:10px 32px; margin-bottom:6px; }}
  .chart-cell img {{ width:100%; height:auto; display:block; }}
  .chart-title {{ font-size:16px; font-weight:700; text-transform:uppercase; letter-spacing:.5px;
    color:{B.DARK_GREY}; margin:16px 0 8px; }}
  .footnote {{ font-size:13px; color:{B.TEXT}; margin-top:6px; font-style:italic; }}
  .legend {{ font-size:13px; color:{B.TEXT}; margin-top:10px; }}
  .legend .box {{ display:inline-block; width:12px; height:12px; margin-right:5px;
    vertical-align:middle; border-radius:2px; }}
  @page {{ size:landscape; margin:8mm; }}
  @media (max-width:1100px) {{ .tiles {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} }}
  @media print {{ .header .wrap, .doc {{ max-width:none; }}
    .chart-cell, .tile {{ break-inside:avoid; }} .section-band {{ break-before:page; }}
    .section-band.first {{ break-before:auto; }} }}
"""

b = lambda c: f'<span class="box" style="background:{c};"></span>'
service_legend = (
    '<div class="legend"><strong>Service band:</strong> '
    f'{b(B.GREEN)}On target &ge;95%&nbsp;&nbsp; {b(B.AMBER)}Watch 90-95%&nbsp;&nbsp; '
    f'{b(B.ACCENT_RED)}Below target &lt;90%</div>')

HTML = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Inventory and Supply Dashboard</title>
<style>{CSS}</style></head>
<body>
<div class="header"><div class="wrap">
  <h1>Inventory and Supply Dashboard</h1>
  <div class="asof">Current state as of {AS_OF} &middot; {N_ITEMS} active items &middot; {N_SUPPLIERS} suppliers</div>
</div></div>
<div class="doc">

  <div class="section-band first">Inventory and Service Scorecard</div>
  <div class="tiles">{''.join(tiles)}</div>
  {service_legend}
  <div class="footnote">Turns are annual consumption value (${ANNUAL_CONS_VALUE/1e6:.1f}M) divided by
    average inventory value (${INV_VALUE/1e6:.2f}M). Figures reflect the current purchasing policy on
    the clean item master over the window {WINDOW}.</div>

  <div class="section-band">Consumption Value Trend</div>
  <div class="chart-grid">
    {cell("Monthly Consumption Value: Trailing 24 Months", "consumption_monthly")}
    {cell("Weekly Consumption Value: Recent 13 Weeks", "consumption_weekly")}
  </div>
  <div class="footnote">Consumption is valued at each item's standard cost. Monthly demand holds near the
    trailing 12-month average of ${cm.groupby('month')['value'].sum().tail(12).mean()/1000:,.0f}K,
    with the expected week-to-week swing on the erratic and lumpy items.</div>

  <div class="section-band">Service and Supplier Performance</div>
  <div class="chart-grid">
    {cell("Fill Rate by ABC Class", "fill_abc")}
    {cell("Supplier Lead Time: Median vs P90", "supplier_lead")}
  </div>
  <div class="footnote">Fill rate is cost-weighted within each class. A-class items, the highest-value
    lines, sit furthest below the 95% target, so they carry the most service risk. Lead-time spread
    between the median and the 90th percentile marks the suppliers where safety stock is doing the
    most work.</div>

  <div class="section-band">Demand Mix</div>
  <div class="chart-grid">
    {cell("Consumption Value Share by Item Class", "class_share")}
    {cell("Active Items by Demand Pattern", "segment_share")}
  </div>
  <div class="footnote">Fasteners and bar stock together account for roughly 40% of consumption value.
    Only about 30% of items show smooth demand; the remainder are erratic, lumpy, or intermittent, which
    is where forecasting and safety-stock policy earn their keep.</div>

</div>
</body></html>"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML, encoding="utf-8")
print(f"KPI dashboard written to {OUT}")
print(f"  Tiles: {N_TILES}  Charts: {N_CHARTS}")
print(f"  Inv ${INV_VALUE/1e6:.2f}M  Turns {TURNS:.1f}x  Fill {FILL*100:.0f}%  "
      f"Stockouts {STOCKOUTS:,}  Expedites {EXPEDITES:,}  Lead {LEAD_MEDIAN:.0f}d")
