"""Inventory Analytics Diagnostic Report -> docs/reports/analytics_report.html

Audience: plant manager, operations director, finance. Answers where the inventory
dollars sit, what demand patterns the shop is buying against, how supplier lead
times are actually performing, and what a forecast-driven reorder policy releases
in working capital at the same service level.

Mirrors the Case 03 report layout (brand.py: TOC, numbered sections, KPI cards,
data tables, house palette charts). Every chart is preceded by a descriptive
paragraph that states the takeaway first.

Run from repo root:
  PYTHONIOENCODING=utf-8 "../mfg-oee-maintenance/.venv/Scripts/python.exe" -m ml.reports.generate_analytics_report
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import brand as B
from .brand import (DARK_GREY, DARK_BLUE, LIGHT_BLUE, ACCENT_RED, MUTED_RED,
                    GREEN, AMBER, MED_GREY, LIGHT_GREY, TEXT, mticker)

REPO    = Path(__file__).resolve().parents[2]
MARTS   = REPO / "ml" / "data" / "marts"
POLICY  = REPO / "ml" / "data" / "policy"
DQ      = REPO / "ml" / "data" / "data_quality"
RAW     = REPO / "data_source" / "raw" / "erp"
OUT     = REPO / "docs" / "reports" / "analytics_report.html"

WINDOW_LABEL = "September 2023 to August 2026"
# Cost assumptions used to value the policy comparison, stated in the report.
EXPEDITE_PER_EVENT = 250     # dollars per expedited replenishment
CARRYING_RATE      = 0.22    # annual carrying cost as a share of inventory value

# ── Segment planning implications (audience-facing, no jargon) ────────────────
SEGMENT_LABEL = {"smooth": "Smooth", "erratic": "Erratic",
                 "lumpy": "Lumpy", "intermittent": "Intermittent"}
SEGMENT_COLOR = {"smooth": DARK_BLUE, "erratic": LIGHT_BLUE,
                 "lumpy": AMBER, "intermittent": MED_GREY}
SEGMENT_PLANNING = {
    "smooth": "Steady, predictable draw. Statistical reorder points work well and "
              "safety stock can be trimmed toward the low end.",
    "erratic": "Regular ordering but volatile sizes. These items carry the most "
               "dollars, so a demand forecast pays off most here.",
    "lumpy": "Sporadic and large when it comes. Reorder points sized to average "
             "demand either stock out or overstock; forecasting the timing matters.",
    "intermittent": "Long gaps with small quantities. A simple min/max rule is "
                    "usually enough; heavy modeling adds little.",
}


def usd(x):
    """Compact currency: $1.4M or $438K, unit chosen by how the value rounds."""
    return f"${x / 1e6:.1f}M" if round(x / 1e3) >= 1000 else f"${x / 1e3:.0f}K"


# ── Load data ─────────────────────────────────────────────────────────────────
ia = pd.read_parquet(MARTS / "item_attributes.parquet").rename(
    columns={"canonical_item_number": "item_number"})
sp = pd.read_parquet(MARTS / "supplier_performance.parquet")
pc = pd.read_parquet(POLICY / "policy_comparison.parquet")
psum = json.loads((POLICY / "policy_summary.json").read_text(encoding="utf-8"))
dq = json.loads((DQ / "summary.json").read_text(encoding="utf-8"))
item_master = pd.read_csv(RAW / "item_master.csv")[["item_number", "primary_supplier_id"]]
suppliers = pd.read_csv(RAW / "suppliers.csv")

N_ITEMS = len(ia)
TOTAL_VALUE = float(ia["annual_value"].sum())

# ── ABC roll-up ───────────────────────────────────────────────────────────────
ABC_ORDER = ["A", "B", "C"]
abc_counts = {k: int((ia["abc"] == k).sum()) for k in ABC_ORDER}
abc_value = {k: float(ia.loc[ia["abc"] == k, "annual_value"].sum()) for k in ABC_ORDER}
abc_vshare = {k: abc_value[k] / TOTAL_VALUE for k in ABC_ORDER}
a_item_share = abc_counts["A"] / N_ITEMS

# ── Value by item class ───────────────────────────────────────────────────────
class_value = ia.groupby("item_class")["annual_value"].sum().sort_values(ascending=False)

# ── Demand segment roll-up ────────────────────────────────────────────────────
SEG_ORDER = ["smooth", "erratic", "lumpy", "intermittent"]
seg_count = {s: int((ia["segment"] == s).sum()) for s in SEG_ORDER}
seg_ishare = {s: seg_count[s] / N_ITEMS for s in SEG_ORDER}
seg_value = {s: float(ia.loc[ia["segment"] == s, "annual_value"].sum()) for s in SEG_ORDER}
seg_vshare = {s: seg_value[s] / TOTAL_VALUE for s in SEG_ORDER}

# ── Supplier lead time: master (on file) vs corrected (actual) per supplier ───
sup = ia.merge(item_master, on="item_number", how="left")
sup_lead = (sup.groupby("primary_supplier_id")
              .agg(items=("item_number", "size"),
                   master=("master_lead_time_days", "median"),
                   actual=("corrected_lead_days", "median"))
              .reset_index())
sup_lead["drift"] = sup_lead["actual"] - sup_lead["master"]
sup_lead = sup_lead.merge(sp[["supplier_id", "median_lead", "p90_lead"]],
                          left_on="primary_supplier_id", right_on="supplier_id", how="left")
sup_name = dict(zip(suppliers["supplier_id"], suppliers["supplier_name"]))

# The one supplier whose actual lead time drifted materially above the file value.
drift_row = sup_lead.sort_values("drift", ascending=False).iloc[0]
DRIFT_ID = drift_row["primary_supplier_id"]
DRIFT_NAME = sup_name.get(DRIFT_ID, DRIFT_ID)
DRIFT_MASTER = float(drift_row["master"])
DRIFT_ACTUAL = float(drift_row["actual"])
DRIFT_GAP = DRIFT_ACTUAL - DRIFT_MASTER
d2 = dq["D2"]
DRIFT_ITEMS = int(d2["items"])
DRIFT_STOCKOUTS = float(d2["est_stockouts_year"])
DRIFT_EXPEDITE = float(d2["est_expedite_cost"])
# The rest of the book: how flat every other supplier sits.
other_max_drift = float(sup_lead.loc[sup_lead["primary_supplier_id"] != DRIFT_ID, "drift"].abs().max())

# ── Three-policy comparison ───────────────────────────────────────────────────
POL = ["current", "corrected", "forecast"]
POL_LABEL = {"current": "Current reorder points",
             "corrected": "Corrected lead times",
             "forecast": "Forecast-driven policy"}
POL_COLOR = {"current": MED_GREY, "corrected": LIGHT_BLUE, "forecast": DARK_BLUE}
fill = {p: float(psum[p]["fill"]) for p in POL}
inv = {p: float(psum[p]["inv"]) for p in POL}
stockouts = {p: float(psum[p]["stockouts"]) for p in POL}
expedite = {p: float(psum[p]["expedite"]) for p in POL}

# Working capital released by the forecast policy against the corrected-lead policy,
# compared at the same (roughly 97%) service level.
wc_released = inv["corrected"] - inv["forecast"]
wc_pct = wc_released / inv["corrected"]
carrying_saved = wc_released * CARRYING_RATE
expedite_cut = expedite["current"] - expedite["forecast"]

# ── T6 phantom on-order: never-closed PO lines that fake an inbound position ───
TXN = DQ / "txn"
t6 = pd.read_parquet(TXN / "t6_phantom.parquet")
po_all = pd.read_csv(RAW / "purchase_orders.csv")[["po_id", "item_number"]]
# Resolve the raw item numbers on the phantom PO lines to canonical items so the
# item count lines up with the deduplicated catalogue used everywhere else.
_dups = pd.read_csv(DQ / "d1_duplicates.csv")
_xwalk = {}
for _, _r in _dups.iterrows():
    for _raw in ast.literal_eval(_r["item_number"]):
        _xwalk[_raw] = _r["canonical_item_number"]
_phantom_po = po_all[po_all["po_id"].isin(t6["po_id"])].copy()
_phantom_po["canonical"] = _phantom_po["item_number"].map(lambda x: _xwalk.get(x, x))
PHANTOM_PO_LINES = int(len(t6))
PHANTOM_ITEMS = int(_phantom_po["canonical"].nunique())
PHANTOM_SHARE = float(psum["phantom_stockout_share"])
PHANTOM_EVENTS = int(round(PHANTOM_SHARE * stockouts["current"]))


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def chart_abc_pareto():
    """Cumulative share of annual consumption value against cumulative share of
    items, ranked most valuable first. The A/B/C cut points are marked."""
    s = ia.sort_values("annual_value", ascending=False).reset_index(drop=True)
    cum_val = s["annual_value"].cumsum() / TOTAL_VALUE * 100
    x = (np.arange(1, len(s) + 1)) / len(s) * 100
    fig, ax = B.make_fig(h=4.0)
    ax.plot(x, cum_val, color=DARK_BLUE, linewidth=2.4, zorder=4)
    ax.plot([0, 100], [0, 100], color=MED_GREY, linestyle=":", linewidth=1.2, zorder=1)
    # A / B / C boundaries as vertical cut points at their cumulative-item share.
    a_x = abc_counts["A"] / N_ITEMS * 100
    b_x = (abc_counts["A"] + abc_counts["B"]) / N_ITEMS * 100
    ax.axvspan(0, a_x, color=DARK_BLUE, alpha=0.07, zorder=0)
    ax.axvspan(a_x, b_x, color=LIGHT_BLUE, alpha=0.10, zorder=0)
    for xc in (a_x, b_x):
        ax.axvline(xc, color=LIGHT_GREY, linewidth=1.0, zorder=2)
    ax.annotate(f"A items\n{abc_counts['A']} items ({a_x:.0f}%)\n{abc_vshare['A']:.0%} of value",
                xy=(a_x, 80), xytext=(a_x + 6, 55), fontsize=9.5, color=DARK_GREY,
                fontweight="bold", va="center")
    ax.text(a_x / 2, 90, "A", ha="center", fontsize=13, fontweight="bold", color=DARK_BLUE)
    ax.text((a_x + b_x) / 2, 90, "B", ha="center", fontsize=13, fontweight="bold", color=LIGHT_BLUE)
    ax.text((b_x + 100) / 2, 90, "C", ha="center", fontsize=13, fontweight="bold", color=MED_GREY)
    ax.scatter([a_x], [abc_vshare["A"] * 100], color=ACCENT_RED, s=40, zorder=5)
    ax.set_xlabel("Cumulative share of items (ranked by annual value)")
    ax.set_ylabel("Cumulative share of value")
    ax.set_xlim(0, 100); ax.set_ylim(0, 102)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_value_by_class():
    d = class_value.sort_values()
    fig, ax = B.make_fig(h=3.8)
    ax.barh(range(len(d)), d.values / 1e3, color=DARK_BLUE, height=0.62)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index)
    for i, v in enumerate(d.values):
        ax.text(v / 1e3 + TOTAL_VALUE / 1e3 * 0.01, i, f"{usd(v)}  ({v / TOTAL_VALUE:.0%})",
                va="center", fontsize=9.5, color=TEXT)
    ax.set_xlabel("Annual consumption value")
    ax.set_xlim(0, class_value.max() / 1e3 * 1.28)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}K"))
    B.chart_style(ax); ax.xaxis.grid(True, color=LIGHT_GREY); ax.yaxis.grid(False)
    fig.tight_layout()
    return B.b64(fig)


def chart_segment_mix():
    """Share of items versus share of annual value for each demand segment."""
    x = np.arange(len(SEG_ORDER)); w = 0.38
    ishare = [seg_ishare[s] * 100 for s in SEG_ORDER]
    vshare = [seg_vshare[s] * 100 for s in SEG_ORDER]
    fig, ax = B.make_fig(h=3.8)
    b1 = ax.bar(x - w / 2, ishare, w, color=LIGHT_BLUE, label="Share of items")
    b2 = ax.bar(x + w / 2, vshare, w, color=DARK_BLUE, label="Share of annual value")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels([SEGMENT_LABEL[s] for s in SEG_ORDER])
    ax.set_ylabel("Share")
    ax.set_ylim(0, max(max(ishare), max(vshare)) * 1.22)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(fontsize=9.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_supplier_lead():
    """Lead time on file versus the actual lead time now observed, one point per
    supplier. Points on the diagonal are performing to plan; the drift supplier
    sits well above it."""
    d = sup_lead.copy()
    is_drift = d["primary_supplier_id"] == DRIFT_ID
    fig, ax = B.make_fig(h=4.0)
    lim = max(d["master"].max(), d["actual"].max()) + 5
    ax.plot([0, lim], [0, lim], color=MED_GREY, linestyle=":", linewidth=1.3, zorder=1)
    ax.scatter(d.loc[~is_drift, "master"], d.loc[~is_drift, "actual"],
               color=DARK_BLUE, s=55, edgecolor="white", zorder=3, label="Suppliers on plan")
    ax.scatter(d.loc[is_drift, "master"], d.loc[is_drift, "actual"],
               color=ACCENT_RED, s=130, edgecolor="white", zorder=4, label="Lead time drifted upward")
    ax.annotate(f"{DRIFT_NAME}\n{DRIFT_MASTER:.0f}d on file, {DRIFT_ACTUAL:.0f}d actual",
                xy=(DRIFT_MASTER, DRIFT_ACTUAL), xytext=(DRIFT_MASTER + 6, DRIFT_ACTUAL + 1),
                fontsize=9.5, color=ACCENT_RED, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACCENT_RED, linewidth=1.0))
    ax.text(lim * 0.62, lim * 0.55, "On plan\n(actual = on file)", fontsize=9,
            color=MED_GREY, rotation=32, ha="center", va="center")
    ax.set_xlabel("Lead time on file in the ERP (days)")
    ax.set_ylabel("Actual lead time now observed (days)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=9.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_policy():
    """Two panels: fill rate and inventory value under each of the three policies."""
    import matplotlib.pyplot as plt
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(B.CHART_W, 3.9))
    x = np.arange(len(POL))
    colors = [POL_COLOR[p] for p in POL]
    # Left: fill rate.
    fills = [fill[p] * 100 for p in POL]
    axl.bar(x, fills, width=0.6, color=colors)
    axl.axhline(96, color=ACCENT_RED, linestyle="--", linewidth=1.2)
    axl.text(len(POL) - 0.5, 96.4, "96% service target", ha="right", va="bottom",
             fontsize=8.5, color=ACCENT_RED)
    for i, v in enumerate(fills):
        axl.text(i, v + 0.6, f"{v:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axl.set_ylim(0, 108); axl.set_title("Service level (fill rate)", fontsize=11, fontweight="bold")
    axl.set_xticks(x); axl.set_xticklabels([POL_LABEL[p].replace(" ", "\n", 1) for p in POL], fontsize=8.5)
    axl.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    B.chart_style(axl)
    # Right: inventory value.
    invs = [inv[p] / 1e3 for p in POL]
    axr.bar(x, invs, width=0.6, color=colors)
    for i, v in enumerate(invs):
        axr.text(i, v + max(invs) * 0.015, usd(v * 1e3), ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    axr.set_ylim(0, max(invs) * 1.20)
    axr.set_title("Inventory value", fontsize=11, fontweight="bold")
    axr.set_xticks(x); axr.set_xticklabels([POL_LABEL[p].replace(" ", "\n", 1) for p in POL], fontsize=8.5)
    axr.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}K"))
    B.chart_style(axr)
    fig.tight_layout()
    return B.b64(fig)


def chart_phantom_stockouts():
    """Current-policy stockout events split into those attributable to material
    phantom on-order (never-closed POs the ERP still counts as inbound) and all
    other causes. A single stacked bar keeps the share in view."""
    total = stockouts["current"]
    phantom = PHANTOM_EVENTS
    other = total - phantom
    fig, ax = B.make_fig(h=2.5)
    ax.barh([0], [other], color=MED_GREY, height=0.5)
    ax.barh([0], [phantom], left=[other], color=ACCENT_RED, height=0.5)
    ax.text(other / 2, 0, f"Other causes\n{other:,.0f} ({other / total:.0%})",
            ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.annotate(f"Phantom on-order\n{phantom:,.0f} ({phantom / total:.0%})",
                xy=(other + phantom / 2, 0.28), xytext=(other + phantom / 2, 0.62),
                ha="center", va="bottom", fontsize=9.5, color=ACCENT_RED, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ACCENT_RED, linewidth=1.0))
    ax.set_xlim(0, total * 1.02); ax.set_ylim(-0.5, 0.9)
    ax.set_yticks([])
    ax.set_xlabel("Annual stockout events under current reorder points")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    B.chart_style(ax); ax.yaxis.grid(False); ax.xaxis.grid(True, color=LIGHT_GREY)
    fig.tight_layout()
    return B.b64(fig)


print("Generating charts...")
charts = {
    "abc": chart_abc_pareto(),
    "class": chart_value_by_class(),
    "segment": chart_segment_mix(),
    "supplier": chart_supplier_lead(),
    "policy": chart_policy(),
    "phantom": chart_phantom_stockouts(),
}


# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════
def abc_table():
    rows = []
    for k in ABC_ORDER:
        rows.append([
            B.badge(f"Class {k}", {"A": DARK_BLUE, "B": LIGHT_BLUE, "C": MED_GREY}[k]),
            f"{abc_counts[k]:,}",
            f"{abc_counts[k] / N_ITEMS:.0%}",
            usd(abc_value[k]),
            f"{abc_vshare[k]:.0%}",
        ])
    rows.append([
        "<td style='font-weight:700;'>All items</td>",
        f"<td style='text-align:right;font-weight:700;'>{N_ITEMS:,}</td>",
        "<td style='text-align:right;font-weight:700;'>100%</td>",
        f"<td style='text-align:right;font-weight:700;'>{usd(TOTAL_VALUE)}</td>",
        "<td style='text-align:right;font-weight:700;'>100%</td>",
    ])
    return B.data_table(["Class", "Items", "Item share", "Annual value", "Value share"],
                        rows, right={1, 2, 3, 4})


def segment_table():
    rows = []
    for s in SEG_ORDER:
        rows.append([
            B.badge(SEGMENT_LABEL[s], SEGMENT_COLOR[s]),
            f"{seg_count[s]:,}",
            f"{seg_ishare[s]:.0%}",
            f"{seg_vshare[s]:.0%}",
            SEGMENT_PLANNING[s],
        ])
    return B.data_table(["Demand pattern", "Items", "Item share", "Value share",
                         "What it implies for planning"], rows, right={1, 2, 3})


def policy_table():
    def row(p, highlight=False):
        style = ' style="font-weight:700;"' if highlight else ""
        return [
            f"<td{style}>{POL_LABEL[p]}</td>",
            f'<td style="text-align:right;{"font-weight:700;" if highlight else ""}">{fill[p]:.0%}</td>',
            f'<td style="text-align:right;{"font-weight:700;" if highlight else ""}">{usd(inv[p])}</td>',
            f'<td style="text-align:right;">{stockouts[p]:,.0f}</td>',
            f'<td style="text-align:right;">{usd(expedite[p])}</td>',
        ]
    rows = [row("current"), row("corrected"), row("forecast", highlight=True)]
    return B.data_table(["Policy", "Fill rate", "Inventory value", "Annual stockouts",
                         "Annual expedite cost"], rows, right={1, 2, 3, 4})


# ══════════════════════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════════════════════
toc = (
    '<a href="#summary">Executive Summary</a><hr>'
    '<a href="#dollars">Where the Dollars Sit</a>'
    '<a href="#abc" class="sub">ABC Segmentation</a>'
    '<a href="#byclass" class="sub">Value by Item Class</a><hr>'
    '<a href="#demand">Demand Patterns</a>'
    '<a href="#supplier">Supplier Lead Times</a>'
    '<a href="#policy">The Policy Comparison</a>'
    '<a href="#phantom" class="sub">Phantom On-Order and Stockouts</a>'
)

n_a = abc_counts["A"]
top_class = class_value.index[0]

body = f"""
{B.section("summary", "Section 1", "Executive Summary")}
<p>The shop buys about <strong>{usd(TOTAL_VALUE)}</strong> of purchased material a year across
<strong>{N_ITEMS:,} active items</strong>. This report looks at where those dollars sit, what
demand patterns the shop is buying against, how supplier lead times are actually performing,
and what a forecast-driven reorder policy would release in working capital. The window is
{WINDOW_LABEL}.</p>
<p>Three findings drive the rest of the report. First, spending is highly concentrated: the
top <strong>{n_a} items ({a_item_share:.0%} of the catalog)</strong> account for
<strong>{abc_vshare['A']:.0%} of annual consumption value</strong>, so a small slice of the
book carries the working capital. Second, one supplier, <strong>{DRIFT_NAME}</strong>, has let
its actual lead time drift from <strong>{DRIFT_MASTER:.0f} days on file to about
{DRIFT_ACTUAL:.0f} days</strong>, and the stale figure quietly undersizes reorder points on
roughly <strong>{DRIFT_ITEMS} items</strong>. Third, correcting the lead times and forecasting
demand lifts the fill rate from <strong>{fill['current']:.0%} to {fill['forecast']:.0%}</strong>
while holding inventory close to today's level, and at an equal service level it releases about
<strong>{usd(wc_released)}</strong> of working capital against a corrected-lead policy that
simply buys more safety stock.</p>
{B.kpi_row(
    B.kpi_card(usd(TOTAL_VALUE), "Annual consumption value", f"across {N_ITEMS:,} items", DARK_GREY),
    B.kpi_card(usd(inv['current']), "Inventory value today", "under current reorder points", DARK_BLUE),
    B.kpi_card(f"{fill['current']:.0%}", "Fill rate today", "current policy service level", ACCENT_RED),
    B.kpi_card(usd(wc_released), "Working capital opportunity", f"released at equal service ({wc_pct:.0%})", GREEN))}

{B.section("dollars", "Section 2", "Where the Inventory Dollars Sit")}
<p>Before any forecasting, the first question for a plant manager is where the money is tied up.
The answer is a textbook Pareto: a small number of high-value items dominate consumption, and the
long tail of low-value parts barely moves the working-capital needle. That shape decides where
tighter planning is worth the effort and where a simple rule is fine.</p>

{B.section("abc", "Section 2.1", "ABC Segmentation")}
<p>Ranking every item by its annual consumption value and accumulating from the top produces the
curve below. The takeaway is the concentration: the Class A items, the top
<strong>{a_item_share:.0%} of the catalog</strong>, already account for
<strong>{abc_vshare['A']:.0%} of the value</strong>, while the entire Class C tail, nearly half
the catalog, is only <strong>{abc_vshare['C']:.0%}</strong>. Planning attention and safety-stock
dollars should follow that curve, concentrated on A and B and kept light on C.</p>
{B.chart("Annual Consumption Value Is Concentrated in a Few Items", charts["abc"])}
{abc_table()}
<p>A items ({n_a} parts) are where forecasting and lead-time accuracy earn their keep: a
percentage point of service on these items moves real dollars. C items ({abc_counts['C']} parts)
carry so little value that a plain min/max rule with a generous buffer costs almost nothing to
hold and saves planning effort for where it matters.</p>

{B.section("byclass", "Section 2.2", "Value by Item Class")}
<p>Cut the same spending by material type and it spreads more evenly than the item-level Pareto,
which tells the buying team that no single commodity dominates the book. <strong>{top_class}</strong>
is the largest single class at <strong>{class_value.iloc[0] / TOTAL_VALUE:.0%}</strong> of annual
value, followed closely by bar stock and hardware. The practical read is that supplier and
lead-time risk is diversified across several commodity groups rather than sitting behind one
vendor relationship.</p>
{B.chart("Annual Consumption Value by Item Class", charts["class"])}

{B.section("demand", "Section 3", "Demand Pattern Distribution")}
<p>Not every item should be planned the same way, because they do not sell the same way. Each part
is classified by how regularly it is consumed and how variable the quantities are, into four
patterns: smooth, erratic, lumpy, and intermittent. The chart contrasts each segment's share of
the catalog with its share of the dollars, and the gap between the two bars is the point: the
erratic and lumpy items are a minority of parts but the majority of the money, so that is where a
demand forecast, rather than a static reorder point, pays off.</p>
{B.chart("Demand Segments: Share of Items vs Share of Value", charts["segment"])}
<p>The erratic and lumpy segments together are
<strong>{(seg_ishare['erratic'] + seg_ishare['lumpy']):.0%} of items</strong> but
<strong>{(seg_vshare['erratic'] + seg_vshare['lumpy']):.0%} of the value</strong>. These are the
parts that order regularly in volatile sizes, or arrive in sporadic large lots, and a reorder
point tuned to average demand either stocks out or overstocks them. The smooth items plan cleanly
with statistical reorder points, and the intermittent tail is best left on a simple min/max rule.
The table below states the implication for each pattern.</p>
{segment_table()}

{B.section("supplier", "Section 4", "Supplier Lead-Time Performance")}
<p>Reorder points are only as good as the lead time behind them, so the next check is whether the
lead times on file still match what suppliers actually deliver. Comparing the figure stored in the
ERP against the lead time now observed at receiving, almost every supplier sits on the diagonal:
the actual lead time matches the file within about <strong>{other_max_drift:.0f} day</strong>. One
supplier is the exception. <strong>{DRIFT_NAME}</strong> is delivering in roughly
<strong>{DRIFT_ACTUAL:.0f} days</strong> against a file value of <strong>{DRIFT_MASTER:.0f} days</strong>,
a gap of <strong>{DRIFT_GAP:.0f} days</strong> that has crept up unnoticed.</p>
<p>One measurement caveat shapes how these lead times are read. Receiving tends to post
several deliveries together in end-of-week batches, so the receipt date on file lands a
little after the goods actually arrived. Left uncorrected that batching stretches the
apparent order-to-receipt gap and biases the raw lead time upward by about
<strong>two days</strong>. The figures here use a robust median across each supplier's
receipts rather than a mean, so a handful of batched postings does not pull the number,
and the drift shown below is a real change in supplier performance rather than an artifact
of how receipts are keyed.</p>
{B.chart("Lead Time on File vs Actual Lead Time by Supplier", charts["supplier"])}
<p>The drift matters because the stale {DRIFT_MASTER:.0f}-day figure feeds the reorder-point
formula on about <strong>{DRIFT_ITEMS} items</strong> bought from this vendor, so those items are
reordered too late for how the supplier actually ships today. The estimated cost of that single
stale number is roughly <strong>{DRIFT_STOCKOUTS:.0f} stockouts a year</strong> and about
<strong>{usd(DRIFT_EXPEDITE)}</strong> in expedite premiums to recover them. Correcting one lead
time in the item master removes most of that exposure, and it is the first of the two levers the
policy comparison below pulls.</p>

{B.section("policy", "Section 5", "The Three-Policy Comparison")}
<p>This is the payoff. The same {N_ITEMS:,} items were replayed over a twelve-month holdout under
three replenishment policies: today's reorder points as they stand, the same points recomputed
with the corrected lead times, and a forecast-driven policy that sizes each item from its demand
forecast. Fill rate is priced with expedite events at <strong>${EXPEDITE_PER_EVENT}</strong> each
and inventory carried at <strong>{CARRYING_RATE:.0%}</strong> a year. The comparison below holds
service roughly constant so the inventory figures are read like for like.</p>
{B.chart("Fill Rate and Inventory Value Under Each Policy", charts["policy"])}
<p>Reading the panels together tells the story. Today's policy runs a
<strong>{fill['current']:.0%}</strong> fill rate on <strong>{usd(inv['current'])}</strong> of
inventory, and pays <strong>{usd(expedite['current'])}</strong> a year in expedite premiums to
cover the gaps. Simply correcting the lead times lifts fill to
<strong>{fill['corrected']:.0%}</strong>, but it gets there by buying safety stock: inventory
climbs to <strong>{usd(inv['corrected'])}</strong>. The forecast-driven policy reaches essentially
the same service, <strong>{fill['forecast']:.0%}</strong>, on just
<strong>{usd(inv['forecast'])}</strong>, because it holds stock where demand variability actually
warrants it rather than padding every item.</p>
{policy_table()}
<p>The headline is the working-capital release. Held at the same roughly
<strong>{fill['forecast']:.0%}</strong> service level, the forecast-driven policy carries about
<strong>{usd(wc_released)}</strong> less inventory than the corrected-lead policy, a
<strong>{wc_pct:.0%}</strong> reduction, worth roughly <strong>{usd(carrying_saved)}</strong> a
year in carrying cost alone. Against today's policy it also lifts fill from
<strong>{fill['current']:.0%}</strong> to <strong>{fill['forecast']:.0%}</strong> and cuts expedite
premiums from <strong>{usd(expedite['current'])}</strong> to
<strong>{usd(expedite['forecast'])}</strong>, a saving of <strong>{usd(expedite_cut)}</strong>. In
short, better information, corrected lead times plus a demand forecast, buys higher service and
lower working capital at the same time, rather than trading one for the other.</p>

{B.section("phantom", "Section 5.1", "Phantom On-Order and Stockouts")}
<p>Part of today's stockout count is self-inflicted, and it is fixable without any
forecasting. The ledger still carries <strong>{PHANTOM_PO_LINES:,} never-closed purchase
order lines</strong> against <strong>{PHANTOM_ITEMS} items</strong>, quantities the ERP
counts as inbound even though the receipt never posted and never will. Because the
on-order position looks covered, the reorder logic sees no shortfall and holds off buying.
About <strong>{PHANTOM_EVENTS} stockout events a year</strong>, roughly
<strong>{PHANTOM_SHARE:.0%}</strong> of the <strong>{stockouts['current']:,.0f}</strong>
under today's policy, trace back to this phantom on-order rather than to a genuine demand
surprise.</p>
{B.chart("Current Stockouts Attributable to Phantom On-Order", charts["phantom"])}
<p>The read for the buying team is that closing the dead POs is a clerical cleanup, not a
planning project, yet it recovers about a tenth of the stockouts on its own. It also
compounds with the policy changes above: the fill-rate gains from corrected lead times and
a demand forecast assume the on-order position the system reports is real, so purging the
phantom lines is a prerequisite for the reorder logic to act on the shortfalls it is meant
to catch.</p>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(B.page("Inventory Analytics Diagnostic Report",
                      f"Where the dollars sit, how lead times perform, and what a forecast-driven policy releases &middot; {WINDOW_LABEL}",
                      toc, body), encoding="utf-8")
print(f"Analytics report written to {OUT}")
