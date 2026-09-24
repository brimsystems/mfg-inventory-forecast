"""ML Model Overview and Performance Report -> docs/reports/model_overview.html

Plain-language report on the demand model and on what six months of running the
shop on it delivered. The forward window (1H26) is compared two ways: against
the two halves of 2025 as the shop actually ran them, and against the same
1H26 demand replayed with nothing fixed and with the cleaned records but no
model. Every figure is read from the generation run and the backtest.

    PYTHONIOENCODING=utf-8 "../mfg-oee-maintenance/.venv/Scripts/python.exe" -m ml.reports.generate_model_overview
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import brand as B
from .brand import DARK_BLUE, LIGHT_BLUE, MED_GREY, GREEN, DARK_GREY

REPO = Path(__file__).resolve().parents[2]
BACKTEST = REPO / "ml" / "data" / "backtest"
POLICY = REPO / "ml" / "data" / "policy"
TRUTH = REPO / "data_source" / "truth"
OUT = REPO / "docs" / "reports" / "model_overview.html"

# ── data ──────────────────────────────────────────────────────────────────────
metrics = json.loads((BACKTEST / "model_metrics.json").read_text(encoding="utf-8"))
tw = json.loads((BACKTEST / "threeway_overall.json").read_text(encoding="utf-8"))
sched = json.loads((POLICY / "rop_schedule.json").read_text(encoding="utf-8"))
fr = json.loads((TRUTH / "forward_results.json").read_text(encoding="utf-8"))
cross = json.loads((TRUTH / "crosswalks.json").read_text(encoding="utf-8"))
txd = json.loads((TRUTH / "txn_defects.json").read_text(encoding="utf-8"))

REC = fr["as_recorded"]
FW = fr["forward"]
H1, H2 = REC["1H25"], REC["2H25"]
F = FW["1H26"]
M13, M46 = FW["1H26_months_1_3"], FW["1H26_months_4_6"]
VARIANTS = [("dirty", "Nothing fixed"), ("clean_rule", "Cleaned records, recomputed rule"), ("model", "Cleaned records, demand model")]
SEG_ORDER = ["smooth", "erratic", "lumpy", "intermittent"]


def _wape_on(name, items=None):
    t = pd.read_parquet(BACKTEST / f"threeway_{name}.parquet")
    if items is not None:
        t = t[t["item"].isin(items)]
    return float((t["target"] - t["pred"]).abs().sum() / t["target"].abs().sum())


prim = {c["primary"] for c in cross["duplicate_clusters"].values()}
unrec = {r["item_number"] for r in txd["t1"]}
repaired = prim | unrec
tw_rep = {k: _wape_on(k, repaired) for k in ["raw", "master", "fully"]}
tw_dup = {k: _wape_on(k, prim) for k in ["raw", "fully"]}


def money(x):
    return f"${x:,.0f}"


def k(x):
    return f"${x/1e6:.2f}M" if abs(x) >= 1e6 else f"${x/1e3:.0f}K"


def pct(x, d=1):
    return f"{x*100:.{d}f}%"


def widths(table_html, w):
    cols = "".join(f'<col style="width:{v}%;">' for v in w)
    return table_html.replace('<table class="data-table">',
                              f'<table class="data-table" style="table-layout:fixed;"><colgroup>{cols}</colgroup>', 1)


def sub(t):
    return f'<p style="font-size:18px;font-weight:700;color:{DARK_GREY};margin-top:30px;">{t}</p>'


# ── charts ────────────────────────────────────────────────────────────────────
def chart_halves():
    """Stockout events, days short and rush spend by half-year, as the shop ran it."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(B.CHART_W, 3.2))
    labels = ["1H25", "2H25", "1H26"]
    series = [H1, H2, F["model"]]
    panels = [("Stockout events", [s["stockout_episodes"] for s in series], "{:,.0f}"),
              ("Jobs held for material", [s["jobs_delayed"] for s in series], "{:,.0f}"),
              ("Rush spend ($000)", [s["rush_spend"] / 1000 for s in series], "${:,.0f}K")]
    for ax, (title, vals, fmt) in zip(axes, panels):
        bars = ax.bar(labels, vals, color=[MED_GREY, MED_GREY, DARK_BLUE], width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.02, fmt.format(v), ha="center", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=10, color=DARK_GREY, pad=8)
        ax.set_ylim(0, max(vals) * 1.22); ax.set_yticks([])
        B.chart_style(ax)
    fig.tight_layout(w_pad=2.0)
    return B.b64(fig)


def chart_variants():
    """The same 1H26 demand three ways, in steady state (months 4 to 6)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(B.CHART_W, 3.3))
    names = ["Nothing\nfixed", "Cleaned,\nrule", "Cleaned,\nmodel"]
    cols = [MED_GREY, LIGHT_BLUE, DARK_BLUE]
    vals = [M46[v] for v, _ in VARIANTS]
    panels = [("A-class fill rate", [x["fill_rate_by_abc"]["A"] * 100 for x in vals], "{:.1f}%", True),
              ("Stockout events", [x["stockout_episodes"] for x in vals], "{:,.0f}", False),
              ("Average inventory ($000)", [x["avg_inventory_value"] / 1000 for x in vals], "${:,.0f}K", False)]
    for ax, (title, v, fmt, zoom) in zip(axes, panels):
        bars = ax.bar(names, v, color=cols, width=0.62)
        lo = min(v) - 3 if zoom else 0
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x + (max(v) - lo) * 0.02, fmt.format(x), ha="center", fontsize=9, fontweight="bold")
        if zoom:
            ax.axhline(98, color=GREEN, linewidth=1, linestyle="--"); ax.text(-0.45, 98.15, "target 98%", fontsize=8, color=GREEN, ha="left", va="bottom")
        ax.set_title(title, fontsize=10, color=DARK_GREY, pad=8)
        ax.set_ylim(lo, max(v) + (max(v) - lo) * 0.18); ax.set_yticks([]); ax.tick_params(axis="x", labelsize=8.5)
        B.chart_style(ax)
    fig.tight_layout(w_pad=2.0)
    return B.b64(fig)


def chart_accuracy():
    """Forecast error at three stages of cleaning, all items and the repaired items."""
    fig, ax = B.make_fig(3.3)
    groups = ["All live items", f"Items the cleanup repaired ({len(repaired)})"]
    stages = [("As recorded", "raw", MED_GREY), ("Records merged", "master", LIGHT_BLUE), ("Fully cleaned", "fully", DARK_BLUE)]
    x = np.arange(2); w = 0.26
    for i, (lab, key, col) in enumerate(stages):
        v = [tw[key] * 100, tw_rep[key] * 100]
        bars = ax.bar(x + (i - 1) * w, v, width=w, color=col, label=lab)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.8, f"{val:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Forecast error, WAPE (%)"); ax.set_ylim(0, max(tw["raw"], tw_rep["raw"]) * 100 * 1.25)
    B.chart_style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=3, fontsize=9)
    return B.b64(fig)


# ── tables ────────────────────────────────────────────────────────────────────
def halves_table():
    rows = []
    m = F["model"]
    spec = [
        ("Fill rate", lambda s: pct(s["fill_rate"])),
        ("Fill rate, A items", lambda s: pct(s["fill_rate_by_abc"]["A"])),
        ("Fill rate, B items", lambda s: pct(s["fill_rate_by_abc"]["B"])),
        ("Fill rate, C items", lambda s: pct(s["fill_rate_by_abc"]["C"])),
        ("Stockout events", lambda s: f"{s['stockout_episodes']:,}"),
        ("Stockout days (item-days short)", lambda s: f"{s['stockout_days']:,}"),
        ("Jobs held for material", lambda s: f"{s['jobs_delayed']:,}"),
        ("Days lost on held jobs", lambda s: f"{s['job_delay_days']:,}"),
        ("Rush order lines", lambda s: f"{s['rush_lines']:,}"),
        ("Rush freight and premiums", lambda s: money(s["rush_spend"])),
        ("Average inventory value", lambda s: money(s["avg_inventory_value"])),
        ("Days of supply", lambda s: f"{s['days_of_supply']:.0f}"),
        ("Order lines placed", lambda s: f"{s['order_lines']:,}"),
        ("Purchases", lambda s: money(s["purchases"])),
    ]
    for label, fn in spec:
        rows.append([label, fn(H1), fn(H2), fn(m)])
    return widths(B.data_table(["Measure", "1H25", "2H25", "1H26"], rows, right=[1, 2, 3]), [37, 21, 21, 21])


def variants_table(window):
    rows = []
    spec = [
        ("Fill rate", lambda s: pct(s["fill_rate"])),
        ("Fill rate, A / B / C", lambda s: " / ".join(pct(s["fill_rate_by_abc"][a]) for a in "ABC")),
        ("Stockout events", lambda s: f"{s['stockout_episodes']:,}"),
        ("Stockout days, A items", lambda s: f"{s['stockout_days_by_abc']['A']:,}"),
        ("Jobs held for material", lambda s: f"{s['jobs_delayed']:,}"),
        ("Rush freight and premiums", lambda s: money(s["rush_spend"])),
        ("Average inventory value", lambda s: money(s["avg_inventory_value"])),
        ("Days of supply", lambda s: f"{s['days_of_supply']:.0f}"),
        ("Safety stock (as set)", lambda s: money(s["safety_stock_value"])),
        ("Cycle stock value", lambda s: money(s["cycle_stock_value"])),
        ("Excess stock (over 12 months of supply)", lambda s: f"{money(s['excess_value'])} ({s['excess_items']} items)"),
        ("Order lines placed", lambda s: f"{s['order_lines']:,}"),
        ("Purchases", lambda s: money(s["purchases"])),
        ("Consumption at cost", lambda s: money(s["consumption_value"])),
    ]
    for label, fn in spec:
        rows.append([label] + [fn(window[v]) for v, _ in VARIANTS])
    return widths(B.data_table(["Measure"] + [n for _, n in VARIANTS], rows, right=[1, 2, 3]), [31, 23, 23, 23])


def purchases_table():
    months = sorted(F["model"]["purchases_by_month"])
    rows = []
    for m in months:
        rows.append([pd.Timestamp(m).strftime("%b %Y")] + [money(F[v]["purchases_by_month"].get(m, 0.0)) for v, _ in VARIANTS])
    return widths(B.data_table(["Month"] + [n for _, n in VARIANTS], rows, right=[1, 2, 3]), [22, 26, 26, 26])


def accuracy_table():
    seg = {s["segment"]: s for s in metrics["segments"]}
    fbias = F["model"].get("forecast_bias_by_segment") or {}
    rows = []
    for s in SEG_ORDER:
        if s not in seg:
            continue
        r = seg[s]
        fb = fbias.get(s)
        rows.append([s.capitalize(), pct(r["wape_model"], 0), pct(r["wape_baseline"], 0), f"{r['lift']*100:+.0f}%",
                     f"{r['bias']*100:+.0f}%", f"{fb*100:+.0f}%" if fb is not None else "n/a"])
    rows.append(["<strong>All items</strong>", f"<strong>{pct(metrics['overall']['model'], 0)}</strong>",
                 f"<strong>{pct(metrics['overall']['baseline'], 0)}</strong>",
                 f"<strong>{(metrics['overall']['baseline']-metrics['overall']['model'])/metrics['overall']['baseline']*100:+.0f}%</strong>", "", ""])
    return widths(B.data_table(["Demand pattern", "Model error", "Best simple method", "Improvement",
                                "Bias before correction", "Bias in 1H26, corrected"], rows, right=[1, 2, 3, 4, 5]),
                  [20, 15, 18, 15, 16, 16])


# ── report ────────────────────────────────────────────────────────────────────
mod, rule, dirty = F["model"], F["clean_rule"], F["dirty"]
mod46, rule46, dirty46 = M46["model"], M46["clean_rule"], M46["dirty"]
k_abc, z = sched["safety_factor"], sched["normal_z"]
bias = sched["bias_correction"]
churn = sched["churn"]

attrs = pd.read_parquet(REPO / "ml" / "data" / "marts" / "item_attributes.parquet")
monthly = pd.read_parquet(REPO / "ml" / "data" / "marts" / "consumption_monthly.parquet")
seg_n = attrs["segment"].value_counts().to_dict(); abc_n = attrs["abc"].value_counts().to_dict()
n_items = len(attrs)
cand = metrics["candidates"]
CAND_LABEL = {"Linear": "Linear regression (ridge)", "RandomForest": "Random forest", "XGBoost": "XGBoost"}


def candidate_table():
    rows = [[CAND_LABEL[c], pct(cand[c]["val_wape"]), pct(cand[c]["test_wape"]),
             "Selected" if c == metrics["winner"] else ""] for c in ["Linear", "RandomForest", "XGBoost"]]
    return widths(B.data_table(["Candidate", "Validation error (WAPE)", "Test error (WAPE)", ""], rows, right=[1, 2]),
                  [34, 24, 24, 18])


hist = monthly[monthly["month"] < "2026-01-01"].merge(
    attrs[["canonical_item_number", "segment", "abc", "standard_cost"]], left_on="canonical", right_on="canonical_item_number")
hist["value"] = hist["consumption"] * hist["standard_cost"].fillna(attrs["standard_cost"].median())
SEG_COL = {"smooth": DARK_BLUE, "erratic": LIGHT_BLUE, "lumpy": "#8093A4", "intermittent": "#D5DCE1"}


def chart_history_value():
    """Monthly consumption at standard cost over the three years, by demand pattern."""
    fig, ax = B.make_fig(3.6)
    w = hist.pivot_table(index="month", columns="segment", values="value", aggfunc="sum").fillna(0)[SEG_ORDER] / 1000
    ax.stackplot(w.index, [w[c] for c in SEG_ORDER], colors=[SEG_COL[c] for c in SEG_ORDER],
                 labels=[c.capitalize() for c in SEG_ORDER], alpha=0.95)
    tot = w.sum(axis=1); x = np.arange(len(tot)); fit = np.polyfit(x, tot.to_numpy(), 1)
    ax.plot(w.index, np.polyval(fit, x), color=DARK_GREY, linestyle="--", linewidth=1.2, label="Trend")
    ax.set_ylabel("Consumption at cost ($000 / month)")
    B.chart_style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=5, fontsize=9)
    return B.b64(fig)


def chart_examples():
    """One representative item per demand pattern, 36 months of monthly consumption."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(B.CHART_W, 2.6), sharey=False)
    for ax, seg in zip(axes, SEG_ORDER):
        g = hist[hist["segment"] == seg]
        tot = g.groupby("canonical")["consumption"].sum()
        item = tot.sort_values().index[int(len(tot) * 0.6)]          # a typical, not an extreme, item
        sr = g[g["canonical"] == item].set_index("month")["consumption"].reindex(
            pd.date_range("2023-01-01", "2025-12-01", freq="MS"), fill_value=0)
        ax.bar(sr.index, sr.values, width=20, color=SEG_COL[seg] if seg != "intermittent" else "#8093A4")
        ax.set_title(f"{seg.capitalize()}\n{item}", fontsize=9, color=DARK_GREY)
        ax.tick_params(axis="x", labelsize=7, rotation=0); ax.tick_params(axis="y", labelsize=7)
        ax.xaxis.set_major_locator(__import__("matplotlib.dates", fromlist=["YearLocator"]).YearLocator())
        ax.xaxis.set_major_formatter(__import__("matplotlib.dates", fromlist=["DateFormatter"]).DateFormatter("%Y"))
        B.chart_style(ax)
    axes[0].set_ylabel("Units / month", fontsize=8)
    fig.tight_layout(w_pad=1.2)
    return B.b64(fig)


def chart_pareto():
    """Share of consumption value against share of items, with the ABC cut-offs."""
    fig, ax = B.make_fig(3.3)
    v = hist[hist["month"] >= "2025-01-01"].groupby("canonical")["value"].sum().sort_values(ascending=False)
    cum = v.cumsum() / v.sum() * 100; share = np.arange(1, len(v) + 1) / len(v) * 100
    ax.plot(share, cum.values, color=DARK_BLUE, linewidth=2)
    nA, nB = abc_n.get("A", 0), abc_n.get("B", 0)
    for n_, lab in ((nA, "A"), (nA + nB, "B")):
        x_ = n_ / len(v) * 100; y_ = float(cum.iloc[min(n_, len(v)) - 1])
        ax.axvline(x_, color=MED_GREY, linestyle=":", linewidth=1)
        ax.text(x_ + 1, 8, f"{lab} items end\n{x_:.0f}% of items, {y_:.0f}% of value", fontsize=8, color=DARK_GREY)
    ax.set_xlabel("Share of items, highest consumption value first (%)"); ax.set_ylabel("Cumulative share of value (%)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 102)
    B.chart_style(ax)
    return B.b64(fig)


FLOW_HTML = (
    '<div style="display:flex;align-items:stretch;gap:0;margin:22px 0;flex-wrap:wrap;">'
    '<div style="flex:1;min-width:190px;background:#F3F5F7;border-radius:8px;padding:16px 18px;border-top:4px solid #381FA1;">'
    '<div style="font-weight:700;color:#322B4B;margin-bottom:6px;">1. What it reads</div>'
    '<div style="font-size:16px;line-height:1.55;">Three years of cleaned consumption history for every stocked item, '
    'with its demand pattern, value class, cost and corrected supplier lead time.</div></div>'
    '<div style="align-self:center;font-size:26px;color:#8093A4;padding:0 12px;">&rarr;</div>'
    '<div style="flex:1;min-width:190px;background:#F3F5F7;border-radius:8px;padding:16px 18px;border-top:4px solid #381FA1;">'
    '<div style="font-weight:700;color:#322B4B;margin-bottom:6px;">2. What it predicts</div>'
    '<div style="font-size:16px;line-height:1.55;">Once a month, one forecast per item: how many units the shop will use '
    'before a new order placed today could arrive.</div></div>'
    '<div style="align-self:center;font-size:26px;color:#8093A4;padding:0 12px;">&rarr;</div>'
    '<div style="flex:1;min-width:190px;background:#F3F5F7;border-radius:8px;padding:16px 18px;border-top:4px solid #381FA1;">'
    '<div style="font-weight:700;color:#322B4B;margin-bottom:6px;">3. What it produces</div>'
    '<div style="font-size:16px;line-height:1.55;">A reorder point per item, the forecast plus a safety buffer, loaded into '
    'the ERP, which checks every item against it daily and flags what to order.</div></div></div>')

import base64
_png = REPO / "docs" / "screenshots" / "erp_queue.png"
erp_b64 = base64.b64encode(_png.read_bytes()).decode() if _png.exists() else ""

charts = {"halves": chart_halves(), "variants": chart_variants(), "acc": chart_accuracy(),
          "hist": chart_history_value(), "examples": chart_examples(), "pareto": chart_pareto()}

toc = ('<a href="#summary">Executive Summary</a><hr>'
       '<a href="#modeloverview">Model Overview</a>'
       '<a href="#what" class="sub">What This Model Does</a>'
       '<a href="#data" class="sub">Training Data Overview</a><hr>'
       '<a href="#performance">Model Performance</a>'
       '<a href="#scoring" class="sub">Scoring Summary</a>'
       '<a href="#accuracy" class="sub">Accuracy and Validation</a>'
       '<a href="#source" class="sub">Where the Improvement Came From</a>'
       '<a href="#limits" class="sub">What It Can and Cannot Predict</a>')

body = f"""
{B.section("summary", "Section 1", "Executive Summary")}
<p>From January through June 2026 the shop ran its purchasing on the cleaned records and on the model's reorder
points. Against the first half of 2025, run the same way the shop had always run it, the fill rate rose from
{pct(H1['fill_rate'])} to {pct(mod['fill_rate'])}, stockout events fell from {H1['stockout_episodes']:,} to
{mod['stockout_episodes']:,}, jobs held for material from {H1['jobs_delayed']} to {mod['jobs_delayed']}, and rush
freight and premiums from {k(H1['rush_spend'])} to {k(mod['rush_spend'])}. The shop holds more stock to do it:
{k(mod['avg_inventory_value'])} on average against {k(H1['avg_inventory_value'])}.</p>
{B.chart("Stockouts, held jobs and rush spend by half-year", charts["halves"])}
<p>Most of that improvement is the cleanup, not the model. Replaying the same six months of demand with nothing
fixed gives {dirty['stockout_episodes']:,} stockouts; with the cleaned records and reorder points recomputed by
the simple rule, {rule['stockout_episodes']:,}; with the model, {mod['stockout_episodes']:,}. The model adds
service on top of the cleanup, but it pays for it with inventory: once the new policy has settled (months 4 to
6), it lifts A-item fill from {pct(rule46['fill_rate_by_abc']['A'])} to {pct(mod46['fill_rate_by_abc']['A'])}
against a 98% target, on {k(mod46['avg_inventory_value'])} of stock against {k(rule46['avg_inventory_value'])}.
Held to the target, the simpler rule gets there more cheaply. The model's safety stock is sized for every order
cycle, and the shop's large order quantities already protect most of them; sizing it against a fill-rate target
instead is the next step (Section 3.4).</p>

{B.kpi_row(
    B.kpi_card(f"{pct(H1['fill_rate'])} &rarr; {pct(mod['fill_rate'])}", "Fill rate", "1H25 to 1H26", DARK_BLUE),
    B.kpi_card(f"{H1['stockout_episodes']:,} &rarr; {mod['stockout_episodes']:,}", "Stockout events", "1H25 to 1H26", GREEN),
    B.kpi_card(f"{k(H1['rush_spend'])} &rarr; {k(mod['rush_spend'])}", "Rush spend", "1H25 to 1H26", GREEN),
    B.kpi_card(f"{pct(tw['raw'], 0)} &rarr; {pct(tw['fully'], 0)}", "Forecast error", "before and after cleaning", DARK_BLUE))}

{B.section("modeloverview", "Section 2", "Model Overview")}

{B.section("what", "Section 2.1", "What This Model Does")}
<p>Since January 2026 the demand forecasting model has been the shop's only source of reorder points. Every
month it sets, for each of the {n_items:,} stocked items, the level of stock at which the ERP should reorder, and
the ERP places its purchase suggestions against those points. The points rest on three things: a forecast of how
much of the part the shop will use before a new order can arrive, learned from three years of cleaned
consumption history; the part's corrected supplier lead time; and a safety buffer sized to how far off the
forecast has been for parts like it.</p>
<p>The model is built on XGBoost, a gradient-boosted decision-tree algorithm. It answers one question for every
stocked item, once a month: <strong>how much of this part will the shop use before a new order placed today could
arrive?</strong> That window differs by part. A fastener that arrives in two weeks and a gearmotor that takes two
months are different questions, and the reorder decision only cares about the demand that lands before the next
delivery does.</p>
{FLOW_HTML}
<p>The model makes one forecast per item per month; the ERP does the daily work. Each forecast becomes a reorder
point: the expected usage over the lead time, plus a safety buffer for the months when demand runs above the
forecast, larger for the high-value parts the shop can least afford to run out of. The ERP then compares every
item's stock on hand and on order against its point each day, and suggests an order when an item falls to it.
Points only change when the new forecast moves them meaningfully, so buyers are not chasing small month-to-month
changes.</p>
<p>The model is embedded in the ERP's purchasing screen. The reorder queue below ranks every stocked item against
this month's point: items at or below it are marked ORDER NOW, items within two weeks of it ORDER SOON, and each
line shows the forecast, the safety stock and the suggested order quantity, with tags where the cleanup changed
the item (a merged record, a corrected lead time or bill, restored history) so the buyer sees why a number moved.</p>
<div class="chart-wrap" style="padding:6px;">
  <img src="data:image/png;base64,{erp_b64}" alt="ERP reorder queue with the demand model's reorder points"
       style="width:100%;height:auto;display:block;border:1px solid #D5DCE1;">
</div>

{B.section("data", "Section 2.2", "Training Data Overview")}
<p>The model learns from the shop's monthly consumption of every stocked item, January 2023 onward: parts
backflushed to production jobs, parts issued to service orders, and parts pulled by hand. This is the history the
data quality audit cleaned, with duplicate records merged, free-text purchases returned to their items and the
components missing from the bills restored, so each series is the part's actual usage. Over the three years
before the forward window the shop consumed about {k(hist['value'].sum()/3)} of parts a year at standard cost,
across {n_items:,} items.</p>
{B.chart("Monthly consumption at cost, January 2023 to December 2025, by demand pattern", charts["hist"])}
<p>Demand is steady in aggregate, roughly flat over the three years, but the total hides very different behaviour
underneath. The items fall into four demand patterns, which the model treats differently:
{seg_n.get('smooth', 0)} smooth (regular and steady), {seg_n.get('erratic', 0)} erratic (regular but variable),
{seg_n.get('lumpy', 0)} lumpy (irregular and variable) and {seg_n.get('intermittent', 0)} intermittent (many months
with no demand at all). One typical item of each is shown below.</p>
{B.chart("Three years of monthly consumption, one typical item per demand pattern", charts["examples"])}
<p>Value is concentrated. A small share of items carries most of the consumption value, which is why the safety
buffers are set by value class: {abc_n.get('A', 0)} A items, {abc_n.get('B', 0)} B items and {abc_n.get('C', 0)} C
items, with the A items held to the highest service level.</p>
{B.chart("Consumption value concentration, 2025", charts["pareto"])}

{B.section("performance", "Section 3", "Model Performance")}

{B.section("scoring", "Section 3.1", "Scoring Summary")}
<p>From January to June 2026 the model set {n_items:,} reorder points every month. The table compares that half-
year with the two halves of 2025, each as the shop actually ran it. The two 2025 halves are close to each
other, which makes them a fair baseline; 2H25 includes the ten weeks of remediation, but the cleaned records and
the new reorder points only took over on 1 January. A job counts as held for material when its production order
hit a material-shortage hold, the same definition the data quality audit used for 2025.</p>
{halves_table()}
<p>Purchases rose in 1H26 ({k(mod['purchases'])} against {k(H1['purchases'])}) because the new reorder points
rebuilt stock on items the stale parameters had been running short; the extra
{k(mod['avg_inventory_value'] - H1['avg_inventory_value'])} of average inventory is where that money went.
Different halves carry different demand, so the size of each change also reflects the season and the product
mix. Section 3.3 removes that by holding demand fixed.</p>

{B.section("accuracy", "Section 3.2", "Accuracy and Validation")}
<p>Accuracy is measured as weighted absolute percentage error (WAPE) over each item's lead time, which reads like
the familiar percentage error for a steady part and stays defined for the third of parts with months of zero
demand. The model learned on the cleaned history through 2024, was tuned on a block of validation months, and
was then scored on a held-out year of rolling forecasts in 2025. Three candidate algorithms were each tuned and
compared on the validation months; XGBoost was carried forward.</p>
{candidate_table()}
<p>The cleaning matters to what the model can learn.
The same model, with the same features, was run on the history at three stages of cleaning. Across all items
the error falls from {pct(tw['raw'])} to {pct(tw['fully'])}; on the {len(repaired)} items whose history the
cleanup actually repaired (duplicates merged, unrecorded consumption restored) it falls from
{pct(tw_rep['raw'])} to {pct(tw_rep['fully'])}, and on the merged duplicates alone from {pct(tw_dup['raw'], 0)} to
{pct(tw_dup['fully'], 0)}. The gain is modest overall because only three of the sixteen errors touch the demand
history; the other thirteen corrupt what the forecast is used for, which is why Section 3.3 is where the
cleanup shows its value.</p>
{B.chart("Forecast error at three stages of cleaning", charts["acc"])}
<p>On the held-out year the model beats the best simple method for each demand pattern (a moving average, last
year's month or Croston's method, whichever did best):</p>
{accuracy_table()}
<p>Bias is the diagnostic accuracy hides. Before correction the model ran
{', '.join(f"{abs(s_['bias'])*100:.0f}% low on {s_['segment']}" for s_ in metrics['segments'])} parts; after the
correction described in Section 2.1, the forecasts in the six forward months ran within a few percent of actual
demand for every pattern.</p>

{B.section("source", "Section 3.3", "Where the Improvement Came From")}
<p>To separate what the cleanup did from what the model did, the same January to June 2026 demand, with the same
supplier deliveries, was replayed three ways. With nothing fixed, the shop keeps the stale lead times and
reorder points, the duplicate records, the phantom on-order and the unrecorded pulls. With the cleaned records
and the recomputed rule, it runs on the corrected masters with reorder points recomputed from twelve months of
usage over the corrected lead time. With the demand model, the corrected masters are the same, and the reorder
point each month comes from the model's forecast. The only difference between the last two is the forecast
behind the reorder point. None of the three can see future demand: the reorder decision knows scheduled
production and nothing else, as the ERP does.</p>
{B.chart("The same demand three ways, months 4 to 6 (steady state)", charts["variants"])}
<p>The first three months are a transition: stock is being rebuilt on items that were short, and the old stale
orders are still arriving. The comparison that matters is months 4 to 6, once the new policy has settled.
Across the full six months:</p>
{variants_table(F)}
{sub("Months 1 to 3 (transition)")}
{variants_table(M13)}
{sub("Months 4 to 6 (steady state)")}
{variants_table(M46)}
<p>The cleanup does the heavy lifting: in steady state, stockout events fall from {dirty46['stockout_episodes']:,}
with nothing fixed to {rule46['stockout_episodes']:,} on the cleaned records, and rush spend from
{k(dirty46['rush_spend'])} to {k(rule46['rush_spend'])}. The model then cuts stockouts to
{mod46['stockout_episodes']:,} and A-item days short from {rule46['stockout_days_by_abc']['A']:,} to
{mod46['stockout_days_by_abc']['A']:,}, but with {k(mod46['safety_stock_value'] - rule46['safety_stock_value'])} more safety
stock and {k(mod46['excess_value'] - rule46['excess_value'])} more excess. The order-line counts show the new policy
does not buy its service with a flood of small orders: lines placed are within about ten percent across the three.</p>
{sub("Purchases by month")}
<p>Purchases should converge to consumption under any sound policy. The transition shows up as buying above
consumption while stock is rebuilt, then settling.</p>
{purchases_table()}

{B.section("limits", "Section 3.4", "What It Can and Cannot Predict")}
<ul class="limitation-list">
  <li><strong>It forecasts demand, not supply.</strong> The model predicts how much the shop will use; it takes the
      supplier's lead time from the recomputed master and does not predict a late delivery.</li>
  <li><strong>The forward window is a simulation.</strong> The six months are replayed from the generated demand
      and supplier behaviour, not observed. Demand, deliveries and the forecast are identical across the three
      variants, so the differences between them are the policy; the size of each difference is an estimate.</li>
  <li><strong>Safety stock is sized per cycle, not per unit filled.</strong> The model's buffer protects every
      order cycle to the class's service level, and on items with large order quantities most cycles are already
      protected by the order itself, so the model overshoots the A-item fill target. Setting safety stock against
      a fill-rate target, which accounts for order quantity, would bring inventory down toward the rule's while
      keeping the model's advantage on the parts that stop the line. That is the next change to make.</li>
  <li><strong>The first three months are a transition.</strong> Purchases run above consumption while stock is
      rebuilt; the steady-state comparison is months 4 to 6, and three months is a short steady state.</li>
</ul>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
html = B.page("ML Model Overview and Performance", "", toc, body)
html = html.replace("</style></head>", ".section-title-block.sub .section-title{font-size:18px;font-weight:700;}</style></head>", 1)
OUT.write_text(html, encoding="utf-8")
print(f"Model overview written to {OUT}  ({len(html)//1024} KB)")
