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
    """Stockout episodes, days short and rush spend by half-year, as the shop ran it."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(B.CHART_W, 3.2))
    labels = ["1H25", "2H25", "1H26"]
    series = [H1, H2, F["model"]]
    panels = [("Stockout episodes", [s["stockout_episodes"] for s in series], "{:,.0f}"),
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
              ("Stockout episodes", [x["stockout_episodes"] for x in vals], "{:,.0f}", False),
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
        ("Stockout episodes", lambda s: f"{s['stockout_episodes']:,}"),
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
        ("Stockout episodes", lambda s: f"{s['stockout_episodes']:,}"),
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

charts = {"halves": chart_halves(), "variants": chart_variants(), "acc": chart_accuracy()}

toc = ('<a href="#summary">Summary</a>'
       '<a href="#halves">Six Months on the New System</a>'
       '<a href="#source">Where the Improvement Came From</a>'
       '<a href="#accuracy">Forecast Accuracy</a>'
       '<a href="#rop">From Forecast to Reorder Point</a>'
       '<a href="#limits">Limitations and Next Step</a>')

body = f"""
{B.section("summary", "Section 1", "Summary")}
<p>From January through June 2026 the shop ran its purchasing on the cleaned records and on reorder points set
each month by the demand model. Against the first half of 2025, run the same way the shop had always run it,
the fill rate rose from {pct(H1['fill_rate'])} to {pct(mod['fill_rate'])}, stockout episodes fell from
{H1['stockout_episodes']:,} to {mod['stockout_episodes']:,}, jobs held for material from {H1['jobs_delayed']} to
{mod['jobs_delayed']}, and rush freight and premiums from {k(H1['rush_spend'])} to {k(mod['rush_spend'])}. The
shop holds more stock to do it: {k(mod['avg_inventory_value'])} on average against {k(H1['avg_inventory_value'])}.</p>
<p>Most of that improvement is the cleanup, not the model. Replaying the same six months of demand with nothing
fixed gives {dirty['stockout_episodes']:,} stockouts; with the cleaned records and reorder points recomputed by
the simple rule, {rule['stockout_episodes']:,}; with the model, {mod['stockout_episodes']:,}. The model adds
service on top of the cleanup, but it pays for it with inventory: once the new policy has settled (months 4 to
6), it lifts A-item fill from {pct(rule46['fill_rate_by_abc']['A'])} to {pct(mod46['fill_rate_by_abc']['A'])}
against a 98% target, on {k(mod46['avg_inventory_value'])} of stock against {k(rule46['avg_inventory_value'])}.
Held to the target, the simpler rule gets there more cheaply. The model's safety stock is sized for every order
cycle, and the shop's large order quantities already protect most of them; sizing it against a fill-rate target
instead is the next step (Section 6).</p>

{B.kpi_row(
    B.kpi_card(f"{pct(H1['fill_rate'])} &rarr; {pct(mod['fill_rate'])}", "Fill rate", "1H25 to 1H26", DARK_BLUE),
    B.kpi_card(f"{H1['stockout_episodes']:,} &rarr; {mod['stockout_episodes']:,}", "Stockout episodes", "1H25 to 1H26", GREEN),
    B.kpi_card(f"{k(H1['rush_spend'])} &rarr; {k(mod['rush_spend'])}", "Rush spend", "1H25 to 1H26", GREEN),
    B.kpi_card(f"{pct(tw['raw'], 0)} &rarr; {pct(tw['fully'], 0)}", "Forecast error", "before and after cleaning", DARK_BLUE))}

{B.section("halves", "Section 2", "Six Months on the New System")}
<p>The first comparison is the one the shop lived through: the forward half-year against the two halves of
2025, each as the shop actually ran it. The two 2025 halves are close to each other, which makes them a fair
baseline; 2H25 includes the ten weeks of remediation, but the cleaned records and the new reorder points only
took over on 1 January. A job counts as held for material when its production order hit a material-shortage
hold, the same definition the data quality audit used for 2025.</p>
{B.chart("Stockouts, held jobs and rush spend by half-year", charts["halves"])}
{halves_table()}
<p>Purchases rose in 1H26 ({k(mod['purchases'])} against {k(H1['purchases'])}) because the new reorder points
rebuilt stock on items the stale parameters had been running short; the extra {k(mod['avg_inventory_value'] - H1['avg_inventory_value'])}
of average inventory is where that money went. Different halves carry different demand, so the size of each
change also reflects the season and the product mix. The next section removes that by holding demand fixed.</p>

{B.section("source", "Section 3", "Where the Improvement Came From")}
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
<p>The cleanup does the heavy lifting: in steady state, stockout episodes fall from {dirty46['stockout_episodes']:,}
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

{B.section("accuracy", "Section 4", "Forecast Accuracy")}
<p>The model forecasts each item's demand over its own replenishment lead time, because that is the quantity a
reorder point has to cover. Accuracy is measured as weighted absolute percentage error (WAPE) on a held-out
year, which reads like the familiar percentage error for a steady part and stays defined for the third of parts
with months of zero demand. The same model, with the same features, was run on the history at three stages of
cleaning. Across all items the error falls from {pct(tw['raw'])} to {pct(tw['fully'])}; on the {len(repaired)}
items whose history the cleanup actually repaired (duplicates merged, unrecorded consumption restored) it falls
from {pct(tw_rep['raw'])} to {pct(tw_rep['fully'])}, and on the merged duplicates alone from {pct(tw_dup['raw'], 0)}
to {pct(tw_dup['fully'], 0)}. The gain is modest overall because only three of the sixteen errors touch the demand
history; the other thirteen corrupt what the forecast is used for, which is why Section 3 is where the cleanup
shows its value.</p>
{B.chart("Forecast error at three stages of cleaning", charts["acc"])}
<p>On the cleaned history the model beats the best simple method for each demand pattern (a moving average,
last year's month or Croston's method, whichever did best):</p>
{accuracy_table()}
<p>Bias is the diagnostic accuracy hides. Trained on log demand, the model forecasts something nearer the median
than the mean, and ran {', '.join(f"{abs(s['bias'])*100:.0f}% low on {s['segment']}" for s in metrics['segments'])}
parts. A reorder point needs the mean, so the forecast is corrected per demand pattern by the ratio of actual to
forecast on the 2025 backtest; in the six forward months the corrected forecast ran within a few percent of
actual demand for every pattern.</p>

{B.section("rop", "Section 5", "From Forecast to Reorder Point")}
<p>At the start of each forward month the model was retrained on the cleaned history to date, and each item's
reorder point was set to the corrected forecast over the lead time plus a safety stock of k times the item's
forecast error over the lead time. The factor k was calibrated on the 2025 backtest as the quantile of the
errors at each class's service level, rather than assumed from the normal curve: A {k_abc['A']:.2f} (normal
{z['A']:.2f}), B {k_abc['B']:.2f} ({z['B']:.2f}), C {k_abc['C']:.2f} ({z['C']:.2f}). The errors have slightly heavier
tails than the normal curve assumes, so the calibrated buffers are a little larger.</p>
<p>Buyers stop trusting numbers that jump around, so a point only moves when the fresh value differs from the
current one by more than {sched['hysteresis']*100:.0f}%. Without that band every point would change every month,
and {churn['without_hysteresis']*100:.0f}% of them by more than a fifth; with it, {churn['with_hysteresis']*100:.0f}% of
item-months see a change at all, each one large enough to be worth acting on.</p>

{B.section("limits", "Section 6", "Limitations and Next Step")}
<ul class="limitation-list">
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
