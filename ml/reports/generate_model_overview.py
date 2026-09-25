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
metrics = json.loads((BACKTEST / "weekly_metrics.json").read_text(encoding="utf-8"))
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
    """Inventory, stockout events, held jobs and rush spend by half-year, as the shop ran it."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(B.CHART_W, 3.2))
    halves = ["1H25", "2H25", "1H26"]
    series = [H1, H2, F["model"]]
    hc = [MED_GREY, MED_GREY, DARK_BLUE]
    # inventory: the average month of 2025 against June 2026, the model's sixth month
    panels = [("Inventory ($000)", ["2025 avg\nmonth", "June\n2026"], [avg25 / 1000, jun_avg / 1000], "${:,.0f}K",
               [MED_GREY, DARK_BLUE]),
              ("Stockout events", halves, [s["stockout_episodes"] for s in series], "{:,.0f}", hc),
              ("Jobs held for material", halves, [s["jobs_delayed"] for s in series], "{:,.0f}", hc),
              ("Rush spend ($000)", halves, [s["rush_spend"] / 1000 for s in series], "${:,.0f}K", hc)]
    for ax, (title, labels, vals, fmt, cols_) in zip(axes, panels):
        bars = ax.bar(labels, vals, color=cols_, width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.02, fmt.format(v), ha="center", fontsize=8.5, fontweight="bold")
        ax.set_title(title, fontsize=9.5, color=DARK_GREY, pad=8)
        ax.set_ylim(0, max(vals) * 1.22); ax.set_yticks([]); ax.tick_params(axis="x", labelsize=8.5)
        B.chart_style(ax)
    fig.tight_layout(w_pad=1.6)
    return B.b64(fig)


def chart_inventory_months():
    """Average inventory value by month, the same demand three ways, against 1H25 as the shop ran it."""
    fig, ax = B.make_fig(3.4)
    cols = {"dirty": MED_GREY, "clean_rule": LIGHT_BLUE, "model": DARK_BLUE}
    for v, name in VARIANTS:
        mm = F[v]["inventory_by_month"]
        xs = [pd.Timestamp(m_) for m_ in sorted(mm)]
        ys = [mm[m_] / 1e6 for m_ in sorted(mm)]
        ax.plot(xs, ys, marker="o", color=cols[v], linewidth=2.2 if v == "model" else 1.6, label=name)
        ax.text(xs[-1] + pd.Timedelta(days=4), ys[-1], f"${ys[-1]:.2f}M", fontsize=8.5, va="center", color=cols[v],
                fontweight="bold" if v == "model" else "normal")
    ax.axhline(avg25 / 1e6, color=DARK_GREY, linestyle="--", linewidth=1)
    ax.text(pd.Timestamp("2026-01-01"), avg25 / 1e6 + 0.02, "2025 monthly average, as the shop ran it",
            fontsize=8, color=DARK_GREY)
    ax.set_ylabel("Average inventory ($M)")
    ax.xaxis.set_major_formatter(__import__("matplotlib.dates", fromlist=["DateFormatter"]).DateFormatter("%b"))
    ax.set_xlim(pd.Timestamp("2025-12-20"), pd.Timestamp("2026-07-10"))
    B.chart_style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=3, fontsize=9)
    return B.b64(fig)


def chart_variants():
    """The same 1H26 demand three ways, in steady state (months 4 to 6)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(B.CHART_W, 3.3))
    names = ["Nothing\nfixed", "Cleaned,\nrule", "Cleaned,\nmodel"]
    cols = [MED_GREY, LIGHT_BLUE, DARK_BLUE]
    vals = [M46[v] for v, _ in VARIANTS]
    panels = [("Stockout events", [x["stockout_episodes"] for x in vals], "{:,.0f}", False),
              ("Jobs held for material", [x["jobs_delayed"] for x in vals], "{:,.0f}", False),
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
        ("Fill rate, line-critical / service / standard parts",
         lambda s: " / ".join(pct(s["fill_rate_by_tier"][t_]) for t_ in ["line", "service", "standard"])),
        ("Stockout events", lambda s: f"{s['stockout_episodes']:,}"),
        ("Stockout days (item-days short)", lambda s: f"{s['stockout_days']:,}"),
        ("Jobs held for material", lambda s: f"{s['jobs_delayed']:,}"),
        ("Days lost on held jobs", lambda s: f"{s['job_delay_days']:,}"),
        ("Rush order lines", lambda s: f"{s['rush_lines']:,}"),
        ("Rush freight and premiums", lambda s: money(s["rush_spend"])),
        ("Order lines placed", lambda s: f"{s['order_lines']:,}"),
        ("Purchases", lambda s: money(s["purchases"])),
    ]
    for label, fn in spec:
        rows.append([label, fn(H1), fn(H2), fn(m)])
    return widths(B.data_table(["Measure", "1H25", "2H25", "1H26"], rows, right=[1, 2, 3]), [37, 21, 21, 21])


def inventory_table():
    """Inventory: the average month of 2025 as the shop ran it, against June 2026 on the model."""
    c25 = (H1["consumption_value"] + H2["consumption_value"]) / (H1["days"] + H2["days"])
    c26 = F["model"]["consumption_value"] / F["model"]["days"]
    rows = [["Inventory value", money(avg25), money(jun_avg), f"-{fall(avg25, jun_avg)}"],
            ["Days of supply", f"{avg25 / c25:.0f}", f"{jun_avg / c26:.0f}", f"{jun_avg / c26 - avg25 / c25:+.0f} days"],
            ["June 2026 with manual reordering continued", "", money(jun_dirty), ""]]
    return widths(B.data_table(["Measure", "2025, average month", "June 2026", "Change"], rows, right=[1, 2, 3]),
                  [37, 21, 21, 21])


def variants_table(window):
    rows = []
    spec = [
        ("Fill rate", lambda s: pct(s["fill_rate"])),
        ("Fill rate, line-critical / service / standard parts",
         lambda s: " / ".join(pct(s["fill_rate_by_tier"][t_]) for t_ in ["line", "service", "standard"])),
        ("Stockout events", lambda s: f"{s['stockout_episodes']:,}"),
        ("Stockout events, line-critical parts", lambda s: f"{s['stockout_episodes_by_tier']['line']:,}"),
        ("Jobs held for material", lambda s: f"{s['jobs_delayed']:,}"),
        ("Rush freight and premiums", lambda s: money(s["rush_spend"])),
        ("Average inventory value", lambda s: money(s["avg_inventory_value"])),
        ("Inventory at the end of the window", lambda s: money(s["end_inventory_value"])),
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
V_days = int(round(sched["mrp_visibility_days"]))
VM_days = int(round(sched.get("model_visibility_days", sched["mrp_visibility_days"])))
BK_days = int(round(sched.get("order_book_days", 0)))
FT = sched["fill_rate_target"]
tier_n = pd.Series(sched["criticality"]).value_counts().to_dict()
LOT_LO, LOT_HI = sched["lot_days"]
end_mod, end_dirty = F["model"]["end_inventory_value"], F["dirty"]["end_inventory_value"]
end_rule = F["clean_rule"]["end_inventory_value"]
dec_avg, dec_end = H2["inventory_by_month"]["2025-12"], H2["end_inventory_value"]
jun_avg = F["model"]["inventory_by_month"]["2026-06"]
jun_dirty = F["dirty"]["inventory_by_month"]["2026-06"]
_m, _r = M46["model"]["avg_inventory_value"], M46["clean_rule"]["avg_inventory_value"]
RULE_SVC = ("improves service somewhat" if (rule46["fill_rate"] > dirty46["fill_rate"] + 0.003
                                             or rule46["jobs_delayed"] < dirty46["jobs_delayed"])
            else "barely moves service")
if end_mod <= end_rule:
    END_NOTE = f"ends June with the least stock of the three ({k(end_mod)})"
else:
    END_NOTE = (f"ends June with {k(end_mod)} of stock, well below manual reordering and close to the rule's "
                f"{k(end_rule)}, because it holds larger buffers on the parts whose demand is hardest to predict,")
if _m <= _r:
    AVG_NOTE = (f"It also carries the least stock of the three on average over months 4 to 6 ({k(_m)}), though it "
                "gets there gradually: it first builds buffers on the parts that were running short, then runs down the rest.")
else:
    AVG_NOTE = ""   # the June comparison above already explains the model's position against the rule
avg25 = float(np.mean(list(H1["inventory_by_month"].values()) + list(H2["inventory_by_month"].values())))


def tgt(x):
    return f"{round(x * 100, 1):g}%"


def chg(a_, b_):
    return f"{(b_ - a_) / a_ * 100:+.0f}%"


def fall(a_, b_):
    return f"{(a_ - b_) / a_ * 100:.0f}%"
fb_max = max(abs(v) for v in (mod.get("forecast_bias_by_segment") or {"all": 0.0}).values())

attrs = pd.read_parquet(REPO / "ml" / "data" / "marts" / "item_attributes.parquet")
monthly = pd.read_parquet(REPO / "ml" / "data" / "marts" / "consumption_monthly.parquet")
seg_n = attrs["segment"].value_counts().to_dict(); abc_n = attrs["abc"].value_counts().to_dict()
n_items = len(attrs)
cand = metrics["candidates"]
CAND_LABEL = {"Linear": "Linear regression (ridge)", "RandomForest": "Random forest", "XGBoost": "XGBoost"}
WIN = metrics["winner"]
WIN_DESC = {"RandomForest": "a random forest, an ensemble of several hundred decision trees whose forecasts are averaged",
            "XGBoost": "XGBoost, a gradient-boosted decision-tree algorithm",
            "Linear": "a regularized linear regression"}[WIN]
_vals = sorted(cand, key=lambda c: cand[c]["val_wape"])
if _vals[0] == WIN:
    WIN_WHY = (f"{CAND_LABEL[WIN]} had the lowest validation error, "
               f"{(cand[_vals[1]]['val_wape'] - cand[WIN]['val_wape']) * 100:.1f} points below {CAND_LABEL[_vals[1]] if _vals[1] == "XGBoost" else CAND_LABEL[_vals[1]].lower()}, and was carried forward.")
else:
    WIN_WHY = (f"The two tree methods finished within half a point of each other, and {CAND_LABEL[WIN]} was carried "
               "forward as the lighter of the two to retrain every month.")


def candidate_table():
    rows = [[CAND_LABEL[c], pct(cand[c]["val_wape"]), pct(cand[c]["test_wape"]),
             "Selected" if c == metrics["winner"] else ""] for c in ["Linear", "RandomForest", "XGBoost"]]
    return widths(B.data_table(["Candidate", "Validation error (WAPE)", "Test error (WAPE)", ""], rows, right=[1, 2]),
                  [34, 24, 24, 18])


hist = monthly[monthly["month"] < "2026-01-01"].merge(
    attrs[["canonical_item_number", "segment", "abc", "standard_cost"]], left_on="canonical", right_on="canonical_item_number")
hist["value"] = hist["consumption"] * hist["standard_cost"].fillna(attrs["standard_cost"].median())
SEG_COL = {"smooth": DARK_BLUE, "erratic": LIGHT_BLUE, "lumpy": "#8093A4", "intermittent": "#D5DCE1"}
# the part's category, read from its item-number prefix (the class field carries a
# generic MISC placeholder on some records)
PREFIX_CAT = {"MEC": "Mechanical", "ELE": "Electrical", "RM": "Raw material", "FAS": "Fasteners",
              "HDW": "Hardware", "FIT": "Fittings", "CON": "Consumables", "SVC": "Outside service"}
CAT_ORDER = ["Mechanical", "Electrical", "Raw material", "Outside service", "Fittings", "Consumables",
             "Hardware", "Fasteners"]
CAT_COL = dict(zip(CAT_ORDER, [DARK_BLUE, "#5B45C0", LIGHT_BLUE, "#9ADCF2", "#322B4B", "#8093A4", "#B7C2CC",
                               "#D5DCE1"]))
attrs["category"] = attrs["canonical_item_number"].map(lambda n: PREFIX_CAT.get(str(n).split("-")[0], "Other"))
hist["category"] = hist["canonical"].map(attrs.set_index("canonical_item_number")["category"])
cat_seg = pd.crosstab(attrs["category"], attrs["segment"]).reindex(index=CAT_ORDER, columns=SEG_ORDER).fillna(0)
cat_share = cat_seg.div(cat_seg.sum(axis=1), axis=0)


def _avg_box(ax, text):
    ax.text(0.5, 0.97, text, transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=DARK_GREY,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#D5DCE1"))


def chart_value_by_category():
    """Monthly consumption at standard cost over the three years, by item category."""
    fig, ax = B.make_fig(3.9)
    w = hist.pivot_table(index="month", columns="category", values="value", aggfunc="sum").fillna(0)
    w = w.reindex(columns=[c for c in CAT_ORDER if c in w.columns]) / 1000
    ax.stackplot(w.index, [w[c] for c in w.columns], colors=[CAT_COL[c] for c in w.columns],
                 labels=list(w.columns), alpha=0.95)
    tot = w.sum(axis=1); x = np.arange(len(tot)); fit = np.polyfit(x, tot.to_numpy(), 1)
    ax.plot(w.index, np.polyval(fit, x), color=DARK_GREY, linestyle="--", linewidth=1.2, label="Trend")
    ax.set_ylim(0, tot.max() * 1.35)
    ax.set_ylabel("Consumption at cost ($000 / month)")
    _avg_box(ax, f"Average: ${tot.mean():,.0f}K of parts consumed per month")
    B.chart_style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=5, fontsize=8.5)
    fig.tight_layout()
    return B.b64(fig)


def chart_category_patterns():
    """Number of items in each category, stacked by demand pattern."""
    fig, ax = B.make_fig(3.7)
    bottom = np.zeros(len(cat_seg))
    for seg in SEG_ORDER:
        vals = cat_seg[seg].to_numpy()
        ax.bar(cat_seg.index, vals, bottom=bottom, color=SEG_COL[seg], width=0.65, label=seg.capitalize())
        bottom += vals
    for i, tot_ in enumerate(bottom):
        ax.text(i, tot_ + 2, f"{int(tot_)}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Items")
    ax.set_ylim(0, bottom.max() * 1.15)
    ax.set_xticks(range(len(cat_seg))); ax.set_xticklabels([c.replace(" ", "\n") for c in cat_seg.index])
    ax.tick_params(axis="x", labelsize=8.5)
    B.chart_style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=4, fontsize=9)
    fig.tight_layout()
    return B.b64(fig)


def _top(seg, n=2):
    s_ = cat_share[seg].sort_values(ascending=False).head(n)
    return " and ".join(f"{c.lower()} ({v*100:.0f}%)" for c, v in s_.items())


CAT_TEXT = (f"Demand pattern follows the kind of part, though every category carries all four. Smooth demand is most "
            f"common among {_top('smooth')}, the floor stock pulled week in, week out. Erratic demand is spread "
            f"widely, highest among {_top('erratic')}. Lumpy demand concentrates in {_top('lumpy')}, whose use "
            f"follows the job mix, and intermittent demand in {_top('intermittent')}, parts used a few to a job or "
            f"sold as occasional spares.")


def chart_history_value():
    """Monthly consumption in units over the three years, by demand pattern."""
    fig, ax = B.make_fig(3.9)
    w = hist.pivot_table(index="month", columns="segment", values="consumption", aggfunc="sum").fillna(0)[SEG_ORDER] / 1000
    ax.stackplot(w.index, [w[c] for c in SEG_ORDER], colors=[SEG_COL[c] for c in SEG_ORDER],
                 labels=[c.capitalize() for c in SEG_ORDER], alpha=0.95)
    tot = w.sum(axis=1); x = np.arange(len(tot)); fit = np.polyfit(x, tot.to_numpy(), 1)
    ax.plot(w.index, np.polyval(fit, x), color=DARK_GREY, linestyle="--", linewidth=1.2, label="Trend")
    ax.set_ylim(0, tot.max() * 1.35)
    ax.set_ylabel("Units consumed (000 / month)")
    _avg_box(ax, f"Average: {round(tot.mean() * 10) * 100:,.0f} units consumed per month")
    B.chart_style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=5, fontsize=9)
    fig.tight_layout()
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
    'with its demand pattern, value class, cost and supplier lead time.</div></div>'
    '<div style="align-self:center;font-size:26px;color:#8093A4;padding:0 12px;">&rarr;</div>'
    '<div style="flex:1;min-width:190px;background:#F3F5F7;border-radius:8px;padding:16px 18px;border-top:4px solid #381FA1;">'
    '<div style="font-weight:700;color:#322B4B;margin-bottom:6px;">2. What it predicts</div>'
    '<div style="font-size:16px;line-height:1.55;">Every week, one forecast per item: how many units the shop will use '
    'before a new order placed today could arrive.</div></div>'
    '<div style="align-self:center;font-size:26px;color:#8093A4;padding:0 12px;">&rarr;</div>'
    '<div style="flex:1;min-width:190px;background:#F3F5F7;border-radius:8px;padding:16px 18px;border-top:4px solid #381FA1;">'
    '<div style="font-weight:700;color:#322B4B;margin-bottom:6px;">3. What it produces</div>'
    '<div style="font-size:16px;line-height:1.55;">For every item, when to reorder and how much, loaded into the ERP, '
    'which checks every item daily and flags what to order.</div></div></div>')

import base64
_png = REPO / "docs" / "screenshots" / "erp_queue.png"
erp_b64 = base64.b64encode(_png.read_bytes()).decode() if _png.exists() else ""

charts = {"halves": chart_halves(), "variants": chart_variants(), "acc": chart_accuracy(),
          "invm": chart_inventory_months(), "valcat": chart_value_by_category(), "catseg": chart_category_patterns(),
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
points and order quantities, and it carried less stock while running short less often. Inventory fell from an
average of {k(avg25)} a month in 2025 under manual reordering to {k(jun_avg)} in June 2026, {fall(avg25, jun_avg)}
lower. Part of that decline would have happened anyway: replaying 2026 with manual reordering continued leaves
June at {k(jun_dirty)}, so the model's own effect is the {fall(jun_dirty, jun_avg)} gap between the two Junes.
Against the first half of 2025, run the way the shop had always run it, stockout events fell from {H1['stockout_episodes']:,} to
{mod['stockout_episodes']:,}, jobs held for material from {H1['jobs_delayed']} to {mod['jobs_delayed']}, and rush
freight and premiums from {k(H1['rush_spend'])} to {k(mod['rush_spend'])}.</p>
{B.chart("Inventory (2025 average month against June 2026); stockouts, held jobs and rush spend by half-year", charts["halves"])}
<p>The manual process held more stock than the shop needed, in the wrong places. Reorder points set at go-live
and never revisited were far too high on some parts and too low on others, and buyers bought about four and a half months
of a part at a time whatever it cost. The model moves the stock to where it prevents a stoppage. Each part's
buffer is sized to how far off its own forecasts and deliveries have been, jobs already booked are netted weeks
before they draw parts, and expensive parts are bought more often in smaller lots. Replaying the same six months of demand with nothing fixed shows the gain once the new
policy has settled (months 4 to 6): stockout events fall {fall(dirty46['stockout_episodes'], mod46['stockout_episodes'])},
jobs held for material {fall(dirty46['jobs_delayed'], mod46['jobs_delayed'])}
and rush spend {fall(dirty46['rush_spend'], mod46['rush_spend'])}. Inventory is still falling at the end of June,
as excess on slow parts is used up and not replaced.</p>

{B.kpi_row(
    B.kpi_card(f"{k(avg25)} &rarr; {k(jun_avg)}", "Inventory", "2025 monthly average to June 2026", GREEN),
    B.kpi_card(f"{H1['stockout_episodes']:,} &rarr; {mod['stockout_episodes']:,}", "Stockout events", "1H25 to 1H26", GREEN),
    B.kpi_card(f"{H1['jobs_delayed']} &rarr; {mod['jobs_delayed']}", "Jobs held for material", "1H25 to 1H26", GREEN),
    B.kpi_card(f"{k(H1['rush_spend'])} &rarr; {k(mod['rush_spend'])}", "Rush spend", "1H25 to 1H26", GREEN))}

{B.section("modeloverview", "Section 2", "Model Overview")}

{B.section("what", "Section 2.1", "What This Model Does")}
<p>Over the past six months (January 2026 to June 2026), the demand forecasting model has set the shop's reorder
decisions. Every week, the model predicts which items need to be reordered, and these predictions are then fed
directly into the ERP for each of the {n_items:,} stocked items. The model's reorder decisions are based on the
current stock on hand and on order for each item, the forecasted consumption over each supplier's delivery time,
and a safety buffer based on how unpredictable each part's demand and deliveries have been.</p>
{FLOW_HTML}
<p>The model refreshes its forecasts every Monday and is retrained on the latest history once a month. The
model's predictions are loaded directly into the ERP, which flags when each item should be reordered and how much
to order. To keep working capital low, the recommendations for how much to order are informed by the item's cost:
expensive items are bought every few weeks in small lots, and cheap items a few times a year in larger lots.</p>
<p>The ERP's reorder queue ranks every stocked item against the model's weekly reorder point: items at or below it
are marked "ORDER NOW", items within two weeks of it "ORDER SOON", and items outside of it "OK". Each line also
shows the part's criticality, the stock on hand, allocated to released jobs and on order, the forecast, the safety
stock and the suggested order quantity, as seen in the screenshot of the ERP system below:</p>
<div class="chart-wrap" style="padding:6px;">
  <img src="data:image/png;base64,{erp_b64}" alt="ERP reorder queue with the demand model's reorder points"
       style="width:100%;height:auto;display:block;border:1px solid #D5DCE1;">
</div>

{B.section("data", "Section 2.2", "Training Data Overview")}
<p>The model learns from the shop's weekly consumption of every stocked item. It was originally trained on three
years of this consumption data (2023 through 2025), and is continually trained on every new month of data.</p>
{B.chart("Monthly consumption in value, January 2023 to December 2025, by item category", charts["valcat"])}
<p>Demand is steady in aggregate and roughly flat over the three years, but individual items exhibit very different
behavior. We've categorized the individual demand patterns of items into four patterns, which the model treats
differently: {seg_n.get('smooth', 0)} smooth (regular and steady), {seg_n.get('erratic', 0)} erratic (regular but
variable), {seg_n.get('lumpy', 0)} lumpy (irregular and variable) and {seg_n.get('intermittent', 0)} intermittent
(many months with no demand at all). The model is calibrated against these demand patterns. One representative item
within each pattern is shown below.</p>
{B.chart("Three years of monthly consumption, one representative item per demand pattern", charts["examples"])}
<p>{CAT_TEXT}</p>
{B.chart("Items in each category, by demand pattern", charts["catseg"])}
{B.chart("Monthly consumption in units, January 2023 to December 2025, by demand pattern", charts["hist"])}
<p>Value is concentrated. A small share of items carries most of the consumption value: {abc_n.get('A', 0)} A
items, {abc_n.get('B', 0)} B items and {abc_n.get('C', 0)} C items. That concentration is why order quantities are
set by cost: buying the A items in smaller, more frequent lots frees most of the working capital, while the
buffers that protect against stockouts are set by how unpredictable each part is.</p>
{B.chart("Consumption value concentration, 2025", charts["pareto"])}

{B.section("performance", "Section 3", "Model Performance")}

{B.section("scoring", "Section 3.1", "Scoring Summary")}
<p>From January to June 2026 the model refreshed the reorder points of all {n_items:,} items every week. The table compares that half-
year with the two halves of 2025, each as the shop actually ran it. The two 2025 halves are close to each
other, which makes them a fair baseline; 2H25 includes the ten weeks of remediation, but the cleaned records and
the new reorder points only took over on 1 January. A job counts as held for material when its production order
hit a material-shortage hold, the same definition the data quality audit used for 2025.</p>
{halves_table()}
<p>Inventory is compared on the average month of 2025, as the shop ran it, against June 2026, the model's sixth
month. A half-year average would mix in the first months of 2026, when the model was still working through the
stock the manual process had built. The replay with manual reordering continued shows how much of the decline
would have happened anyway.</p>
{inventory_table()}
<p>Purchases fell in 1H26 ({k(mod['purchases'])} against {k(H1['purchases'])}) because the model stopped
reordering parts that already held more than they needed and let that stock run down. Different halves carry
different demand, so the size of each change also reflects the season and the product mix. Section 3.3 removes
that by holding demand fixed.</p>

{B.section("accuracy", "Section 3.2", "Accuracy and Validation")}
<p>Accuracy is measured as weighted absolute percentage error (WAPE) over each item's lead time, which reads like
the familiar percentage error for a steady part and stays defined for the many parts with weeks of zero demand.
The model learned on the cleaned weekly history through 2024, was tuned on a block of validation weeks, and was
then scored on a held-out year of weekly rolling forecasts in 2025. Three candidate algorithms were each tuned
and compared on the validation weeks. {WIN_WHY}</p>
{candidate_table()}
<p>The cleaning matters to what the model can learn.
The same model, with the same features, was run on the history at three stages of cleaning (scored on monthly forecasts over the lead time). Across all items
the error falls from {pct(tw['raw'])} to {pct(tw['fully'])}; on the {len(repaired)} items whose history the
cleanup actually repaired (duplicates merged, unrecorded consumption restored) it falls from
{pct(tw_rep['raw'])} to {pct(tw_rep['fully'])}, and on the merged duplicates alone from {pct(tw_dup['raw'], 0)} to
{pct(tw_dup['fully'], 0)}. The gain is modest overall because only three of the sixteen errors touch the demand
history. The other thirteen corrupt the lead times, bills and stock records the reorder point is built on, and
the model's results in Section 3.3 rest on those corrections.</p>
{B.chart("Forecast error at three stages of cleaning", charts["acc"])}
<p>On the held-out year the model beats the best simple method for each demand pattern (a moving average, last
year's month or Croston's method, whichever did best):</p>
{accuracy_table()}
<p>Bias is the diagnostic accuracy hides. Before correction the model ran
{', '.join(f"{abs(s_['bias'])*100:.0f}% low on {s_['segment']}" for s_ in metrics['segments'])} parts. Each
pattern's forecasts are scaled up by the ratio of actual to forecast demand in the 2025 backtest, and in the six
forward months the corrected forecasts ran within {fb_max*100:.0f}% of actual demand for every pattern.</p>
<p>The safety buffer is calibrated on the same backtest. The buffer is set so that the units a part is expected to
run short between deliveries stay within its fill-rate target, measured on the model's actual 2025 errors rather
than on a normal curve. Those errors have fatter tails than a normal curve, so textbook multiples would leave the
buffer short. The calculation accounts for the order quantity (a large lot protects most of its own cycle), for
the part of demand the ERP already sees on released jobs, and for delivery variability from each item's own
receipt history.</p>

{B.section("source", "Section 3.3", "Where the Improvement Came From")}
<p>To separate what the cleanup did from what the model did, the same January to June 2026 demand, with the same
supplier deliveries, was replayed three ways. With nothing fixed, the shop keeps the stale lead times and
reorder points, the duplicate records, the phantom on-order and the buyers' four-and-a-half-month lots. With the cleaned
records and the recomputed rule, it runs on the corrected masters with reorder points recomputed each month from
the last twelve months of usage over the corrected lead time, plus a standard buffer, and keeps the buyers' lots.
With the demand model, the corrected masters are the same, and the reorder points and order quantities come from
the model. Under the rule the ERP nets the parts committed to released jobs, which it sees about {V_days} days
ahead; the model also reads the order book, which extends that to about {VM_days} days, and keeps supplier lead
times current. In every variant the purchasing manager expedites the same parts she always has. None of the three can
see future demand beyond that, as the ERP cannot.</p>
{B.chart("The same demand three ways, months 4 to 6 (steady state)", charts["variants"])}
{B.chart("Average inventory by month, the same demand three ways", charts["invm"])}
<p>The first three months are a transition: orders placed under the old points are still arriving, and excess
stock is being used up. The comparison that matters is months 4 to 6, once the new policy has settled. Across the
full six months:</p>
{variants_table(F)}
{sub("Months 1 to 3 (transition)")}
{variants_table(M13)}
{sub("Months 4 to 6 (steady state)")}
{variants_table(M46)}
<p>The recomputed rule alone lowers inventory (a June 30 balance of {k(end_rule)} against {k(end_dirty)} with
nothing fixed) and {RULE_SVC}: fill is {pct(rule46['fill_rate'])} against {pct(dirty46['fill_rate'])}, and
{rule46['jobs_delayed']} jobs are held against {dirty46['jobs_delayed']}. But its textbook buffer assumes steadier
demand than the shop's parts show, so the lumpy, hard-to-predict parts still run short. The model, on the same cleaned records,
{END_NOTE} and cuts stockout events to
{mod46['stockout_episodes']:,}, stockouts on line-critical parts from {dirty46['stockout_episodes_by_tier']['line']}
to {mod46['stockout_episodes_by_tier']['line']}, and jobs held to {mod46['jobs_delayed']}. {AVG_NOTE} It places more order
lines ({mod46['order_lines']:,} against {dirty46['order_lines']:,}) because the expensive parts are bought more often;
that is the cost of carrying less of them.</p>
{sub("Purchases by month")}
<p>Purchases should converge to consumption under any sound policy. Under the model, purchases run above
consumption at first while buffers are built on the parts that were running short, then below it as excess on
the rest is used up.</p>
{purchases_table()}

{B.section("limits", "Section 3.4", "What It Can and Cannot Predict")}
<ul class="limitation-list">
  <li><strong>It forecasts demand, not supply.</strong> The model predicts how much the shop will use; it takes each
      supplier's lead time from its recent deliveries, refreshed monthly, and does not predict a late delivery. The buffer covers
      the usual spread of deliveries, not a supplier failure.</li>
  <li><strong>The forward window is a simulation.</strong> The six months are replayed from the generated demand
      and supplier behaviour, not observed. Demand, deliveries and the forecast are identical across the three
      variants, so the differences between them are the policy; the size of each difference is an estimate.</li>
  <li><strong>Excess on slow parts takes time to clear.</strong> {k(mod46['excess_value'])} of stock in months 4 to 6
      still sits on {mod46['excess_items']} items holding more than a year of supply. The model stops reordering
      them, but a part used a few times a year takes that long to draw down. Returning or selling the worst of it
      would release the cash sooner; that is a disposition decision for purchasing and finance, not a forecast.</li>
  <li><strong>The order book is an assumption.</strong> The shop's records carry no booking date for customer
      orders, so each job is assumed booked four to eight weeks before its release, typical of a job shop quoting
      lead times of that length. With less notice, the model would see less of the coming demand.</li>
  <li><strong>The comparison rule is a textbook rule.</strong> The recomputed rule uses the standard buffer a
      planner would set by hand. A planner could pad it further and buy more service with more stock; what the
      model adds is putting that stock on the parts whose demand is hardest to predict.</li>
  <li><strong>Expediting is held constant.</strong> All three variants expedite the same parts the purchasing
      manager tracked before go-live, so rush spend reflects how often those parts were at risk, not a change in
      how hard the shop chases suppliers.</li>
  <li><strong>The first three months are a transition.</strong> Orders placed under the old points are still
      arriving while excess is used up; the steady-state comparison is months 4 to 6, and three months is a short
      steady state.</li>
</ul>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
html = B.page("ML Model Overview and Performance", "", toc, body)
html = html.replace("</style></head>", ".section-title-block.sub .section-title{font-size:18px;font-weight:700;}</style></head>", 1)
OUT.write_text(html, encoding="utf-8")
print(f"Model overview written to {OUT}  ({len(html)//1024} KB)")
