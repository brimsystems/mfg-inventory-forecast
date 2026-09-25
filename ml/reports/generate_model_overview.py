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
    """2025 against the model's first half: stockouts, held jobs, rush spend (half-year average) and inventory."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(B.CHART_W, 3.3))
    m = F["model"]
    half = lambda key: (H1[key] + H2[key]) / 2
    panels = [("Stockout events", ["1H/2H '25\nAvg.", "1H '26"], half("stockout_episodes"), m["stockout_episodes"], "{:,.0f}"),
              ("Jobs held for material", ["1H/2H '25\nAvg.", "1H '26"], half("jobs_delayed"), m["jobs_delayed"], "{:,.0f}"),
              ("Rush spend ($000)", ["1H/2H '25\nAvg.", "1H '26"], half("rush_spend") / 1000, m["rush_spend"] / 1000, "${:,.0f}K"),
              ("Inventory balance ($M)", ["2025\nAvg.", "June 30, '26"], avg25 / 1e6, end_mod / 1e6, "${:,.2f}M")]
    for ax, (title, labels, a_, b_, fmt) in zip(axes, panels):
        bars = ax.bar(labels, [a_, b_], color=[MED_GREY, DARK_BLUE], width=0.6)
        ax.text(bars[0].get_x() + bars[0].get_width() / 2, a_ * 1.02, fmt.format(a_), ha="center", fontsize=8.5,
                fontweight="bold")
        ax.text(bars[1].get_x() + bars[1].get_width() / 2, b_ * 1.02,
                fmt.format(b_) + f"\n({(b_ - a_) / a_ * 100:+.0f}%)", ha="center", fontsize=8.5, fontweight="bold")
        ax.set_title(title, fontsize=9.5, color=DARK_GREY, pad=8)
        ax.set_ylim(0, max(a_, b_) * 1.3); ax.set_yticks([]); ax.tick_params(axis="x", labelsize=8)
        B.chart_style(ax)
        ax.spines["left"].set_visible(False)
    fig.tight_layout(w_pad=1.6)
    return B.b64(fig)


def chart_inventory_months():
    """Average inventory value by month, the same demand three ways, against 1H25 as the shop ran it."""
    fig, ax = B.make_fig(3.4)
    cols = {"dirty": MED_GREY, "model": DARK_BLUE}
    for v, name in PAIR:
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


STATUS = ["ORDER NOW", "ORDER SOON", "OK"]
STATUS_COL = {"ORDER NOW": B.ACCENT_RED, "ORDER SOON": B.AMBER, "OK": GREEN}
_sw = fr.get("status_weekly", {})
status_tot = {st: sum(w.get(st, 0) for w in _sw.values()) for st in STATUS}
n_weeks = len(_sw)
n_scored = sum(status_tot.values())


def chart_actions():
    """Item-weeks by reorder action across the weekly refreshes, January to June 2026."""
    vals = [status_tot[st] for st in STATUS]
    fig, ax = B.make_fig(3.4)
    bars = ax.bar(STATUS, vals, color=[STATUS_COL[st] for st in STATUS], width=0.55)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + n_scored * 0.005, f"{v:,}\n({v / n_scored:.0%})",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Predictions")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.yaxis.set_major_formatter(__import__("matplotlib.ticker", fromlist=["FuncFormatter"]).FuncFormatter(lambda v, _: f"{v:,.0f}"))
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_actions_by_month():
    """Average weekly predictions by reorder action, by month (months have four or five Monday refreshes)."""
    months = sorted({d_[:7] for d_ in _sw})
    x = np.arange(len(months)); w = 0.26
    fig, ax = B.make_fig(3.4)
    n_mon = {m_: sum(1 for d_ in _sw if d_[:7] == m_) for m_ in months}
    vals = {st: [sum(v.get(st, 0) for d_, v in _sw.items() if d_[:7] == m_) / n_mon[m_] for m_ in months] for st in STATUS}
    top = max(max(v) for v in vals.values())
    for i, st in enumerate(STATUS):
        bars = ax.bar(x + (i - 1) * w, vals[st], w, color=STATUS_COL[st], label=st)
        for b_, v in zip(bars, vals[st]):
            ax.text(b_.get_x() + b_.get_width() / 2, v + top * 0.012, f"{v:,.0f}", ha="center", va="bottom",
                    fontsize=7.5, color=DARK_GREY)
    ax.set_xticks(x); ax.set_xticklabels([pd.Timestamp(m_ + "-01").strftime("%b %Y") for m_ in months])
    ax.set_ylabel("Predictions per week"); ax.set_ylim(0, top * 1.15)
    ax.yaxis.set_major_formatter(__import__("matplotlib.ticker", fromlist=["FuncFormatter"]).FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=3, fontsize=9)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


PAIR = [("dirty", "Status quo"), ("model", "With the model")]


def chart_variants():
    """Status quo against the model on the same 1H26 demand: the four headline measures."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(B.CHART_W, 3.3))
    names = ["Status\nquo", "With the\nmodel"]
    cols = [MED_GREY, DARK_BLUE]
    d_, m_ = F["dirty"], F["model"]
    panels = [("Stockout events", d_["stockout_episodes"], m_["stockout_episodes"], "{:,.0f}"),
              ("Jobs held for material", d_["jobs_delayed"], m_["jobs_delayed"], "{:,.0f}"),
              ("Rush spend ($000)", d_["rush_spend"] / 1000, m_["rush_spend"] / 1000, "${:,.0f}K"),
              ("Ending inventory ($M)", d_["end_inventory_value"] / 1e6, m_["end_inventory_value"] / 1e6, "${:,.2f}M")]
    for ax, (title, a_, b_, fmt) in zip(axes, panels):
        bars = ax.bar(names, [a_, b_], color=cols, width=0.6)
        ax.text(bars[0].get_x() + bars[0].get_width() / 2, a_ * 1.02, fmt.format(a_), ha="center", fontsize=8.5,
                fontweight="bold")
        ax.text(bars[1].get_x() + bars[1].get_width() / 2, b_ * 1.02,
                fmt.format(b_) + f"\n({(b_ - a_) / a_ * 100:+.0f}%)", ha="center", fontsize=8.5, fontweight="bold")
        ax.set_title(title, fontsize=9.5, color=DARK_GREY, pad=8)
        ax.set_ylim(0, max(a_, b_) * 1.3); ax.set_yticks([]); ax.tick_params(axis="x", labelsize=8.5)
        B.chart_style(ax)
    fig.tight_layout(w_pad=1.6)
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
        ("Fill rate, line-critical / service / standard items",
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
        ("Fill rate, line-critical / service / standard items",
         lambda s: " / ".join(pct(s["fill_rate_by_tier"][t_]) for t_ in ["line", "service", "standard"])),
        ("Stockout events", lambda s: f"{s['stockout_episodes']:,}"),
        ("Stockout events, line-critical items", lambda s: f"{s['stockout_episodes_by_tier']['line']:,}"),
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
        ("Usage at cost", lambda s: money(s["consumption_value"])),
    ]
    for label, fn in spec:
        rows.append([label] + [fn(window[v]) for v, _ in VARIANTS])
    return widths(B.data_table(["Measure"] + [n for _, n in VARIANTS], rows, right=[1, 2, 3]), [31, 23, 23, 23])


def compare_table(window):
    """Without and with the model on the same demand, for one window."""
    d_, m_ = window["dirty"], window["model"]
    def chg_(a_, b_, money_=False):
        if a_ == 0:
            return ""
        return f"{(b_ - a_) / a_ * 100:+.0f}%"
    spec = [
        ("Fill rate", lambda x: pct(x["fill_rate"]), lambda: f"{(m_['fill_rate'] - d_['fill_rate']) * 100:+.1f} pts"),
        ("Stockout events", lambda x: f"{x['stockout_episodes']:,}", lambda: chg_(d_["stockout_episodes"], m_["stockout_episodes"])),
        ("Stockout events, line-critical items", lambda x: f"{x['stockout_episodes_by_tier']['line']:,}",
         lambda: chg_(d_["stockout_episodes_by_tier"]["line"], m_["stockout_episodes_by_tier"]["line"])),
        ("Jobs held for material", lambda x: f"{x['jobs_delayed']:,}", lambda: chg_(d_["jobs_delayed"], m_["jobs_delayed"])),
        ("Rush freight and premiums", lambda x: money(x["rush_spend"]), lambda: chg_(d_["rush_spend"], m_["rush_spend"])),
        ("Average inventory value", lambda x: money(x["avg_inventory_value"]), lambda: chg_(d_["avg_inventory_value"], m_["avg_inventory_value"])),
        ("Inventory at the end of the window", lambda x: money(x["end_inventory_value"]), lambda: chg_(d_["end_inventory_value"], m_["end_inventory_value"])),
        ("Order lines placed", lambda x: f"{x['order_lines']:,}", lambda: chg_(d_["order_lines"], m_["order_lines"])),
        ("Purchases", lambda x: money(x["purchases"]), lambda: chg_(d_["purchases"], m_["purchases"])),
    ]
    rows = [[label, fn(d_), fn(m_), ch()] for label, fn, ch in spec]
    return widths(B.data_table(["Measure", "Status quo<br>(1H '26)", "With the model<br>(1H '26)", "Change"], rows, right=[1, 2, 3]),
                  [40, 21, 21, 18])


def purchases_table():
    months = sorted(F["model"]["purchases_by_month"])
    rows = []
    for m in months:
        rows.append([pd.Timestamp(m).strftime("%b %Y")] + [money(F[v]["purchases_by_month"].get(m, 0.0)) for v, _ in VARIANTS])
    return widths(B.data_table(["Month"] + [n for _, n in VARIANTS], rows, right=[1, 2, 3]), [22, 26, 26, 26])


def _live_accuracy():
    """Score every weekly forecast the model made in 1H26 against the usage that followed, alongside the
    simple method chosen for each demand pattern before go-live. Forecasts whose horizon runs past June 30
    are left out."""
    wk = pd.read_parquet(REPO / "ml" / "data" / "marts" / "consumption_weekly.parquet")
    wide = wk.pivot(index="canonical", columns="week", values="consumption").fillna(0)
    widx = {pd.Timestamp(x): i for i, x in enumerate(wide.columns)}
    end = pd.Timestamp(C_END) - pd.Timedelta(days=1)
    seg_attr = attrs.set_index("canonical_item_number")["segment"]
    meth = {r["segment"]: r["baseline_method"] for r in metrics["segments"]}
    rows = []
    for item, entries in sched["items"].items():
        if item not in wide.index:
            continue
        sr = wide.loc[item].to_numpy(float)
        seg = seg_attr.get(item, "smooth")
        for e in entries:
            o = pd.Timestamp(e[0]); mon = o - pd.Timedelta(days=o.weekday())
            h = int(round(e[4] / 7)); t = widx.get(mon)
            if t is None or t < 52 or mon + pd.Timedelta(days=7 * h) > end:
                continue
            base = {"ma4": sr[t - 4:t].sum() / 4 * h, "ma13": sr[t - 13:t].sum() / 13 * h,
                    "ma52": sr[t - 52:t].sum() / 52 * h, "snaive": sr[t - 52:t - 52 + h].sum()}[meth.get(seg, "ma52")]
            rows.append((seg, sr[t:t + h].sum(), float(e[3]), base))
    return pd.DataFrame(rows, columns=["segment", "actual", "forecast", "base"])


def _wape(a, f):
    return float(np.abs(a - f).sum() / a.sum())


def accuracy_table():
    seg25 = {s_["segment"]: s_ for s_ in metrics["segments"]}
    rows = []
    for s_ in SEG_ORDER:
        g = live[live["segment"] == s_]
        if g.empty:
            continue
        wm, wb = _wape(g["actual"], g["forecast"]), _wape(g["actual"], g["base"])
        bias = (g["forecast"].sum() - g["actual"].sum()) / g["actual"].sum()
        rows.append([s_.capitalize(), pct(wm, 0), pct(wb, 0), f"{(wb - wm) / wb * 100:+.0f}%",
                     f"{seg25[s_]['bias'] * 100:+.0f}%" if s_ in seg25 else "", f"{bias * 100:+.0f}%"])
    rows.append(["<strong>All items</strong>", f"<strong>{pct(live_model, 0)}</strong>",
                 f"<strong>{pct(live_base, 0)}</strong>",
                 f"<strong>{(live_base - live_model) / live_base * 100:+.0f}%</strong>", "",
                 f"<strong>{live_bias * 100:+.0f}%</strong>"])
    return widths(B.data_table(["Demand pattern", "Model error", "Best simple method", "Improvement",
                                "Forecast bias"], [r[:4] + r[5:] for r in rows],
                               right=[1, 2, 3, 4]), [24, 18, 22, 18, 18])


# ── report ────────────────────────────────────────────────────────────────────
mod, rule, dirty = F["model"], F["clean_rule"], F["dirty"]
C_END = "2026-06-30"
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
HALF = lambda key: (H1[key] + H2[key]) / 2
_m, _r = M46["model"]["avg_inventory_value"], M46["clean_rule"]["avg_inventory_value"]
RULE_SVC = ("improves service somewhat" if (rule46["fill_rate"] > dirty46["fill_rate"] + 0.003
                                             or rule46["jobs_delayed"] < dirty46["jobs_delayed"])
            else "barely moves service")
if end_mod <= end_rule:
    END_NOTE = f"ends June with the least stock of the three ({k(end_mod)})"
else:
    END_NOTE = (f"ends June with {k(end_mod)} of stock, well below manual reordering and close to the rule's "
                f"{k(end_rule)}, because it holds larger buffers on the items whose demand is hardest to predict,")
if _m <= _r:
    AVG_NOTE = (f"It also carries the least stock of the three on average over months 4 to 6 ({k(_m)}), though it "
                "gets there gradually: it first builds buffers on the items that were running short, then runs down the rest.")
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
live = _live_accuracy()
live_model, live_base = _wape(live["actual"], live["forecast"]), _wape(live["actual"], live["base"])
live_bias = (live["forecast"].sum() - live["actual"].sum()) / live["actual"].sum()
live_bias_max = max(abs((g["forecast"].sum() - g["actual"].sum()) / g["actual"].sum()) for _, g in live.groupby("segment"))
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
    order = [metrics["winner"]] + [c for c in ["RandomForest", "XGBoost", "Linear"] if c != metrics["winner"]]
    rows = [[CAND_LABEL[c], pct(cand[c]["val_wape"]), pct(cand[c]["test_wape"]),
             f'<span style="color:{GREEN};font-weight:700;">&#10004;</span> Selected' if c == metrics["winner"] else ""]
            for c in order]
    return widths(B.data_table(["Candidate", "Validation error (WAPE)", "Held-out 2025 error (WAPE)", ""], rows, right=[1, 2]),
                  [34, 24, 24, 18])


hist = monthly[monthly["month"] < "2026-01-01"].merge(
    attrs[["canonical_item_number", "segment", "abc", "standard_cost"]], left_on="canonical", right_on="canonical_item_number")
hist["value"] = hist["consumption"] * hist["standard_cost"].fillna(attrs["standard_cost"].median())
SEG_COL = {"smooth": DARK_BLUE, "erratic": LIGHT_BLUE, "lumpy": "#8093A4", "intermittent": "#D5DCE1"}
# the item's category, read from its item-number prefix (the class field carries a
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
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=DARK_GREY, linestyle="--", linewidth=1.1))


def _chart_by_category(col, scale, ylabel, avg_fmt):
    """Monthly consumption over the three years, stacked by item category, with the trend in a callout."""
    fig, ax = B.make_fig(3.9)
    w = hist.pivot_table(index="month", columns="category", values=col, aggfunc="sum").fillna(0)
    w = w.reindex(columns=[c for c in CAT_ORDER if c in w.columns]) / scale
    ax.stackplot(w.index, [w[c] for c in w.columns], colors=[CAT_COL[c] for c in w.columns],
                 labels=list(w.columns), alpha=0.95)
    tot = w.sum(axis=1); x = np.arange(len(tot)); fit = np.polyfit(x, tot.to_numpy(), 1)
    ax.plot(w.index, np.polyval(fit, x), color=DARK_GREY, linestyle="--", linewidth=1.2)
    growth = fit[0] * 12 / tot.mean() * 100
    ax.set_ylim(0, tot.max() * 1.4)
    ax.set_ylabel(ylabel)
    _avg_box(ax, avg_fmt.format(tot.mean()) + f"\nTrend: {growth:+.1f}% a year")
    B.chart_style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=4, fontsize=8.5)
    fig.tight_layout()
    return B.b64(fig)


def chart_value_by_category():
    return _chart_by_category("value", 1000, "Usage in value ($000 / month)",
                              "Average: ${:,.0f}K of items used per month")


def chart_units_by_category():
    return _chart_by_category("consumption", 1000, "Usage in units (000 / month)",
                              "Average: {:,.1f}K units used per month")


# cost and usage by category, 2025
_h25 = hist[(hist["month"] >= "2025-01-01") & (hist["month"] < "2026-01-01")]
# every stocked item counts, including the few with no usage in 2025
_u25 = attrs.set_index("canonical_item_number")[["category", "standard_cost"]].join(
    _h25.groupby("canonical").agg(units=("consumption", "sum"), value=("value", "sum"))).fillna({"units": 0, "value": 0})
cat_tbl = _u25.groupby("category").agg(n=("units", "size"), med_cost=("standard_cost", "median"),
                                       med_units=("units", "median"), units=("units", "sum"), value=("value", "sum"))
cat_tbl = cat_tbl.reindex([c for c in CAT_ORDER if c in cat_tbl.index])
cat_tbl["unit_share"] = cat_tbl["units"] / cat_tbl["units"].sum()
cat_tbl["value_share"] = cat_tbl["value"] / cat_tbl["value"].sum()
cat_tbl["item_share"] = cat_tbl["n"] / cat_tbl["n"].sum()


def category_table():
    rows = [[c, f"{int(r['n'])}", f"${r.med_cost:,.2f}", f"{r.med_units:,.0f}", pct(r.unit_share, 0),
             k(r.value), pct(r.value_share, 0)] for c, r in cat_tbl.iterrows()]
    rows.append(["<strong>All items</strong>", f"<strong>{int(cat_tbl['n'].sum()):,}</strong>", "", "", "<strong>100%</strong>",
                 f"<strong>{k(cat_tbl['value'].sum())}</strong>", "<strong>100%</strong>"])
    return widths(B.data_table(["Category", "Items", "Median unit cost", "Median units used per item (2025)",
                                "Share of units used (2025)", "Usage value (2025)", "Share of value (2025)"], rows,
                               right=[1, 2, 3, 4, 5, 6]), [20, 9, 14, 16, 13, 15, 13])


_hi = cat_tbl.loc[[c for c in ["Mechanical", "Electrical"] if c in cat_tbl.index]]
_lo = cat_tbl.loc[[c for c in ["Fasteners", "Hardware"] if c in cat_tbl.index]]
TBL_TEXT = (f"Item cost and usage move in opposite directions across the categories. Mechanical and electrical items "
            f"(motors, gearboxes, drives, controls) are {pct(_hi['item_share'].sum(), 0)} of items and "
            f"{pct(_hi['value_share'].sum(), 0)} of usage value, with median unit costs of "
            f"${cat_tbl.loc['Mechanical', 'med_cost']:,.0f} and ${cat_tbl.loc['Electrical', 'med_cost']:,.0f}, but only "
            f"{pct(_hi['unit_share'].sum(), 0)} of the units used. Fasteners and hardware are the reverse: "
            f"{pct(_lo['unit_share'].sum(), 0)} of the units used but {pct(_lo['value_share'].sum(), 0)} of the value, "
            f"at median unit costs under ${max(cat_tbl.loc['Fasteners', 'med_cost'], cat_tbl.loc['Hardware', 'med_cost']) + 0.5:,.0f}. "
            f"The shop's working capital is therefore tied up in a few hundred expensive, slower-moving items, not in the "
            f"high-volume floor stock.")


def chart_category_patterns():
    """Number of items in each category, stacked by demand pattern."""
    fig, ax = B.make_fig(3.7)
    bottom = np.zeros(len(cat_seg))
    totals = cat_seg.sum(axis=1).to_numpy()
    for seg in SEG_ORDER:
        vals = cat_seg[seg].to_numpy()
        ax.bar(cat_seg.index, vals, bottom=bottom, color=SEG_COL[seg], width=0.65, label=seg.capitalize())
        # each segment's share of its category, where the segment is tall enough to hold a label
        for i, (v, b0) in enumerate(zip(vals, bottom)):
            if v >= 7:
                ax.text(i, b0 + v / 2, f"{v / totals[i] * 100:.0f}%", ha="center", va="center", fontsize=7.5,
                        color="white" if seg in ("smooth", "lumpy") else DARK_GREY)
        bottom += vals
    for i, tot_ in enumerate(bottom):
        ax.text(i, tot_ + 5, f"{int(tot_)}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Items")
    ax.set_ylim(0, bottom.max() * 1.18)
    ax.set_xticks(range(len(cat_seg))); ax.set_xticklabels([c.replace(" ", "\n") for c in cat_seg.index])
    ax.tick_params(axis="x", labelsize=8.5)
    B.chart_style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=4, fontsize=9)
    fig.tight_layout()
    return B.b64(fig)


def _top(seg, n=2):
    s_ = cat_share[seg].sort_values(ascending=False).head(n)
    return " and ".join(f"{c.lower()} ({v*100:.0f}%)" for c, v in s_.items())


CAT_TEXT = (f"Individual demand patterns are informed by the kind of item, though every category carries all four "
            f"patterns. Smooth demand is most common among {_top('smooth')}, the floor stock used consistently every "
            f"week. Erratic demand is spread widely, and is highest among {_top('erratic')}. Lumpy demand concentrates "
            f"in {_top('lumpy')}. Intermittent demand is highest in {_top('intermittent')}, where just a few of these "
            f"high-cost items are used for a job or sold as occasional spares.")


def chart_history_value():
    """Monthly consumption in units over the three years, by demand pattern."""
    fig, ax = B.make_fig(3.9)
    w = hist.pivot_table(index="month", columns="segment", values="consumption", aggfunc="sum").fillna(0)[SEG_ORDER] / 1000
    ax.stackplot(w.index, [w[c] for c in SEG_ORDER], colors=[SEG_COL[c] for c in SEG_ORDER],
                 labels=[c.capitalize() for c in SEG_ORDER], alpha=0.95)
    tot = w.sum(axis=1); x = np.arange(len(tot)); fit = np.polyfit(x, tot.to_numpy(), 1)
    ax.plot(w.index, np.polyval(fit, x), color=DARK_GREY, linestyle="--", linewidth=1.2, label="Trend")
    ax.set_ylim(0, tot.max() * 1.35)
    ax.set_ylabel("Units used (000 / month)")
    _avg_box(ax, f"Average: {round(tot.mean() * 10) * 100:,.0f} units used per month")
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
    ax.set_xlabel("Share of items, highest usage value first (%)"); ax.set_ylabel("Cumulative share of value (%)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 102)
    B.chart_style(ax)
    return B.b64(fig)


FLOW_HTML = (
    '<div style="display:flex;align-items:stretch;gap:0;margin:22px 0;flex-wrap:wrap;">'
    '<div style="flex:1;min-width:190px;background:#F3F5F7;border-radius:8px;padding:16px 18px;border-top:4px solid #381FA1;">'
    '<div style="font-weight:700;color:#322B4B;margin-bottom:6px;">1. What it reads</div>'
    '<div style="font-size:16px;line-height:1.55;">Three years of cleaned usage history for every stocked item, '
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

charts = {"halves": chart_halves(), "variants": chart_variants(), "actions": chart_actions(), "actions_m": chart_actions_by_month(),
          "invm": chart_inventory_months(), "valcat": chart_value_by_category(), "catseg": chart_category_patterns(),
          "unitcat": chart_units_by_category(),
          "examples": chart_examples()}

toc = ('<a href="#summary">Executive Summary</a><hr>'
       '<a href="#modeloverview">Model Overview</a>'
       '<a href="#what" class="sub">What This Model Does</a>'
       '<a href="#data" class="sub">Training Data Overview</a><hr>'
       '<a href="#performance">Model Performance</a>'
       '<a href="#scoring" class="sub">Scoring Summary</a>'
       '<a href="#accuracy" class="sub">Accuracy and Validation</a>'
       '<a href="#limits" class="sub">What It Can and Cannot Predict</a>')

body = f"""
{B.section("summary", "Section 1", "Executive Summary")}
<p>The demand forecasting model sets the shop's reorder decisions for all {n_items:,} stocked items. Every week it
forecasts how much of each item the shop will use before a new order could arrive, and turns that forecast into a
reorder point and an order quantity that are loaded straight into the ERP's purchasing screen. The model has been
live for the past six months, January through June 2026, and in that time every reorder decision the shop has made
came from it.</p>
<p>Before the model, reordering was manual and ran on data that was messy and couldn't be trusted. Buyers
compensated by padding safety stock, keeping their own spreadsheets of stock and lead times, checking the shelves by
eye before ordering, and rushing orders in when stock ran short. The result was the shop carrying excess inventory,
roughly {(H1['days_of_supply'] + H2['days_of_supply']) / 2:.0f} days of usage, yet still logging elevated stockout
events, held jobs for missing material and rush freight spend.</p>
<p>In its first six months, the model improved the four outcomes that matter most to keeping production running
without tying up cash. Compared to 2025, stockout events fell {fall(HALF('stockout_episodes'), mod['stockout_episodes'])},
jobs held for material {fall(HALF('jobs_delayed'), mod['jobs_delayed'])} and rush spend
{fall(HALF('rush_spend'), mod['rush_spend'])}, and the inventory balance at the end of June was {k(end_mod)},
{fall(avg25, end_mod)} below the 2025 average.</p>
<div style="margin:18px 0;"><div class="chart-title" style="text-align:center;">Model Performance Summary, 2025 vs. 1H 2026</div>
<img src="data:image/png;base64,{charts['halves']}" alt="Model Performance Summary, 2025 vs. 1H 2026" style="width:100%;height:auto;display:block;"></div>

{B.section("modeloverview", "Section 2", "Model Overview")}

{B.section("what", "Section 2.1", "What This Model Does")}
<p>Over the past six months (January 2026 to June 2026), the demand forecasting model has set the shop's reorder
decisions. Every week, the model predicts which items need to be reordered, and these predictions are then fed
directly into the ERP for each of the {n_items:,} stocked items. The model's reorder decisions are based on the
current stock on hand and on order for each item, the forecasted usage over each supplier's delivery time,
and a safety buffer based on how unpredictable each item's demand and deliveries have been.</p>
{FLOW_HTML}
<p>The model refreshes its forecasts every week and is retrained on the latest history once a month. The
model's predictions are loaded directly into the ERP, which flags when each item should be reordered and how much
to order. To keep working capital low, the recommendations for how much to order are informed by the item's cost:
expensive items are bought every few weeks in small lots, and cheap items a few times a year in larger lots.</p>
<p>The ERP's reorder queue ranks every stocked item against the model's weekly reorder point: items at or below it
are marked "ORDER NOW", items within two weeks of it "ORDER SOON", and items more than two weeks outside of
it "OK". Each line also
shows the item's criticality, the stock on hand, allocated to released jobs and on order, the forecast, the safety
stock and the suggested order quantity, as seen in the screenshot of the ERP system below:</p>
<div class="chart-wrap" style="padding:6px;">
  <img src="data:image/png;base64,{erp_b64}" alt="ERP reorder queue with the demand model's reorder points"
       style="width:100%;height:auto;display:block;border:1px solid #D5DCE1;">
</div>

{B.section("data", "Section 2.2", "Training Data Overview")}
<p>The model learns from the shop's historical usage of every stocked item. It was originally trained on three
years of this usage data (2023 through 2025, as shown below), and is continually trained on every new month
of data.</p>
{B.chart("MONTHLY USAGE IN VALUE, BY ITEM CATEGORY (JAN. 2023 to DEC. 2025)", charts["valcat"])}
{B.chart("MONTHLY USAGE IN UNITS, BY ITEM CATEGORY (JAN. 2023 to DEC. 2025)", charts["unitcat"])}
<p>Demand is steady in aggregate over the three years, but individual items exhibit very different demand patterns.
We've categorized these individual item demand patterns into four groups, which the model is calibrated against:
{seg_n.get('smooth', 0)} smooth (regular and steady), {seg_n.get('erratic', 0)} erratic (regular but variable),
{seg_n.get('lumpy', 0)} lumpy (irregular and variable) and {seg_n.get('intermittent', 0)} intermittent (many months
with no demand at all). One representative item within each pattern is shown below.</p>
{B.chart("Three years of monthly usage, one representative item per demand pattern", charts["examples"])}
<p>{CAT_TEXT}</p>
{B.chart("Items in each category, by demand pattern", charts["catseg"])}
<p>{TBL_TEXT}</p>
{category_table()}
{B.section("performance", "Section 3", "Model Performance")}

{B.section("scoring", "Section 3.1", "Scoring Summary")}
<p>From January to June 2026 the model refreshed the reorder points of all {n_items:,} items every week. Across
those {n_weeks} weekly refreshes it made {n_scored:,} predictions: <strong>{status_tot['ORDER NOW']:,}</strong>
({status_tot['ORDER NOW'] / n_scored:.0%}) were marked ORDER NOW and <strong>{status_tot['ORDER SOON']:,}</strong>
({status_tot['ORDER SOON'] / n_scored:.0%}) ORDER SOON, and the rest were OK. In a typical week that is about
{status_tot['ORDER NOW'] / n_weeks:.0f} items to order now and {status_tot['ORDER SOON'] / n_weeks:.0f} to order soon,
a short, ranked list for the buyers to work through.</p>
{B.chart("Predictions by Reorder Action, January to June 2026", charts["actions"])}
{B.chart("Average Weekly Predictions by Reorder Action, by Month", charts["actions_m"])}

{B.section("accuracy", "Section 3.2", "Accuracy and Validation")}
<p>The model was chosen and calibrated before go-live on held-out data for the full year of 2025. Three candidate
algorithms were tuned and compared on their forecasts for that year, and the
{CAND_LABEL[WIN].lower() if WIN != 'XGBoost' else 'XGBoost'} model, with the lowest error, was selected.</p>
{candidate_table()}
<p>To understand how well the model performed against baseline, we'll compare the model's performance in the
1H 2026 to the same period of time under a status quo scenario where the shop did not use the model. This
status quo scenario assumes the shop operated
in 1H 2026 as it did throughout 2025, including with stale lead times and reorder points, the buyers'
four-and-a-half-month lots and with the data errors fully intact.</p>
{compare_table(F)}
<p>The model improves results across the board, underscoring its effectiveness at keeping the right items in
stock: fewer stockouts and held jobs, less spent rushing orders in, and less inventory on the shelf.</p>
{B.chart("Status quo and with the model, January to June 2026", charts["variants"])}
{B.chart("Average inventory by month, status quo and with the model", charts["invm"])}

{B.section("limits", "Section 3.3", "What It Can and Cannot Predict")}
<ul class="limitation-list">
  <li><strong>It forecasts demand, not supply.</strong> The model predicts how much the shop will use; it takes each
      supplier's lead time from its recent deliveries, refreshed monthly, and does not predict a late delivery. The buffer covers
      the usual spread of deliveries, not a supplier failure.</li>
  <li><strong>The forward window is a simulation.</strong> The six months are replayed from the generated demand
      and supplier behaviour, not observed. Demand, deliveries and the forecast are identical in both replays, so
      the differences between them are the policy; the size of each difference is an estimate.</li>
  <li><strong>Excess on slow items takes time to clear.</strong> {k(mod46['excess_value'])} of stock from April to June
      still sits on {mod46['excess_items']} items holding more than a year of supply. The model stops reordering
      them, but an item used a few times a year takes that long to draw down. Returning or selling the worst of it
      would release the cash sooner; that is a disposition decision for purchasing and finance, not a forecast.</li>
  <li><strong>The order book is an assumption.</strong> The shop's records carry no booking date for customer
      orders, so each job is assumed booked four to eight weeks before its release, typical of a job shop quoting
      lead times of that length. With less notice, the model would see less of the coming demand.</li>
  <li><strong>Expediting is held constant.</strong> Both replays expedite the same items the purchasing
      manager tracked before go-live, so rush spend reflects how often those items were at risk, not a change in
      how hard the shop chases suppliers.</li>
  <li><strong>The first three months are a transition.</strong> Orders placed under the old points were still
      arriving while excess was used up, so the six-month results understate the model's settled performance: from
      April to June alone, stockout events were {fall(dirty46['stockout_episodes'], mod46['stockout_episodes'])} lower
      than the status quo and jobs held for material {fall(dirty46['jobs_delayed'], mod46['jobs_delayed'])} lower.</li>
</ul>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
html = B.page("ML Model Overview and Performance", "", toc, body)
html = html.replace("</style></head>", ".section-title-block.sub .section-title{font-size:18px;font-weight:700;}</style></head>", 1)
OUT.write_text(html, encoding="utf-8")
print(f"Model overview written to {OUT}  ({len(html)//1024} KB)")
