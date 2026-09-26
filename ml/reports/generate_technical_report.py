"""ML Model Technical Overview for the demand forecast and reorder policy -> docs/reports/technical_report.html

Mirrors Case 02's technical overview (Model Card, Training Data, Model Selection &
Performance, Feature Importance, Known Limitations, Deployment & Operations) and adds
the section this case needs: how a usage forecast becomes a reorder point and an
order quantity. Every figure is read from the weekly backtest, the explainability
step, the reorder-point schedule and the forward replay.

    python -m ml.src.explain_weekly        (once, after forward_policy)
    PYTHONIOENCODING=utf-8 "../mfg-oee-maintenance/.venv/Scripts/python.exe" -m ml.reports.generate_technical_report
"""
import json
from math import erf, exp, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from . import brand as B
from .brand import DARK_BLUE, LIGHT_BLUE, MED_GREY, LIGHT_GREY, GREEN, AMBER, ACCENT_RED, DARK_GREY

REPO = Path(__file__).resolve().parents[2]
BACKTEST = REPO / "ml" / "data" / "backtest"
EXPLAIN = BACKTEST / "explain"
MARTS = REPO / "ml" / "data" / "marts"
POLICY = REPO / "ml" / "data" / "policy"
TRUTH = REPO / "data_source" / "truth"
OUT = REPO / "docs" / "reports" / "technical_report.html"

metrics = json.loads((BACKTEST / "weekly_metrics.json").read_text(encoding="utf-8"))
split = json.loads((EXPLAIN / "split_summary.json").read_text(encoding="utf-8"))
sched = json.loads((POLICY / "rop_schedule.json").read_text(encoding="utf-8"))
fr = json.loads((TRUTH / "forward_results.json").read_text(encoding="utf-8"))
bt = pd.read_parquet(BACKTEST / "weekly_backtest.parquet")
imp = pd.read_csv(EXPLAIN / "shap_importance.csv")
lc = pd.read_csv(EXPLAIN / "learning_curve.csv")
fcorr = pd.read_csv(EXPLAIN / "feature_corr.csv", index_col=0)["spearman"]
cmat = pd.read_csv(EXPLAIN / "feature_corr_matrix.csv", index_col=0)
tsample = pd.read_parquet(EXPLAIN / "target_sample.parquet")
attrs = pd.read_parquet(MARTS / "item_attributes.parquet").set_index("canonical_item_number")
weekly = pd.read_parquet(MARTS / "consumption_weekly.parquet")

WIN = metrics["winner"]
N_TRIALS = 12          # Optuna trials per candidate (forward_policy.N_TRIALS)
from data_source.generate import config as _GC
SEASONAL_SHARE = _GC.SEASONAL_ITEM_SHARE
LABEL = {"Linear": "Linear regression (ridge)", "RandomForest": "Random forest", "XGBoost": "XGBoost"}
SEG_ORDER = ["smooth", "erratic", "lumpy", "intermittent"]
TIERS = [("line", "Production items"), ("service", "Spare parts"), ("standard", "Shop supplies")]
FT = sched["fill_rate_target"]
tier_of = sched["criticality"]
MOD = fr["forward"]["1H26"]["model"]
DIRTY = fr["forward"]["1H26"]["dirty"]
M46M = fr["forward"]["1H26_months_4_6"]["model"]


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def tgt(x):
    return f"{round(x * 100, 1):g}%"


def wape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.abs(a - f).sum() / a.sum())


def widths(table_html, w):
    cols = "".join(f'<col style="width:{v}%;">' for v in w)
    return table_html.replace('<table class="data-table">',
                              f'<table class="data-table" style="table-layout:fixed;"><colgroup>{cols}</colgroup>', 1)


# ── live forecasts, January to June 2026 ────────────────────────────────────
def _live():
    """Every weekly forecast made in 1H26, scored against the usage that followed. Forecasts whose
    horizon runs past June 30 are left out."""
    wide = weekly.pivot(index="canonical", columns="week", values="consumption").fillna(0)
    widx = {pd.Timestamp(x): i for i, x in enumerate(wide.columns)}
    end = pd.Timestamp("2026-06-29")
    meth = {r["segment"]: r["baseline_method"] for r in metrics["segments"]}
    rows = []
    for item, entries in sched["items"].items():
        if item not in wide.index or item not in attrs.index:
            continue
        sr = wide.loc[item].to_numpy(float)
        seg = attrs.loc[item, "segment"]
        for e in entries:
            o = pd.Timestamp(e[0]); mon = o - pd.Timedelta(days=o.weekday())
            h = int(round(e[4] / 7)); t = widx.get(mon)
            if t is None or t < 52 or mon + pd.Timedelta(days=7 * h) > end:
                continue
            base = {"ma4": sr[t - 4:t].sum() / 4 * h, "ma13": sr[t - 13:t].sum() / 13 * h,
                    "ma52": sr[t - 52:t].sum() / 52 * h, "snaive": sr[t - 52:t - 52 + h].sum()}[meth.get(seg, "ma52")]
            rows.append((item, seg, attrs.loc[item, "abc"], tier_of.get(item, "standard"), sr[t:t + h].sum(),
                         float(e[3]), float(e[2]), base))
    return pd.DataFrame(rows, columns=["item", "segment", "abc", "tier", "actual", "forecast", "ss", "base"])


live = _live()
live_w, live_b = wape(live["actual"], live["forecast"]), wape(live["actual"], live["base"])
live_bias = (live["forecast"].sum() - live["actual"].sum()) / live["actual"].sum()
live["covered"] = live["actual"] <= live["forecast"] + live["ss"]
test_w = metrics["overall"]["model"]
test_bias = (bt["pred_c"].sum() - bt["actual"].sum()) / bt["actual"].sum()
test_w_c = wape(bt["actual"], bt["pred_c"])
val_w = metrics["candidates"][WIN]["val_wape"]

# standardized 2025 errors, as the buffer calibration uses them
_resid = (bt["actual"] - bt["pred_c"]).groupby(bt["item"]).transform("std")
z_err = ((bt["actual"] - bt["pred_c"]) / _resid).replace([np.inf, -np.inf], np.nan).dropna()

# the latest schedule entry per item: lot size in days of forecast usage
_last = []
for item, entries in sched["items"].items():
    e = entries[-1]
    d = e[3] / max(1.0, e[4])
    if d > 0 and len(e) > 7 and item in attrs.index:
        _last.append((attrs.loc[item, "abc"], e[7] / d, e[2] / d))
lots = pd.DataFrame(_last, columns=["abc", "lot_days", "ss_days"])

_m25 = pd.read_parquet(MARTS / "consumption_monthly.parquet")
_u25 = _m25[(_m25["month"] >= "2025-01-01") & (_m25["month"] < "2026-01-01")].groupby("canonical")["consumption"].sum()
_v = attrs.join(_u25.rename("u25")).fillna({"u25": 0})
_v["v25"] = _v["u25"] * _v["standard_cost"]
ABCV = _v.groupby("abc").agg(med_val=("v25", "median"), p10=("v25", lambda x: x.quantile(0.1)),
                              med_cost=("standard_cost", "median"))


def k_(x):
    return f"${x / 1000:,.1f}K" if x >= 1000 else f"${x:,.0f}"


# ── charts ──────────────────────────────────────────────────────────────────
def chart_volume():
    """Total weekly usage across all items, shaded by the window each week feeds."""
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    tot = weekly.groupby("week")["consumption"].sum()
    tot = tot[tot.index < pd.Timestamp("2026-06-29")] / 1000
    bounds = {k: [pd.Timestamp(x) for x in split[k]["window"].split(" to ")] for k in ["train", "validation", "test"]}

    def col(w):
        if w < bounds["train"][0]:
            return LIGHT_GREY, "History only (feature warm-up)"
        if w <= bounds["train"][1]:
            return DARK_BLUE, "Train"
        if w <= bounds["validation"][1]:
            return LIGHT_BLUE, "Validation"
        if w < pd.Timestamp("2026-01-01"):
            return AMBER, "Held-out test (2025)"
        return GREEN, "Live (1H 2026)"
    fig, ax = B.make_fig(3.1)
    ax.bar(tot.index, tot.values, width=6, color=[col(w)[0] for w in tot.index])
    ax.set_ylabel("Units used (000 / week)")
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    seen = {}
    for w in tot.index:
        c, l = col(w); seen[l] = c
    ax.legend(handles=[Patch(color=c, label=l) for l, c in seen.items()], fontsize=8, ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_target():
    fig, ax = B.make_fig(3.1)
    bins = np.linspace(0, np.log1p(tsample["target"].quantile(0.995)), 40)
    for name, c in [("Train", DARK_BLUE), ("Validation", LIGHT_BLUE), ("Test (2025)", AMBER)]:
        v = np.log1p(tsample.loc[tsample["split"] == name, "target"])
        ax.hist(v, bins=bins, density=True, histtype="step", linewidth=2, color=c, label=name)
    ticks = [0, 1, 10, 100, 1000, 10000]
    ax.set_xticks([np.log1p(t) for t in ticks if np.log1p(t) <= bins[-1]])
    ax.set_xticklabels([f"{t:,}" for t in ticks if np.log1p(t) <= bins[-1]])
    ax.set_xlabel("Units used over the lead time (target, log scale)"); ax.set_ylabel("Density"); ax.legend()
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_corr():
    from matplotlib.colors import LinearSegmentedColormap
    feats = list(cmat.columns)
    cmap = LinearSegmentedColormap.from_list("brand_div", [DARK_BLUE, "#FFFFFF", ACCENT_RED])
    fig, ax = B.make_fig(5.6)
    im = ax.imshow(cmat.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=7.5); ax.set_yticklabels(feats, fontsize=7.5)
    ax.set_xticks(np.arange(-.5, len(feats), 1), minor=True); ax.set_yticks(np.arange(-.5, len(feats), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6); ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    return B.b64(fig)


def chart_learning():
    fig, ax = B.make_fig(3.1)
    ax.plot(lc["train_rows"], lc["train_wape"] * 100, "o-", color=DARK_BLUE, lw=2, label="Training rows (in sample)")
    ax.plot(lc["train_rows"], lc["test_wape"] * 100, "s-", color=LIGHT_BLUE, lw=2, label="Held-out 2025")
    ax.set_xlabel("Training rows"); ax.set_ylabel("WAPE (%)"); ax.legend()
    ax.xaxis.set_major_formatter(__import__("matplotlib.ticker", fromlist=["FuncFormatter"]).FuncFormatter(lambda v, _: f"{v:,.0f}"))
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_calibration():
    """Mean actual against mean forecast within forecast deciles, held-out 2025 and live 1H26."""
    fig, ax = B.make_fig(3.5)
    top = 0
    for df, a_, f_, c, lab in [(bt, "actual", "pred_c", MED_GREY, "Held-out 2025"),
                               (live, "actual", "forecast", DARK_BLUE, "Live 1H 2026")]:
        d = df[df[f_] > 0].copy()
        d["bin"] = pd.qcut(d[f_].rank(method="first"), 10, labels=False)
        g = d.groupby("bin")[[f_, a_]].mean()
        ax.plot(g[f_], g[a_], "o-", color=c, lw=2, label=lab)
        top = max(top, g[f_].max(), g[a_].max())
    ax.plot([0.5, top * 1.1], [0.5, top * 1.1], color=ACCENT_RED, ls="--", lw=1.2, label="Perfectly calibrated")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Mean forecast in decile (units, log scale)"); ax.set_ylabel("Mean actual usage (units, log scale)")
    ax.legend(loc="upper left")
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_by_pattern():
    fig, ax = B.make_fig(3.1)
    x = np.arange(len(SEG_ORDER)); w = 0.36
    m_ = [wape(live.loc[live.segment == s, "actual"], live.loc[live.segment == s, "forecast"]) * 100 for s in SEG_ORDER]
    b_ = [wape(live.loc[live.segment == s, "actual"], live.loc[live.segment == s, "base"]) * 100 for s in SEG_ORDER]
    for off, vals, c, lab in [(-w / 2, b_, MED_GREY, "Best simple method"), (w / 2, m_, DARK_BLUE, "Model")]:
        bars = ax.bar(x + off, vals, w, color=c, label=lab)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in SEG_ORDER])
    ax.set_ylabel("WAPE, live 1H 2026 (%)"); ax.set_ylim(0, max(b_ + m_) * 1.2); ax.legend(ncol=2)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_pred_actual():
    import matplotlib.pyplot as plt
    d = live[(live["actual"] > 0) & (live["forecast"] > 0)].sample(n=min(6000, len(live)), random_state=3)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.scatter(d["actual"], d["forecast"], s=6, alpha=0.18, color=DARK_BLUE, edgecolors="none")
    lo, hi = 0.8, max(d["actual"].max(), d["forecast"].max()) * 1.2
    ax.plot([lo, hi], [lo, hi], color=ACCENT_RED, ls="--", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual usage over the lead time (units)"); ax.set_ylabel("Forecast (units)")
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_shap():
    d = imp.head(14).iloc[::-1]
    names = [FEAT_DESC.get(f, (f, ""))[0] for f in d["feature"]]
    fig, ax = B.make_fig(4.2)
    ax.barh(names, d["mean_abs_shap"], color=DARK_BLUE, height=0.66)
    for i, v in enumerate(d["mean_abs_shap"]):
        ax.text(v + d["mean_abs_shap"].max() * 0.01, i, f"{v:.2f}", va="center", fontsize=8.5, color=MED_GREY)
    ax.set_xlabel("Mean |SHAP| (effect on log usage)")
    ax.tick_params(axis="y", labelsize=8.5)
    B.chart_style(ax); ax.xaxis.grid(True, color=LIGHT_GREY); ax.yaxis.grid(False)
    fig.tight_layout()
    return B.b64(fig)


def _normal_loss(k):
    phi = exp(-k * k / 2) / sqrt(2 * pi)
    return phi - k * (1 - 0.5 * (1 + erf(k / sqrt(2))))


def chart_loss():
    ks = np.linspace(0, 3.5, 71)
    zs = z_err.to_numpy()
    emp = [np.maximum(zs - k, 0).mean() for k in ks]
    fig, ax = B.make_fig(3.2)
    ax.plot(ks, [_normal_loss(k) for k in ks], color=MED_GREY, lw=2, ls="--", label="Normal distribution (textbook)")
    ax.plot(ks, emp, color=DARK_BLUE, lw=2.2, label="The model's actual 2025 errors")
    ax.set_yscale("log")
    ax.set_xlabel("Safety buffer, in standard errors of the forecast (k)")
    ax.set_ylabel("Expected shortfall per cycle\n(standard errors, log scale)")
    ax.legend()
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_lots():
    fig, ax = B.make_fig(3.0)
    bins = np.linspace(sched["lot_days"][0], sched["lot_days"][1], 22)
    for ab, c in [("A", DARK_BLUE), ("B", LIGHT_BLUE), ("C", MED_GREY)]:
        ax.hist(lots.loc[lots.abc == ab, "lot_days"], bins=bins, histtype="step", lw=2, color=c,
                label=f"{ab} items (median {lots.loc[lots.abc == ab, 'lot_days'].median():.0f} days)")
    ax.axvline(sched["buyer_lot_days"], color=ACCENT_RED, ls="--", lw=1.3)
    ax.text(sched["buyer_lot_days"] - 2, ax.get_ylim()[1] * 0.9, "buyers' old lot", ha="right", fontsize=8, color=ACCENT_RED)
    ax.set_xlabel("Order quantity, in days of forecast usage"); ax.set_ylabel("Items"); ax.legend(fontsize=8.5)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_coverage():
    fig, ax = B.make_fig(3.0)
    names = [n for _, n in TIERS]
    cov = [live.loc[live.tier == t, "covered"].mean() * 100 for t, _ in TIERS]
    fill = [MOD["fill_rate_by_tier"][t] * 100 for t, _ in TIERS]
    x = np.arange(len(TIERS)); w = 0.36
    for off, vals, c, lab in [(-w / 2, cov, LIGHT_BLUE, "Lead-time windows covered by forecast + buffer"),
                              (w / 2, fill, DARK_BLUE, "Fill rate achieved (units)")]:
        bars = ax.bar(x + off, vals, w, color=c, label=lab)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(75, 102); ax.set_ylabel("%")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=8.5)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


# ── tables ──────────────────────────────────────────────────────────────────
FEAT_DESC = {
    "s4": ("Usage, last 4 weeks", "Rolling usage"), "s13": ("Usage, last 13 weeks", "Rolling usage"),
    "s26": ("Usage, last 26 weeks", "Rolling usage"), "s52": ("Usage, last 52 weeks", "Rolling usage"),
    "nz13": ("Weeks with usage, last 13", "Rolling usage"), "nz52": ("Weeks with usage, last 52", "Rolling usage"),
    "since_nz": ("Weeks since last usage", "Rolling usage"), "cv13": ("Week-to-week variability, last 13 weeks", "Rolling usage"),
    "trend": ("Recent trend (4-week rate less 13-week rate)", "Rolling usage"),
    "ly": ("Usage in the same weeks last year", "Rolling usage"), "mean_all": ("Average weekly usage since history began", "Rolling usage"),
    "woy": ("Week of year", "Calendar"), "month": ("Month", "Calendar"),
    "std_cost": ("Standard unit cost", "Item attribute"), "lead": ("Supplier lead time (days)", "Item attribute"),
    "annual": ("Annual usage", "Item attribute"), "seg_code": ("Demand pattern", "Item attribute"),
    "abc_code": ("Value class (ABC)", "Item attribute"), "cls_code": ("Item category", "Item attribute"),
    "h": ("Forecast horizon (lead time, weeks)", "Item attribute"),
}
TCOL = {"Rolling usage": DARK_BLUE, "Calendar": AMBER, "Item attribute": LIGHT_BLUE}


def feature_table():
    rows = ""
    for f, (desc, typ) in FEAT_DESC.items():
        c = fcorr.get(f, np.nan)
        rows += (f'<tr><td style="font-family:monospace;font-size:13px;">{f}</td><td>{desc}</td>'
                 f'<td>{B.badge(typ, TCOL[typ])}</td>'
                 f'<td style="text-align:right;">{"-" if pd.isna(c) else f"{c:+.2f}"}</td></tr>')
    return (f'<table class="data-table"><thead><tr><th>Feature</th><th>Description</th><th>Type</th>'
            f'<th style="text-align:right;">Corr. with target</th></tr></thead><tbody>{rows}</tbody></table>')


def split_table():
    rows = [[name, split[k]["window"].replace(" to ", " to "), f"{split[k]['rows']:,}", f"{split[k]['items']:,}",
             f"{split[k]['target_median']:,.0f}", pct(split[k]["zero_share"], 0)]
            for name, k in [("Train", "train"), ("Validation", "validation"), ("Test", "test")]]
    rows.append(["Live", "2026-01-01 to 2026-06-29", f"{len(live):,}", f"{live['item'].nunique():,}",
                 f"{live['actual'].median():,.0f}", pct((live['actual'] == 0).mean(), 0)])
    return widths(B.data_table(["Split", "Forecast dates", "Forecasts", "Items", "Median target (units)", "Zero-usage share"],
                               rows, right=[2, 3, 4, 5]), [16, 28, 13, 10, 18, 15])


def candidate_table():
    order = [WIN] + [c for c in ["RandomForest", "XGBoost", "Linear"] if c != WIN]
    rows = ""
    for c in order:
        r = metrics["candidates"][c]
        sel = c == WIN
        tick = f' <span style="color:{GREEN};font-weight:700;">&#10004; Selected</span>' if sel else ""
        bg = ' style="font-weight:700;"' if sel else ""
        params = ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in r["params"].items())
        rows += (f'<tr{bg}><td>{LABEL[c]}{tick}</td><td style="text-align:right;">{pct(r["val_wape"])}</td>'
                 f'<td style="text-align:right;">{pct(r["test_wape"])}</td><td style="font-size:12.5px;">{params}</td></tr>')
    return widths(f'<table class="data-table"><thead><tr><th>Model</th><th style="text-align:right;">Validation WAPE</th>'
                  f'<th style="text-align:right;">Held-out 2025 WAPE</th><th>Tuned hyperparameters</th></tr></thead>'
                  f'<tbody>{rows}</tbody></table>', [26, 16, 18, 40])


def metrics_table():
    rows = [["WAPE, raw forecast", pct(val_w), pct(test_w), "n/a"],
            ["WAPE, bias-corrected forecast (used for reorder points)", "n/a", pct(test_w_c), f"<strong>{pct(live_w)}</strong>"],
            ["Best simple method, WAPE", "n/a", pct(metrics["overall"]["baseline"]), pct(live_b)],
            ["Forecast bias, after correction", "n/a", f"{test_bias * 100:+.1f}%", f"{live_bias * 100:+.1f}%"],
            ["Forecasts scored", f"{split['validation']['rows']:,}", f"{split['test']['rows']:,}", f"{len(live):,}"]]
    return widths(B.data_table(["Metric", "Validation", "Held-out 2025", "Live 1H 2026"], rows, right=[1, 2, 3]),
                  [40, 20, 20, 20])


def pattern_table():
    seg25 = {r["segment"]: r for r in metrics["segments"]}
    rows = []
    for s in SEG_ORDER:
        g = live[live.segment == s]
        r = seg25[s]
        lb = (g["forecast"].sum() - g["actual"].sum()) / g["actual"].sum()
        rows.append([s.capitalize(), pct(r["wape_model"], 0), pct(r["wape_baseline"], 0), r["baseline_method"],
                     pct(wape(g["actual"], g["forecast"]), 0), pct(wape(g["actual"], g["base"]), 0), f"{lb * 100:+.0f}%"])
    return widths(B.data_table(["Demand pattern", "Model, 2025", "Simple, 2025", "Simple method", "Model, 1H26",
                                "Simple, 1H26", "Bias, 1H26"], rows, right=[1, 2, 4, 5, 6]),
                  [17, 13, 13, 15, 14, 14, 14])


def bias_table():
    seg25 = {r["segment"]: r for r in metrics["segments"]}
    rows = [[s.capitalize(), f"{seg25[s]['bias'] * 100:+.0f}%", f"&times;{metrics['bias_correction'][s]:.2f}",
             f"{((live.loc[live.segment == s, 'forecast'].sum() / live.loc[live.segment == s, 'actual'].sum()) - 1) * 100:+.0f}%"]
            for s in SEG_ORDER]
    return widths(B.data_table(["Demand pattern", "Raw bias, 2025", "Correction factor", "Corrected bias, 1H26"], rows,
                               right=[1, 2, 3]), [28, 24, 24, 24])


def policy_table():
    ch = sched["churn"]
    rows = [
        ["Reorder point", "Forecast usage over the lead time, less scheduled usage already visible, plus safety buffer"],
        ["Fill-rate targets", f"Production items {tgt(FT['line'])}, spare parts {tgt(FT['service'])}, shop supplies {tgt(FT['standard'])}"],
        ["Safety buffer", "k &times; &sigma;, with k the smallest multiple whose expected shortfall per cycle, on the "
                          "model's own error distribution, fits (1 - fill target) &times; order quantity"],
        ["&sigma; (uncertainty, in units)", "The typical size of the gap between the usage forecast and actual usage "
                                    "over the item's lead time (one standard deviation): &radic;(forecast error&sup2; + "
                                    "(daily usage &times; lead-time standard deviation)&sup2;). The forecast error is the "
                                    "standard deviation of the item's own 2025 forecast misses, scaled to its lead time "
                                    "and reduced for the share of demand visible on booked jobs; the second term is the "
                                    "variability of the supplier's delivery time"],
        ["Order-book visibility", f"Released jobs about {sched['mrp_visibility_days']:.0f} days ahead; booked jobs about "
                                  f"{sched['model_visibility_days']:.0f} days ahead.<br><em>Released:</em> a job's parts come off "
                                  f"the shelf when it is completed, and the ERP sees the job from its release date, so "
                                  f"{sched['mrp_visibility_days']:.0f} days is the median time from release to completion of "
                                  f"2025 jobs that finished on time (delayed jobs are excluded, since their extra days are "
                                  f"waiting on material). <em>Booked:</em> adds the time from booking the customer order to "
                                  f"releasing the job. The records carry no booking date, so this is an assumption: jobs "
                                  f"booked four to eight weeks before release, a median of about "
                                  f"{sched['order_book_days']:.0f} days."],
        ["Order quantity", f"Economic lot: &radic;(2 &times; annual usage &times; ${sched['order_line_cost']:.0f} per order line "
                           f"&divide; ({sched['holding_rate'] * 100:.0f}% holding &times; unit cost)), kept between "
                           f"{sched['lot_days'][0]} and {sched['lot_days'][1]} days of usage.<br>"
                           f"<em>Annual usage:</em> the model's usage forecast, at a yearly rate. "
                           f"<em>Unit cost:</em> standard cost from the item master. "
                           f"<em>${sched['order_line_cost']:.0f} per order line (assumption):</em> the fully loaded cost of "
                           f"placing and processing one line (buyer time, receiving and inspection, putaway, invoice "
                           f"matching); published estimates for manufacturers run roughly $20 to $100+ per purchase order, "
                           f"less per line. <em>{sched['holding_rate'] * 100:.0f}% holding (assumption):</em> the usual "
                           f"20 to 30% rule of thumb covering cost of capital, storage, handling, insurance and "
                           f"obsolescence. <em>{sched['lot_days'][0]} to {sched['lot_days'][1]} days (judgment):</em> no "
                           f"item ordered more often than about every two weeks, and no more than four months of a cheap "
                           f"item at once, below the buyers' old {sched['buyer_lot_days']}-day lot. Supplier minimum "
                           f"order quantities, price breaks and freight are not modelled."],
        ["Change threshold", f"A reorder point moves only when the new value differs by more than "
                             f"{sched['hysteresis'] * 100:.0f}%: {ch['raw_any_change'] * 100:.0f}% of item-weeks would "
                             f"otherwise change, {ch['applied_any_change'] * 100:.0f}% do"],
        ["Lead times", "Refreshed monthly from each item's last six months of receipts"],
        ["Inbound orders", "Pushed back when not yet needed; never on production items"],
    ]
    return widths(B.data_table(["Element", "Rule"], rows), [24, 76])


def ops_table():
    rows = [
        ["Scoring cadence", f"Weekly (Monday): a usage forecast, reorder point and order quantity for all "
                            f"{len(sched['items']):,} stocked items; the ERP checks every item against its point daily"],
        ["Retraining", "Monthly, on the first forecast date of each month, with the tuned hyperparameters held fixed; "
                       "a full re-tune of the three candidates on each annual review or retraining trigger"],
        ["Run time", f"The weekly refresh scores {len(sched['items']):,} items in seconds; the monthly retrain of a "
                     f"{metrics['candidates'][WIN]['params'].get('n_estimators', '')}-tree forest on about "
                     f"{(split['train']['rows'] + split['validation']['rows']) // 1000 * 1000:,}+ rows takes about a minute"],
        ["Outputs", "rop_schedule.json: per item and week, the reorder level, safety stock, forecast, horizon, "
                    "reorder point and order quantity, loaded into the ERP reorder queue"],
        ["Data lineage", "ERP and WMS extracts &rarr; dbt staging, intermediate and data-quality models &rarr; weekly "
                         "usage and item attribute marts &rarr; forward_policy.py (train, forecast, policy) &rarr; "
                         "rop_schedule.json &rarr; ERP reorder queue"],
        ["Retraining triggers", "Live WAPE for any demand pattern more than 5 points above its 2025 level for two "
                                "consecutive months; corrected bias outside &plusmn;10% for a pattern; achieved fill "
                                "rate more than 1 point below target for a criticality group; stockout events or jobs "
                                "held for material above their 1H 2026 monthly level for two consecutive months"],
    ]
    return widths(B.data_table(["Specification", "Detail"], rows), [22, 78])


charts = {"volume": chart_volume(), "target": chart_target(), "corr": chart_corr(), "learning": chart_learning(),
          "calib": chart_calibration(), "pattern": chart_by_pattern(), "pva": chart_pred_actual(),
          "shap": chart_shap(), "loss": chart_loss(), "lots": chart_lots()}

p_ = metrics["candidates"][WIN]["params"]
PROSE = {"ly": "usage in the same weeks last year", "lead": "supplier lead time", "h": "the forecast horizon",
         "s4": "the last 4 weeks of usage", "s13": "the last 13 weeks of usage", "s26": "the last 26 weeks of usage",
         "s52": "the last 52 weeks of usage", "annual": "annual usage", "mean_all": "average weekly usage",
         "seg_code": "the demand pattern", "nz13": "the weeks with usage in the last 13",
         "since_nz": "the weeks since last usage", "std_cost": "unit cost"}
top3 = [PROSE.get(f, FEAT_DESC.get(f, (f, ""))[0].lower()) for f in imp["feature"].head(3)]
k_emp = {}
zs = z_err.to_numpy()
for kk in [1.0, 2.0, 3.0]:
    k_emp[kk] = (np.maximum(zs - kk, 0).mean(), _normal_loss(kk))

toc = ('<a href="#card">Model Card</a><hr>'
       '<a href="#data">Training Data</a><hr>'
       '<a href="#modelperf">Model Selection &amp; Performance</a>'
       '<a href="#selection" class="sub">Model Selection</a>'
       '<a href="#performance" class="sub">Model Performance</a><hr>'
       '<a href="#shap">Feature Importance</a><hr>'
       '<a href="#policy">From Forecast to Reorder Decision</a><hr>'
       '<a href="#limits">Known Limitations</a><hr>'
       '<a href="#ops">Deployment &amp; Operations</a>')

body = f"""
{B.section("card", "Section 1", "Model Card")}
<div class="model-card"><div class="model-card-grid">
  <div><div class="mc-label">Model Name</div><div class="mc-value">demand_forecaster (weekly)</div></div>
  <div><div class="mc-label">Model Type</div><div class="mc-value">{LABEL[WIN]} regressor (scikit-learn)</div></div>
  <div><div class="mc-label">Target</div><div class="mc-value">Units an item will use over its supplier lead time, from the forecast date</div></div>
  <div><div class="mc-label">Prediction Type</div><div class="mc-value">Regression on log(1 + usage), one forecast per item per week</div></div>
  <div><div class="mc-label">Features</div><div class="mc-value">{len(FEAT_DESC)} (rolling usage, calendar, item attributes)</div></div>
  <div><div class="mc-label">Tuning</div><div class="mc-value">Optuna, {N_TRIALS} trials per candidate, validation WAPE objective</div></div>
  <div><div class="mc-label">Cadence</div><div class="mc-value">Refreshed every Monday; retrained monthly</div></div>
  <div><div class="mc-label">Outputs</div><div class="mc-value">Reorder point and order quantity per item, loaded into the ERP</div></div>
  <div><div class="mc-label">Held-out 2025 WAPE</div><div class="mc-value">{pct(test_w)}</div></div>
  <div><div class="mc-label">Live 1H 2026 WAPE</div><div class="mc-value">{pct(live_w)}</div></div>
  <div style="grid-column:1/-1;"><div class="mc-label">Purpose</div><div class="mc-value">Forecasts each stocked item's usage over its supplier lead time and turns it into a reorder point and order quantity for the ERP's reorder queue. Intended as decision support for reordering: buyers are required to release the suggested orders.</div></div>
</div></div>

{B.section("data", "Section 2", "Training Data")}
<p>The model learns from the shop's weekly usage of each of its {attrs.shape[0]:,} stocked items. Each training row
is one item on one forecast date. Its features describe the item's usage up to that date, and its target is the
usage over the following lead time, rounded to whole weeks.</p>
<p>The historical data is split by date and never shuffled. The first 52 weeks of history (through 2023) are used
only to build the rolling features, because every forecast needs a full year of history behind it, so the first
usable forecast date is January 2024. The 2024 data was used for tuning, split into a January to June training
window and a July to December validation window on which the three candidates were compared. Once the winner was
chosen, it was refit on all of 2024 before being scored on 2025 data. We chose to use the full 2025 year of data for
model testing, because the bias correction for each demand pattern and the safety-buffer multiples are calibrated on
its errors, and both need a full year to reflect the seasonal demand of certain items.</p>
<p>The first six months of 2026 are the model's live output. In live use the model is retrained monthly on every
forecast date whose outcome is already known, which by June 2026 spans 2024 through early 2026. Because 2025 serves
both to calibrate the model and to report held-out accuracy, the fully out-of-sample test is the live performance
from January to June 2026.</p>
{split_table()}
{B.chart("Weekly Usage by Split", charts["volume"])}
<p>The model's {len(FEAT_DESC)} features fall into three groups. The first is Rolling Usage, including usage windows (4, 13,
26 and 52 weeks), the count of weeks with any usage, the time since the last usage and the recent variability. The
second is Calendar Features, to let the model learn seasonal demand patterns. The third is Item Attributes,
including cost, lead time, demand pattern, value class and category, to tell the model how to read each item's
usage history: the lead time sets how long a window it is forecasting, the demand pattern how far recent usage can
be trusted, and the cost, value class and category how similar items tend to behave. The correlation figures
presented are calculated as Spearman rank correlations on the training rows, which suit the skewed, zero-heavy
usage.</p>
{feature_table()}
<p>The rolling usage windows are strongly correlated with one another, since they overlap in time. The
weeks-since-last-usage and variability features move against the usage windows, as expected for intermittent items.
Unit cost runs against usage ({cmat.loc['std_cost', 's13']:+.2f} to {cmat.loc['std_cost', 'mean_all']:+.2f}), because
expensive items are used in smaller quantities. Last year's usage correlates only {cmat.loc['s52', 'ly']:+.2f} with the
52-week total, so its timing adds information beyond the average level.</p>
{B.chart("Feature Correlation Heatmap (numeric features, Spearman)", charts["corr"])}
<p>The target is heavily right-skewed: most lead-time windows use a few dozen units, while fasteners and
consumables run to thousands. The model is therefore trained on log(1 + usage), so the few very high-volume items do
not dominate the fit and the model learns proportional patterns across items of very different sizes. This
distribution is stable across the three splits, as shown below.</p>
{B.chart("Target Distribution: Train / Validation / Held-out Test", charts["target"])}

{B.section("modelperf", "Section 3", "Model Selection & Performance")}

{B.section("selection", "Section 3.1", "Model Selection")}
<p>Three candidates, a ridge regression, a random forest and an XGBoost gradient-boosted model, were each tuned
with Optuna ({N_TRIALS} trials per model) on the validation weeks using weighted absolute percentage error (WAPE) as
the objective. The models were then refit on train plus validation (i.e., full year 2024) and scored once on the
held-out 2025 year. The {LABEL[WIN].lower()} had the lowest validation error and was selected.</p>
{candidate_table()}
<p>The best {LABEL[WIN].lower()} model has {p_.get('n_estimators')} trees with a maximum depth of
{p_.get('max_depth')} and at least {p_.get('min_samples_leaf')} rows per leaf. The relatively large leaf-size floor
keeps the model from fitting individual spikes and instead learning the items' typical usage patterns.</p>

{B.section("performance", "Section 3.2", "Model Performance")}
<p>Accuracy is measured as WAPE: the total absolute gap between forecast and actual usage across all forecasts,
divided by total actual usage. Results are shown for the validation weeks used in tuning, the held-out 2025 year
used for selection and calibration, and the live January to June 2026 forecasts, which are the honest test of the
model in use. The reorder points use the bias-corrected forecast (Section 5), which trades a little accuracy
({pct(test_w)} to {pct(test_w_c)} on the held-out year) for forecasts that do not run systematically low.</p>
{metrics_table()}
<p>A live WAPE of {pct(live_w, 0)} means that for every 100 units an item used over its lead time, the forecast was
off by about {live_w * 100:.0f} units in either direction. That is typical for item-level forecasts of lumpy,
job-driven demand, where a single job or spare-parts order can double an item's usage in a week. <strong>The errors
largely cancel across items: total forecast usage was within {abs(live_bias) * 100:.1f}% of actual.</strong> The model's
gain over the best simple method ({pct(live_b, 0)}) is modest, as is usual for forecasts built from usage history
alone. Most of the operational improvement comes from how the forecast is used, covered in Section 5.</p>
<p>The learning curve refits the selected model on growing random shares of the training rows. The held-out error
flattens well before the full training set, so more of the same history would add little; better inputs, such as
the quantities already on booked jobs, would.</p>
{B.chart("Learning Curve", charts["learning"])}
<p>Calibration compares the mean forecast with the mean actual usage within each tenth of the forecasts, from
smallest to largest. Points on the diagonal mean a forecast of, say, 50 units really does average about 50 units
of usage. After the bias correction described in Section 5, the forecasts track actual usage closely from about
20 units up, in both the held-out year and live use. For the smallest forecasts, actual usage runs above the
forecast (an item forecast at 2 units averages about 3), which the safety buffer absorbs.</p>
{B.chart("Calibration: Mean Forecast vs Mean Actual, by Forecast Decile", charts["calib"])}
<p>Error varies by demand pattern. Smooth items are forecast best and intermittent items worst, since an item used
a few times a year at random offers little to learn from. The model beats the best simple method for each pattern
in both periods.</p>
{pattern_table()}
{B.chart("Live WAPE by Demand Pattern, Model vs Best Simple Method", charts["pattern"])}
<p>Plotted against actual usage on log scales, the live forecasts cluster around the diagonal across four orders
of magnitude, meaning the same model tracks items that use a handful of units per lead time as well as those that use
thousands, rather than working only for high- or low-volume items. The spread around the diagonal is the item-level
error that the safety buffer absorbs.</p>
<div style="max-width:520px;margin:18px auto;">{B.chart("Forecast vs Actual Usage, Live 1H 2026", charts["pva"])}</div>

{B.section("shap", "Section 4", "Feature Importance (SHAP)")}
<p>SHAP values measure each feature's average contribution to a forecast, here on a sample of the held-out 2025
forecasts and in units of log usage. The model leans most on {top3[0]}, {top3[1]} and {top3[2]}. Last year's usage
over the same weeks is by far the strongest, because it is the only feature measured over exactly the window being
forecast: the same number of weeks as the item's lead time, at the same time of year. Every other usage feature
covers a fixed span (4, 13, 26 or 52 weeks) that the model has to rescale. More recent usage then adjusts that
starting point up or down, and the item's demand pattern tells the model how reliably this can be trusted.</p>
<p>Calendar and category features contribute little. About {SEASONAL_SHARE:.0%} of items are seasonal, but each peaks
at its own time of year, and last year's same-weeks usage already carries each item's own seasonal timing; a shared
week-of-year or month feature adds little on top of it. Category adds little for a similar reason, since cost, usage
level and demand pattern already capture most of what distinguishes the categories.</p>
{B.chart("Mean Absolute SHAP Value by Feature", charts["shap"])}

{B.section("policy", "Section 5", "From Forecast to Reorder Decision")}
<p>The model makes one prediction: the usage forecast, how many units each item will use over its supplier lead
time. The reorder point and order quantity presented in the ERP are then calculated from that usage forecast using
the rules below.</p>
{policy_table()}
<p><strong>Bias correction.</strong> Trained on log usage, the model's back-transformed forecasts estimate the
median rather than the mean, so they run low, most of all for intermittent items. Each pattern's forecasts are then
scaled by the ratio of actual to forecast usage in the held-out 2025 year. This correction removes almost all of the
systematic under-forecasting in live use, as shown below.</p>
{bias_table()}
<p><strong>Safety buffer.</strong> The buffer is calculated as k standard errors of the item's own forecast
error: k &times; &sigma;, where &sigma; combines the forecast's miss over the lead time with the variability of the
supplier's delivery time, and k sets how many of those standard errors to hold. Two refinements are important to
highlight here. First, the part of an item's demand already visible on booked production jobs carries no forecast
error, so the error is scaled down by that share. Second, k is not taken from a normal curve. It is set on the
model's actual 2025 errors so that the expected shortfall per replenishment cycle stays within the fill-rate target
for the item's criticality group. The errors have much fatter tails than a normal curve: at k = 2 the expected
shortfall is {k_emp[2.0][0] / k_emp[2.0][1]:.1f} times what a normal distribution implies, and at k = 3,
{k_emp[3.0][0] / k_emp[3.0][1]:.1f} times. This is important to ensure an adequate buffer size. Because the shortfall allowance scales with the order
quantity, an item ordered in large lots needs less buffer (its own lot protects most of the cycle), and an item
ordered often needs more.</p>
{B.chart("Expected Shortfall vs Buffer Size: Actual Errors vs Normal", charts["loss"])}
<p><strong>Order quantity.</strong> The recommended order quantity for each item, calculated from its usage
forecast, balances the cost of placing an order line against the cost of holding stock. Expensive, heavily used items
are ordered every few weeks, and cheap items a few times a year.</p>
<p>The chart below groups items by value class. The classes are set by annual usage value (units used &times; unit
cost), not by unit price: A items are those that together make up the first 80% of usage value, B items the next 15%
and C items the last 5%. In 2025 the typical A item used about {k_(ABCV.loc['A', 'med_val'])} of stock a year (most
above {k_(ABCV.loc['A', 'p10'])}) at a median unit cost of ${ABCV.loc['A', 'med_cost']:,.0f}; a typical B item about
{k_(ABCV.loc['B', 'med_val'])} a year at ${ABCV.loc['B', 'med_cost']:,.0f}; and a typical C item about
{k_(ABCV.loc['C', 'med_val'])} a year at ${ABCV.loc['C', 'med_cost']:,.0f}. The A items, where most of the money sits,
are bought in the smallest lots.</p>
{B.chart("Order Quantity in Days of Usage, by Value Class", charts["lots"])}

{B.section("limits", "Section 6", "Known Limitations")}
<ul class="limitation-list">
  <li><strong>Forecasts from history.</strong> The forecast uses past usage only; booked jobs are netted afterward
  rather than fed in as a feature. Adding the booked quantities as an input is the clearest route to a more
  accurate forecast for production items.</li>
  <li><strong>Irreducible item-level error.</strong> Lumpy and intermittent demand carries randomness no model can
  learn; much of the item-level WAPE of {pct(live_w, 0)} reflects that rather than the fit.</li>
  <li><strong>Fill rate, not events.</strong> The buffer targets the share of units supplied. It does not directly
  target the probability of a stockout event or of a job being held, though these important metrics are closely
  monitored.</li>
  <li><strong>Supply is taken as given.</strong> Lead times are refreshed from recent receipts and their variability
  enters the buffer, but late deliveries are not forecast.</li>
  <li><strong>Order-book assumption.</strong> The records carry no booking date for customer orders; jobs are
  assumed booked four to eight weeks before release.</li>
  <li><strong>Simulated data.</strong> The model is trained and evaluated on synthetic ERP data with embedded
  patterns and deliberate noise. Real-world accuracy depends on the signal in the shop's own data.</li>
</ul>

{B.section("ops", "Section 7", "Deployment & Operations")}
<p>How the model runs in production and where its inputs come from.</p>
{ops_table()}
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(B.page("ML Model Technical Overview: Demand Forecast and Reorder Policy", "", toc, body),
               encoding="utf-8")
print(f"Technical report written to {OUT}")
