"""ML technical report for the global demand model -> docs/reports/technical_report.html

Written for a technical reader. Mirrors the Case 02 ml_technical generator in
structure, section helpers, KPI cards, tables and tone, restyled to the current
brand kit. Documents the method actually implemented in ml/src: demand
segmentation by ADI and CV squared, the leakage-guarded feature frame, the single
global gradient-boosted model, the lead-time demand target, the rolling-origin
backtest, the WAPE and MASE metrics, the results by segment and ABC class, the
XGBoost feature importances, the calibration of forecast error into safety stock,
the duplicate-cleaning before and after study, and the residual analysis.

Run:
  PYTHONIOENCODING=utf-8 "../mfg-oee-maintenance/.venv/Scripts/python.exe" \
      -m ml.reports.generate_technical_report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from . import brand as B
from .brand import (DARK_BLUE, LIGHT_BLUE, ACCENT_RED, MUTED_RED, AMBER, GREEN,
                    MED_GREY, LIGHT_GREY, DARK_GREY)
from ml.src.features import FEATURE_COLS

REPO     = Path(__file__).resolve().parents[2]
MARTS    = REPO / "ml" / "data" / "marts"
BACKTEST = REPO / "ml" / "data" / "backtest"
MODELS   = REPO / "ml" / "models"
OUT      = REPO / "docs" / "reports" / "technical_report.html"

ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49
SEG_ORDER = ["smooth", "erratic", "lumpy", "intermittent"]
SEG_COLOR = {"smooth": DARK_BLUE, "erratic": LIGHT_BLUE, "lumpy": AMBER, "intermittent": ACCENT_RED}
ABC_ORDER = ["A", "B", "C"]
# Service levels used to calibrate the safety-stock z multiplier by ABC class.
ABC_SERVICE = {"A": 0.98, "B": 0.95, "C": 0.90}
ABC_Z = {"A": 2.054, "B": 1.645, "C": 1.282}

# ── Load the modeled artifacts ───────────────────────────────────────────────
metrics = json.loads((BACKTEST / "model_metrics.json").read_text(encoding="utf-8"))
mb      = pd.read_parquet(BACKTEST / "model_backtest.parquet")
cba     = pd.read_parquet(BACKTEST / "clean_before_after.parquet")
attrs   = pd.read_parquet(MARTS / "item_attributes.parquet")
monthly = pd.read_parquet(MARTS / "consumption_monthly.parquet")
model   = joblib.load(MODELS / "demand_model.joblib")

winner   = metrics["winner"]
cand     = metrics["candidates"]
seg_rows = {r["segment"]: r for r in metrics["segments"]}
overall  = metrics["overall"]
N_ORIGINS = mb["origin"].nunique()
N_ITEMS   = mb["item"].nunique()
HORIZONS  = sorted(int(h) for h in mb["horizon"].unique())
N_MONTHS  = monthly["month"].nunique()


def wape(a, f):
    a = np.asarray(a, float); f = np.asarray(f, float)
    d = a.sum()
    return np.abs(a - f).sum() / d if d else np.nan


# ── Segmentation recomputation (ADI vs CV squared per item) ──────────────────
def _adi_cv2(series: np.ndarray):
    nz = series[series > 0]
    if len(nz) < 2:
        return np.nan, np.nan
    adi = len(series) / len(nz)
    cv2 = (nz.std() / nz.mean()) ** 2 if nz.mean() > 0 else 0.0
    return adi, cv2


_wide = monthly.pivot(index="canonical", columns="month", values="consumption").fillna(0)
_seg = attrs.set_index("canonical_item_number")["segment"]
_srows = []
for item, row in _wide.iterrows():
    adi, cv2 = _adi_cv2(row.to_numpy(float))
    _srows.append({"item": item, "adi": adi, "cv2": cv2, "segment": _seg.get(item)})
seg_scatter = pd.DataFrame(_srows).dropna(subset=["adi", "cv2", "segment"])

seg_mix = attrs["segment"].value_counts()
seg_share = attrs["segment"].value_counts(normalize=True)
abc_mix = attrs["abc"].value_counts()

# Baseline WAPE the shop's moving average produces, and the overall model figure.
ma_wape = wape(mb["actual"], mb["ma3"])
model_wape = overall["model"]
base_wape = overall["baseline"]
overall_lift = (base_wape - model_wape) / base_wape

# ── ABC results recomputed from the backtest ─────────────────────────────────
def _abc_result(a):
    g = mb[mb["abc"] == a]
    wm, wb = wape(g["actual"], g["pred"]), wape(g["actual"], g["base"])
    nmae = np.abs(g["actual"] - g["naive"]).mean()
    mase = np.abs(g["actual"] - g["pred"]).mean() / nmae if nmae else np.nan
    bias = (g["pred"] - g["actual"]).sum() / g["actual"].sum()
    return {"wm": wm, "wb": wb, "lift": (wb - wm) / wb, "mase": mase, "bias": bias}


abc_res = {a: _abc_result(a) for a in ABC_ORDER}

# ── Residuals by segment (pred minus actual, lead-time demand units) ─────────
mb = mb.assign(resid=mb["pred"] - mb["actual"])


# ── Charts ───────────────────────────────────────────────────────────────────
def chart_segment_scatter():
    fig, ax = B.make_fig(h=B.CHART_H_T)
    for seg in SEG_ORDER:
        d = seg_scatter[seg_scatter["segment"] == seg]
        ax.scatter(d["adi"], d["cv2"], s=16, alpha=0.55, edgecolors="none",
                   color=SEG_COLOR[seg], label=f"{seg} ({len(d)})")
    ax.axvline(ADI_THRESHOLD, color=MED_GREY, ls="--", lw=1.3)
    ax.axhline(CV2_THRESHOLD, color=MED_GREY, ls="--", lw=1.3)
    ax.text(ADI_THRESHOLD + 0.02, ax.get_ylim()[1] * 0.96, f"ADI = {ADI_THRESHOLD}",
            fontsize=8, color=MED_GREY, va="top")
    ax.text(seg_scatter["adi"].max() * 0.99, CV2_THRESHOLD + 0.03,
            f"CV² = {CV2_THRESHOLD}", fontsize=8, color=MED_GREY, ha="right")
    ax.set_xlabel("Average inter-demand interval (ADI)")
    ax.set_ylabel("Squared coefficient of variation (CV²)")
    ax.set_ylim(0, seg_scatter["cv2"].quantile(0.99) * 1.05)
    ax.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_wape_by_segment():
    fig, ax = B.make_fig(h=B.CHART_H)
    x = np.arange(len(SEG_ORDER)); w = 0.38
    wm = [seg_rows[s]["wape_model"] * 100 for s in SEG_ORDER]
    wb = [seg_rows[s]["wape_baseline"] * 100 for s in SEG_ORDER]
    ax.bar(x - w / 2, wb, w, color=MED_GREY, label="Relevant baseline")
    ax.bar(x + w / 2, wm, w, color=DARK_BLUE, label="Global model")
    for i, (b, m) in enumerate(zip(wb, wm)):
        ax.text(i - w / 2, b + 1, f"{b:.0f}", ha="center", va="bottom", fontsize=9, color=MED_GREY)
        better = m < b
        ax.text(i + w / 2, m + 1, f"{m:.0f}", ha="center", va="bottom", fontsize=9,
                color=GREEN if better else ACCENT_RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([s.title() for s in SEG_ORDER])
    ax.set_ylabel("WAPE (%)"); ax.set_ylim(0, max(wb + wm) * 1.18)
    ax.legend(fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_feature_importance():
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
    d = imp.tail(14)
    fig, ax = B.make_fig(h=B.CHART_H_T)
    ax.barh(range(len(d)), d.values, color=DARK_BLUE, height=0.68)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index, fontsize=9, fontfamily="monospace")
    for i, v in enumerate(d.values):
        ax.text(v + d.max() * 0.01, i, f"{v:.02f}", va="center", fontsize=8.5, color=MED_GREY)
    ax.set_xlabel("XGBoost gain importance (share of total)")
    ax.set_xlim(0, d.max() * 1.16)
    B.chart_style(ax); ax.xaxis.grid(True, color=LIGHT_GREY); ax.yaxis.grid(False)
    fig.tight_layout()
    return B.b64(fig)


def chart_clean_before_after():
    fig, ax = B.make_fig(h=B.CHART_H)
    ax.scatter(cba["before"] * 100, cba["after"] * 100, s=26, alpha=0.7,
               color=DARK_BLUE, edgecolors="white", linewidth=0.5)
    lim = max(cba["before"].max(), cba["after"].max()) * 100 * 1.08
    ax.plot([0, lim], [0, lim], color=MED_GREY, ls="--", lw=1.3, label="No change")
    worse = cba[cba["after"] > cba["before"]]
    ax.scatter(worse["before"] * 100, worse["after"] * 100, s=30, color=ACCENT_RED,
               edgecolors="white", linewidth=0.5, label=f"Worse after merge ({len(worse)})")
    ax.set_xlabel("Per-item WAPE before merge (%)")
    ax.set_ylabel("Per-item WAPE after merge (%)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_residuals_by_segment():
    fig, ax = B.make_fig(h=B.CHART_H)
    data = [mb.loc[mb["segment"] == s, "resid"].to_numpy() for s in SEG_ORDER]
    # clip the plotted range so the central mass stays legible; whiskers keep shape
    lo = min(np.quantile(d, 0.02) for d in data)
    hi = max(np.quantile(d, 0.98) for d in data)
    bp = ax.boxplot(data, orientation="vertical", patch_artist=True, showfliers=False,
                    widths=0.55, medianprops=dict(color=DARK_GREY, linewidth=1.6))
    for patch, s in zip(bp["boxes"], SEG_ORDER):
        patch.set_facecolor(SEG_COLOR[s]); patch.set_alpha(0.55); patch.set_edgecolor(MED_GREY)
    for wk in bp["whiskers"] + bp["caps"]:
        wk.set_color(MED_GREY)
    ax.axhline(0, color=ACCENT_RED, ls="--", lw=1.3)
    ax.set_xticklabels([s.title() for s in SEG_ORDER])
    ax.set_ylabel("Residual: predicted minus actual (units)")
    ax.set_ylim(lo * 1.1, hi * 1.1)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


charts = {
    "scatter": chart_segment_scatter(),
    "wape": chart_wape_by_segment(),
    "imp": chart_feature_importance(),
    "clean": chart_clean_before_after(),
    "resid": chart_residuals_by_segment(),
}


# ── Tables ───────────────────────────────────────────────────────────────────
def segmentation_table():
    quad = {"smooth": ("ADI &lt; 1.32", "CV&sup2; &lt; 0.49"),
            "erratic": ("ADI &lt; 1.32", "CV&sup2; &ge; 0.49"),
            "intermittent": ("ADI &ge; 1.32", "CV&sup2; &lt; 0.49"),
            "lumpy": ("ADI &ge; 1.32", "CV&sup2; &ge; 0.49")}
    rows = []
    for s in SEG_ORDER:
        adi_c, cv_c = quad[s]
        rows.append([f'{B.badge(s, SEG_COLOR[s])}', adi_c, cv_c,
                     f"{seg_mix[s]:,}", f"{seg_share[s]*100:.0f}%"])
    return B.data_table(["Segment", "Interval", "Dispersion", "Items", "Share"], rows, right={3, 4})


def candidate_table():
    labels = {"Linear": "Linear (Ridge on standardized features)",
              "RandomForest": "Random Forest", "XGBoost": "XGBoost (gradient boosted)"}
    order = sorted(cand, key=lambda k: cand[k]["val_wape"])
    rows = ""
    for k in order:
        sel = k == winner
        win = f' <span style="color:{GREEN};font-weight:700;">&#10003; Selected</span>' if sel else ""
        bg = f' style="background:{B.BG_GREY};font-weight:700;"' if sel else ""
        rows += (f'<tr{bg}><td>{labels.get(k, k)}{win}</td>'
                 f'<td style="text-align:right;">{cand[k]["val_wape"]*100:.1f}%</td>'
                 f'<td style="text-align:right;">{cand[k]["test_wape"]*100:.1f}%</td></tr>')
    return (f'<table class="data-table"><thead><tr><th>Candidate</th>'
            f'<th style="text-align:right;">Validation WAPE</th>'
            f'<th style="text-align:right;">Test WAPE</th></tr></thead><tbody>{rows}</tbody></table>')


def segment_result_table():
    rows = ""
    for s in SEG_ORDER:
        r = seg_rows[s]
        lift = r["lift"] * 100
        lift_col = GREEN if lift > 0 else ACCENT_RED
        rows += (f'<tr><td>{B.badge(s, SEG_COLOR[s])}</td>'
                 f'<td style="text-align:right;">{r["wape_model"]*100:.1f}%</td>'
                 f'<td style="text-align:right;">{r["wape_baseline"]*100:.1f}%</td>'
                 f'<td style="text-align:right;color:{lift_col};font-weight:700;">{lift:+.1f}%</td>'
                 f'<td style="text-align:right;">{r["mase"]:.2f}</td>'
                 f'<td style="text-align:right;">{r["bias"]*100:+.0f}%</td>'
                 f'<td style="font-family:monospace;font-size:13px;">{r["baseline_method"]}</td></tr>')
    return (f'<table class="data-table"><thead><tr><th>Segment</th>'
            f'<th style="text-align:right;">Model WAPE</th>'
            f'<th style="text-align:right;">Baseline WAPE</th>'
            f'<th style="text-align:right;">Lift</th>'
            f'<th style="text-align:right;">MASE</th>'
            f'<th style="text-align:right;">Bias</th><th>Baseline</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def abc_result_table():
    rows = ""
    for a in ABC_ORDER:
        r = abc_res[a]
        lift = r["lift"] * 100
        lift_col = GREEN if lift > 0 else ACCENT_RED
        rows += (f'<tr><td>{B.badge("Class " + a, DARK_GREY)}</td>'
                 f'<td style="text-align:right;">{abc_mix[a]:,}</td>'
                 f'<td style="text-align:right;">{r["wm"]*100:.1f}%</td>'
                 f'<td style="text-align:right;">{r["wb"]*100:.1f}%</td>'
                 f'<td style="text-align:right;color:{lift_col};font-weight:700;">{lift:+.1f}%</td>'
                 f'<td style="text-align:right;">{r["mase"]:.2f}</td>'
                 f'<td style="text-align:right;">{r["bias"]*100:+.0f}%</td></tr>')
    return (f'<table class="data-table"><thead><tr><th>ABC class</th>'
            f'<th style="text-align:right;">Items</th>'
            f'<th style="text-align:right;">Model WAPE</th>'
            f'<th style="text-align:right;">Baseline WAPE</th>'
            f'<th style="text-align:right;">Lift</th>'
            f'<th style="text-align:right;">MASE</th>'
            f'<th style="text-align:right;">Bias</th></tr></thead><tbody>{rows}</tbody></table>')


def feature_family_table():
    rows = [
        ["Lags", "monospace", "lag_1, lag_2, lag_3, lag_6, lag_12",
         "Consumption in each of five prior months, carrying autocorrelation and last-year recall."],
        ["Rolling statistics", "monospace", "roll_mean_{3,6,12}, roll_std_{3,6,12}, nz_{3,6,12}",
         "Level, volatility and non-zero count over 3, 6 and 12 month windows."],
        ["Recency", "monospace", "months_since_nz",
         "Months since the last non-zero demand, the core intermittency signal."],
        ["Level and dispersion", "monospace", "mean_all, cv_all",
         "Lifetime mean and coefficient of variation of the series to date."],
        ["Direction", "monospace", "trend_3_6, ratio_1_6",
         "Short minus medium window level, and last month over its six month mean."],
        ["Calendar", "monospace", "month, quarter, is_quarter_start",
         "Position in the year and the quarter-start flag for release-driven demand."],
        ["Item attributes", "monospace", "std_cost, price, annual, corrected_lead, horizon",
         "Cost, price, annual volume, corrected lead time and the horizon in months."],
        ["Encoded class", "monospace", "seg_code, abc_code, cls_code",
         "Demand segment, ABC class and item class as integer codes."],
    ]
    body = ""
    for name, _, cols, desc in rows:
        body += (f'<tr><td style="font-weight:700;">{name}</td>'
                 f'<td style="font-family:monospace;font-size:12.5px;color:{DARK_GREY};">{cols}</td>'
                 f'<td>{desc}</td></tr>')
    return (f'<table class="data-table"><thead><tr><th>Family</th><th>Columns</th>'
            f'<th>What it encodes</th></tr></thead><tbody>{body}</tbody></table>')


def safety_stock_table():
    rows = []
    for a in ABC_ORDER:
        rows.append([f'{B.badge("Class " + a, DARK_GREY)}',
                     f"{ABC_SERVICE[a]*100:.0f}%", f"{ABC_Z[a]:.3f}",
                     f"{ABC_Z[a]:.2f} &times; &sigma;<sub>LTD</sub>"])
    return B.data_table(["ABC class", "Service level", "z multiplier", "Safety stock"],
                        rows, right={1, 2, 3})


def metric_table():
    rows = [
        ["WAPE", "&Sigma;|actual &minus; forecast| / &Sigma;actual",
         "Volume-weighted absolute error. Defined whenever total demand is positive, so it survives zero months."],
        ["MASE", "mean|actual &minus; forecast| / mean|actual &minus; naive|",
         "Error scaled by the naive one-step error. Below 1.0 means the model beats a last-value forecast."],
        ["Bias", "&Sigma;(forecast &minus; actual) / &Sigma;actual",
         "Signed volume error. Negative means the forecast sits below realized demand on average."],
    ]
    body = ""
    for name, formula, desc in rows:
        body += (f'<tr><td style="font-weight:700;">{name}</td>'
                 f'<td style="font-family:monospace;font-size:12.5px;">{formula}</td>'
                 f'<td>{desc}</td></tr>')
    return (f'<table class="data-table"><thead><tr><th>Metric</th><th>Definition</th>'
            f'<th>Reading</th></tr></thead><tbody>{body}</tbody></table>')


# ── KPI cards ────────────────────────────────────────────────────────────────
kpis = B.kpi_row(
    B.kpi_card(f"{model_wape*100:.0f}%", "Test WAPE", f"{winner}, held-out origins", DARK_BLUE),
    B.kpi_card(f"{ma_wape*100:.0f}%", "Moving-average WAPE", "what the shop runs today", MED_GREY),
    B.kpi_card(f"{overall_lift*100:+.1f}%", "Lift vs relevant baseline", "volume-weighted, per segment", GREEN),
    B.kpi_card(f"{N_ITEMS:,}", "Items backtested", f"{N_ORIGINS} rolling origins each", DARK_GREY),
)

cba_before = cba["before"].median()
cba_after = cba["after"].median()
cba_flat = (cba["rel_improve"] <= 0.02).mean()
cba_worse = int((cba["after"] > cba["before"]).sum())

# ── TOC ──────────────────────────────────────────────────────────────────────
toc = ('<a href="#seg">1. Demand Segmentation</a><hr>'
       '<a href="#feat">2. Feature Engineering</a><hr>'
       '<a href="#global">3. Why a Global Model</a><hr>'
       '<a href="#target">4. Target Construction</a><hr>'
       '<a href="#backtest">5. Backtest Design</a><hr>'
       '<a href="#metrics">6. Metric Definitions</a><hr>'
       '<a href="#results">7. Results by Segment and Class</a><hr>'
       '<a href="#importance">8. Feature Importance</a><hr>'
       '<a href="#safety">9. Error to Safety Stock</a><hr>'
       '<a href="#resid">10. Residual Analysis</a>')

# ── Body ─────────────────────────────────────────────────────────────────────
body = f"""
<p>This report documents the global demand model behind the reorder queue for a precision
machining shop of roughly $25M in revenue, three buyers and about 800 active purchased items,
over a 36 month window ending August 2026. It is written for a technical reader: every method
below is the one implemented in <code>ml/src</code>, and every number is read from the backtest
artifacts the training run wrote, not restated from memory. The headline is that a single
gradient-boosted model, scored over the same rolling origins the baselines used, cuts
volume-weighted error from the moving average the shop runs today at about
<strong>{ma_wape*100:.0f}%</strong> WAPE to about <strong>{model_wape*100:.0f}%</strong>, while
holding an honest, per-segment comparison against the strongest classical method for each demand
pattern.</p>
{kpis}

{B.section("seg", "Section 1", "Demand Segmentation Method")}
<p>Purchased-part demand is not one problem but four, so the first step is to separate the items by
how their demand behaves rather than forecasting them all the same way. Two statistics do the
separation. The average inter-demand interval, ADI, is the number of months divided by the number
of months with non-zero demand, so it measures how often an item moves. The squared coefficient of
variation, CV squared, is the variance of the non-zero quantities over their squared mean, so it
measures how uneven the sizes are when the item does move. The Syntetos-Boylan-Croston thresholds
split each axis: <strong>ADI at {ADI_THRESHOLD}</strong> and <strong>CV squared at
{CV2_THRESHOLD}</strong>. The four quadrants are smooth (frequent and steady), erratic (frequent
but variable in size), intermittent (sporadic but even when present), and lumpy (sporadic and
variable, the hardest to forecast).</p>
<p>The cleaned catalog spreads across all four quadrants with no single segment dominating, which is
why a one-size baseline underperforms: about a third of items are smooth and forecastable, but
nearly half are intermittent or lumpy where a naive average is badly placed. The distribution below
is computed by classifying each item's cleaned monthly series.</p>
{segmentation_table()}
<p>The scatter places every item by its own ADI and CV squared, colored by the segment it lands in,
with the two thresholds drawn as dashed lines. The takeaway is that the segments are a genuine
partition of the data rather than a label pasted on top: items cluster into the four quadrants the
thresholds define, with smooth items pinned at ADI near 1.0 and low dispersion in the lower left,
and lumpy items pushed to the upper right where both sporadic timing and uneven size compound.</p>
{B.chart("Demand Segmentation: ADI vs CV squared by Segment", charts["scatter"])}

{B.section("feat", "Section 2", "Feature Engineering")}
<p>The model reads one row per item and origin month. Every feature is computed from consumption
strictly before the origin, and the target is demand over the horizon that starts at the origin, so
no feature can encode any part of the forecast period. That leakage guard is the reason the feature
builder slices history as <code>past = hist[:t]</code> and never touches <code>hist[t:]</code>. The
features fall into eight families that give a single global model the autocorrelation, recency,
volatility and calendar structure the univariate baselines cannot see at once.</p>
{feature_family_table()}
<p>Two design choices are worth calling out. Recency is carried explicitly by
<code>months_since_nz</code>, the count of months since the last non-zero demand, which is the
signal that lets one model treat an intermittent item differently from a smooth one without a
separate model. And the calendar features exist because this shop's demand is partly release-driven:
<code>is_quarter_start</code> flags the months where quarterly production releases land. Feature
NaNs from the items missing a master field are imputed with train-split medians so the linear and
forest candidates compare fairly against the NaN-native booster, and the split boundaries keep each
origin's whole horizon inside one split so no training row can overlap a validation or test target.</p>

{B.section("global", "Section 3", "Why a Global Model, Not One per Item")}
<p>The decisive reason to fit one model across all items rather than 800 per-item models is data.
A single item contributes only a few dozen monthly observations, far too few to estimate lags,
rolling statistics and calendar effects without overfitting, and the intermittent items barely move
at all. Pooling every item into one training frame lets the model borrow strength: it learns the
shared shape of autocorrelation, quarterly release and size dispersion from thousands of item-origin
rows, then conditions on <code>seg_code</code>, <code>abc_code</code>, <code>cls_code</code> and the
per-item level and cost features to specialize per item. A per-item approach would also be
operationally unworkable for three buyers to retrain and monitor, whereas one registered model
scores the whole catalog in a single pass. The classical baselines, by contrast, are genuinely
per-item and univariate, which is exactly the handicap the global model is built to overcome.</p>

{B.section("target", "Section 4", "Target Construction")}
<p>The target is not next month's demand but total demand over the item's lead time, because that is
the quantity a reorder point actually has to cover. Each item's horizon is its corrected lead time
in months, <code>h = round(corrected_lead_days / 30)</code> with a floor of one, so a fast-turning
bar with a two week lead is forecast over one month and a long-lead casting over three. Across the
catalog the horizons span <strong>{HORIZONS[0]} to {HORIZONS[-1]} months</strong>. The target for a
row at origin <code>t</code> is <code>series[t : t + h].sum()</code>, the realized demand the
purchase order would need to satisfy. Training on this lead-time aggregate rather than a fixed
one-step horizon means the forecast drops straight into the safety-stock and reorder logic without a
separate horizon-stacking step, and it lets the same model serve items with very different lead
times.</p>

{B.section("backtest", "Section 5", "Backtest Design")}
<p>Every number in this report comes from a rolling-origin backtest, not an in-sample fit. Each item
is evaluated over <strong>{N_ORIGINS} monthly origins</strong>, at least the twelve the harness
requires, with the horizon rolled forward one month at a time so the model is always predicting a
window it has not seen. The split is by time and never shuffled: earlier origins train, a block of
six origins validates and selects, and the last stretch of origins is the held-out test set.
Crucially the boundaries are drawn so each origin's whole horizon sits inside a single split, which
removes the subtle leak where a training target would overlap a test origin. The model reuses the
exact origins and horizons the baselines ran on, so the comparison is like for like rather than a
re-tuned rematch.</p>
<p>Candidates were tuned with Optuna on validation WAPE. A linear ridge, a random forest and an
XGBoost booster each got an independent search, and the winner was chosen by validation WAPE alone,
then refit on train plus validation and scored once on test. {winner} won and was registered.</p>
{candidate_table()}
<p>The linear model is off the table: even standardized and regularized it cannot represent the
interaction between intermittency and size, so its WAPE runs above 100%. The two tree ensembles are
close, and XGBoost edges the forest on validation WAPE, which is the selection criterion, so it is
the registered model. Its held-out test WAPE of <strong>{cand[winner]['test_wape']*100:.1f}%</strong>
is the honest estimate carried through the rest of this report.</p>

{B.section("metrics", "Section 6", "Metric Definitions")}
<p>Accuracy is reported with three metrics, chosen because the demand here is full of zeros and
spikes that break the usual percentage error. WAPE is the primary metric: it weights every unit of
error equally regardless of which item or month it lands in, and it stays defined as long as total
demand over the group is positive. MASE scales the model's error by the naive forecast's error, so a
value below 1.0 states plainly that the model beats simply carrying the last value forward. Bias is
the signed counterpart to WAPE, exposing whether the model systematically sits above or below realized
demand.</p>
{metric_table()}
<p>MAPE is deliberately absent. Mean absolute percentage error divides by the actual, so a single zero
month makes it undefined and a near-zero month makes it explode, which is precisely the regime the
intermittent and lumpy segments live in. Reporting MAPE on this catalog would either drop the hardest
items or be dominated by a handful of tiny denominators, so WAPE and MASE carry the accuracy story
instead.</p>

{B.section("results", "Section 7", "Full Results by Segment and ABC Class")}
<p>The comparison is deliberately hard on the model: each segment is scored against the single
strongest classical method for that segment, chosen by pooled WAPE, not against a weak straw man. On
that footing the model wins overall, cutting volume-weighted error to
<strong>{model_wape*100:.1f}%</strong> against a blended baseline of
<strong>{base_wape*100:.1f}%</strong>, a lift of <strong>{overall_lift*100:+.1f}%</strong>, and it
wins clearly where structure exists to exploit. It does not win everywhere, and the table reports
that honestly.</p>
{segment_result_table()}
<p>The pattern is consistent with what the features can and cannot see. On erratic and lumpy items
the model gains {seg_rows['erratic']['lift']*100:.0f} to {seg_rows['lumpy']['lift']*100:.0f} percent
over the best baseline, because the lags, rolling volatility and calendar structure let it anticipate
size and timing that a univariate method smears out. On smooth items the strong Croston baseline is
already near the achievable floor, so the model essentially ties it at a small
{seg_rows['smooth']['lift']*100:.1f} percent. On intermittent items the model loses by about
{abs(seg_rows['intermittent']['lift'])*100:.0f} percent: when demand is sporadic but even in size,
Croston's rate estimate is genuinely hard to beat, and the model's tendency to shade forecasts down
(bias {seg_rows['intermittent']['bias']*100:+.0f} percent) costs it here. The MASE column confirms
the model beats the naive forecast in every segment, most decisively on lumpy demand
({seg_rows['lumpy']['mase']:.2f}).</p>
<p>By value class the story is steadier, which is what matters for inventory dollars: the model beats
the baseline across A, B and C items, with the largest lift on the B and C classes where the mixed
demand patterns concentrate. The negative bias across classes, forecasts landing below realized
demand, is the one behavior to watch and is the reason the safety-stock calibration in Section 9 adds
an explicit buffer rather than trusting the point forecast.</p>
{abc_result_table()}
<p>The chart puts the segment comparison side by side. The takeaway is that the two bars are close on
smooth and intermittent, where the classical baseline is already strong, and separate visibly on
erratic and lumpy, where the global model earns its place.</p>
{B.chart("WAPE by Segment: Global Model vs Relevant Baseline", charts["wape"])}

{B.section("importance", "Section 8", "Feature Importance")}
<p>The booster exposes a gain-based importance over the {len(FEATURE_COLS)} feature columns, and it
lines up with the demand structure the segmentation described. Recent lags carry the most weight, led
by <code>lag_3</code>, confirming that near-term autocorrelation is the dominant signal. The segment
code is the second heaviest driver, which is the model doing internally what a per-segment split would
do externally: it reads the demand pattern and adapts. The horizon and corrected lead time rank next,
since the target scales with how many months the forecast has to cover.</p>
<p>The takeaway from the chart is that no single feature dominates to the point of fragility: the
importance decays smoothly from the top lags through the rolling means and the encoded class, so the
model rests on a spread of correlated signals rather than one brittle input.</p>
{B.chart("XGBoost Feature Importance (top features by gain)", charts["imp"])}

{B.section("safety", "Section 9", "Calibrating Forecast Error into Safety Stock")}
<p>A point forecast is only half of a reorder policy. Because the model carries a mild negative bias
and a residual spread that widens with demand size, the reorder point adds a safety buffer sized to
the forecast error and to how much service each item warrants. Safety stock is set as
<strong>z &times; &sigma;<sub>LTD</sub></strong>, where the standard deviation of lead-time demand
error comes from the backtest residuals for the item's segment, and z is the service-level multiplier
set by ABC class. A items get the highest protection because a stockout there is the most expensive;
C items get the least because carrying cost dominates their economics.</p>
{safety_stock_table()}
<p>The second half of this section is the duplicate-cleaning study, which quantifies how much the
entity-resolution step is worth to the model specifically. For each of the
<strong>{len(cba)}</strong> items whose ERP records were split across duplicate part numbers, the same
model forecasts the merged canonical series (after) and, separately, forecasts each duplicate fragment
and sums the forecasts (before). Median per-item WAPE falls from <strong>{cba_before*100:.0f}%</strong>
before the merge to <strong>{cba_after*100:.0f}%</strong> after, a real gain that comes purely from
giving the model one complete series instead of two sparse ones.</p>
<p>The honest finding is that cleaning does not help every item. The takeaway from the scatter below,
which plots each item's WAPE before against after with the diagonal marking no change, is that most
items sit below the line and improve, but <strong>{cba_worse}</strong> of {len(cba)} land above it and
get worse, and about {cba_flat*100:.0f}% are flat or worse within a two-point tolerance. Those are
items where the split happened to route demand cleanly enough that two series were each individually
forecastable, so merging added little and occasionally blurred a pattern. The net is a clear median
improvement with a visible minority that does not benefit, which is the sort of result worth stating
rather than averaging away.</p>
{B.chart("Duplicate Cleaning: Per-item WAPE Before vs After Merge", charts["clean"])}

{B.section("resid", "Section 10", "Residual Analysis")}
<p>Residuals, defined here as predicted minus actual over the lead-time window, show where the model
sits relative to realized demand by segment. Across all four segments the residual distributions
center slightly below zero, the same negative bias the results table flagged: the model is a little
conservative, shading forecasts down rather than over-ordering. The takeaway from the chart is that the
bias is small and stable on smooth items, where the boxes are tight around zero, and grows in both
spread and downward pull on erratic and lumpy items, where large sporadic orders arrive that the model
smooths under.</p>
{B.chart("Residuals by Segment (predicted minus actual)", charts["resid"])}
<p>This is the residual behavior the safety-stock calibration is built to absorb. The wider,
downward-leaning residuals on the harder segments are exactly why the reorder point adds a
segment-sized buffer on top of the point forecast rather than trusting it directly, and why the A
class items carry the highest service multiplier. The smooth segment needs little protection; the
lumpy and intermittent segments need the buffer to convert an honest but conservative forecast into a
policy that holds its service level.</p>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    B.page("ML Technical Report: Global Demand Model",
           "Precision machining purchased-parts forecast, 36 month window ending August 2026",
           toc, body),
    encoding="utf-8")
print(f"Technical report written to {OUT}")
print(f"  sections 10, charts {len(charts)}, items {N_ITEMS}, origins {N_ORIGINS}")
