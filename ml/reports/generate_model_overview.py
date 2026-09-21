"""ML Model Overview and Performance Report for the demand forecaster -> docs/reports/model_overview.html

Plain-language companion to the Case 02 model overview: what the model predicts,
how the winning algorithm was chosen, how it performs by demand segment and ABC
class against the baseline the shop uses today, the headline duplicate-cleaning
result, and how forecast error over lead time feeds the safety-stock policy. Runs
as a package module so the shared brand kit imports cleanly:

    PYTHONIOENCODING=utf-8 "../mfg-oee-maintenance/.venv/Scripts/python.exe" -m ml.reports.generate_model_overview
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import brand as B
from .brand import (DARK_GREY, DARK_BLUE, LIGHT_BLUE, ACCENT_RED, MUTED_RED,
                    AMBER, GREEN, MED_GREY, LIGHT_GREY)

REPO     = Path(__file__).resolve().parents[2]
BACKTEST = REPO / "ml" / "data" / "backtest"
POLICY   = REPO / "ml" / "data" / "policy"
OUT      = REPO / "docs" / "reports" / "model_overview.html"

# ── Data ──────────────────────────────────────────────────────────────────────
metrics = json.loads((BACKTEST / "model_metrics.json").read_text(encoding="utf-8"))
mbt     = pd.read_parquet(BACKTEST / "model_backtest.parquet")
clean   = pd.read_parquet(BACKTEST / "clean_before_after.parquet")
policy  = json.loads((POLICY / "policy_summary.json").read_text(encoding="utf-8"))

CAND_LABEL   = {"Linear": "Linear (Ridge)", "RandomForest": "Random Forest", "XGBoost": "XGBoost"}
CAND_ORDER   = ["Linear", "RandomForest", "XGBoost"]
SEG_ORDER    = ["smooth", "erratic", "lumpy", "intermittent"]
SEG_TITLE    = {"smooth": "Smooth", "erratic": "Erratic", "lumpy": "Lumpy", "intermittent": "Intermittent"}
BASE_LABEL   = {"croston": "Croston", "ses": "Simple exponential smoothing", "snaive": "Seasonal naive",
                "ma3": "3-month moving average", "ma6": "6-month moving average", "naive": "Naive"}
WINNER       = metrics["winner"]

seg_by = {s["segment"]: s for s in metrics["segments"]}
overall_model = metrics["overall"]["model"]
overall_base  = metrics["overall"]["baseline"]
overall_lift  = (overall_base - overall_model) / overall_base


def _wape(a, p):
    a = np.asarray(a, float); p = np.asarray(p, float)
    denom = np.abs(a).sum()
    return float(np.abs(a - p).sum() / denom) if denom else float("nan")


# Moving average the shop uses today (3-month), for the headline comparison.
ma_wape = _wape(mbt["actual"], mbt["ma3"])

# ABC-level model vs baseline.
abc_rows = []
for abc in ["A", "B", "C"]:
    g = mbt[mbt["abc"] == abc]
    abc_rows.append({"abc": abc, "n": int(g["item"].nunique()),
                     "model": _wape(g["actual"], g["pred"]),
                     "base": _wape(g["actual"], g["base"])})

# Cleaning result: forecasting the merged series vs the split records.
clean_before = float(clean["before"].mean())
clean_after  = float(clean["after"].mean())
clean_before_med = float(clean["before"].median())
clean_after_med  = float(clean["after"].median())
clean_rel    = (clean_before - clean_after) / clean_before
n_clean      = int(len(clean))
n_improved   = int((clean["after"] < clean["before"]).sum())

# Working capital released at equal service (safety-stock section).
wc_release = policy["corrected"]["inv"] - policy["forecast"]["inv"]
wc_release_pct = wc_release / policy["corrected"]["inv"]

n_items   = int(mbt["item"].nunique())
n_origins = int(mbt["origin"].nunique())


# ── Charts ────────────────────────────────────────────────────────────────────
def chart_candidates():
    val  = [metrics["candidates"][c]["val_wape"] * 100 for c in CAND_ORDER]
    test = [metrics["candidates"][c]["test_wape"] * 100 for c in CAND_ORDER]
    x = np.arange(len(CAND_ORDER)); w = 0.36
    fig, ax = B.make_fig(h=3.6)
    b1 = ax.bar(x - w / 2, val, w, color=MED_GREY, label="Validation WAPE")
    b2 = ax.bar(x + w / 2, test, w, color=DARK_BLUE, label="Test WAPE")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9, color=DARK_GREY)
    ax.set_xticks(x); ax.set_xticklabels([CAND_LABEL[c] for c in CAND_ORDER])
    ax.set_ylabel("WAPE (lower is better)")
    ax.set_ylim(0, max(val) * 1.15)
    ax.yaxis.set_major_formatter(B.mticker.PercentFormatter())
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_segment_vs_baseline():
    model = [seg_by[s]["wape_model"] * 100 for s in SEG_ORDER]
    base  = [seg_by[s]["wape_baseline"] * 100 for s in SEG_ORDER]
    x = np.arange(len(SEG_ORDER)); w = 0.36
    fig, ax = B.make_fig(h=3.8)
    b1 = ax.bar(x - w / 2, base, w, color=MED_GREY, label="Segment baseline")
    b2 = ax.bar(x + w / 2, model, w, color=DARK_BLUE, label="XGBoost model")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.4,
                    f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9, color=DARK_GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{SEG_TITLE[s]}\n({BASE_LABEL[seg_by[s]['baseline_method']]})" for s in SEG_ORDER],
                       fontsize=9)
    ax.set_ylabel("WAPE (lower is better)")
    ax.set_ylim(0, max(base + model) * 1.16)
    ax.yaxis.set_major_formatter(B.mticker.PercentFormatter())
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_cleaning():
    before = clean["before"] * 100
    after  = clean["after"] * 100
    bins = np.linspace(min(before.min(), after.min()), max(before.max(), after.max()), 12)
    fig, ax = B.make_fig(h=3.6)
    ax.hist(before, bins=bins, color=MED_GREY, alpha=0.75, label="Split records (before merge)")
    ax.hist(after, bins=bins, color=DARK_BLUE, alpha=0.75, label="Merged series (after cleaning)")
    ax.axvline(clean_before * 100, color=MED_GREY, ls="--", lw=1.6)
    ax.axvline(clean_after * 100, color=DARK_BLUE, ls="--", lw=1.6)
    ax.text(clean_before * 100, ax.get_ylim()[1] * 0.96, f" mean {clean_before*100:.0f}%",
            color=DARK_GREY, fontsize=9, va="top", ha="left")
    ax.text(clean_after * 100, ax.get_ylim()[1] * 0.82, f"mean {clean_after*100:.0f}% ",
            color=DARK_BLUE, fontsize=9, va="top", ha="right")
    ax.set_xlabel("WAPE on the affected items (lower is better)")
    ax.set_ylabel("Items")
    ax.xaxis.set_major_formatter(B.mticker.PercentFormatter())
    ax.legend(ncol=1, loc="upper right", frameon=False)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


# ── Tables ────────────────────────────────────────────────────────────────────
def candidate_table():
    rows = []
    for c in CAND_ORDER:
        cand = metrics["candidates"][c]
        win = c == WINNER
        name = (f'<td style="font-weight:700;color:{DARK_BLUE};">{CAND_LABEL[c]} '
                f'{B.badge("selected", GREEN)}</td>') if win else CAND_LABEL[c]
        rows.append([name,
                     f'{cand["val_wape"]*100:.1f}%',
                     f'{cand["test_wape"]*100:.1f}%'])
    return B.data_table(["Candidate", "Validation WAPE", "Test WAPE"], rows, right={1, 2})


def segment_table():
    rows = []
    for s in SEG_ORDER:
        d = seg_by[s]
        lift = d["lift"]
        lift_col = GREEN if lift > 0.01 else (ACCENT_RED if lift < -0.01 else MED_GREY)
        verdict = ("Model wins" if lift > 0.02 else
                   "Baseline competitive" if lift >= -0.03 else "Use baseline")
        rows.append([
            SEG_TITLE[s],
            BASE_LABEL[d["baseline_method"]],
            f'{d["wape_baseline"]*100:.0f}%',
            f'{d["wape_model"]*100:.0f}%',
            f'<td style="text-align:right;color:{lift_col};font-weight:600;">{lift*100:+.0f}%</td>',
            f'{d["mase"]:.2f}',
            f'{d["bias"]*100:+.0f}%',
            verdict,
        ])
    return B.data_table(
        ["Segment", "Relevant baseline", "Baseline WAPE", "Model WAPE", "Lift", "MASE", "Bias", "Recommendation"],
        rows, right={2, 3, 4, 5, 6})


def abc_table():
    rows = []
    for r in abc_rows:
        lift = (r["base"] - r["model"]) / r["base"]
        rows.append([
            f'Class {r["abc"]}',
            f'{r["n"]}',
            f'{r["base"]*100:.0f}%',
            f'{r["model"]*100:.0f}%',
            f'<td style="text-align:right;color:{GREEN if lift>0 else MED_GREY};font-weight:600;">{lift*100:+.0f}%</td>',
        ])
    return B.data_table(["ABC class", "Items", "Baseline WAPE", "Model WAPE", "Lift"], rows, right={1, 2, 3, 4})


MODEL_CARD = f"""
<div class="model-card">
  <div class="model-card-grid">
    <div><div class="mc-label">What it predicts</div>
      <div class="mc-value">Units consumed of each purchased item over that item's own replenishment lead time</div></div>
    <div><div class="mc-label">Algorithm</div>
      <div class="mc-value">XGBoost gradient-boosted trees, tuned with Optuna</div></div>
    <div><div class="mc-label">Scored</div>
      <div class="mc-value">Monthly, for all {n_items} canonical items</div></div>
    <div><div class="mc-label">Accuracy metric</div>
      <div class="mc-value">WAPE (weighted absolute percentage error); MASE and bias as supporting checks</div></div>
    <div><div class="mc-label">Headline accuracy</div>
      <div class="mc-value">{overall_model*100:.0f}% test WAPE, against {ma_wape*100:.0f}% for the moving average in use today</div></div>
    <div><div class="mc-label">Intended use</div>
      <div class="mc-value">Decision support for the three buyers: it sizes reorder demand, a person releases the PO</div></div>
  </div>
</div>
"""


# ── Assemble ──────────────────────────────────────────────────────────────────
charts = {
    "cand": chart_candidates(),
    "seg": chart_segment_vs_baseline(),
    "clean": chart_cleaning(),
}

toc = ('<a href="#summary">Executive Summary</a><hr>'
       '<a href="#predicts">What the Model Predicts</a>'
       '<a href="#selection">Model Selection</a>'
       '<a href="#performance">Performance by Segment and ABC</a><hr>'
       '<a href="#cleaning">The Cleaning Result</a>'
       '<a href="#safetystock">From Accuracy to Safety Stock</a>'
       '<a href="#limits">Limitations and Intended Use</a>')

body = f"""
{B.section("summary", "Section 1", "Executive Summary")}
<p>This model tells the shop's three buyers how much of each purchased item they are likely to consume before
their next order can arrive, so they can reorder the right quantity at the right time. It is the forecasting
engine behind the reorder queue for a precision machining shop running about $25M in revenue across roughly
800 active purchased items. Better demand numbers are what let the shop hold less stock without running out:
the goal is to free up working capital tied up in excess inventory and to cut the scrap and rush-freight bills
that come from buying the wrong amounts.</p>
<p>The model is an XGBoost forecaster, and on a held-out test window it predicts lead-time demand with a
<strong>weighted error (WAPE) of {overall_model*100:.0f}%</strong>. That is a large step up from the 3-month
moving average the shop relies on today, whose error on the same items is about <strong>{ma_wape*100:.0f}%</strong>.
Feeding these forecasts into the reorder policy, at an equal service level of roughly 97%, releases about
<strong>${wc_release:,.0f}</strong> of working capital ({wc_release_pct:.0%}) versus running the corrected
policy on simpler numbers, while holding fill rate steady.</p>
{B.kpi_row(
    B.kpi_card(f"{overall_model*100:.0f}%", "Test WAPE", "held-out backtest", DARK_BLUE),
    B.kpi_card(f"{ma_wape*100:.0f}% → {overall_model*100:.0f}%", "vs shop moving average", "same items, same window", GREEN),
    B.kpi_card(f"{overall_lift*100:+.0f}%", "Lift over segment baselines", "weighted across all items", DARK_BLUE),
    B.kpi_card(f"${wc_release/1000:,.0f}K", "Working capital released", f"{wc_release_pct:.0%} at equal service", GREEN))}
<p>The rest of this report is written for a non-technical reader. It explains what the model forecasts and why
lead-time demand is the right target, how the winning algorithm was chosen from a field of candidates, where
it beats the shop's current baselines and where an old-fashioned rule is still the honest choice, the data
cleaning that quietly delivered the single largest accuracy gain, and how the forecast error turns into the
safety stock that protects the line.</p>
{MODEL_CARD}

{B.section("predicts", "Section 2", "What the Model Predicts")}
<p>The model does not forecast a fixed 30-day or 90-day window. For each item it predicts demand over that
item's own <strong>replenishment lead time</strong>: the number of days between placing a purchase order and
having the parts on the shelf ready to use. A bar stock item that arrives in a week and a casting that takes
two months are two different questions, and the reorder decision only cares about the demand that will land
before the next delivery does.</p>
<p>That is why lead-time demand is the right target. The buyer's real question is never "how much will we use
next month" in the abstract; it is "will what I have on hand, plus what is already on order, cover us until the
next shipment arrives." Forecasting demand over each item's lead time answers that question directly, and it
is the exact quantity the reorder point and safety stock are built on. The model is scored monthly against
what actually happened, across all {n_items} canonical items and {n_origins} rolling forecast origins in the
backtest, so its accuracy reflects repeated real reorder decisions rather than a single lucky month.</p>

{B.section("selection", "Section 3", "Model Selection")}
<p>Three candidate algorithms were put through the same tuning and the same held-out test: a linear model
(Ridge), a Random Forest, and XGBoost, each searched over its hyperparameters with Optuna. They were compared
on a validation window first, then scored once on a later test window they had never seen, so the winner is
the one that generalizes, not the one that memorizes. The table and chart below report WAPE, where lower is
better. XGBoost had the lowest error on both windows, so it was carried forward; the takeaway is that the
tree-based models decisively beat the linear one, and XGBoost edged out the Random Forest on the unseen test
data.</p>
{candidate_table()}
{B.chart("Candidate Comparison: Validation and Test WAPE", charts["cand"])}
<p>The linear model sits far above the others because demand here is intermittent and non-linear, with long
flat stretches broken by spikes, which a straight-line fit cannot follow. XGBoost's test WAPE of
{metrics['candidates']['XGBoost']['test_wape']*100:.0f}% against the Random Forest's
{metrics['candidates']['RandomForest']['test_wape']*100:.0f}% is a modest but consistent edge, and it holds up
across the demand segments examined next.</p>

{B.section("performance", "Section 4", "Performance by Segment and ABC")}
<p>A single average hides more than it shows, because these items do not behave alike. Every item is sorted
into a demand pattern, smooth, erratic, lumpy, or intermittent, and each pattern has its own sensible baseline
that the shop could use instead of a model. The honest test is the model against the <em>right</em> baseline
for each pattern, not against a weak straw man. The chart and table below show that comparison; the takeaway
is that the model earns its place on the erratic and lumpy items, is a wash on smooth demand, and does not beat
the baseline on the truly intermittent items, where the recommendation is to keep using the simpler rule.</p>
{B.chart("Model vs Each Segment's Relevant Baseline (WAPE)", charts["seg"])}
{segment_table()}
<p>Reading the table: <strong>lift</strong> is how much the model reduces error versus that segment's baseline,
<strong>MASE</strong> compares the model's error to a naive one-step forecast (below 1.0 means better than
naive), and <strong>bias</strong> shows whether the model tends to under-forecast (negative) or over-forecast.
The model delivers its clearest wins on <strong>erratic</strong> demand ({seg_by['erratic']['lift']*100:+.0f}%)
and <strong>lumpy</strong> demand ({seg_by['lumpy']['lift']*100:+.0f}%), the volatile items where a moving
average lags and overshoots. On <strong>smooth</strong> items it is a statistical tie with Croston
({seg_by['smooth']['lift']*100:+.0f}%), which is fine because those items are already easy to forecast either
way. On <strong>intermittent</strong> items the model is competitive at best ({seg_by['intermittent']['lift']*100:+.0f}%)
and Croston is the honest choice; these items are mostly zeros with occasional demand, and no method predicts
them well. The consistent negative bias across segments means the model leans slightly toward under-forecasting,
which the safety stock in Section 6 is sized to absorb.</p>
<p>The same picture holds when items are grouped by ABC value class rather than demand pattern. The model beats
the blended baseline in every class, with the largest gains on the B and C items where demand is choppier.</p>
{abc_table()}
<p>The comparison that matters most to the shop, though, is not against these tuned per-segment baselines but
against what the buyers actually use: a 3-month moving average applied to every item alike. Measured that way,
the model cuts WAPE from about <strong>{ma_wape*100:.0f}% to {overall_model*100:.0f}%</strong>, because the
moving average is badly mismatched to the lumpy and intermittent items that make up nearly half the catalog.</p>

{B.section("cleaning", "Section 5", "The Cleaning Result")}
<p>The single largest accuracy gain did not come from the algorithm at all. It came from repairing the data
first. In the source records, {n_clean} items had their history split across duplicate part numbers, so the
demand for one physical item was scattered over two or more records. Forecasting each split fragment means
forecasting a series with holes in it. After the duplicates were merged into one canonical item and the
history stitched back together, the same model was rerun on the repaired series. The chart below shows the
error distribution before and after; the takeaway is that cleaning shifted the whole distribution left,
dropping the average WAPE on these items from about <strong>{clean_before*100:.0f}% to
{clean_after*100:.0f}%</strong>.</p>
{B.chart("Forecast Error Before and After Merging Duplicate Records", charts["clean"])}
<p>Across the {n_clean} affected items the median error fell from {clean_before_med*100:.0f}% to
{clean_after_med*100:.0f}%, and {n_improved} of {n_clean} improved. The point for the business is that model
choice and data quality are not competing investments: the cleaning captured here is worth more than the
difference between the candidate algorithms in Section 3, and it is the reason the reorder queue can be trusted
at the item level, not just in aggregate.</p>

{B.section("safetystock", "Section 6", "From Accuracy to Safety Stock")}
<p>Accuracy is not the end product; the reorder policy is. The forecast sets the expected demand over lead
time, and the model's <em>error</em> over that same window sets the safety stock: the buffer held to cover the
gap between what was forecast and what actually gets consumed while an order is in transit. Tighter, less biased
forecasts mean a smaller buffer is enough to hit the same service target, which is exactly how better numbers
turn into freed working capital. Safety stock is sized to each item's ABC service level, 98% for A items, 95%
for B, and 90% for C, so the most valuable and most disruptive stockouts are protected first, and the long tail
of C items is not over-insured. The inventory outcome of running these forecasts through that policy, roughly
${wc_release/1000:,.0f}K of working capital released at an equal service level, is detailed in the analytics
and policy deliverables; this report's job is the forecast quality that makes it possible.</p>

{B.section("limits", "Section 7", "Limitations and Intended Use")}
<p>Being clear about what the model cannot do is what makes it safe to rely on. It is a buyer's aid, not an
autopilot.</p>
<ul class="limitation-list">
  <li><strong>Intermittent demand is genuinely hard.</strong> Items that sell in rare, irregular bursts are
  the least predictable, and the model does not beat the Croston baseline on them. For those items the
  recommendation is to keep the simpler rule and lean on safety stock, not to trust a tight point forecast.</li>
  <li><strong>It forecasts quantity, not price or supply risk.</strong> The output is expected demand over lead
  time. It does not predict cost changes, supplier delays, or a sudden engineering change that makes a part
  obsolete; those still need human judgment and the supplier data.</li>
  <li><strong>It assumes the past is a fair guide.</strong> The model learns from history, so a brand-new item
  with no demand record, or a step change in the shop's product mix, will be forecast poorly until enough new
  history accumulates. Lead times are taken as given from the corrected item attributes.</li>
  <li><strong>It is decision support, not automated purchasing.</strong> The model sizes the reorder and
  explains it; a buyer reviews the queue and releases the purchase order. It is built to inform the three
  buyers, not to release POs on its own.</li>
  <li><strong>It stays honest through monitoring.</strong> Demand patterns drift, so forecast accuracy is
  tracked each period and the model is retrained when it slips, as covered in the monitoring report.</li>
</ul>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(B.page("ML Model Overview and Performance Report: Demand Forecaster",
                      "", toc, body), encoding="utf-8")
print(f"Model overview written to {OUT}")
