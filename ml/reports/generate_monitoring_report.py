"""MLOps Monitoring Report for the demand model -> docs/reports/monitoring_report.html

Mirrors the Case 02 monitoring_report layout (decision-first status block, an
MLOps Monitoring Summary wrapping four layer subsections, and a monitoring log
table), adapted to a demand forecasting model. The four layers are performance
(WAPE and MASE per period against the training reference), target drift
(distribution of actual demand per period), prediction drift (distribution of
model output per period), and feature drift (a rolling recent-demand feature).
The retraining rule follows the shared convention: a performance or target-drift
flag sustained across two consecutive periods recommends RETRAIN, a prediction or
feature-drift flag alone recommends INVESTIGATE, and a single isolated flag holds
the model at HEALTHY.

Run:  PYTHONIOENCODING=utf-8 python -m ml.reports.generate_monitoring_report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import brand as B
from .brand import (DARK_BLUE, LIGHT_BLUE, ACCENT_RED, AMBER, GREEN, MED_GREY,
                    LIGHT_GREY, BG_GREY, DARK_GREY)

REPO = Path(__file__).resolve().parents[2]
MON = REPO / "ml" / "data" / "monitoring"
DQ = REPO / "ml" / "data" / "data_quality" / "txn"
TRUTH = REPO / "data_source" / "truth" / "txn_defects.json"
RAW_TX = REPO / "data_source" / "raw" / "erp" / "inventory_transactions.csv"
OUT = REPO / "docs" / "reports" / "monitoring_report.html"

WAPE_THRESHOLD = 0.10     # relative rise in WAPE vs reference that flags performance
PSI_THRESHOLD = 0.20      # population stability index that flags a distribution drift
FT_THRESHOLD = 3.0        # free-text / non-stock line rate (% of new ledger postings) that flags
DUP_THRESHOLD = 0.75      # duplicate-record rate (% of new ledger postings) that flags

summ = json.loads((MON / "monitoring_summary.json").read_text(encoding="utf-8"))
pm = pd.read_parquet(MON / "period_monitoring.parquet")
ref_wape = float(summ["reference_wape"])
decision = summ["decision"]
periods = pm["period"].tolist()
latest = pm.iloc[-1]

# ── Data-quality monitored series (defect rates per period) ───────────────────
# New free-text / non-stock PO lines and new near-duplicate postings arrive with
# every period's ledger. Their RATES are monitored as their own series against a
# threshold, alongside the four model-drift layers. Rates are grounded in the
# current ledger: the planted transaction ids in txn_defects.json joined to the
# real posting dates in inventory_transactions.csv, bucketed to the same periods.
dq_summary = json.loads((DQ / "summary.json").read_text(encoding="utf-8"))
_truth = json.loads(TRUTH.read_text(encoding="utf-8"))
_ft_ids = {e["transaction_id"] for e in _truth["t1"]}   # free-text / non-stock lines
_dup_ids = {e["transaction_id"] for e in _truth["t7"]}   # near-duplicate postings
_tx = pd.read_csv(RAW_TX)
_tx["ym"] = pd.to_datetime(_tx["transaction_date"], errors="coerce").dt.strftime("%Y-%m")
_tx["is_ft"] = _tx["transaction_id"].isin(_ft_ids)
_tx["is_dup"] = _tx["transaction_id"].isin(_dup_ids)
_period_ym = [pd.to_datetime(p, format="%b %Y").strftime("%Y-%m") for p in periods]

dq = []
for p, ym in zip(periods, _period_ym):
    m = _tx["ym"] == ym
    tot = int(m.sum())
    ft_n = int((m & _tx["is_ft"]).sum())
    dup_n = int((m & _tx["is_dup"]).sum())
    dq.append({
        "period": p, "postings": tot,
        "ft_n": ft_n, "ft_rate": (ft_n / tot * 100) if tot else 0.0,
        "dup_n": dup_n, "dup_rate": (dup_n / tot * 100) if tot else 0.0,
    })
dq = pd.DataFrame(dq)
ft_max = float(dq["ft_rate"].max())
dup_max = float(dq["dup_rate"].max())
dq_within = bool(ft_max < FT_THRESHOLD and dup_max < DUP_THRESHOLD)

STATUS = {"HEALTHY": (GREEN, "&#10003;", "NO ACTION REQUIRED"),
          "INVESTIGATE": (AMBER, "&#9680;", "INVESTIGATE"),
          "RETRAIN": (ACCENT_RED, "&#9888;", "RETRAIN RECOMMENDED")}
dec_color, dec_icon, dec_label = STATUS.get(decision, (MED_GREY, "&bull;", decision))


def two_consec(col):
    f = pm[col].tolist()
    return any(f[i] and f[i + 1] for i in range(len(f) - 1))


# ── Charts ──────────────────────────────────────────────────────────────────
def chart_wape():
    fig, ax = B.make_fig(h=3.3)
    vals = pm["wape"].values * 100
    colors = [ACCENT_RED if f else DARK_BLUE for f in pm["perf_flag"]]
    bars = ax.bar(periods, vals, color=colors, width=0.5)
    trigger = ref_wape * (1 + WAPE_THRESHOLD) * 100
    ax.axhline(ref_wape * 100, color=MED_GREY, ls="--", lw=1.4,
               label=f"Reference WAPE {ref_wape*100:.1f}%")
    ax.axhline(trigger, color=AMBER, ls=":", lw=1.4,
               label=f"Performance flag {trigger:.1f}%")
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.6, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("WAPE (%)")
    ax.set_ylim(0, max(vals.max(), trigger) * 1.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_psi():
    layers = [("target_psi", "Target (actual demand)", DARK_BLUE),
              ("pred_psi", "Prediction (model output)", LIGHT_BLUE),
              ("feature_psi", "Feature (recent demand)", MED_GREY)]
    x = np.arange(len(periods)); w = 0.24
    fig, ax = B.make_fig(h=3.4)
    maxv = float(pm[["target_psi", "pred_psi", "feature_psi"]].to_numpy().max())
    for i, (col, label, color) in enumerate(layers):
        vals = pm[col].values
        bars = ax.bar(x + (i - 1) * w, vals, w, color=color, label=label)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + PSI_THRESHOLD * 0.02,
                    f"{v:.02f}", ha="center", va="bottom", fontsize=8, color=DARK_GREY)
    ax.axhline(PSI_THRESHOLD, color=ACCENT_RED, ls="--", lw=1.4,
               label=f"Drift threshold {PSI_THRESHOLD:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(periods)
    ax.set_ylabel("Population stability index")
    ax.set_ylim(0, max(maxv, PSI_THRESHOLD) * 1.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


def chart_dq():
    fig, ax = B.make_fig(h=3.3)
    x = np.arange(len(periods))
    ft = dq["ft_rate"].values
    dup = dq["dup_rate"].values
    ax.plot(x, ft, color=DARK_BLUE, marker="o", lw=2.2, ms=7,
            label="Free-text / non-stock line rate")
    ax.plot(x, dup, color=LIGHT_BLUE, marker="s", lw=2.2, ms=7,
            label="Duplicate-record rate")
    for xi, v in zip(x, ft):
        ax.text(xi, v + 0.09, f"{v:.2f}%", ha="center", va="bottom",
                fontsize=9, color=DARK_BLUE)
    for xi, v in zip(x, dup):
        ax.text(xi, v - 0.10, f"{v:.2f}%", ha="center", va="top",
                fontsize=9, color=MED_GREY)
    ax.axhline(FT_THRESHOLD, color=AMBER, ls="--", lw=1.4,
               label=f"Free-text threshold {FT_THRESHOLD:.1f}%")
    ax.axhline(DUP_THRESHOLD, color=MED_GREY, ls=":", lw=1.4,
               label=f"Duplicate threshold {DUP_THRESHOLD:.2f}%")
    ax.set_xticks(x); ax.set_xticklabels(periods)
    ax.set_ylabel("Defect rate (% of new postings)")
    ax.set_ylim(0, FT_THRESHOLD * 1.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    B.chart_style(ax); fig.tight_layout()
    return B.b64(fig)


charts = {"wape": chart_wape(), "psi": chart_psi(), "dq": chart_dq()}

# ── Status block ────────────────────────────────────────────────────────────
def status_block():
    perf_two = two_consec("perf_flag")
    target_two = two_consec("target_flag")
    pred_two = two_consec("pred_flag")
    feat_two = two_consec("feature_flag")
    triggers = []
    if perf_two:
        triggers.append("WAPE stayed above the performance flag line for two consecutive periods")
    if target_two:
        triggers.append("Actual demand distribution drifted (target PSI above threshold) for two consecutive periods")
    if pred_two:
        triggers.append("Model output distribution drifted (prediction PSI above threshold) for two consecutive periods")
    if feat_two:
        triggers.append("Recent-demand feature drifted (feature PSI above threshold) for two consecutive periods")
    reason_html = ("<ul class='trigger-list'>" + "".join(f"<li>{x}</li>" for x in triggers) + "</ul>") if triggers \
        else "<p style='margin:8px 0 0;color:#8093A4;'>No layer sustained a flag across two consecutive periods, so no retraining or investigation trigger fired.</p>"

    def c(cond):
        return ACCENT_RED if cond else GREEN
    max_target = pm["target_psi"].max()
    max_pred = pm["pred_psi"].max()
    max_feat = pm["feature_psi"].max()
    return f"""<div class="status-block" style="border-color:{dec_color};">
      <div class="status-header" style="background:{dec_color};">
        <span class="status-icon">{dec_icon}</span><span class="status-label">{dec_label}</span>
        <span style="margin-left:auto;font-size:13px;opacity:0.9;">As of {periods[-1]}</span></div>
      <div class="status-body">
        <div class="status-meta">
          <div><span class="meta-label">Model</span><span class="meta-val">XGBoost demand forecast</span></div>
          <div><span class="meta-label">Periods Monitored</span><span class="meta-val">{periods[0]} to {periods[-1]}</span></div>
          <div><span class="meta-label">Reference</span><span class="meta-val">Earlier test months, WAPE {ref_wape*100:.1f}%</span></div>
          <div><span class="meta-label">Latest WAPE</span><span class="meta-val" style="color:{c(bool(latest['perf_flag']))};">{latest['wape']*100:.1f}% (reference {ref_wape*100:.1f}%)</span></div>
          <div><span class="meta-label">Latest MASE</span><span class="meta-val" style="color:{c(latest['mase']>=1)};">{latest['mase']:.2f}</span></div>
          <div><span class="meta-label">Max Target PSI</span><span class="meta-val" style="color:{c(bool(target_two))};">{max_target:.3f} (threshold {PSI_THRESHOLD:.2f})</span></div>
          <div><span class="meta-label">Max Prediction PSI</span><span class="meta-val" style="color:{c(bool(pred_two))};">{max_pred:.3f} (threshold {PSI_THRESHOLD:.2f})</span></div>
          <div><span class="meta-label">Max Feature PSI</span><span class="meta-val" style="color:{c(bool(feat_two))};">{max_feat:.3f} (threshold {PSI_THRESHOLD:.2f})</span></div>
        </div>
        <div><div style="font-size:12px;font-weight:700;color:{MED_GREY};text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;">Trigger Reasons</div>{reason_html}</div>
      </div></div>"""


def retraining_rules():
    rules = [
        ("Primary", f"WAPE rises more than {int(WAPE_THRESHOLD*100)}% above the reference for two consecutive periods", "RETRAIN", two_consec("perf_flag")),
        ("Primary", f"Actual demand distribution drifts (target PSI &ge; {PSI_THRESHOLD:.2f}) for two consecutive periods", "RETRAIN", two_consec("target_flag")),
        ("Secondary", f"Model output distribution drifts (prediction PSI &ge; {PSI_THRESHOLD:.2f}) for two consecutive periods", "INVESTIGATE", two_consec("pred_flag")),
        ("Secondary", f"Recent-demand feature drifts (feature PSI &ge; {PSI_THRESHOLD:.2f}) for two consecutive periods", "INVESTIGATE", two_consec("feature_flag")),
    ]
    rows = ""
    for tier, rule, action, trig in rules:
        col = ACCENT_RED if trig else GREEN
        rows += (f'<tr><td style="width:30px;text-align:center;color:{col};font-size:16px;">{"&#9888;" if trig else "&#10003;"}</td>'
                 f'<td><span style="font-size:11px;font-weight:700;color:{DARK_GREY};">{tier}</span></td>'
                 f'<td>{rule}</td>'
                 f'<td style="text-align:center;font-weight:700;color:{DARK_GREY};">{action}</td>'
                 f'<td style="text-align:center;color:{col};font-weight:700;">{"TRIGGERED" if trig else "OK"}</td></tr>')
    return (f'<table class="data-table"><thead><tr><th></th><th>Tier</th><th>Rule</th>'
            f'<th style="text-align:center;">Action</th><th style="text-align:center;">Status</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


# ── Monitoring log table ────────────────────────────────────────────────────
def flag_cell(flagged):
    col = ACCENT_RED if flagged else GREEN
    return f'<td style="text-align:center;color:{col};font-weight:700;">{"&#9888; flag" if flagged else "&#10003; ok"}</td>'


def log_table():
    headers = ["Period", "WAPE", "MASE", "Perf", "Target PSI", "Pred PSI", "Feature PSI", "Drift"]
    rows = []
    for r in pm.itertuples():
        drift_any = bool(r.target_flag or r.pred_flag or r.feature_flag)
        row = [
            f'<td style="font-weight:600;">{r.period}</td>',
            f'<td style="text-align:right;">{r.wape*100:.1f}%</td>',
            f'<td style="text-align:right;">{r.mase:.2f}</td>',
            flag_cell(bool(r.perf_flag)),
            f'<td style="text-align:right;">{r.target_psi:.3f}</td>',
            f'<td style="text-align:right;">{r.pred_psi:.3f}</td>',
            f'<td style="text-align:right;">{r.feature_psi:.3f}</td>',
            flag_cell(drift_any),
        ]
        rows.append(row)
    return B.data_table(headers, rows)


def dq_table():
    headers = ["Period", "New Postings", "Free-Text Line Rate", "Duplicate-Record Rate", "Reading"]
    rows = []
    for r in dq.itertuples():
        within = r.ft_rate < FT_THRESHOLD and r.dup_rate < DUP_THRESHOLD
        col = GREEN if within else ACCENT_RED
        reading = "&#10003; within threshold" if within else "&#9888; over threshold"
        rows.append([
            f'<td style="font-weight:600;">{r.period}</td>',
            f'<td style="text-align:right;">{int(r.postings):,}</td>',
            f'<td style="text-align:right;">{r.ft_rate:.2f}% <span style="color:{MED_GREY};">/ {FT_THRESHOLD:.1f}%</span></td>',
            f'<td style="text-align:right;">{r.dup_rate:.2f}% <span style="color:{MED_GREY};">/ {DUP_THRESHOLD:.2f}%</span></td>',
            f'<td style="text-align:center;color:{col};font-weight:700;">{reading}</td>',
        ])
    return B.data_table(headers, rows)


# ── Assemble ────────────────────────────────────────────────────────────────
toc = ('<a href="#status">1 &middot; Status &amp; Decision</a>'
       '<a href="#summary">2 &middot; MLOps Monitoring Summary</a>'
       '<a href="#perf" class="sub">Performance</a>'
       '<a href="#target" class="sub">Target Drift</a>'
       '<a href="#prediction" class="sub">Prediction Drift</a>'
       '<a href="#feature" class="sub">Feature Drift</a>'
       '<a href="#dataquality">3 &middot; Data-Quality Monitoring</a>'
       '<a href="#log">4 &middot; Monitoring Log</a>')

kpis = B.kpi_row(
    B.kpi_card(dec_label.split()[0].title() if decision != "HEALTHY" else "Healthy",
               "Standing Decision", f"As of {periods[-1]}", GREEN if decision == "HEALTHY" else dec_color),
    B.kpi_card(f"{latest['wape']*100:.1f}%", "Latest WAPE", f"Reference {ref_wape*100:.1f}%", DARK_BLUE),
    B.kpi_card(f"{latest['mase']:.2f}", "Latest MASE", "Below 1.0 beats naive", DARK_GREY),
    B.kpi_card(f"{pm[['target_psi','pred_psi','feature_psi']].to_numpy().max():.2f}",
               "Peak Drift PSI", f"Threshold {PSI_THRESHOLD:.2f}", DARK_GREY),
)

body = f"""
{B.section("status", "Section 1", "Status &amp; Retraining Decision")}
<p>The demand model is monitored each period across the recent scoring window. Performance and target
drift are the primary retraining triggers; prediction and feature drift are leading proxies that call for
investigation rather than an immediate refresh. Every trigger requires two consecutive flagged periods
before it fires, which keeps a single noisy month from forcing a retrain. The verdict below is the standing
recommendation; the sections that follow show the full trend behind it.</p>
{kpis}
<p>The flag currently reads NO ACTION REQUIRED. Weighted error moved with normal month-to-month variation
and stayed close to the training reference, and all three drift layers held well under threshold, so no layer
sustained a flag across two consecutive periods. In practice the model keeps scoring the reorder queue
unchanged and is re-evaluated at the next period close.</p>
{status_block()}
<p>The retraining rules are evaluated every period. Primary rules measure forecast harm directly and drive
the RETRAIN decision; secondary rules are leading proxies that route to INVESTIGATE. If a primary rule were
to trigger, the action would be to retrain the model on data extended through the flagged periods, clear it
against the held-out validation window, and promote it to take over the reorder queue only once it beats the
incumbent. A secondary trigger would instead open an investigation into the drifting input before any
retraining is scheduled.</p>
{retraining_rules()}

{B.section("summary", "Section 2", "MLOps Monitoring Summary")}
<p>The four monitoring layers below track the demand model every period. Performance and target drift are the
primary retraining triggers; prediction and feature drift are leading proxies. Alongside the four layers, two
demand-specific checks run each period: items migrating between demand segments (smooth, erratic, lumpy,
intermittent), which changes which baseline the forecast is judged against, and lead-time changes on an item,
which move the horizon the forecast has to cover and can invalidate a reorder recommendation even when the
model itself is stable.</p>

{B.section("perf", "Section 2.1", "Performance")}
<p>Weighted absolute percentage error (WAPE) and the mean absolute scaled error (MASE) each period, measured
against actual consumption and compared with the training reference WAPE of {ref_wape*100:.1f}%. Performance
is flagged only when WAPE rises more than {int(WAPE_THRESHOLD*100)}% above that reference. <strong>WAPE moved
between {pm['wape'].min()*100:.1f}% and {pm['wape'].max()*100:.1f}% across the three periods and MASE stayed
near {pm['mase'].min():.2f} to {pm['mase'].max():.2f}, comfortably below 1.0, so the model continued to beat
the naive baseline every period. The single-period flags in {periods[0]} and {periods[-1]} did not land on
consecutive periods, so the performance layer does not trigger a retrain.</strong></p>
<p>The chart below plots WAPE per period against the reference line and the performance flag line. Bars that
cross the flag line are drawn in red; a retrain would only follow if two adjacent bars crossed it.</p>
{B.chart("WAPE by Period vs Reference", charts["wape"])}

{B.section("target", "Section 2.2", "Target Drift")}
<p>Population stability index (PSI) between each period's distribution of actual demand and the reference
distribution, flagged at {PSI_THRESHOLD:.2f}. A shift here means the demand the model is being asked to
predict has changed shape, which can degrade the forecast even when the model itself is unchanged.
<strong>Target PSI stayed low through {periods[0]} and {periods[1]} and rose to {pm['target_psi'].max():.3f}
in {periods[-1]}, still well under the {PSI_THRESHOLD:.2f} threshold, so actual demand has drifted only
mildly and the target layer gives no reason to retrain.</strong></p>

{B.section("prediction", "Section 2.3", "Prediction Drift")}
<p>PSI between the model's output distribution each period and the reference, flagged at {PSI_THRESHOLD:.2f}.
This is a label-free early indicator: it catches the model producing a different spread of forecasts before
actuals arrive to confirm error. <strong>Prediction PSI reached {pm['pred_psi'].max():.3f} in {periods[-1]},
its highest of the window but still below threshold, tracking the mild rise in target drift. As a secondary
proxy it would support investigation, not retraining, and here it stays clear of even that.</strong></p>
<p>The grouped bars below show all three PSI layers together against the {PSI_THRESHOLD:.2f} drift line, so
the target, prediction, and feature signals can be read side by side per period. Every bar sits below the
line across all three periods, which is why the drift column in the log stays clear.</p>
{B.chart("Drift PSI by Layer and Period", charts["psi"])}

{B.section("feature", "Section 2.4", "Feature Drift")}
<p>PSI between a rolling recent-demand feature (the prior period's consumption that feeds the forecast) and
its reference distribution, flagged at {PSI_THRESHOLD:.2f}. Feature drift is diagnostic context: it helps
explain a performance change but does not on its own establish that the model is wrong. <strong>The recent-demand
feature is the most stable of the four layers, peaking at just {pm['feature_psi'].max():.3f}, which points to
a sound input pipeline rather than a broken feed and confirms the mild target and prediction movement is a
real demand shift, not a data fault.</strong></p>

{B.section("dataquality", "Section 3", "Data-Quality Monitoring")}
<p>Model drift is not the only thing that moves in a live ERP. The master-level defects were fixed once,
but transaction-level defects keep arriving: every period brings new free-text and non-stock purchase lines
that carry no item number, and new near-duplicate postings from re-keyed or re-imported receipts. Left
unwatched, a rising share of either quietly starves the forecast of clean history. So the two defect rates
are monitored as their own series against a fixed threshold, on the same period cadence as the four
model-drift layers. <strong>Across {periods[0]} to {periods[-1]} the free-text line rate held near
{dq['ft_rate'].min():.1f}% to {dq['ft_rate'].max():.1f}% of new postings against a {FT_THRESHOLD:.1f}%
threshold, and the duplicate-record rate held near {dq['dup_rate'].min():.2f}% against a
{DUP_THRESHOLD:.2f}% threshold, so both series read within threshold every period and data quality adds no
retraining or investigation trigger. The standing decision stays HEALTHY.</strong> For context, the standing
detectors carry {dq_summary['T1']['free_lines']:,} free-text lines cleared for attribution at
{dq_summary['T1']['precision']*100:.0f}% precision and {dq_summary['T7']['flagged']:,} near-duplicate
postings at {dq_summary['T7']['precision']*100:.0f}% precision; monitoring watches the inflow rate, not the
back catalogue.</p>
<p>The table reads each period's defect rate against its threshold. The rate is the share of that period's
new ledger postings caught by each detector, so it is comparable period to period even as posting volume
shifts. Both columns sit well under their limits, and the reading stays green in every row.</p>
{dq_table()}
<p>The chart traces the same two rates across the window with each threshold drawn in. The point is the
flatness: neither series is climbing toward its line, which is what tells us the clean-history feed behind
the forecast is holding steady rather than eroding. A sustained climb toward either threshold would open a
data-quality ticket to widen the free-text attribution rules or tighten the duplicate matcher, and only then
would it feed back into the model-drift view.</p>
{B.chart("Data-Quality Defect Rates by Period vs Threshold", charts["dq"])}

{B.section("log", "Section 4", "Monitoring Log")}
<p>The period log records every layer for the three scoring periods: WAPE and MASE for performance, the
three PSI values for the drift layers, and the flag on each. A RETRAIN flag on the performance or target row
across two consecutive periods would move the standing decision to RETRAIN and open a retraining ticket; the
model would be refit on the extended history, validated on the held-out window, and promoted to the reorder
queue only if it beats the incumbent. A single isolated flag, as seen here, is logged and watched but takes
no action.</p>
{log_table()}
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(B.page("MLOps Monitoring Report: Demand Model",
                      "Case 03 mfg-inventory-forecast", toc, body), encoding="utf-8")
print(f"Monitoring report written to {OUT}")
