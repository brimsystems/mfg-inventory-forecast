# Case 03 deliverable-update brief (transaction-defects addendum)

Read `ml/reports/BUILD_NOTES.md` first for conventions (brand.py, no em dashes, no
"simulation/synthetic/demo/case study", descriptive lead-in before every chart,
house palette, run as `python -m ml.reports.generate_<name>`). This file gives the
NEW numbers and per-deliverable changes. KEEP all existing sections unless told to
replace; ADD the transaction-level content. Every number must come from the data
files, not this brief. Re-run and verify dashes=0 after.

## New data files
- `ml/data/backtest/threeway_overall.json`: {raw, master, fully} = WAPE fractions (0.4735, 0.4575, 0.4400).
- `ml/data/backtest/threeway_raw.parquet`, `threeway_master.parquet`, `threeway_fully.parquet`: cols item, origin, target, pred, segment, abc (compute WAPE = sum|target-pred|/sum(target), overall / by segment / by abc; and subsets).
- `ml/data/data_quality/txn/summary.json`: {T1:{free_lines,attributed,held_for_review,precision,recall}, T2:{flagged,confirmed,precision,recall}, T4:{flagged_items,precision,recall}, T6:{flagged,precision,recall}, T7:{flagged,confirmed,precision,recall}, confirmed_share}.
- `ml/data/data_quality/txn/*.parquet`: t1_attribution (transaction_id, quantity, probable_item, confidence, confirmed), t2_corrections, t4_adjustments (item_number, adjustment_count?, net_adjustment?, implied_usage), t6_phantom (po_id, supplier_id, days_open, phantom_qty), t7_duplicates.
- `ml/data/policy/policy_summary.json`: current/corrected/forecast {fill, inv, stockouts, expedite} + phantom_stockout_share.
- `ml/data/marts/consumption_raw/master/fully/true.parquet`.
- `data_source/truth/txn_defects.json`: lists t1..t7 (use len() for planted counts; t1 entries have is_oneoff).

## Key validated numbers (state where relevant)
- Two tiers of defects: master-level D1-D6 (detectable by lookup/similarity, stated as fact) and transaction-level T1-T7 (probabilistic findings from tens of thousands of ledger lines, split into CONFIRMED vs PROBABLE). This distinction is the framing.
- Transaction defects: T1 free-text lines 1498 (963 attributed at 99% precision/recall, 535 held for review incl genuine one-offs); T2 keying errors 1352 flagged, 16 confirmed at 88% precision (deliberately small high-precision set, the rest flagged for review); T4 25 items with chronic negative adjustments (100%); T5 receipt batching ~32% of receipts, median 2-day upward lead-time bias; T6 816 never-closed PO lines (100%); T7 575 near-duplicate postings, 343 confirmed at 62% precision. Confirmed share ~51% (rest flagged for human review).
- T1 attribution recovers ~15% additional consumption on affected items.
- Three-way decomposition (WAPE vs true demand over lead time): raw 47.3% -> master 45.8% -> fully 44.0%. Master-level cleaning worth +3.4% overall (and rescues duplicate items from ~61% to ~26% WAPE); transaction cleaning worth +3.8% overall and ~+9.7% on the free-text items. Report honestly; transaction gain is real but each tier's largest effect is on the items it repairs.
- Policy on fully-cleaned data: current fill 85.7% / $902K inventory / 1,194 stockouts / $299K expedite; corrected 94.4% / $1.89M; forecast 95.6% / $1.37M. At equal ~96% service, forecast releases ~$528K (~28%) vs corrected. T6 phantom on-order: 493 items, ~123 stockout events (~10% of total) attributable to material phantom on-order.

## Per-deliverable changes
1. **data_quality_audit** (`generate_data_quality_audit.py`): KEEP the master D1-D6 sections. ADD an opening line distinguishing the two tiers (master defects are straightforward to detect but need careful remediation; transaction defects need detection methods with real error rates and leave a residue needing human review). ADD a new top-level section "Transaction-level data quality" with a subsection per T1-T7: detection method (one line), counts, and CONFIRMED vs PROBABLE findings with precision/recall from summary.json + txn_defects.json. Add a chart of confirmed-vs-probable by defect. Cross-reference the forecast gain, do not restate it.
2. **analytics_report** (`generate_analytics_report.py`): In the stockout analysis, ADD the T6 phantom on-order finding (493 items, ~123 events, ~10% of stockouts, from policy_summary.json + t6_phantom.parquet). In supplier lead-time performance, note the T5 receipt-batching biases raw lead times upward ~2 days and that a robust median is used. Refresh policy numbers from policy_summary.json.
3. **dashboard** (`generate_dashboard.py`): Refresh all numbers from the current data (fill ~86%, etc.). No structural change; still current-state on clean data.
4. **model_overview** (`generate_model_overview.py`): REPLACE the two-way cleaning comparison with the THREE-WAY decomposition in plain language: what each stage of cleaning was worth (raw -> master -> fully; master-level +3.4%, transaction-level +3.8% overall, +9.7% on free-text items). Chart the three-tier WAPE. Keep the candidate comparison and segment performance.
5. **technical_report** (`generate_technical_report.py`): ADD full three-way results by segment (from threeway_*.parquet), the T1 attribution model's precision and recall (99%/99%, threshold, confirmed/probable), the robust lead-time estimation and an explicit statement that T5 receipt batching sets a floor on lead-time precision (honest limitation). Chart three-way by segment.
6. **monitoring_report** (`generate_monitoring_report.py`): ADD a data-quality monitoring section: new duplicates and new free-text PO lines appear continuously, so defect rates are monitored as their own series with thresholds, alongside model drift. Keep the four model-drift layers and the HEALTHY decision.

Output paths unchanged (docs/reports/<name>.html). Verify each: run OK, dashes=0, no banned words.
