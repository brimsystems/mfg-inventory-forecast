"""Central configuration for the inventory-forecasting and ERP data-quality platform.

Every generative choice that determines whether the case works lives here as a
named constant: the observation window, the item universe, the demand-segment
mix and its parameters, the ABC value distribution, and the scale of each
planted data-quality defect. The generators read this module and nothing else
for their calibration, so the whole simulation can be tuned from one place.
"""
from datetime import date, timedelta
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
RAW_DIR     = REPO_ROOT / "data_source" / "raw"
SAMPLES_DIR = REPO_ROOT / "data_source" / "samples"

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
SAMPLE_SIZE = 200

# ── Observation window ──────────────────────────────────────────────────────
# A modeled span used for training, validation and the rolling-origin backtest,
# followed by a three-month forward window that stands in as the current period.
# The reorder queue is generated as of the final day of the forward window.
START_DATE      = date(2023, 4, 1)
MODEL_SPAN_END  = date(2026, 5, 31)   # last month included in the backtest origins
FORWARD_START   = date(2026, 6, 1)
END_DATE        = date(2026, 8, 31)   # last day of generated history
AS_OF_DATE      = date(2026, 8, 31)   # "today" for the reorder queue and on-hand

# ── Source system → output subdirectory ─────────────────────────────────────
# The five datasets stand in for the modules of a single ERP and its bolt-on
# inventory and purchasing records.
TABLE_SYSTEM_MAP = {
    "item_master":            "erp",
    "inventory_transactions": "erp",
    "purchase_orders":        "erp",
    "suppliers":              "erp",
    "cycle_counts":           "wms",
}

# ── Item universe ───────────────────────────────────────────────────────────
N_ITEMS = 800

# Purchased-item classes and their share of the catalog. Fasteners and hardware
# are the high-count, high-transaction classes where duplicate records and
# unit-of-measure problems concentrate in practice.
ITEM_CLASS_SHARES = {
    "Bar Stock":       0.17,
    "Sheet":           0.11,
    "Fasteners":       0.22,
    "Hardware":        0.14,
    "Tooling":         0.15,
    "Consumables":     0.13,
    "Outside Service": 0.08,
}

# Material grades carried for the metal classes. Grades within a family are
# substitutable, so their demand shares a latent family factor (see below).
MATERIAL_FAMILIES = {
    "Steel":    ["1018 CRS", "1045 CRS", "4140 HR", "A36 Plate"],
    "Stainless":["304 SS", "316 SS", "17-4 PH"],
    "Aluminum": ["6061-T6", "7075-T6", "2024-T3"],
    "Titanium": ["Ti-6Al-4V"],
}
METAL_CLASSES = ["Bar Stock", "Sheet"]

# ── Demand segments ─────────────────────────────────────────────────────────
# The mix is chosen so the case can report a genuine overall lift while still
# showing a segment (intermittent) where a simple method is the right call.
SEGMENTS = ["smooth", "erratic", "lumpy", "intermittent"]
# Target classified mix (spec) is 25 / 30 / 25 / 20. The generation shares below
# are offset from that target to cancel the classifier's leakage (a minority of
# erratic items read as smooth, a few lumpy items as intermittent), so the
# classified distribution lands on target. See the checkpoint report.
SEGMENT_MIX = {
    "smooth":       0.22,
    "erratic":      0.34,
    "lumpy":        0.27,
    "intermittent": 0.17,
}

# Per-segment demand process parameters. Monthly demand is built as a base level
# shaped by autocorrelation, calendar and seasonal structure, then multiplied by
# irreducible lognormal noise. The noise sigma sets the error floor: it is the
# part no feature can explain, which keeps achievable accuracy realistic.
#
#   ar_phi        first-order autocorrelation of the demand level
#   noise_sigma   sigma of multiplicative lognormal noise (the irreducible floor)
#   active_prob   probability a given month has any demand (1.0 = every month)
#   qty_cv        coefficient of variation of order quantity within active months
SEGMENT_PARAMS = {
    "smooth":       {"ar_phi": 0.55, "noise_sigma": 0.20, "active_prob": 1.00, "qty_cv": 0.18},
    "erratic":      {"ar_phi": 0.78, "noise_sigma": 0.45, "active_prob": 0.95, "qty_cv": 0.55},
    "lumpy":        {"ar_phi": 0.20, "noise_sigma": 0.28, "active_prob": 1.00, "qty_cv": 0.35},
    "intermittent": {"ar_phi": 0.10, "noise_sigma": 0.28, "active_prob": 0.55, "qty_cv": 0.60},
}

# Base monthly demand (units) is lognormal; a heavy right tail creates the ABC
# value concentration once multiplied by unit cost.
BASE_DEMAND_LOG_MEAN = 3.2     # exp(3.2) ~ 25 units/month median
BASE_DEMAND_LOG_STD  = 1.05

# Lumpy, calendar-driven items: a subset of repeat customers release on a
# quarterly schedule, so demand concentrates in one month of each quarter with a
# large multiplier and stays low otherwise. The release month is item-specific
# (phase 0-2) so the spikes are not all simultaneous.
LUMPY_QUARTER_SPIKE_MULT = 6.0
LUMPY_BASELINE_FRACTION  = 0.15   # off-quarter demand as a fraction of the level

# Seasonal materials: a share of items carry a sinusoidal annual factor keyed to
# their material family, so seasonality is shared across substitutable grades.
SEASONAL_ITEM_SHARE   = 0.30
SEASONAL_AMP_RANGE    = (0.15, 0.40)

# Substitutable-grade coupling: fraction of an item's month-to-month movement
# that comes from a shared material-family factor rather than its own draw.
FAMILY_FACTOR_WEIGHT  = 0.35

# ── Cost, ABC and current (stale) inventory policy ──────────────────────────
STANDARD_COST_LOG_MEAN = 1.6    # exp(1.6) ~ $5 median unit cost
STANDARD_COST_LOG_STD  = 1.15
# ABC by cumulative share of annual consumption value.
ABC_A_CUM = 0.80
ABC_B_CUM = 0.95
SERVICE_LEVEL_BY_ABC = {"A": 0.98, "B": 0.95, "C": 0.90}

# The reorder points and safety stocks already in the ERP were set years ago on
# the master lead time and a rough monthly demand guess, then never revisited.
# They are deliberately miscalibrated: a multiplicative error is baked in so the
# recomputed policy has room to improve.
CURRENT_POLICY_ERROR_STD = 0.35

# ── Supplier master ─────────────────────────────────────────────────────────
N_SUPPLIERS      = 24
SUPPLIER_TYPES   = ["Material", "Fasteners", "Tooling", "Outside Service", "Distributor"]
PAYMENT_TERMS    = ["Net 30", "Net 45", "Net 60", "2/10 Net 30"]
BASE_LEAD_TIME_RANGE = (7, 75)    # master lead-time days at item creation (spans 1-3 month horizons)

# ── Planted defects ─────────────────────────────────────────────────────────
# Each scale is chosen so the defect is material to the outcome, not cosmetic.

# D1 Duplicate item records. Concentrated in higher-consumption hardware and
# material whose combined demand is forecastable but whose split halves are not.
D1_N_CLUSTERS          = 35
D1_CLUSTER_CLASSES     = ["Fasteners", "Hardware", "Bar Stock"]
D1_MEMBERS_RANGE       = (2, 3)   # records per duplicate cluster
D1_SPLIT_MIN_SHARE     = 0.30     # each half gets at least this share of demand

# D2 Stale lead times. One supplier's true lead time drifts upward over 18
# months while every master_lead_time_days stays at its creation value.
D2_N_ITEMS             = 120
D2_DRIFT_START_DAYS    = 12
D2_DRIFT_END_DAYS      = 27
D2_DRIFT_MONTHS        = 18       # drift plays out over the final 18 months

# D3 Unit-of-measure inconsistency: purchased by the box, issued by the each,
# with no conversion factor recorded, inflating apparent demand and on-hand.
D3_N_ITEMS             = 40
D3_BOX_SIZES           = [25, 50, 100, 250]

# D4 Phantom inventory: cycle counts diverge from system quantity, worse for
# items with UOM or duplicate problems, and growing with time since last count.
D4_PHANTOM_SHARE       = 0.15
D4_VARIANCE_BASE       = 0.04     # baseline count variance
D4_VARIANCE_PER_MONTH  = 0.015    # added variance per month since last count

# D5 Inconsistent supplier records: one real vendor split across three spellings
# and two IDs, fragmenting spend and lead-time history.
D5_SPELLING_VARIANTS   = 3
D5_SECOND_ID_SHARE     = 0.45     # share of the vendor's POs booked to the 2nd id

# D6 Missing fields: blank reorder point, missing standard cost, or null primary
# supplier on a share of active items.
D6_MISSING_SHARE       = 0.08

# ── Transaction volume ──────────────────────────────────────────────────────
# Issues are the demand signal; receipts follow purchase orders; adjustments and
# returns are the small remainder that make the ledger read like a real one.
ADJUSTMENT_RATE = 0.02    # adjustments as a share of issue lines
RETURN_RATE     = 0.015   # returns as a share of issue lines
LOCATIONS       = ["MAIN", "FLOOR", "CRIB"]

# Cycle counting covers part of the catalog each period (an ABC-weighted cadence).
CYCLE_COUNT_ANNUAL_COVERAGE = 0.85   # share of items counted at least once a year

# ── Transaction-level defects (addendum T1-T7) ──────────────────────────────
# These are created by people keying transactions day to day. They hide in the
# ledger and purchase orders and each needs its own detection method. Rates are
# parameterized so they can be tuned at the validation checkpoint. The master
# defects D1-D6 above are unchanged.

# T1 Free-text / non-stock lines. Consumption for affected items is diverted to
# generic item codes with a typed description, so demand is understated until the
# free text is attributed back. A share of the free-text lines are genuine
# one-offs that must NOT be attributed (held back as true negatives).
GENERIC_ITEM_CODES     = ["NONSTOCK", "MISC", "SHOPSUPPLY"]
T1_AFFECTED_ITEMS      = 70       # real items whose demand is partly diverted
T1_DIVERTED_SHARE      = 0.14     # share of an affected item's issues diverted to free text
T1_ONEOFF_RATIO        = 0.55     # genuine one-off free-text lines as a multiple of diverted lines

# T2 Quantity keying errors: order-of-magnitude (x10) and box-as-each (xN) errors,
# weighted toward the D3 unit-of-measure items.
T2_TXN_SHARE           = 0.008    # share of issue/receipt lines with a keying error
T2_D3_WEIGHT           = 4.0      # relative over-weighting of D3 items

# T3 Issues posted to the wrong item, within the same family or duplicate cluster.
T3_ISSUE_SHARE         = 0.015    # share of issue lines misposted to a similar item

# T4 Unrecorded consumption written off through negative adjustments (chronic).
T4_ITEMS               = 25       # items with a persistent negative adjustment pattern
T4_MONTHLY_ADJ_PROB    = 0.55     # probability of a write-off adjustment in a given month
T4_UNRECORDED_SHARE    = 0.30     # share of the item's true usage that escapes as adjustments

# T5 Receipt-date batching: postings cluster on Mondays and month-end, displaced
# 1-4 days from actual arrival, biasing computed lead times upward.
T5_BATCH_SHARE         = 0.30     # share of receipts displaced
T5_MAX_DISPLACEMENT    = 4        # days

# T6 Purchase orders never closed: partial receipt, balance never received, line
# left open, inflating on-order quantity (phantom inbound).
T6_OPEN_SHARE          = 0.05     # share of PO lines older than 90 days left open

# T7 Duplicate transaction postings: the same line posted twice.
T7_DUP_SHARE           = 0.003    # share of transactions posted a second time


def month_starts(start: date = START_DATE, end: date = END_DATE):
    """Return the first day of every month in the window, inclusive."""
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(date(y, m, 1))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def working_days(start: date = START_DATE, end: date = END_DATE):
    """Return every Monday-Friday date within the window."""
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days
