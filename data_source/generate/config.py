"""Central configuration for the inventory-forecasting and ERP data-quality platform.

Every generative choice that determines whether the case works lives here as a
named constant: the observation window, the item universe, the multi-level bill
of materials, the demand-segment mix and its parameters, the ABC value
distribution, the scale of each planted data-quality defect, and the ten-week
remediation period. The generators read this module and nothing else for their
calibration, so the whole build can be tuned from one place.

The company is an industrial equipment builder: conveyors and material-handling
modules, industrial mixers and agitators, and custom enclosures and frames.
Fabrication of frames and weldments is in house; machined parts, motors,
gearboxes, bearings, controls and hardware are purchased. Products carry
multi-level bills of materials (product -> subassembly -> component), and
consumption is recorded by backflush at each level, by service-parts issues, and
by manual issues.
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
# A 36-month history used for training, validation and the rolling-origin
# backtest, followed by a three-month forward window that stands in as the
# current period. The reorder queue is generated as of the final day of the
# forward window. The remediation period is the final ten weeks of the history.
START_DATE      = date(2023, 1, 1)
MODEL_SPAN_END  = date(2025, 12, 31)   # last month of history; end of remediation
FORWARD_START   = date(2026, 1, 1)
END_DATE        = date(2026, 3, 31)     # last day of generated history
AS_OF_DATE      = date(2026, 3, 31)     # "today" for the reorder queue and on-hand

# The engagement: the final ten weeks of the history window.
REMEDIATION_WEEKS = 10
REMEDIATION_END   = MODEL_SPAN_END
REMEDIATION_START = REMEDIATION_END - timedelta(weeks=REMEDIATION_WEEKS)  # ~2025-10-22

# ── Source system → output subdirectory ─────────────────────────────────────
# The nine datasets stand in for the modules of a single ERP, its bolt-on
# inventory and purchasing records, and the purchasing manager's spreadsheet.
TABLE_SYSTEM_MAP = {
    "item_master":            "erp",
    "supplier_master":        "erp",
    "bill_of_materials":      "erp",
    "production_orders":      "erp",
    "service_orders":         "erp",
    "purchase_orders":        "erp",
    "inventory_transactions": "erp",
    "cycle_counts":           "wms",
    "buyer_spreadsheet":      "purchasing",
}

# ── Item universe ───────────────────────────────────────────────────────────
# Live canonical items carry demand and drive the case. Dead records (defect M1)
# and duplicate members (defect M4) are added on top to reach a ~2,400-record
# master of which ~58% is live before remediation.
N_LIVE_ITEMS = 1300         # canonical live purchased items (one row per physical item)
N_DEAD_ITEMS = 980          # inactive records never deactivated (M1)

# Purchased-item classes and their share of the LIVE catalog. Hardware,
# fasteners, fittings, bearings and electrical are the high-count classes where
# duplicate records, unit-of-measure problems and BOM omissions concentrate.
ITEM_CLASS_SHARES = {
    "Raw Material":    0.14,   # structural steel, tube, plate, sheet
    "Mechanical":      0.20,   # motors, gear reducers, bearings, sprockets, belts, rollers
    "Fittings":        0.08,   # hydraulic and pneumatic fittings
    "Electrical":      0.16,   # drives, sensors, contactors, enclosures, wire, cable
    "Fasteners":       0.16,
    "Hardware":        0.12,
    "Consumables":     0.08,   # weld wire, abrasives, paint, powder
    "Outside Service": 0.06,   # plating, powder coat, heat treat
}

# Relative unit-cost scale by class. Motors, gear reducers and drives are the
# expensive, low-line-count buys; hardware and consumables are cheap and bought
# in long multi-line orders. This shapes the spend and consumption concentration.
CLASS_COST_MULT = {
    "Raw Material":    2.0,
    "Mechanical":      6.0,
    "Fittings":        1.1,
    "Electrical":      4.0,
    "Fasteners":       0.25,
    "Hardware":        0.5,
    "Consumables":     0.8,
    "Outside Service": 1.6,
}

# Classes whose purchase orders carry many lines (distributor/hardware buys)
# versus classes bought one to a few lines at a time (motors, drives, services).
MANY_LINE_CLASSES = ["Fasteners", "Hardware", "Consumables", "Fittings"]
FEW_LINE_CLASSES  = ["Mechanical", "Electrical", "Outside Service"]

# Material grades carried for the raw-material class. Grades within a family are
# substitutable, so their demand shares a latent family factor (see below).
MATERIAL_FAMILIES = {
    "Structural Steel": ["A36 Angle", "A500 Tube", "A36 Plate", "A36 Channel"],
    "Stainless":        ["304 Sheet", "316 Sheet", "304 Tube"],
    "Aluminum":         ["6061 Extrusion", "5052 Sheet", "6063 Tube"],
}
METAL_CLASSES = ["Raw Material"]

# ── Bill of materials ───────────────────────────────────────────────────────
# Standard products, configurable, built from subassemblies and purchased
# components. Consumption is recorded by backflush through this structure.
N_PRODUCTS       = 80          # standard catalog products (conveyors, mixers, enclosures)
N_SUBASSEMBLIES  = 140         # shared subassemblies
PRODUCT_FAMILIES = ["Conveyor", "Mixer", "Enclosure"]
SUBASSY_PER_PRODUCT_RANGE   = (1, 3)    # subassemblies per product
COMPONENTS_PER_PRODUCT_RANGE= (2, 5)    # direct purchased components per product
COMPONENTS_PER_SUBASSY_RANGE= (2, 5)    # purchased components per subassembly
BOM_QTY_PER_RANGE           = (1, 8)    # qty of a component per parent

# Backflush from production is the dominant channel for the high-runner
# components on BOMs; service parts and manual pulls are smaller item-level
# channels. Lumpy and intermittent items are mostly service/manual (kept off the
# BOM) so the item-level segment mix survives the product-driven smoothing.
BOM_COMPONENT_SEGMENTS = ["smooth", "erratic"]   # segments eligible to sit on BOMs
SERVICE_ITEM_SHARE = 0.20      # share of items that also carry a service channel
SERVICE_SCALE      = 0.25      # service demand as a fraction of the item's engine demand

# ── Demand segments ─────────────────────────────────────────────────────────
# The mix is chosen so the case can report a genuine overall lift while still
# showing a segment (intermittent) where a simple method is the right call.
SEGMENTS = ["smooth", "erratic", "lumpy", "intermittent"]
# Target classified mix (spec) is 25 / 30 / 25 / 20. The generation shares below
# are offset from that target to cancel the classifier's leakage, so the
# classified distribution lands on target. See the checkpoint report.
SEGMENT_MIX = {
    "smooth":       0.17,
    "erratic":      0.40,
    "lumpy":        0.29,
    "intermittent": 0.14,
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
BASE_DEMAND_LOG_MEAN = 2.5     # exp(2.5) ~ 12 units/month median (item-level channel)
BASE_DEMAND_LOG_STD  = 0.70

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
STANDARD_COST_LOG_MEAN = 1.72   # exp(1.17) ~ $3.2 median before the class multiplier; sized so
                                # annual purchases land near $10M for a ~$28M builder
STANDARD_COST_LOG_STD  = 0.75
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
N_SUPPLIERS      = 40
SUPPLIER_TYPES   = ["Material", "Mechanical", "Electrical", "Fasteners",
                    "Distributor", "Outside Service"]
PAYMENT_TERMS    = ["Net 30", "Net 45", "Net 60", "2/10 Net 30"]
BASE_LEAD_TIME_RANGE = (7, 75)    # master lead-time days at item creation (1-3 month horizons)

# Map each item class to the supplier type that serves it.
CLASS_SUPPLIER_TYPE = {
    "Raw Material":    "Material",
    "Mechanical":      "Mechanical",
    "Fittings":        "Distributor",
    "Electrical":      "Electrical",
    "Fasteners":       "Fasteners",
    "Hardware":        "Fasteners",
    "Consumables":     "Distributor",
    "Outside Service": "Outside Service",
}

# ══════════════════════════════════════════════════════════════════════════
# Planted defects. Two tiers, master-level (M1-M7) and transaction-level
# (T1-T8), generated as artifacts of how people work. Every scale is chosen so
# the defect is material to the outcome, not cosmetic, and is parameterized so
# it can be tuned at the validation checkpoint.
# ══════════════════════════════════════════════════════════════════════════

# ── Master-level ────────────────────────────────────────────────────────────
# M1 Dead records never deactivated: active items with no movement in 24+ months.
M1_DEAD_SHARE          = 0.39     # dead records as a share of the whole master (35-45%)
M1_KEEP_REORDER_SHARE  = 0.30     # share of dead items that still carry a reorder point
M1_STRAY_TXN_SHARE     = 0.03     # share of dead items with one stray recent transaction
M1_RESCUE_SHARE        = 0.10     # dead items rescued in remediation (seasonal / safety-critical)

# M2 Stale parameters: every live item's lead time, reorder point, safety stock
# and standard cost date from go-live; actual lead times have drifted, upward for
# most suppliers and sharply for one.
M2_DRIFT_SHARE         = 0.26     # live items drifted enough to matter (>7 days subset)
M2_GENERAL_DRIFT_DAYS  = (2, 9)   # general upward drift range (days) over the history
M2_DRIFT_SUPPLIER_MONTHS = 18     # the sharply-drifting supplier ramps over the final 18 months
M2_DRIFT_SUPPLIER_START  = 14     # its lead time early in the history (days)
M2_DRIFT_SUPPLIER_RATIO  = (1.8, 2.4)  # end-of-history actual / master ratio

# M3 BOM omissions: hardware, fittings, wire, weld consumables and finishing
# materials missing from subassembly and product BOMs, so backflush never
# consumes them; omissions compound through levels.
M3_OMISSION_CLASSES    = ["Hardware", "Fasteners", "Fittings", "Consumables", "Outside Service"]
M3_ITEM_SHARE          = 0.30     # share of items in those classes omitted somewhere
M3_PRODUCT_SHARE       = 0.24     # share of products directly targeted for omission

# M4 Duplicate item records: same physical item under two or more numbers.
M4_LIVE_SHARE          = 0.065    # share of live items in a duplicate cluster (5-8%)
M4_CLUSTER_CLASSES     = ["Hardware", "Fasteners", "Fittings", "Mechanical", "Electrical"]
M4_MEMBERS_RANGE       = (2, 4)   # records per duplicate cluster
M4_SPLIT_MIN_SHARE     = 0.30     # each member gets at least this share of demand

# M5 UOM mismatch: purchase UOM differs from stock UOM with no conversion.
M5_N_ITEMS             = 48       # items with a box/each, spool/ft, length/ft or gal/oz mismatch
M5_UOM_PAIRS           = {        # (purchase_uom, stock_uom, conversion) by class
    "Hardware":   [("BOX", "EA", 100), ("BOX", "EA", 50), ("BOX", "EA", 25)],
    "Fasteners":  [("BOX", "EA", 100), ("BOX", "EA", 250)],
    "Electrical": [("SPOOL", "FT", 500), ("SPOOL", "FT", 1000)],
    "Raw Material":[("LENGTH", "FT", 20), ("LENGTH", "FT", 24)],
    "Consumables":[("GAL", "OZ", 128)],
}

# M6 Supplier fragmentation: three real suppliers under multiple names and IDs.
M6_FRAGMENTED_SUPPLIERS = 3       # real vendors split into...
M6_TOTAL_RECORDS        = 7       # ...this many supplier records

# M7 Missing and placeholder fields: blank cost, supplier or reorder point;
# item_class = MISC; descriptions in mixed conventions.
M7_BLANK_SHARE         = 0.12     # live items with at least one blocking blank (10-15%)
M7_MISC_SHARE          = 0.12     # live items with item_class = MISC (10-14%)

# ── Transaction-level ───────────────────────────────────────────────────────
# T1 Unrecorded consumption: consequence of M3 plus incomplete manual issues and
# unrecorded service-parts pulls; material leaves with no record, and chronic
# downward adjustments follow the annual count.
T1_UNRECORDED_SHARE    = 0.30     # share of an affected item's true usage that escapes
T1_MONTHLY_ADJ_PROB    = 0.45     # probability of a write-off adjustment in a given month

# T2 Adjustments as catch-all: ADJUST used for unrecorded issues, mis-receipts,
# returns and scrap; most carry blank or generic reason codes.
T2_ADJ_SHARE_OF_QTY    = 0.04     # share of quantity moved that flows through adjustments (15-25%)
T2_BLANK_REASON_SHARE  = 0.60     # adjustments with blank or generic reason (60-75%)
GENERIC_REASON_CODES   = ["", "ADJ", "VAR", "MISC", "COUNT"]
SPECIFIC_REASON_CODES  = ["CYCLE", "DAMAGE", "SCRAP", "RECOUNT", "RETURN"]

# T3 Free-text and non-stock PO lines: generic codes with typed descriptions,
# some corresponding to stocked items (the defect), some genuine ETO buys.
GENERIC_ITEM_CODES     = ["NONSTOCK", "MISC", "SHOPSUPPLY"]
T3_FREETEXT_SHARE      = 0.12     # share of PO lines that are free-text (10-14%)
T3_STOCKED_RATIO       = 0.55     # share of free-text lines matching a stocked item (the defect)

# T4 Batched and backdated postings: receipts cluster on Mondays and month-end;
# job completions reported at week-end; posting lag 1-5 days.
T4_BATCH_SHARE         = 0.58     # share of receipts displaced (50-65%)
T4_MAX_DISPLACEMENT    = 5        # days

# T5 Open documents never closed: partial receipts never closed; completed jobs
# left open.
T5_OPEN_PO_SHARE       = 0.06     # PO lines older than 90 days left open (5-8%)
T5_OPEN_JOB_SHARE      = 0.08     # completed jobs left open (6-10%)

# T6 Wrong references: issues against a similar item or wrong job, often within
# duplicate clusters or the same material family.
T6_ISSUE_SHARE         = 0.015    # share of manual issues misposted (1-2%)

# T7 Quantity and unit errors: order-of-magnitude and box/each keying errors,
# concentrated on M5 items.
T7_TXN_SHARE           = 0.005    # share of transactions with a keying error (0.5-1%)
T7_M5_WEIGHT           = 4.0      # relative over-weighting of M5 items

# T8 Duplicate postings: the same transaction twice within minutes.
T8_DUP_SHARE           = 0.003    # share of transactions posted a second time (0.2-0.4%)

# ── Replenishment replay ────────────────────────────────────────────────────
# Purchase orders are placed by replaying the ERP's own reorder points and lead
# times against book on-hand and book on-order, so the errors in those records
# produce their consequences: suppressed orders, late arrivals, rush buys and
# shortages. Physical stock is tracked alongside the books.
ORDER_COVER_DAYS        = 75      # order quantity ~ this many days of average demand
INITIAL_STOCK_COVER     = 1.5     # opening stock as a multiple of the reorder point
REORDER_GAP_DAYS        = 5       # no second regular order within this many days
PARTIAL_RECEIPT_PROB    = 0.09    # share of receipts that arrive short
MRP_MAX_LOT_MULT        = 4       # a planned order covers the net requirement, up to this many standard lots
SHELF_FULL_COVER_DAYS   = 180     # no regular order when the visible pile already covers this many days
FLOOR_VISUAL_MIN_DAYS   = 10      # after a surprise shortage, the floor reorders by sight at this cover
PARTIAL_NEVER_CLOSED    = 0.85    # of short receipts, the share whose balance never arrives (T5)
PARTIAL_RECEIVED_RANGE  = (0.6, 0.9)

# The purchasing manager's compensation on the items she tracks: she orders on
# her own (closer to actual) lead time and expedites when inbound runs late.
BUYER_LEAD_MULT         = 1.15    # her lead-time note relative to the true actual
BUYER_SAFETY_DAYS       = 28      # extra days of cover she keeps on tracked items
RUSH_LEAD_FRACTION      = 0.15    # a rush order arrives in this fraction of the normal lead time (min 2 days)
RUSH_ESCALATION_PROB    = 0.85    # share of visible stockouts on untracked items that get expedited
RUSH_PRICE_PREMIUM      = (0.10, 0.20)
RUSH_FREIGHT_BY_CLASS   = {       # freight charge on a rush line, by item class
    "Mechanical": (250, 600), "Electrical": (200, 500), "Raw Material": (150, 400),
    "Fittings": (60, 150), "Fasteners": (60, 150), "Hardware": (60, 150),
    "Consumables": (60, 150), "Outside Service": (100, 300),
}
COUNT_NOISE_SD          = 0.02    # counting noise on a physical count, share of quantity

# ── Transaction volume and floor conditions ─────────────────────────────────
LOCATIONS       = ["MAIN", "FLOOR", "RECV", "CRIB"]
SHARED_LOGINS   = ["ASSY1", "FAB1", "RECV"]     # shared floor / receiving logins
OFFICE_USERS    = ["jbuyer", "kbuyer", "pmgr", "scoord", "recvclerk"]
SHARED_LOGIN_SHARE = 0.78        # floor and receiving transactions under shared logins (70-85%)

# The purchasing manager's spreadsheet: the line-stopping components she tracks.
BUYER_SPREADSHEET_ITEMS   = 120
BUYER_DISAGREE_SHARE      = 0.70   # items where on-hand disagrees with the ERP (60-80%)
BUYER_RIGHT_SHARE         = 0.82   # disagreements where the spreadsheet is closer to truth (75-90%)

# Cycle counting: none before remediation; an annual physical inventory each year.
CYCLE_COUNT_ANNUAL_COVERAGE = 0.85


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


def remediation_weeks():
    """Return the (week_number, monday_date) pairs of the remediation period."""
    out = []
    # Start on the Monday on or before REMEDIATION_START.
    monday = REMEDIATION_START - timedelta(days=REMEDIATION_START.weekday())
    for w in range(1, REMEDIATION_WEEKS + 1):
        out.append((w, monday))
        monday += timedelta(weeks=1)
    return out
