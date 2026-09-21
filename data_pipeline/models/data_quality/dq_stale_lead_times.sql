-- Defect D2: stale lead times. Items whose recorded master lead time understates
-- the lead time actually being realized by at least the configured threshold.

select
    canonical_item_number,
    master_lead_time_days,
    corrected_lead_days,
    corrected_lead_days - master_lead_time_days as understatement_days
from {{ ref('mart_item_attributes') }}
where corrected_lead_days - master_lead_time_days >= {{ var('stale_lead_gap_days') }}
