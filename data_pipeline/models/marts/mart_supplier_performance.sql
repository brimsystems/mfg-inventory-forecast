-- Supplier lead-time performance and spend, from received purchase-order lines.
-- Grain: one row per recorded supplier id (fragmented ids are surfaced in the
-- data-quality model dq_supplier_variants and consolidated via the supplier
-- crosswalk in analysis).

with po as (

    select supplier_id, actual_lead_days, quantity_received, unit_price
    from {{ ref('stg_erp__purchase_orders') }}
    where received_date is not null

)

select
    supplier_id,
    count(*)                                   as received_lines,
    median(actual_lead_days)                   as median_lead_days,
    quantile_cont(actual_lead_days, 0.90)      as p90_lead_days,
    round(sum(quantity_received * unit_price), 2) as spend
from po
group by 1
