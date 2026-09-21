-- Corrected lead time per canonical item, recalculated from actual purchase-order
-- receipts over the trailing year, replacing the stale master value (defect D2).

with po as (

    select item_number, order_date, actual_lead_days
    from {{ ref('stg_erp__purchase_orders') }}
    where actual_lead_days is not null

),

crosswalk as (

    select * from {{ ref('item_crosswalk') }}

),

mapped as (

    select
        coalesce(x.canonical_item_number, p.item_number) as canonical_item_number,
        p.order_date,
        p.actual_lead_days
    from po p
    left join crosswalk x on p.item_number = x.item_number

),

recent as (

    select *
    from mapped
    where order_date >= (select max(order_date) from mapped) - interval '365 day'

)

select
    canonical_item_number,
    median(actual_lead_days) as corrected_lead_days,
    quantile_cont(actual_lead_days, 0.90) as p90_lead_days
from recent
group by 1
