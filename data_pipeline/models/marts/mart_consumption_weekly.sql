-- Cleaned weekly consumption by canonical item, for the shorter-horizon views on
-- the dashboard. Grain: one row per canonical item per week with any issues.

with issues as (

    select item_number, transaction_date, quantity
    from {{ ref('stg_erp__inventory_transactions') }}
    where type = 'issue'

),

crosswalk as (

    select * from {{ ref('item_crosswalk') }}

),

mapped as (

    select
        coalesce(x.canonical_item_number, i.item_number) as canonical_item_number,
        date_trunc('week', i.transaction_date)           as week,
        i.quantity
    from issues i
    left join crosswalk x on i.item_number = x.item_number

)

select
    canonical_item_number,
    week,
    sum(quantity) as consumption
from mapped
group by 1, 2
