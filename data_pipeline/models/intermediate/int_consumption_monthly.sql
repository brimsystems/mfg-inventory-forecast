-- Consumption by canonical item and month, derived from issue transactions and
-- merged across duplicate records through the crosswalk. This is the demand
-- signal the forecast and policy read.

with issues as (

    select item_number, transaction_month, quantity
    from {{ ref('stg_erp__inventory_transactions') }}
    where type = 'issue'

),

crosswalk as (

    select * from {{ ref('item_crosswalk') }}

),

mapped as (

    select
        coalesce(x.canonical_item_number, i.item_number) as canonical_item_number,
        i.transaction_month                              as month,
        i.quantity
    from issues i
    left join crosswalk x on i.item_number = x.item_number

)

select
    canonical_item_number,
    month,
    sum(quantity) as consumption
from mapped
group by 1, 2
