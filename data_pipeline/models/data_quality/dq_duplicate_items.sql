-- Defect D1: duplicate item records. One row per canonical item that resolves
-- from more than one recorded number, with the records involved and the
-- consumption value that was split across them.

with crosswalk as (

    select * from {{ ref('item_crosswalk') }}

),

clusters as (

    select
        canonical_item_number,
        count(*)                        as n_records,
        string_agg(item_number, ', ')   as records
    from crosswalk
    group by 1
    having count(*) > 1

),

issues as (

    select item_number, sum(quantity) as issued
    from {{ ref('stg_erp__inventory_transactions') }}
    where type = 'issue'
    group by 1

),

cost as (

    select item_number, standard_cost from {{ ref('stg_erp__item_master') }}

),

value as (

    select
        coalesce(x.canonical_item_number, i.item_number) as canonical_item_number,
        sum(i.issued * coalesce(c.standard_cost, 0))     as consumption_value_split
    from issues i
    left join crosswalk x on i.item_number = x.item_number
    left join cost c on i.item_number = c.item_number
    group by 1

)

select
    cl.canonical_item_number,
    cl.n_records,
    cl.records,
    round(coalesce(v.consumption_value_split, 0), 0) as consumption_value_split
from clusters cl
left join value v on cl.canonical_item_number = v.canonical_item_number
