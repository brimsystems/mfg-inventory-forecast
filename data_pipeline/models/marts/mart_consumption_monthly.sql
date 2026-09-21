-- Cleaned monthly consumption on a dense item-by-month grid, so demand gaps are
-- explicit zeros. Grain: one row per canonical item per month.

with c as (

    select * from {{ ref('int_consumption_monthly') }}

),

items as (

    select distinct canonical_item_number from c

),

spine as (

    select cast(unnest(generate_series(
        (select min(month) from c),
        (select max(month) from c),
        interval '1 month')) as date) as month

),

grid as (

    select i.canonical_item_number, s.month
    from items i
    cross join spine s

)

select
    g.canonical_item_number,
    g.month,
    coalesce(c.consumption, 0) as consumption
from grid g
left join c
    on g.canonical_item_number = c.canonical_item_number
   and g.month = c.month
