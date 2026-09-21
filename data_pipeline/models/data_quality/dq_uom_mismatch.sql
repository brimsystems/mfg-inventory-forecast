-- Defect D3: unit-of-measure inconsistency. Items whose median purchase price is
-- a large multiple of the recorded unit cost, the signature of a box buy booked
-- without a conversion factor against issues in each.

with po as (

    select item_number, median(unit_price) as median_unit_price, count(*) as po_lines
    from {{ ref('stg_erp__purchase_orders') }}
    group by 1

),

im as (

    select item_number, item_class, standard_cost
    from {{ ref('stg_erp__item_master') }}
    where standard_cost > 0

)

select
    im.item_number,
    im.item_class,
    im.standard_cost,
    po.median_unit_price,
    round(po.median_unit_price / im.standard_cost, 1) as price_to_cost_ratio,
    round(po.median_unit_price / im.standard_cost)     as implied_box_size
from im
join po using (item_number)
where po.median_unit_price / im.standard_cost >= {{ var('uom_price_ratio') }}
