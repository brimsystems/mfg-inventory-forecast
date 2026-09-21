-- Recorded inventory position per canonical item, as the net of receipts,
-- returns, issues and adjustments. For box-bought items (defect D3) receipts are
-- booked in boxes against issues in each, so this recorded position is distorted;
-- that distortion is what the cycle-count variance surfaces. Grain: one row per
-- canonical item.

with tx as (

    select item_number, type, quantity
    from {{ ref('stg_erp__inventory_transactions') }}

),

crosswalk as (

    select * from {{ ref('item_crosswalk') }}

),

mapped as (

    select
        coalesce(x.canonical_item_number, t.item_number) as canonical_item_number,
        t.type,
        t.quantity
    from tx t
    left join crosswalk x on t.item_number = x.item_number

),

net as (

    select
        canonical_item_number,
        sum(case type
                when 'receipt' then quantity
                when 'return'  then quantity
                when 'issue'   then -quantity
                when 'adjustment' then quantity
                else 0 end) as on_hand_recorded
    from mapped
    group by 1

),

attrs as (

    select canonical_item_number, standard_cost, corrected_lead_days, abc, segment
    from {{ ref('mart_item_attributes') }}

)

select
    n.canonical_item_number,
    greatest(n.on_hand_recorded, 0)                        as on_hand_recorded,
    a.standard_cost,
    round(greatest(n.on_hand_recorded, 0) * a.standard_cost, 2) as on_hand_value,
    a.corrected_lead_days,
    a.abc,
    a.segment
from net n
left join attrs a on n.canonical_item_number = a.canonical_item_number
