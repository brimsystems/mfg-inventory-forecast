-- Defect T6: purchase orders never closed. A partial receipt where the balance is
-- never received and the line is never closed, leaving quantity on order
-- indefinitely and inflating believed inbound supply (phantom on-order).

with supplier_lead as (

    select supplier_id,
           median(date_diff('day', order_date, received_date)) as median_lead_days
    from {{ ref('stg_erp__purchase_orders') }}
    where received_date is not null
    group by 1

),

open_lines as (

    select
        po_id, item_number, supplier_id, order_date,
        quantity_ordered - coalesce(quantity_received, 0) as phantom_qty,
        date_diff('day', order_date, cast('{{ var("as_of_date") }}' as date)) as days_open
    from {{ ref('stg_erp__purchase_orders') }}
    where po_status = 'OPEN' or received_date is null

)

select
    o.po_id, o.item_number, o.supplier_id, o.days_open, o.phantom_qty,
    coalesce(s.median_lead_days, 30) as supplier_normal_lead
from open_lines o
left join supplier_lead s using (supplier_id)
where o.days_open > coalesce(s.median_lead_days, 30) + 30
