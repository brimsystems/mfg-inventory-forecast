-- Defect T7: duplicate transaction postings. The same receipt or issue posted
-- twice, copying the whole line including the work-order reference. A second
-- posting within a day of an identical line is a confirmed duplicate.

with lines as (

    select
        transaction_id, item_number, type, quantity, transaction_date,
        coalesce(work_order_id, 'NA') as wo,
        coalesce(location, 'NA')      as loc,
        row_number() over (partition by item_number, type, quantity,
            coalesce(work_order_id, 'NA'), coalesce(location, 'NA')
            order by transaction_date)  as seq,
        lag(transaction_date) over (partition by item_number, type, quantity,
            coalesce(work_order_id, 'NA'), coalesce(location, 'NA')
            order by transaction_date)  as prior_date
    from {{ ref('stg_erp__inventory_transactions') }}
    where type in ('issue', 'receipt')

)

select
    transaction_id, item_number, type, quantity, transaction_date,
    date_diff('day', prior_date, transaction_date) as gap_days
from lines
where seq > 1
  and date_diff('day', prior_date, transaction_date) <= 1
