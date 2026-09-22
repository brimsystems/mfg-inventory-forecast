-- Defect T5: receipt-date batching. Receiving posts in batches, so received_date
-- clusters on Mondays rather than reflecting actual arrival. The day-of-week
-- distribution surfaces the batching, which biases computed lead times upward.

with receipts as (

    select dayofweek(received_date) as dow_num,
           dayname(received_date)   as day_of_week
    from {{ ref('stg_erp__purchase_orders') }}
    where received_date is not null

)

select
    dow_num,
    day_of_week,
    count(*)                                              as receipts,
    round(100.0 * count(*) / sum(count(*)) over (), 1)    as pct_of_receipts
from receipts
group by 1, 2
order by 1
