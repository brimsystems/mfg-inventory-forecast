-- Defect T4: unrecorded consumption. Material used but never issued, written off
-- through chronic negative adjustments with vague reason codes. A persistent
-- one-way adjustment pattern means usage is escaping the transaction record.

with adjustments as (

    select
        item_number,
        count(*)      as adjustment_count,
        sum(quantity) as net_adjustment
    from {{ ref('stg_erp__inventory_transactions') }}
    where type = 'adjustment'
    group by 1

)

select
    item_number,
    adjustment_count,
    net_adjustment,
    -net_adjustment as implied_unrecorded_usage
from adjustments
where net_adjustment < -20
  and adjustment_count >= 8
