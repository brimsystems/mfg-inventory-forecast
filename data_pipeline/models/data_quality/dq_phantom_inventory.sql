-- Defect D4: phantom inventory. Items whose most recent cycle count diverges from
-- the system quantity by more than the configured variance threshold.

with latest as (

    select
        *,
        row_number() over (partition by item_number order by count_date desc) as rn
    from {{ ref('stg_wms__cycle_counts') }}

)

select
    item_number,
    count_date,
    system_quantity,
    counted_quantity,
    round(count_variance, 3) as count_variance
from latest
where rn = 1
  and count_variance > {{ var('phantom_variance') }}
