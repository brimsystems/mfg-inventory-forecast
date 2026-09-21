-- Defect D6: missing fields. Active item records missing a standard cost, a
-- reorder point, or a primary supplier.

select
    item_number,
    item_class,
    standard_cost is null          as missing_standard_cost,
    current_reorder_point is null  as missing_reorder_point,
    primary_supplier_id is null    as missing_supplier
from {{ ref('stg_erp__item_master') }}
where status = 'ACTIVE'
  and (standard_cost is null
       or current_reorder_point is null
       or primary_supplier_id is null)
