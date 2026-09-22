-- Defect T1: free-text / non-stock lines. Issues keyed against a generic item
-- code with a typed description instead of the real item number, so the demand is
-- hidden until the text is attributed back. Attribution and its confidence live in
-- the ML layer (ml/src/txn_quality.py); this model emits the affected lines.

select
    transaction_id,
    entered_code,
    description,
    quantity,
    unit_price,
    transaction_date
from {{ ref('stg_erp__inventory_transactions') }}
where item_number in ('NONSTOCK', 'MISC', 'SHOPSUPPLY')
  and description is not null
