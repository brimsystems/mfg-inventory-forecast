-- Defect D5: inconsistent supplier records. Name stems that resolve to more than
-- one supplier id, the signature of a single vendor fragmented across spellings
-- and ids.

with s as (

    select
        supplier_id,
        supplier_name,
        supplier_name_norm,
        split_part(supplier_name_norm, ' ', 1) as name_stem
    from {{ ref('stg_erp__suppliers') }}

)

select
    name_stem,
    count(distinct supplier_id)               as n_ids,
    string_agg(distinct supplier_id, ', ')    as supplier_ids,
    string_agg(distinct supplier_name, ' | ') as spellings
from s
group by 1
having count(distinct supplier_id) > 1
