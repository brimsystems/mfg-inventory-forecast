-- Item master with the resolution crosswalk applied. One row per recorded item,
-- carrying the canonical item it resolves to and whether it is the surviving
-- record. Nothing is deleted; the merge is expressed as an attribute.

with item_master as (

    select * from {{ ref('stg_erp__item_master') }}

),

crosswalk as (

    select * from {{ ref('item_crosswalk') }}

)

select
    im.*,
    coalesce(x.canonical_item_number, im.item_number) as canonical_item_number,
    im.item_number = coalesce(x.canonical_item_number, im.item_number) as is_survivor

from item_master im
left join crosswalk x on im.item_number = x.item_number
