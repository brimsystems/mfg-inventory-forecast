with source as (

    select * from {{ source('wms', 'cycle_counts') }}

),

staged as (

    select
        count_id,
        item_number,
        cast(count_date as date)          as count_date,
        cast(system_quantity as double)   as system_quantity,
        cast(counted_quantity as double)  as counted_quantity,
        counter_id,
        -- Absolute relative variance between counted and system quantity.
        case when system_quantity > 0
             then abs(counted_quantity - system_quantity) / system_quantity
        end                               as count_variance

    from source

)

select * from staged
