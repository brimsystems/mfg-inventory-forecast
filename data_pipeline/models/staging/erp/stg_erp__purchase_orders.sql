with source as (

    select * from {{ source('erp', 'purchase_orders') }}

),

staged as (

    select
        po_id,
        cast(line as integer)               as line,
        item_number,
        supplier_id,
        cast(order_date as date)            as order_date,
        cast(promised_date as date)         as promised_date,
        -- Null until the line is received.
        cast(received_date as date)         as received_date,
        cast(quantity_ordered as double)    as quantity_ordered,
        cast(quantity_received as double)   as quantity_received,
        cast(unit_price as double)          as unit_price,
        -- Actual lead time realized on received lines, for lead-time recalculation.
        case
            when received_date is not null
            then date_diff('day', cast(order_date as date), cast(received_date as date))
        end                                 as actual_lead_days

    from source

)

select * from staged
