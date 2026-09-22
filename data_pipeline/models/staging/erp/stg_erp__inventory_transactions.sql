with source as (

    select * from {{ source('erp', 'inventory_transactions') }}

),

staged as (

    select
        transaction_id,
        item_number,
        cast(transaction_date as date)  as transaction_date,
        type,
        cast(quantity as double)        as quantity,
        uom,
        work_order_id,
        location,
        -- Populated only on non-stock free-text lines (defect T1).
        description,
        entered_code,
        cast(unit_price as double)      as unit_price,
        date_trunc('month', cast(transaction_date as date)) as transaction_month

    from source

)

select * from staged
