with source as (

    select * from {{ source('erp', 'item_master') }}

),

staged as (

    select
        item_number,
        description,
        uom,
        item_class,
        -- Nullable: some records are missing a standard cost (defect D6).
        cast(standard_cost as double)          as standard_cost,
        -- Nullable: some records have a blank reorder point (defect D6).
        cast(current_reorder_point as double)  as current_reorder_point,
        cast(current_safety_stock as double)   as current_safety_stock,
        cast(master_lead_time_days as integer) as master_lead_time_days,
        -- Nullable: some records have no primary supplier (defect D6).
        primary_supplier_id,
        status,
        cast(created_date as date)             as created_date

    from source

)

select * from staged
