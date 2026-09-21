with source as (

    select * from {{ source('erp', 'suppliers') }}

),

staged as (

    select
        supplier_id,
        supplier_name,
        supplier_type,
        payment_terms,
        status,
        -- Normalized name for variant detection: lowercase, letters and spaces only.
        trim(regexp_replace(lower(supplier_name), '[^a-z ]', '', 'g')) as supplier_name_norm

    from source

)

select * from staged
