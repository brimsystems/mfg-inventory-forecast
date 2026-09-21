-- Canonical item attributes for the forecast, policy and reorder queue. Grain:
-- one row per canonical item. Carries the demand segment (classified from the
-- cleaned series by average inter-demand interval and squared CV), the ABC class
-- by consumption value, and both the stale master and corrected lead times.

with monthly as (

    select * from {{ ref('mart_consumption_monthly') }}

),

survivor as (

    select * from {{ ref('int_items_resolved') }} where is_survivor

),

leads as (

    select * from {{ ref('int_lead_times') }}

),

annual as (

    select canonical_item_number, sum(consumption) as annual_consumption
    from monthly
    where month >= (select max(month) from monthly) - interval '11 month'
    group by 1

),

seg_stats as (

    select
        canonical_item_number,
        count(*)                                    as n_months,
        count(*) filter (where consumption > 0)     as nz,
        avg(consumption) filter (where consumption > 0)         as mean_nz,
        stddev_pop(consumption) filter (where consumption > 0)  as sd_nz
    from monthly
    group by 1

),

seg as (

    select
        canonical_item_number,
        case
            when nz < 2 then 'intermittent'
            when (n_months::double / nz) < 1.32 then
                case when pow(coalesce(sd_nz, 0) / nullif(mean_nz, 0), 2) < 0.49
                     then 'smooth' else 'erratic' end
            else
                case when pow(coalesce(sd_nz, 0) / nullif(mean_nz, 0), 2) < 0.49
                     then 'intermittent' else 'lumpy' end
        end as segment
    from seg_stats

),

base as (

    select
        s.canonical_item_number,
        s.item_class,
        s.standard_cost,
        s.master_lead_time_days,
        coalesce(l.corrected_lead_days, s.master_lead_time_days) as corrected_lead_days,
        coalesce(a.annual_consumption, 0)                        as annual_consumption,
        coalesce(a.annual_consumption, 0)
            * coalesce(s.standard_cost, (select median(standard_cost) from survivor)) as annual_value,
        coalesce(g.segment, 'intermittent')                     as segment
    from survivor s
    left join annual a on s.canonical_item_number = a.canonical_item_number
    left join leads l  on s.canonical_item_number = l.canonical_item_number
    left join seg g    on s.canonical_item_number = g.canonical_item_number

),

ranked as (

    select
        *,
        sum(annual_value) over (order by annual_value desc
            rows between unbounded preceding and current row)
            / nullif(sum(annual_value) over (), 0) as cum_value_share
    from base

)

select
    canonical_item_number,
    item_class,
    standard_cost,
    master_lead_time_days,
    corrected_lead_days,
    annual_consumption,
    annual_value,
    segment,
    case
        when cum_value_share <= {{ var('abc_a_cum') }} then 'A'
        when cum_value_share <= {{ var('abc_b_cum') }} then 'B'
        else 'C'
    end as abc
from ranked
