WITH RankedCustomers AS (
    SELECT
        customer_id,
        customer_desc,
        latitude_coordinate,
        longitude_coordinate,
        street_address,
        city_name,
        state_province_name,
        postal_code,
        region,
        current_ind,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY customer_id) AS rn
    FROM ccb_prd.dm.d_customer_v
    WHERE current_ind = 'Y'
)
SELECT
    customer_id,
    customer_desc,
    latitude_coordinate,
    longitude_coordinate,
    street_address,
    city_name,
    state_province_name,
    postal_code,
    region,
    current_ind
FROM RankedCustomers
WHERE rn = 1;