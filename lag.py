import duckdb

# Connect to DuckDB
con = duckdb.connect()

input_parquet = 'phase2_spatial_temporal_features.parquet'
final_output = 'final_clean_dataset.parquet'

query = f"""
COPY (
    SELECT * FROM (
        SELECT *,
            -- 1. Primary Physical Lags (3-Day)
            LAG(uo, 3) OVER w AS uo_lag3,
            LAG(vo, 3) OVER w AS vo_lag3,
            LAG(mlotst, 3) OVER w AS mlotst_lag3,
            LAG(zos, 3) OVER w AS zos_lag3,
            LAG(thetao, 3) OVER w AS thetao_lag3,
            LAG(so, 3) OVER w AS so_lag3,
            LAG(ssrd, 3) OVER w AS ssrd_lag3,
            
            -- 2. Spatial Derivative Lags (3-Day)
            LAG(vorticity, 3) OVER w AS vorticity_lag3,
            LAG(okubo_weiss, 3) OVER w AS okubo_weiss_lag3,
            LAG(grad_ssh, 3) OVER w AS grad_ssh_lag3,

            -- 3. Primary Physical Lags (7-Day)
            LAG(uo, 7) OVER w AS uo_lag7,
            LAG(vo, 7) OVER w AS vo_lag7,
            LAG(mlotst, 7) OVER w AS mlotst_lag7,
            LAG(zos, 7) OVER w AS zos_lag7,
            LAG(thetao, 7) OVER w AS thetao_lag7,
            LAG(so, 7) OVER w AS so_lag7,
            LAG(ssrd, 7) OVER w AS ssrd_lag7,
            
            -- 4. Spatial Derivative Lags (7-Day)
            LAG(vorticity, 7) OVER w AS vorticity_lag7,
            LAG(okubo_weiss, 7) OVER w AS okubo_weiss_lag7,
            LAG(grad_ssh, 7) OVER w AS grad_ssh_lag7

        FROM '{input_parquet}'
        WINDOW w AS (PARTITION BY lat, lon ORDER BY time)
    )
    
    WHERE chlor_a IS NOT NULL

) TO '{final_output}' (FORMAT PARQUET);
"""

print("Calculating all physical and spatial lags, then purging OCCCI NaN rows...")
con.execute(query)
print(f"Success! Ultimate ML dataset saved to {final_output}")