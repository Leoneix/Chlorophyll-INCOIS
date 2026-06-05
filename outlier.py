import duckdb

# Connect to DuckDB
con = duckdb.connect()

input_file = 'final_clean_dataset_trimmed.parquet'
output_file = 'final_ml_dataset_3sigma.parquet'

print(f"Applying 3-Sigma filter and writing to {output_file}...")

# DuckDB will calculate the mean and sigma, then instantly filter the rows
query = f"""
    COPY (
        WITH stats AS (
            SELECT 
                AVG(chlor_a) AS mu, 
                STDDEV_POP(chlor_a) AS sigma 
            FROM '{input_file}'
        )
        SELECT data.*
        FROM '{input_file}' AS data, stats
        WHERE data.chlor_a >= (stats.mu - 3 * stats.sigma)
          AND data.chlor_a <= (stats.mu + 3 * stats.sigma)
    ) TO '{output_file}' (FORMAT PARQUET);
"""

con.execute(query)
print("Success! Outliers have been successfully dropped.")

# Let's print a quick before-and-after row count to verify the exact number dropped
verify_query = f"""
    SELECT 
        (SELECT COUNT(*) FROM '{input_file}') AS original_rows,
        (SELECT COUNT(*) FROM '{output_file}') AS final_rows
"""
counts = con.execute(verify_query).df()

original = counts['original_rows'][0]
final = counts['final_rows'][0]
dropped = original - final

print("-" * 50)
print("3-SIGMA PURGE RESULTS")
print("-" * 50)
print(f"Starting Rows: {original:,}")
print(f"Ending Rows:   {final:,}")
print(f"Rows Dropped:  {dropped:,}")
print("-" * 50)