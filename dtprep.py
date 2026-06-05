import xarray as xr
import numpy as np

print("Loading master grid...")
ds = xr.open_dataset('master_merged_grid.nc', chunks={'time': 50})

print("Calculating metric conversions for spatial derivatives...")

# Define metric conversion factors
deg_to_m = 111320.0

cos_lat = np.cos(np.deg2rad(ds.lat))
dx_per_degree = deg_to_m * cos_lat
dy_per_degree = deg_to_m

# Calculate spatial derivatives in standard m/s per meter
du_dx = ds.uo.differentiate('lon') / dx_per_degree
du_dy = ds.uo.differentiate('lat') / dy_per_degree

dv_dx = ds.vo.differentiate('lon') / dx_per_degree
dv_dy = ds.vo.differentiate('lat') / dy_per_degree

dh_dx = ds.zos.differentiate('lon') / dx_per_degree
dh_dy = ds.zos.differentiate('lat') / dy_per_degree

print("Computing Vorticity, Okubo-Weiss, and SSH Gradient...")
# Vorticity
ds['vorticity'] = dv_dx - du_dy

# Okubo-Weiss Parameter
normal_strain = du_dx - dv_dy
shear_strain = dv_dx + du_dy
ds['okubo_weiss'] = (normal_strain**2) + (shear_strain**2) - (ds['vorticity']**2)

ds['grad_ssh'] = np.sqrt((dh_dx**2) + (dh_dy**2))


ds['month_sin'] = np.sin(2 * np.pi * ds['time'].dt.month / 12.0)
ds['month_cos'] = np.cos(2 * np.pi * ds['time'].dt.month / 12.0)

ds['day_sin'] = np.sin(2 * np.pi * ds['time'].dt.dayofyear / 365.25)
ds['day_cos'] = np.cos(2 * np.pi * ds['time'].dt.dayofyear / 365.25)

# Calculate exact boundaries of your grid to scale to [-1, 1]
lat_min, lat_max = ds.lat.min(), ds.lat.max()
lon_min, lon_max = ds.lon.min(), ds.lon.max()

ds['lat_scaled'] = 2 * ((ds.lat - lat_min) / (lat_max - lat_min)) - 1
ds['lon_scaled'] = 2 * ((ds.lon - lon_min) / (lon_max - lon_min)) - 1


vars_to_keep = [
    'chlor_a', 'uo', 'vo', 'mlotst', 'zos', 'thetao', 'so', 'ssrd', 
    'vorticity', 'okubo_weiss', 'grad_ssh',
    'month_sin', 'month_cos', 'day_sin', 'day_cos',
    'lat_scaled', 'lon_scaled'
]

ds_final = ds[vars_to_keep]

print("Squeezing out degenerate dimensions...")
ds_final = ds_final.squeeze().drop_vars(['depth', 'number'], errors='ignore')

print("Unifying Dask chunks...")
ds_final = ds_final.unify_chunks()

# ---> THE UPGRADE: Use Zarr instead of NetCDF <---
temp_zarr = 'phase2_math_checkpoint.zarr'
print(f"Executing mathematical task graph and saving to Zarr: {temp_zarr}")
print("Zarr allows parallel writing, bypassing the NetCDF traffic jam...")

# Note: You may need to run `pip install zarr` in your terminal if you don't have it!
ds_final.to_zarr(temp_zarr, mode='w')
print("Math checkpoint complete!")

# ---------------------------------------------------------
# Now, load the pre-calculated Zarr data back in
temp_zarr = 'phase2_math_checkpoint.zarr'

# ---------------------------------------------------------
# Load the pre-calculated Zarr data back in
print("Loading Zarr checkpoint...")
# ---> THE FIX: Force lat and lon to be unchunked (-1) so flattening requires zero shuffling <---
ds_calculated = xr.open_zarr(temp_zarr, chunks={'time': 50, 'lat': -1, 'lon': -1})

print("Unifying chunks across all variables...")
ds_calculated = ds_calculated.unify_chunks()

# Flatten into a dataframe
print("Flattening 3D Data to Tabular Dask DataFrame...")
df = ds_calculated.to_dask_dataframe()

print("Filtering out missing physical grid rows (landmasses)...")
# Drop rows where the PHYSICAL grid is missing (e.g., landmasses)
df = df.dropna(subset=['uo', 'zos'])

# Save the Dask DataFrame to Parquet
output_parquet = 'phase2_spatial_temporal_features.parquet'
print(f"Executing flatten operation and streaming to {output_parquet}...")

# We specify custom-sized row groups to keep the write smooth and efficient
df.to_parquet(output_parquet, engine='pyarrow', row_group_size=100_000)

print("Save complete! Data is ready for DuckDB.")