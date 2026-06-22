import xarray as xr
import numpy as np


ds_glory = xr.open_dataset(
    r"C:\Users\parth\Documents\GLORY\cmems\cmems_1993_2026_merged.nc",
    chunks={'time': 50} 
)

ds_era5 = xr.open_dataset(
    r"C:\Users\parth\Documents\GLORY\era5\ssrd_merged_1993_2026.nc",
    chunks={'valid_time': 50} 
)

ds_occci = xr.open_dataset(
    r"C:\Users\parth\Documents\GLORY\occci_chlor_a\chlor_a_1997_2025_merged.nc",
    chunks={'time': 50} 
)

# Rename
ds_glory = ds_glory.rename({'latitude': 'lat', 'longitude': 'lon'})
ds_era5 = ds_era5.rename({'latitude': 'lat', 'longitude': 'lon', 'valid_time': 'time'})


ds_glory['lat'].load()
ds_glory['lon'].load()

print("ERA5 Regridding")
ds_era5_regridded = ds_era5.interp(
    lat=ds_glory.lat, 
    lon=ds_glory.lon, 
    method='nearest'
)

print("OCCCI Regridding")
ds_occci_regridded = ds_occci.interp(
    lat=ds_glory.lat, 
    lon=ds_glory.lon, 
    method='nearest'
)

ds_merged = xr.merge(
    [ds_glory, ds_era5_regridded, ds_occci_regridded], 
    join='left' 
)

print("Applying math transformations...")
ds_merged['ssrd'] = ds_merged['ssrd'] / 86400.0
ds_merged['mlotst'] = np.log10(ds_merged['mlotst'])
ds_merged['chlor_a'] = np.log10(ds_merged['chlor_a'])

output_file = 'merged_dataset.nc'
print(f"saved to {output_file}.")

ds_merged.to_netcdf(output_file)
print(xr.open_dataset(output_file))
