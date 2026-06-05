import xarray as xr
from tqdm.dask import TqdmCallback


chunk_dict_occci = {'time': 10, 'lat': 500, 'lon': 500}
chunk_dict_glory = {'time': 10, 'latitude': 500, 'longitude': 500}
chunk_dict_era5  = {'valid_time': 10, 'latitude': 500, 'longitude': 500}

ds_occci = xr.open_dataset('chlor_a_1997_2025_merged.nc', chunks=chunk_dict_occci)
ds_glory = xr.open_dataset('cmems_1993_2026_merged.nc', chunks=chunk_dict_glory)
ds_era5 = xr.open_dataset('ssrd_merged_1993_2026.nc', chunks=chunk_dict_era5)

ds_glory = ds_glory.rename({
    'latitude': 'lat', 
    'longitude': 'lon'
})

ds_era5 = ds_era5.rename({
    'latitude': 'lat', 
    'longitude': 'lon',
    'valid_time': 'time'
})


# Interpolate OCCCI to GLORY grid
occci_interp = ds_occci.interp(
    lat=ds_glory.lat, 
    lon=ds_glory.lon, 
    method='nearest'
)

# Interpolate ERA5 to GLORY grid
era5_interp = ds_era5.interp(
    lat=ds_glory.lat, 
    lon=ds_glory.lon, 
    method='nearest'
)

occci_mask = occci_interp['chlor_a'].notnull()

glory_final = ds_glory.where(occci_mask)
era5_final = era5_interp.where(occci_mask)

with TqdmCallback(desc="Computing GLORY"):
    glory_final.to_netcdf('glory_collocated_masked.nc')

with TqdmCallback(desc="Computing ERA5"):
    era5_final.to_netcdf('era5_collocated_masked.nc')

with TqdmCallback(desc="Computing OCCCI"):
    occci_interp.to_netcdf('occci_collocated.nc')

print("All tasks completed successfully!")