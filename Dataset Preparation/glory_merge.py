import xarray as xr
import glob
import os
from dask.diagnostics import ProgressBar

folder_path = r"C:\Users\parth\Documents\GLORY\cmems"

output_file = os.path.join(
    folder_path,
    "cmems_1993_2006_merged.nc"
)

files = sorted(glob.glob(os.path.join(folder_path, "*.nc")))

print(f"Total files found: {len(files)}")

ds = xr.open_mfdataset(
    files,
    combine="by_coords",
    parallel=False,         
    chunks={
        "time": 30,
        "latitude": 100,
        "longitude": 100
    },
    engine="netcdf4"
)

ds = ds.sortby("time")


time_index = ds.indexes["time"]
unique_index = ~time_index.duplicated()

ds = ds.isel(time=unique_index)

encoding = {}

for var in ds.data_vars:

    dims = ds[var].dims

    # Build safe chunk sizes dynamically
    chunk_sizes = []

    for d in dims:

        if d == "time":
            chunk_sizes.append(min(30, ds.sizes[d]))

        elif d in ["latitude", "lat"]:
            chunk_sizes.append(min(100, ds.sizes[d]))

        elif d in ["longitude", "lon"]:
            chunk_sizes.append(min(100, ds.sizes[d]))

        elif d == "depth":
            chunk_sizes.append(min(1, ds.sizes[d]))

        else:
            chunk_sizes.append(min(10, ds.sizes[d]))

    encoding[var] = {
        "zlib": True,
        "complevel": 3,
        "shuffle": True,
        "dtype": "float32",
        "_FillValue": -9999.0,
        "chunksizes": tuple(chunk_sizes)
    }


print("\nSaving")


with ProgressBar():
    ds.to_netcdf(
        output_file,
        mode="w",
        format="NETCDF4",
        engine="netcdf4",
        encoding=encoding,
        compute=True
    )

print("\nDONE!")
print(output_file)