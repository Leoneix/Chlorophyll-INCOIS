import xarray as xr
import glob
import os
import gc

from tqdm import tqdm
from dask.diagnostics import ProgressBar

base_dir = r"C:\Users\parth\Documents\GLORY\era5"

for year in tqdm(range(1993, 2027), desc="Years"):

    folder = os.path.join(base_dir, str(year))

    if not os.path.exists(folder):
        print(f"Skipping {year}: folder not found")
        continue

    files = sorted(glob.glob(os.path.join(folder, "*.nc")))

    if len(files) == 0:
        print(f"Skipping {year}: no files")
        continue

    out_file = os.path.join(base_dir, f"ssrd_{year}.nc")

    if os.path.exists(out_file):
        print(f"Skipping {year}: already merged")
        continue

    print(f"\nProcessing {year}")
    print(f"Found {len(files)} monthly files")

    try:

        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            chunks={"time": 31},
            parallel=True
        )

        delayed = ds.to_netcdf(
            out_file,
            compute=False
        )

        with ProgressBar():
            delayed.compute()

        print(f"✓ Saved {out_file}")

        # Clean up
        del delayed
        del ds
        gc.collect()

    except Exception as e:
        print(f"✗ Failed for {year}")
        print(e)

print("\nAll done.")