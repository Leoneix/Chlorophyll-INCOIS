import xarray as xr
import matplotlib.pyplot as plt
import glob
import os
import re

folder = r"C:\Users\parth\Documents\GLORY\occci_chlor_a"

all_files = sorted(
    glob.glob(os.path.join(folder, "*.nc"))
)

# Keep only yearly files like chlor_a_1997.nc
files = []

for f in all_files:

    name = os.path.basename(f)

    if re.match(r"chlor_a_\d{4}\.nc$", name):
        files.append(f)

print(f"Total yearly files: {len(files)}")

print("Opening 1997 file for mask...")

ds_mask = xr.open_dataset(
    os.path.join(folder, "chlor_a_1997.nc")
)

mask_1997 = (
    ds_mask["chlor_a"]
    .mean(dim="time")
    .notnull()
)

time_series = []

print("Processing yearly files...")

for file in files:

    year = os.path.basename(file).split("_")[-1].split(".")[0]

    print(f"Processing {year}...")

    ds = xr.open_dataset(
        file,
        chunks={"time": 30}
    )

    chlor_ocean = ds["chlor_a"].where(mask_1997)

    chlor_ocean = chlor_ocean.where(
        (chlor_ocean >= 0) &
        (chlor_ocean <= 20)
    )

    daily_mean = chlor_ocean.mean(
        dim=["lat", "lon"],
        skipna=True
    ).compute()

    time_series.append(daily_mean)

print("Combining time series...")

combined = xr.concat(
    time_series,
    dim="time"
)

combined = combined.sortby("time")

print("Plotting...")

plt.figure(figsize=(16,6))

plt.plot(
    combined["time"],
    combined,
    linewidth=0.5
)

plt.xlabel("Year")
plt.ylabel("Chlorophyll-a (mg m$^{-3}$)")

plt.title("Daily Mean Chlorophyll-a (1997–2025)")

plt.grid()

plt.tight_layout()

plt.show()