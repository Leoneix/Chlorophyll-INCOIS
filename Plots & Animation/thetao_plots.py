import xarray as xr
import glob
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import cmocean

# Open combined dataset

files = sorted(glob.glob(r"C:\Users\parth\Documents\GLORY\cmems\*.nc"))

ds = xr.open_mfdataset(
    files,
    combine="nested",
    concat_dim="time",
    parallel=True
)

# Sort time
ds = ds.sortby("time")

# Remove duplicate timestamps
_, index = np.unique(ds.time, return_index=True)

ds = ds.isel(time=index)

# Compute mean SST over time
mean_sst = ds.thetao.mean(dim="time")

# Plot
plt.figure(figsize=(12,6))

mean_sst.plot(
    cmap=cmocean.cm.thermal,
    robust=True
)

plt.title("Mean Sea Surface Temperature (1993–2006)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.show()