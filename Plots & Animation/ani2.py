import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import imageio
import os
import glob
import numpy as np

# Load dataset
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

# Select SST
if "depth" in ds.zos.dims:
    sst = ds.zos.isel(depth=0)
else:
    sst = ds.zos

# Convert to MONTHLY means
sst = sst.resample(time="1MS").mean()

# Strong downsampling for stability
# sst = sst.coarsen(
#     latitude=4,
#     longitude=4,
#     boundary="trim"
# ).mean()

# Color limits
vmin = float(sst.min())
vmax = float(sst.max())

# Temporary frame folder
frame_dir = r"C:\Users\parth\Documents\GLORY\frames1"

os.makedirs(frame_dir, exist_ok=True)

print("Generating frames...")

# Generate PNG frames
for i in range(len(sst.time)):

    fig, ax = plt.subplots(figsize=(10,5))

    data = sst.isel(time=i)

    mesh = ax.pcolormesh(
        sst.longitude,
        sst.latitude,
        data,
        cmap=cmocean.cm.thermal,
        shading="auto",
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(mesh, ax=ax, label="height (m)")

    time_str = str(data.time.values)[:10]

    ax.set_title(f"Sea surface height\n{time_str}")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    frame_path = os.path.join(frame_dir, f"frame_{i:03d}.png")

    plt.savefig(frame_path, dpi=100)

    plt.close()

    print(f"Saved frame {i+1}/{len(sst.time)}")

print("Frames complete!")

# Create GIF
print("Creating GIF...")

frames = []

for i in range(len(sst.time)):

    frame_path = os.path.join(frame_dir, f"frame_{i:03d}.png")

    frames.append(imageio.imread(frame_path))

gif_path = r"C:\Users\parth\Documents\GLORY\sst_animation.gif"

imageio.mimsave(
    gif_path,
    frames,
    fps=4
)

print("GIF SAVED!")