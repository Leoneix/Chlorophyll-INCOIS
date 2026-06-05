import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import imageio.v2 as imageio
import os
import glob
import numpy as np
import matplotlib.colors as colors


files = sorted(
    glob.glob(
        r"C:\Users\parth\Documents\GLORY\occci_chlor_a\*.nc"
    )
)



print("Creating ocean mask from 1997...")

mask_ds = xr.open_dataset(files[0])

mask_chl = mask_ds["chlor_a"]

# Use first timestep of 1997
ocean_mask = mask_chl.notnull().any(dim="time")
print("Ocean mask created.")



frame_dir = r"C:\Users\parth\Documents\GLORY\chl_frames"

os.makedirs(frame_dir, exist_ok=True)


vmin = 0.01
vmax = 10

print("\nGenerating frames...")


for i, file in enumerate(files):

    year = os.path.basename(file).split("_")[-1].split(".")[0]

    print(f"\nProcessing {year}...")

    ds = xr.open_dataset(file)

    chl = ds["chlor_a"]

    # Remove invalid values
    chl = chl.where(chl > 0)
    chl = chl.where(chl < 100)

    # Compute yearly mean
    data = chl.mean(dim="time", skipna=True)

    # APPLY FIXED OCEAN MASK
    data = data.where(ocean_mask)

    # Skip bad years
    if np.all(np.isnan(data.values)):

        print(f"Skipping {year}")
        continue



    fig, ax = plt.subplots(figsize=(10, 5))

    mesh = ax.pcolormesh(
        data.lon,
        data.lat,
        data,
        cmap=cmocean.cm.algae,
        shading="auto",
        norm=colors.LogNorm(
            vmin=vmin,
            vmax=vmax
        )
    )

    plt.colorbar(
        mesh,
        ax=ax,
        label="Chlorophyll-a (mg m$^{-3}$)"
    )

    ax.set_title(
        f"Yearly Mean Chlorophyll-a\n{year}"
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Remove white borders
    ax.set_xlim(float(data.lon.min()), float(data.lon.max()))
    ax.set_ylim(float(data.lat.min()), float(data.lat.max()))

    # Save frame
    frame_path = os.path.join(
        frame_dir,
        f"frame_{i:03d}.png"
    )

    plt.savefig(
        frame_path,
        dpi=90,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved frame for {year}")


print("\nCreating GIF...")

frames = []

frame_files = sorted(
    glob.glob(
        os.path.join(frame_dir, "*.png")
    )
)

for frame in frame_files:

    frames.append(imageio.imread(frame))

gif_path = (
    r"C:\Users\parth\Documents\GLORY"
    r"\chlorophyll_yearly_animation.gif"
)

imageio.mimsave(
    gif_path,
    frames,
    fps=2
)

print("\nGIF saved!")
print(gif_path)