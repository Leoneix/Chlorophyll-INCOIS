import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import imageio.v2 as imageio
import os
import glob
import numpy as np



files = sorted(
    glob.glob(
        r"C:\Users\parth\Documents\GLORY\occci_chlor_a\*.nc"
    )
)



print("Creating ocean mask from 1997...")

mask_ds = xr.open_dataset(files[0])

mask_chl = mask_ds["chlor_a"]

# Ocean pixels valid at least once during 1997
ocean_mask = mask_chl.notnull().any(dim="time")

print("Ocean mask created.")



frame_dir = (
    r"C:\Users\parth\Documents\GLORY"
    r"\missing_frames"
)

os.makedirs(frame_dir, exist_ok=True)



vmin = 0
vmax = 100

print("\nGenerating frames...")



for i, file in enumerate(files):

    year = os.path.basename(file).split("_")[-1].split(".")[0]

    print(f"\nProcessing {year}...")

    ds = xr.open_dataset(file)

    chl = ds["chlor_a"]



    missing_fraction = (
        chl.isnull().sum(dim="time")
        / chl.sizes["time"]
    ) * 100

    # Apply ocean mask
    missing_fraction = missing_fraction.where(ocean_mask)

    # Skip empty frames
    if np.all(np.isnan(missing_fraction.values)):

        print(f"Skipping {year}")
        continue



    fig, ax = plt.subplots(figsize=(10, 5))

    mesh = ax.pcolormesh(
        missing_fraction.lon,
        missing_fraction.lat,
        missing_fraction,
        cmap=cmocean.cm.ice_r,
        shading="auto",
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(
        mesh,
        ax=ax,
        label="Missing Data (%)"
    )

    ax.set_title(
        f"Yearly Missing Chlorophyll-a Data\n{year}"
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.set_xlim(
        float(missing_fraction.lon.min()),
        float(missing_fraction.lon.max())
    )

    ax.set_ylim(
        float(missing_fraction.lat.min()),
        float(missing_fraction.lat.max())
    )



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
    r"\chlorophyll_missing_animation.gif"
)

imageio.mimsave(
    gif_path,
    frames,
    fps=2
)

print("\nGIF saved!")
print(gif_path)