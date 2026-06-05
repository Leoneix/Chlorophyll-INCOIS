import xarray as xr
import pandas as pd
import glob
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# Folder
folder = r"C:\Users\parth\Documents\GLORY\occci_chlor_a"

files = sorted(glob.glob(os.path.join(folder, "*.nc")))

results = []

for file in files:

    year = os.path.basename(file).split("_")[-1].split(".")[0]

    print(f"Processing {year}...")

    ds = xr.open_dataset(file)

    chl = ds["chlor_a"]

    # Ocean mask:
    # pixel valid at least once during year
    ocean_mask = chl.notnull().any(dim="time")

    # Expand mask to all timesteps
    ocean_mask_3d = ocean_mask.broadcast_like(chl)

    # Count total valid ocean positions
    total_ocean_points = ocean_mask_3d.sum().item()

    # Missing ONLY over ocean
    missing_ocean_points = (
        chl.isnull() & ocean_mask_3d
    ).sum().item()

    # Percentages
    missing_percent = (
        missing_ocean_points / total_ocean_points
    ) * 100

    valid_percent = 100 - missing_percent

    results.append({
        "Year": int(year),
        "Missing %": missing_percent,
        "Valid %": valid_percent
    })

# DataFrame
df = pd.DataFrame(results)

# Sort
df = df.sort_values("Year")

print("\nCorrected Ocean-only Gaps:\n")
print(df)

# Save CSV
csv_path = os.path.join(
    folder,
    "corrected_ocean_gaps.csv"
)

df.to_csv(csv_path, index=False)

print(f"\nSaved CSV:\n{csv_path}")

# Plot
plt.figure(figsize=(12,5))

plt.plot(
    df["Year"],
    df["Missing %"],
    marker="o"
)

plt.xlabel("Year")
plt.ylabel("Missing Data (%)")
plt.title("Corrected Ocean-only Chlorophyll Gaps")

plt.grid(True)

plt.show()