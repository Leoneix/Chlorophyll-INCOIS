# import xarray as xr
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import matplotlib.dates as mdates
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import scienceplots
# import matplotlib.ticker as ticker

# # Set the style
# plt.style.use(['science', 'no-latex', 'nature'])


# spatial_filename = "arabian_sea_spatial_picp_2025.nc"
# print(f"Loading datacube: {spatial_filename}...")
# ds_spatial = xr.open_dataset(spatial_filename)

# picp_spatial = ds_spatial['PICP']

# valid_picp = picp_spatial.values[~np.isnan(picp_spatial.values)]
# mean_spatial_picp = np.mean(valid_picp) * 100

# metrics_text = f"Mean Coverage: {mean_spatial_picp:.2f}%\n"

# lon_min, lon_max = float(ds_spatial.lon.min()), float(ds_spatial.lon.max())
# lat_min, lat_max = float(ds_spatial.lat.min()), float(ds_spatial.lat.max())


# print("Rendering Spatial Map...")
# fig1 = plt.figure(figsize=(8, 6))
# proj = ccrs.PlateCarree()

# def format_map(ax, title):
#     ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
#     ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=3)
#     ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    
#     gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5, zorder=4)
#     gl.top_labels = False
#     gl.right_labels = False
#     ax.set_title(title, fontsize=15, pad=10)

# ax1 = fig1.add_subplot(1, 1, 1, projection=proj)
# format_map(ax1, 'Spatial Prediction Interval Coverage Probability (2025)')

# # Use TwoSlopeNorm to center exactly at 0.80 (target coverage)
# norm = colors.Normalize(vmin=0.40, vmax=1.0)
# im1 = ax1.pcolormesh(ds_spatial.lon, ds_spatial.lat, picp_spatial, 
#                      cmap='jet', transform=proj, norm=norm, zorder=1)

# # Add the 'format' argument to plt.colorbar
# cbar = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, 
#                     format=ticker.PercentFormatter(xmax=1.0))

# # ALIGNMENT FIX: Anchor text to the axes, not the figure
# # x=0.5 centers it to the plot, y=-0.15 pushes it just below the x-axis
# ax1.text(0.5, -0.15, metrics_text, fontsize=14,
#          transform=ax1.transAxes,
#          horizontalalignment='center', 
#          verticalalignment='top', # Changed to 'top' so it hangs downward from the y-coordinate
#          zorder=10,
#          bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

# plt.tight_layout()
# plt.show()

# # Clean up memory
# ds_spatial.close()

# # # ---------------------------------------------------------
# # # 3. LOAD & DRAW TEMPORAL PICP (TIME-SERIES)
# # # ---------------------------------------------------------
# # temporal_filename = "arabian_sea_temporal_picp_2025.csv"
# # print(f"Loading timeseries: {temporal_filename}...")
# # df_temporal = pd.read_csv(temporal_filename)
# # df_temporal['time'] = pd.to_datetime(df_temporal['time'])

# # print("Rendering Temporal Line Plot...")
# # fig2, ax2 = plt.subplots(figsize=(10, 5))

# # # Plot the daily PICP and the target reference line
# # ax2.plot(df_temporal['time'], df_temporal['PICP'], color='teal', linewidth=1.5, label='Daily PICP')
# # ax2.axhline(y=0.80, color='red', linestyle='--', linewidth=1.5, label='Target Coverage (80%)')

# # # --- Overlay Temporal Metrics Text Box ---
# # mean_temp_picp = df_temporal['PICP'].mean() * 100
# # temp_metrics_text = f"Annual Mean: {mean_temp_picp:.1f}%\nTarget: 80.0%"

# # ax2.text(0.02, 0.94, temp_metrics_text, transform=ax2.transAxes, fontsize=12,
# #          verticalalignment='top', zorder=10,
# #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
# # # -----------------------------------------

# # ax2.set_title('Temporal Prediction Interval Coverage Probability (2025)', fontsize=15, pad=10)
# # ax2.set_ylabel('Coverage Fraction', fontsize=12)
# # ax2.set_xlabel('Date', fontsize=12)
# # ax2.set_ylim(0.40, 1.0)
# # ax2.legend(loc='lower right', frameon=True, edgecolor='gray')

# # # Format the x-axis dates
# # ax2.xaxis.set_major_locator(mdates.MonthLocator())
# # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
# # plt.xticks(rotation=45)

# # plt.tight_layout()
# # plt.show()

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scienceplots
import matplotlib.ticker as ticker

# Set the style
plt.style.use(['science', 'no-latex', 'nature'])

spatial_filename = "arabian_sea_spatial_picp_2025.nc"
print(f"Loading datacube: {spatial_filename}...")
ds_spatial = xr.open_dataset(spatial_filename)

picp_spatial = ds_spatial['PICP']

# --- NEW: Mask the data to only keep values >= 80% ---
picp_spatial_filtered = picp_spatial.where(picp_spatial >= 0.80)

# Calculate mean based on the ORIGINAL valid data to keep your metric accurate
valid_picp = picp_spatial.values[~np.isnan(picp_spatial.values)]
mean_spatial_picp = np.mean(valid_picp) * 100

metrics_text = f"Overall Mean Coverage: {mean_spatial_picp:.2f}%\n"

lon_min, lon_max = float(ds_spatial.lon.min()), float(ds_spatial.lon.max())
lat_min, lat_max = float(ds_spatial.lat.min()), float(ds_spatial.lat.max())

print("Rendering Spatial Map...")
fig1 = plt.figure(figsize=(8, 6))
proj = ccrs.PlateCarree()

def format_map(ax, title):
    # Add ocean background so the masked (<80%) areas are clearly distinguished from land
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue', zorder=1)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=3)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5, zorder=4)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title, fontsize=15, pad=10)

ax1 = fig1.add_subplot(1, 1, 1, projection=proj)
format_map(ax1, 'Spatial Prediction Interval Coverage Probability\n(Regions $\geq$ 80%)')

norm = colors.Normalize(vmin=0.80, vmax=1.0)

im1 = ax1.pcolormesh(ds_spatial.lon, ds_spatial.lat, picp_spatial_filtered, 
                     cmap='jet', transform=proj, norm=norm, zorder=2)

cbar = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, 
                    format=ticker.PercentFormatter(xmax=1.0))

ax1.text(0.5, -0.15, metrics_text, fontsize=14,
         transform=ax1.transAxes,
         horizontalalignment='center', 
         verticalalignment='top', 
         zorder=10,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.tight_layout()
plt.show()

# Clean up memory
ds_spatial.close()