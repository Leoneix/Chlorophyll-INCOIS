import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scienceplots
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Set the style
plt.style.use(['science', 'no-latex', 'nature'])

filename = "arabian_sea_chlorophyll_quantiles_2025.nc"
print(f"Loading {filename}...")
ds = xr.open_dataset(filename)

# Calculate RMSE and Correlation for the ENTIRE year/dataset
print("Calculating overall metrics for the entire year...")
actual_all = ds['chlor_a_actual'].values.flatten()
pred_all = ds['chlor_a_q50'].values.flatten()

valid_mask = ~np.isnan(actual_all) & ~np.isnan(pred_all)
act_valid = actual_all[valid_mask]
pred_valid = pred_all[valid_mask]

yearly_rmse = np.sqrt(mean_squared_error(act_valid, pred_valid))
yearly_corr, _ = pearsonr(act_valid, pred_valid)

# Create the text string for the plot (now formatted horizontally)
metrics_text = f"Yearly RMSE: {yearly_rmse:.3f}   |   Yearly Corr (r): {yearly_corr:.3f}"

target_date = '2025-03-05'
print(f"Rendering maps for {target_date}...")
day_data = ds.sel(time=target_date)

actual = day_data['chlor_a_actual']
pred_q50 = day_data['chlor_a_q50']
error = day_data['prediction_error']

lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())
lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())

fig = plt.figure(figsize=(18, 6))
proj = ccrs.PlateCarree()

def format_map(ax, title):
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=3)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5, zorder=4)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title, fontsize=15, pad=10)

ax1 = fig.add_subplot(1, 3, 1, projection=proj)
format_map(ax1, f'Actual Chlorophyll-a\n{target_date}')
im1 = ax1.pcolormesh(day_data.lon, day_data.lat, actual, 
                     cmap='jet', transform=proj,vmin=0,vmax=1.5, 
                      zorder=1)
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='mg/m³')

ax2 = fig.add_subplot(1, 3, 2, projection=proj)
format_map(ax2, f'Predicted Chlorophyll-a (Q50)\n{target_date}')
im2 = ax2.pcolormesh(day_data.lon, day_data.lat, pred_q50, 
                     cmap='jet', transform=proj,vmin=0,vmax=1.5, 
                     zorder=1)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='mg/m³')

ax3 = fig.add_subplot(1, 3, 3, projection=proj)
format_map(ax3, 'Spatial Error\n(Predicted - Actual)')

error_limit = float(np.nanpercentile(np.abs(error), 95))
im3 = ax3.pcolormesh(day_data.lon, day_data.lat, error, 
                     cmap='jet', transform=proj, 
                     vmin=-error_limit, vmax=error_limit, zorder=1)
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Error (mg/m³)')

# Adjust layout to leave room at the bottom for the text
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Add the text box to the figure itself, centered at the bottom
fig.text(0.5, 0.02, metrics_text, fontsize=14,
         horizontalalignment='center', verticalalignment='bottom', zorder=10,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.show()

# Clean up memory
ds.close()