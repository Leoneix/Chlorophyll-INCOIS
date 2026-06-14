import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scienceplots
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Set the style
plt.style.use(['science', 'no-latex', 'nature'])

filename = "arabian_sea_chlorophyll_quantiles_log_2025.nc"
print(f"Loading datacube: {filename}...")
ds = xr.open_dataset(filename)

# --- NEW: Calculate RMSE and Correlation for the ENTIRE year/dataset ---
print("Calculating overall metrics for the entire year...")
actual_all = ds['chlor_a_actual'].values.flatten()
pred_all = ds['chlor_a_q50'].values.flatten()

valid_mask = ~np.isnan(actual_all) & ~np.isnan(pred_all)
act_valid = actual_all[valid_mask]
pred_valid = pred_all[valid_mask]

yearly_rmse = np.sqrt(mean_squared_error(act_valid, pred_valid))
yearly_corr, _ = pearsonr(act_valid, pred_valid)

# Create the text string for the plot (formatted horizontally)
metrics_text = f"Yearly RMSE (log10): {yearly_rmse:.3f}   |   Yearly Corr ($r$): {yearly_corr:.3f}"
# -----------------------------------------------------------------------

target_date = '2025-03-12'
print(f"Slicing data for {target_date}...")
day_data = ds.sel(time=target_date)

actual = day_data['chlor_a_actual']
pred_q50 = day_data['chlor_a_q50']
error = day_data['prediction_error']

vmin = float(np.nanmin([actual.min(), pred_q50.min()]))
vmax = float(np.nanmax([actual.max(), pred_q50.max()]))

lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())
lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())

print("Rendering maps...")
fig = plt.figure(figsize=(18, 6))
proj = ccrs.PlateCarree()

# Reusable function to format the geography of each panel
def format_map(ax, title):
    # Add land and coastlines (NaN values in your NetCDF will just show as this land color)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=3)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5, zorder=4)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title, fontsize=15, pad=10)

# --- MAP 1: Actual Chlorophyll ---
ax1 = fig.add_subplot(1, 3, 1, projection=proj)
format_map(ax1, f'Actual Chlorophyll-a (log10)\n{target_date}')
im1 = ax1.pcolormesh(day_data.lon, day_data.lat, actual, 
                     cmap='jet', transform=proj, vmin=vmin, vmax=vmax, zorder=1)
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='log10(mg/m³)')

# --- MAP 2: Predicted Chlorophyll (Q50) ---
ax2 = fig.add_subplot(1, 3, 2, projection=proj)
format_map(ax2, f'Predicted Chlorophyll-a (Q50)\n{target_date}')
im2 = ax2.pcolormesh(day_data.lon, day_data.lat, pred_q50, 
                     cmap='jet', transform=proj, vmin=vmin, vmax=vmax, zorder=1)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='log10(mg/m³)')

# --- MAP 3: Spatial Error (Difference) ---
ax3 = fig.add_subplot(1, 3, 3, projection=proj)
format_map(ax3, 'Spatial Error\n(Predicted - Actual)')

# Center the error map exactly at 0 so white means "perfect prediction"
error_limit = float(np.nanmax(np.abs(error))) * 0.8  # Cap at 80% to highlight gradients
im3 = ax3.pcolormesh(day_data.lon, day_data.lat, error, 
                     cmap='jet', transform=proj, vmin=-error_limit, vmax=error_limit, zorder=1)
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Error (log10 units)')

# Adjust layout to leave room at the bottom for the text
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Add the text box to the figure itself, centered at the bottom
fig.text(0.5, 0.02, metrics_text, fontsize=14,
         horizontalalignment='center', verticalalignment='bottom', zorder=10,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.show()

# Clean up memory
ds.close()