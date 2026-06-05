import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =====================================================
# Open correlation file
# =====================================================

corr = xr.open_dataarray(
    r"C:\Users\parth\Documents\GLORY\mld_thetao_correlation.nc"
)

# =====================================================
# Open original chlorophyll file to recover coordinates
# =====================================================

chl = xr.open_dataset(
    r"C:\Users\parth\Documents\GLORY\occci_chlor_a\chlor_a_1997_2025_merged.nc"
)

lat = chl.lat.values
lon = chl.lon.values

# Same coarsening used in correlation calculation
lat_coarse = (
    xr.DataArray(lat, dims="lat")
    .coarsen(lat=4, boundary="trim")
    .mean()
    .values
)

lon_coarse = (
    xr.DataArray(lon, dims="lon")
    .coarsen(lon=4, boundary="trim")
    .mean()
    .values
)

# Attach coordinates
corr = corr.assign_coords(
    lat=("lat", lat_coarse),
    lon=("lon", lon_coarse)
)

# =====================================================
# Plot
# =====================================================

fig = plt.figure(figsize=(12,8))

ax = plt.axes(projection=ccrs.PlateCarree())

pcm = ax.pcolormesh(
    corr.lon,
    corr.lat,
    corr,
    cmap="RdBu_r",
    vmin=-0.5,
    vmax=0.5,
    shading="auto",
    transform=ccrs.PlateCarree()
)

# Land only
ax.add_feature(
    cfeature.LAND,
    facecolor="lightgray",
    edgecolor="none"
)

# Coastlines only
ax.coastlines(
    resolution="10m",
    linewidth=0.8,
    color="black"
)

ax.set_extent(
    [65, 100, 5, 25],
    crs=ccrs.PlateCarree()
)

gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.5,
    alpha=0.5,
    linestyle="--"
)

gl.top_labels = False
gl.right_labels = False

plt.colorbar(
    pcm,
    ax=ax,
    label="Correlation"
)

plt.title("MLD vs Chlorophyll Correlation")

plt.show()