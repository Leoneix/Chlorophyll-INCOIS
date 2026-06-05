import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Load the dataset
occci = xr.open_dataset('occci_collocated.nc')

# 2. Slice specifically from the 4th to the 20th
subset = occci.sel(time=slice('1997-09-04', '1997-10-30'))

fig, ax = plt.subplots(figsize=(10, 6))

# Set consistent color scale bounds across the timeline
vmax = subset['chlor_a'].quantile(0.95).values 
vmin = subset['chlor_a'].min().values

def update(frame_index):
    ax.clear()
    day_data = subset['chlor_a'].isel(time=frame_index)
    date_str = str(day_data.time.values)[:10]
    
    # Plot spatial grid map
    day_data.plot(ax=ax, add_colorbar=False, vmin=vmin, vmax=vmax, cmap='viridis')
    ax.set_title(f'OCCCI Chlorophyll-a Timeline\nDate: {date_str}', fontsize=14)

# Create animation loop
ani = animation.FuncAnimation(fig, update, frames=len(subset.time), interval=600)

# Save as GIF (requires 'pillow' library: pip install pillow)
ani.save('chlor_a_4_to_20.gif', writer='pillow', fps=1.5)
plt.close()
print("Successfully generated 'chlor_a_4_to_20.gif'!")