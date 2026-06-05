import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from numpy.polynomial import Polynomial

plt.style.use(['science', 'no-latex', 'nature', 'grid'])

features = ['uo', 'vo', 'mlotst', 'zos', 'thetao', 'so', 'ssrd']
target = 'chlor_a_log'

cols_to_keep = ['lat', 'lon', target] + features 

print("Loading and computing dataset...")
ddf = dd.read_parquet('ocean_ml_dataset_transformed.parquet')

mask_entire = (ddf['lat'] >= 5) & (ddf['lat'] <= 25) & (ddf['lon'] >= 50) & (ddf['lon'] <= 100)
df_entire = ddf[mask_entire][cols_to_keep].sample(frac=0.02).compute()

lon_divider = 77  
df_arabian = df_entire[df_entire['lon'] < lon_divider]
df_bob = df_entire[df_entire['lon'] >= lon_divider]

print("Data loaded! Generating plots...")

from scipy.stats import linregress

def plot_region(ax, df, feature, target, region_title):
    # Drop NaNs
    df_clean = df[[feature, target]].dropna()
    
    # Use .values to convert to raw NumPy arrays, avoiding Pandas index misalignment
    x_data = df_clean[feature].values
    y_data = df_clean[target].values
    
    # linregress calculates both the fit and the correlation (r_value) simultaneously
    slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
    
    # Create the x and y coordinates for the trendline
    x_line = np.array([x_data.min(), x_data.max()])
    y_line = slope * x_line + intercept
    
    # Scatter plot
    ax.scatter(
        x_data, y_data, 
        marker='o', s=25, alpha=0.3, color='#0077BB'
    )
    
    # Fit line
    ax.plot(
        x_line, y_line, 
        color='black', linewidth=2, linestyle='--', 
        label=f'r: {r_value:.2f}'
    )
    
    ax.set_title(region_title, fontsize=14)
    ax.set_xlabel(f'{feature}', fontsize=12)
    
    if region_title == 'Entire Region':
        ax.set_ylabel(f'Log Chlor-a ({target})', fontsize=12)
        
    ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=10)

for feature in features:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle(f'Impact of {feature} on Chlorophyll-a', fontsize=16, fontweight='bold', y=1.05)
    
    plot_region(axes[0], df_entire, feature, target, 'Entire Region')
    plot_region(axes[1], df_arabian, feature, target, 'Arabian Sea')
    plot_region(axes[2], df_bob, feature, target, 'Bay of Bengal')
    
    plt.tight_layout()
    plt.show()