import duckdb
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.stats import linregress

# Set the style for high-quality scientific plots
plt.style.use(['science', 'no-latex', 'nature', 'grid'])

# Target variables
features = ['grad_ssh', 'vorticity', 'okubo_weiss']
target = 'chlor_a' 

print("Executing DuckDB query to load spatial derivatives...")

# Because we only dropped the lagged versions earlier, the base versions are safe in the trimmed file
query = f"""
SELECT lat, lon, chlor_a, grad_ssh, vorticity, okubo_weiss 
FROM read_parquet('final_clean_dataset_trimmed.parquet')
WHERE lat >= 5 AND lat <= 25 
  AND lon >= 50 AND lon <= 100
USING SAMPLE 2 PERCENT
"""

df_entire = duckdb.sql(query).df()
print(f"Data loaded successfully! Shape: {df_entire.shape}")

# Split the dataset into Arabian Sea and Bay of Bengal
lon_divider = 77  
df_arabian = df_entire[df_entire['lon'] < lon_divider]
df_bob = df_entire[df_entire['lon'] >= lon_divider]

print("Generating region-specific spatial plots...")

def plot_spatial_features(df, region_title):
    """
    Creates a 1x3 subplot figure for a specific region, showing 
    grad_ssh, vorticity, and okubo_weiss side-by-side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f'{region_title}: Current-Day Spatial Derivatives vs Chlorophyll-a', 
                 fontsize=16, fontweight='bold', y=1.05)
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
            
        # Drop NaNs strictly for the column we are plotting right now
        df_clean = df[[feature, target]].dropna()
        
        x_data = df_clean[feature].values
        y_data = df_clean[target].values
        
        # Safety catch for empty regions
        if len(x_data) == 0:
            ax.set_title(f'{feature.upper()} (No Data)')
            continue
            
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
        
        # Generate the trendline coordinates
        x_line = np.array([x_data.min(), x_data.max()])
        y_line = slope * x_line + intercept
        
        # Plot Scatter & Trendline
        ax.scatter(x_data, y_data, marker='o', s=25, alpha=0.3, color='#CC6677')
        ax.plot(x_line, y_line, color='black', linewidth=2, linestyle='--', label=f'r: {r_value:.2f}')
        
        # Formatting
        ax.set_title(f"{feature.replace('_', ' ').title()}", fontsize=14)
        ax.set_xlabel(f'{feature}', fontsize=12)
        
        # Only add the Y-axis label to the far-left plot
        if idx == 0:
            ax.set_ylabel(f'Log Chlor-a ({target})', fontsize=12)
            
        ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=10)
        
    plt.tight_layout()
    plt.show()

# Execute the plotting pipeline for each specific region
plot_spatial_features(df_entire, 'Entire Region')
plot_spatial_features(df_arabian, 'Arabian Sea')
plot_spatial_features(df_bob, 'Bay of Bengal')