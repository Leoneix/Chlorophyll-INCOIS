import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import numpy as np

# Set the style for high-quality scientific plots
plt.style.use(['science', 'no-latex', 'nature', 'grid'])

features = ['grad_ssh', 'vorticity', 'okubo_weiss']

print("Executing DuckDB query to load spatial derivatives...")

# We can safely use the trimmed file since we just need the base, current-day features
query = f"""
SELECT grad_ssh, vorticity, okubo_weiss 
FROM read_parquet('final_clean_dataset_trimmed.parquet')
WHERE lat >= 5 AND lat <= 25 
  AND lon >= 50 AND lon <= 100
USING SAMPLE 2 PERCENT
"""

df = duckdb.sql(query).df()
print(f"Data loaded successfully! Shape: {df.shape}")

print("Generating density distribution plots...")

# Create a 1x3 subplot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Density Distributions of Spatial Oceanographic Derivatives', 
             fontsize=16, fontweight='bold', y=1.05)

# Distinct colors for each feature to make the chart pop
colors = ['#0077BB', '#CC6677', '#228833'] 

for idx, feature in enumerate(features):
    ax = axes[idx]
    
    # 1. Drop missing values
    data = df[feature].dropna()
    
    # 2. Trim extreme 1% outliers specifically for visualization
    lower_bound = data.quantile(0.01)
    upper_bound = data.quantile(0.99)
    clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    # 3. Plot the Histogram with the smooth KDE overlay
    sns.histplot(clean_data, bins=50, kde=True, color=colors[idx], 
                 ax=ax, stat="density", alpha=0.4, edgecolor='none')
    
    # 4. Formatting
    ax.set_title(f"{feature.replace('_', ' ').title()}", fontsize=14)
    ax.set_xlabel(f'{feature}', fontsize=12)
    
    # Only add the Y-axis label to the far-left plot for a clean look
    if idx == 0:
        ax.set_ylabel('Density', fontsize=12)
    else:
        ax.set_ylabel('')
        
plt.tight_layout()
plt.show()