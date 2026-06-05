import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

# Keep your high-quality scientific styling
plt.style.use(['science', 'no-latex', 'nature', 'grid'])

# All 7 un-lagged features + the target variable
features = ['uo', 'vo', 'mlotst', 'zos', 'thetao', 'so', 'ssrd', 'chlor_a_log']

print("1. Querying un-lagged baseline data via DuckDB...")

# Convert to SQL string
sql_columns = ", ".join(features)

# Load a safe 2% sample of the baseline variables
query = f"""
SELECT {sql_columns} 
FROM read_parquet('ocean_ml_dataset_transformed.parquet')
USING SAMPLE 2 PERCENT
"""

df_base = duckdb.sql(query).df()
print(f"Data loaded! Shape: {df_base.shape}")

print("2. Generating density distribution plots...")

# Create a 2x4 grid of subplots for our 8 variables
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle('Density Distributions of Baseline (Un-lagged) Ocean Variables', 
             fontsize=18, fontweight='bold', y=1.02)

# Flatten the axes array to make it easy to loop through
axes = axes.flatten()

for idx, feature in enumerate(features):
    ax = axes[idx]
    
    # Drop NaNs just for this specific feature to ensure a clean density curve
    clean_data = df_base[feature].dropna()
    
    # Seaborn's KDE plot generates a smooth probability density curve
    sns.kdeplot(
        data=clean_data, 
        ax=ax, 
        fill=True, 
        color='#0077BB', 
        alpha=0.3, 
        linewidth=2
    )
    
    # Formatting
    ax.set_title(f'Distribution of {feature.upper()}', fontsize=14)
    ax.set_xlabel(f'{feature}', fontsize=12)
    
    # Only add 'Density' to the y-axis of the far-left plots to reduce clutter
    if idx % 4 == 0:
        ax.set_ylabel('Density', fontsize=12)
    else:
        ax.set_ylabel('')
        
plt.tight_layout()
plt.show()