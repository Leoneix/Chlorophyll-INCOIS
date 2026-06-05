import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt


ddf = dd.read_parquet('ocean_ml_dataset_transformed.parquet')
features = ['chlor_a_log', 'thetao', 'so', 'uo', 'vo', 'zos', 'mlotst', 'ssrd']


corr_matrix = ddf[features].corr().compute()

plt.figure(figsize=(10, 8))


sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f", 
    vmin=-1, 
    vmax=1, 
    linewidths=0.5
)

plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()