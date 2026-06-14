import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import duckdb
import joblib
import scienceplots
import os
from sklearn.metrics import mean_squared_error 
from scipy.stats import pearsonr                

# Set style
plt.style.use(['science', 'no-latex', 'nature', 'grid'])

# Configuration
file_path = r'C:\Users\parth\Documents\Chlorophyll\final_ml_dataset_cleanednewwo3sigma.parquet'
target_col = 'chlor_a_log10' 
drop_cols = ['time', 'lat', 'lon', target_col]
tol = 0.1 

# Define the dictionary of target points (Name: (Lat, Lon))
points = {
    # 'A': (8.440403, 76.241972),
    # 'B': (9.355717, 76.051908),
    # 'C': (10.224867, 75.772600),
    # 'D': (11.163408, 75.303564),
    # 'E': (11.791756, 74.788669),
    'F': (13.382222, 74.060553),
    'G': (15.086300, 73.370039),
    'H': (18.765467, 72.218008),
    'I': (23.179939, 66.945581),
    'J': (19.473783, 68.761353),
    'K': (14.952203, 70.632994),
    'L': (10.383008, 72.380481),
    'M': (6.487397, 74.276542),
    'N': (8.710411, 70.776722),
    'O': (12.699800, 68.624753),
    'P': (16.953142, 67.024764),
    'Q': (13.584125, 66.037831),
    'R': (9.952758, 67.409447),
    'S': (6.051781, 69.297297),
    'T': (7.047317, 65.737047)
}

print("Connecting to DuckDB...")
con = duckdb.connect()

print("Loading Q10, Q50, and Q90 models...")
model_q10 = joblib.load(r"C:\Users\parth\Documents\models better rmse diff 2\lgbm_q10_final_arabian.pkl")
model_q50 = joblib.load(r"C:\Users\parth\Documents\models better rmse diff 2\lgbm_q50_final_arabian.pkl")
model_q90 = joblib.load(r"C:\Users\parth\Documents\models better rmse diff 2\lgbm_q90_final_arabian.pkl")

# Create an output directory for the plots if it doesn't exist
output_dir = 'output_plots'
os.makedirs(output_dir, exist_ok=True)

# Loop through each point in the list
for point_name, (target_lat, target_lon) in points.items():
    print(f"\n" + "="*50)
    print(f"Processing POINT {point_name} (Lat: {target_lat}, Lon: {target_lon})...")
    print("="*50)
    
    query_test = f"""
        SELECT * FROM '{file_path}' 
        WHERE EXTRACT(YEAR FROM time) = 2025
        AND lat BETWEEN {target_lat - tol} AND {target_lat + tol}
        AND lon BETWEEN {target_lon - tol} AND {target_lon + tol}
    """
    df_test = con.execute(query_test).df()

    if df_test.empty:
        print(f"WARNING: No data found for Point {point_name}. Skipping to next point.")
        continue

    print(f"Found {len(df_test)} records for Point {point_name} in 2025.")

    test_dates = pd.to_datetime(df_test['time'])
    actual_lat = df_test['lat'].mean()
    actual_lon = df_test['lon'].mean()

    X_test = df_test.drop(columns=drop_cols)
    y_test_actual = df_test[target_col].values

    print("Running predictions...")
    pred_q10 = model_q10.predict(X_test)
    pred_q50 = model_q50.predict(X_test)
    pred_q90 = model_q90.predict(X_test)

    df_results = pd.DataFrame({
        'date': test_dates.dt.date, 
        'actual': y_test_actual,
        'q10': pred_q10,
        'q50': pred_q50,
        'q90': pred_q90
    })

    daily_ts = df_results.groupby('date').mean().reset_index()

    rmse_val = np.sqrt(mean_squared_error(daily_ts['actual'], daily_ts['q50']))
    
    # Handle edge case where standard deviation might be 0 to avoid Pearson correlation nan errors
    if daily_ts['actual'].nunique() > 1 and daily_ts['q50'].nunique() > 1:
        corr_val, _ = pearsonr(daily_ts['actual'], daily_ts['q50'])
    else:
        corr_val = np.nan

    print(f"Metrics for Point {point_name} -> RMSE: {rmse_val:.4f} | Correlation: {corr_val:.4f}")

    # ---------------------------------------------------------
    # DRAW THE TIME SERIES PLOT
    # ---------------------------------------------------------
    print(f"Drawing Plot for Point {point_name}...")
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = pd.to_datetime(daily_ts['date'])

    # Fill the 80% Confidence Interval
    ax.fill_between(dates, daily_ts['q10'], daily_ts['q90'], 
                    color='#0077BB', alpha=0.2, label='80% Confidence Interval (Q10-Q90)')

    # Plot the Quantile Lines
    ax.plot(dates, daily_ts['q90'], linestyle='--', color='#0077BB', alpha=0.6, linewidth=1.2, label='Q90 (Upper Bound)')
    ax.plot(dates, daily_ts['q10'], linestyle='--', color='#0077BB', alpha=0.6, linewidth=1.2, label='Q10 (Lower Bound)')
    ax.plot(dates, daily_ts['q50'], linestyle='-', color='black', linewidth=2, label='Q50 (Median Prediction)')

    # Scatter the Actual Data Points on top
    ax.scatter(dates, daily_ts['actual'], color='#CC3311', s=25, zorder=5, label='Actual Local Chlorophyll (Log10)')

    # Formatting the X-Axis for Dates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)

    # Dynamic Title containing Point letter name
    ax.set_title(f'2025 Time Series (Lat: {actual_lat:.3f}, Lon: {actual_lon:.3f}): Chlorophyll Predictions vs Actuals (POINT {point_name})', fontsize=16)
    ax.set_ylabel('Chlorophyll-a (log10)', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)

    # Add metrics text box
    metrics_text = f"RMSE: {rmse_val:.4f}\nCorr ($r$): {corr_val:.4f}"
    ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    
    # Save image automatically to avoid losing plots between loop iterations
    fig_path = os.path.join(output_dir, f'Point_{point_name}_2025_TimeSeries.png')
    plt.savefig(fig_path, dpi=300)
    print(f"Saved plot to {fig_path}")
    
    plt.show()
    plt.close(fig) # Close figure to free up system memory

print("\nAll points processed successfully!")


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import duckdb
# import joblib
# import scienceplots
# from sklearn.metrics import mean_squared_error 
# from scipy.stats import pearsonr                

# # Set style
# plt.style.use(['science', 'no-latex', 'nature', 'grid'])

# file_path = r'C:\Users\parth\Documents\Chlorophyll\final_ml_dataset_cleanednewwo3sigma.parquet'
# target_col = 'chlor_a_log10' 
# drop_cols = ['time', 'lat', 'lon', target_col]

# # Define the regional bounding box boundaries
# lat_min, lat_max = 8.0, 12.0
# lon_min, lon_max = 74.0, 78.0

# print(f"Connecting to DuckDB to extract regional data (Lat: {lat_min}-{lat_max}N, Lon: {lon_min}-{lon_max}E)...")
# con = duckdb.connect()

# # Querying all data points falling within the bounding box matrix for 2025
# query_test = f"""
#     SELECT * FROM '{file_path}' 
#     WHERE EXTRACT(YEAR FROM time) = 2025
#     AND lat BETWEEN {lat_min} AND {lat_max}
#     AND lon BETWEEN {lon_min} AND {lon_max}
# """
# df_test = con.execute(query_test).df()

# if df_test.empty:
#     raise ValueError("No data found within this regional bounding box for 2025. Please verify your coordinate boundaries.")

# print(f"Found {len(df_test)} total grid records within the specified region for 2025.")

# test_dates = pd.to_datetime(df_test['time'])

# # Features and target split
# X_test = df_test.drop(columns=drop_cols)
# y_test_actual = df_test[target_col].values

# print("Loading Q10, Q50, and Q90 models...")
# model_q10 = joblib.load(r"C:\Users\parth\Documents\models better rmse diff 2\lgbm_q10_final_arabian.pkl")
# model_q50 = joblib.load(r"C:\Users\parth\Documents\models better rmse diff 2\lgbm_q50_final_arabian.pkl")
# model_q90 = joblib.load(r"C:\Users\parth\Documents\models better rmse diff 2\lgbm_q90_final_arabian.pkl")

# print("Running predictions across all regional grid points...")
# pred_q10 = model_q10.predict(X_test)
# pred_q50 = model_q50.predict(X_test)
# pred_q90 = model_q90.predict(X_test)

# # Build results dataframe containing dates and spatial coordinates
# df_results = pd.DataFrame({
#     'date': test_dates.dt.date, 
#     'actual': y_test_actual,
#     'q10': pred_q10,
#     'q50': pred_q50,
#     'q90': pred_q90
# })

# # Grouping by 'date' to get the daily spatial average across the entire region
# daily_ts = df_results.groupby('date').mean().reset_index()

# # Calculate performance metrics on the regional spatial averages
# rmse_val = np.sqrt(mean_squared_error(daily_ts['actual'], daily_ts['q50']))
# corr_val, _ = pearsonr(daily_ts['actual'], daily_ts['q50'])

# print(f"Regional Time Series Metrics -> RMSE: {rmse_val:.4f} | Correlation: {corr_val:.4f}")


# print("Drawing Regional Plot...")
# fig, ax = plt.subplots(figsize=(14, 6))

# dates = pd.to_datetime(daily_ts['date'])

# # Fill the 80% Confidence Interval for the regional average boundary
# ax.fill_between(dates, daily_ts['q10'], daily_ts['q90'], 
#                 color='#0077BB', alpha=0.2, label='Regional 80% CI (Q10-Q90 Average)')

# # Plot the Quantile Lines
# ax.plot(dates, daily_ts['q90'], linestyle='--', color='#0077BB', alpha=0.6, linewidth=1.2, label='Mean Q90 (Upper Bound)')
# ax.plot(dates, daily_ts['q10'], linestyle='--', color='#0077BB', alpha=0.6, linewidth=1.2, label='Mean Q10 (Lower Bound)')
# ax.plot(dates, daily_ts['q50'], linestyle='-', color='black', linewidth=2, label='Mean Q50 (Median Prediction)')

# # Scatter the Actual Spatial Average Data Points on top
# ax.scatter(dates, daily_ts['actual'], color='#CC3311', s=25, zorder=5, label='Actual Regional Avg Chlorophyll (Log10)')

# # Formatting the X-Axis for Dates
# ax.xaxis.set_major_locator(mdates.MonthLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
# plt.xticks(rotation=45)

# # Dynamic title reflecting the exact bounding box requested
# ax.set_title('2025 Regional Average Time Series (74°E-78°E, 8°N-12°N)\nChlorophyll Predictions vs Actuals', fontsize=16)
# ax.set_ylabel('Chlorophyll-a (log10)', fontsize=14)
# ax.set_xlabel('Date', fontsize=14)

# # Add metrics text box to the plot
# metrics_text = f"RMSE: {rmse_val:.4f}\nCorr ($r$): {corr_val:.4f}"
# ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
# ax.grid(True, linestyle=':', alpha=0.7)

# plt.tight_layout()
# plt.show()