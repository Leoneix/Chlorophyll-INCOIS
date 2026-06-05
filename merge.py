import xarray as xr
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

output_dir = 'ocean_ml_dataset_full.parquet'
os.makedirs(output_dir, exist_ok=True)

# 1. Load datasets
occci = xr.open_dataset('occci_collocated.nc').squeeze(drop=True)
glory = xr.open_dataset('glory_collocated_masked.nc').squeeze(drop=True)
era5  = xr.open_dataset('era5_collocated_masked.nc').squeeze(drop=True)

# 2. Standardize Coordinates
# Helper function to ensure all spatial/temporal dims match exactly
def standardize_coords(ds):
    rename_dict = {}
    # Check for ERA5's 'valid_time'
    if 'valid_time' in ds.coords or 'valid_time' in ds.variables:
        rename_dict['valid_time'] = 'time'
    # Check for expanded 'latitude' / 'longitude' names
    if 'latitude' in ds.coords or 'latitude' in ds.variables:
        rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.coords or 'longitude' in ds.variables:
        rename_dict['longitude'] = 'lon'
        
    return ds.rename(rename_dict) if rename_dict else ds

occci = standardize_coords(occci)
glory = standardize_coords(glory)
era5  = standardize_coords(era5)

# Now we can safely grab the standardized time array
all_times = np.unique(np.concatenate([
    occci.time.values, 
    glory.time.values, 
    era5.time.values
]))
all_times.sort() # Ensure it flows from 1993 to the end

step_size = 50 

print(f"Total days to process: {len(all_times)}")
print(f"Dataset spans from {pd.to_datetime(all_times[0]).date()} to {pd.to_datetime(all_times[-1]).date()}")
print("Beginning manual chunking pipeline...")

# Define the cutoff date for the NaN filtering rule
cutoff_date = pd.to_datetime('1997-09-16')

for i in tqdm(range(0, len(all_times), step_size), desc="Processing Time Chunks"):
    
    start_time = all_times[i]
    end_idx = min(i + step_size - 1, len(all_times) - 1)
    end_time = all_times[end_idx]
    
    time_slice = slice(start_time, end_time)

    # 3. Merge Datasets
    merged_slice = xr.merge([
        occci.sel(time=time_slice),
        glory.sel(time=time_slice),
        era5.sel(time=time_slice)
    ])

    # 4. Load into RAM and downcast to float32
    in_memory_slice = merged_slice.astype(np.float32).load()

    # 5. Convert to a standard Pandas DataFrame
    df = in_memory_slice.to_dataframe().reset_index()

    # 6. Drop NaNs using the date condition
    condition_to_keep = ~(df['chlor_a'].isna() & (df['time'] > cutoff_date))
    df_clean = df[condition_to_keep].copy() 

    # 7. Apply transformations
    df_clean['chlor_a'] = np.log(df_clean['chlor_a'])
    df_clean['mlotst'] = np.log(df_clean['mlotst'])
    
    # NEW: Convert ssrd to W/m²
    if 'ssrd' in df_clean.columns:
        df_clean['ssrd'] = df_clean['ssrd'] / 86400.0

    # 8. Save to Parquet
    if not df_clean.empty:
        filename = f"{output_dir}/part_{i:06d}.parquet"
        df_clean.to_parquet(filename, index=False)

print(f"\nSuccess! All clean data safely stored in the '{output_dir}' folder.")