import os
import duckdb
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import gc

file_path = r'C:\Users\parth\Documents\Chlorophyll\final_ml_dataset_cleanednewwo3sigma.parquet'
target_col = 'chlor_a_log10' 
drop_cols = ['time', 'lat', 'lon', target_col]

lat_min, lat_max = 5.0, 25.0
lon_min, lon_max = 65.0, 78.0

con = duckdb.connect()
query_train = f"""
    SELECT * FROM '{file_path}' 
    WHERE EXTRACT(YEAR FROM time) BETWEEN 1998 AND 2024 
    AND lat BETWEEN {lat_min} AND {lat_max}
    AND lon BETWEEN {lon_min} AND {lon_max}
    
"""
df_train = con.execute(query_train).df()
print(f"Success! {len(df_train):,} rows loaded into memory.")

df_train['time'] = pd.to_datetime(df_train['time'])
df_train['year'] = df_train['time'].dt.year

train_mask = df_train['year'] <= 2023
val_mask = df_train['year'] == 2024

X_train_full = df_train[train_mask]
X_val_full = df_train[val_mask]

X_train = X_train_full.drop(columns=drop_cols + ['year'])
y_train = X_train_full[target_col]

X_val = X_val_full.drop(columns=drop_cols + ['year'])
y_val = X_val_full[target_col]

del df_train, X_train_full, X_val_full
gc.collect()

for col in X_train.select_dtypes(include=['float64']).columns:
    X_train[col] = X_train[col].astype('float32')
for col in X_val.select_dtypes(include=['float64']).columns:
    X_val[col] = X_val[col].astype('float32')
gc.collect()

locked_params = {
    'learning_rate': 0.02211,
    'n_estimators': 1000,
    'num_leaves': 86,
    'max_depth': 9,
    'min_child_samples': 1674,
    'reg_alpha': 3.89718,
    'reg_lambda': 6.71138,
    'colsample_bytree': 0.79377,
    'subsample': 0.85547,
    'bagging_freq': 1,
    'random_state': 42,
    'max_bin': 63,
    'n_jobs': -1,
    'verbose': -1
}

model_q10 = lgb.LGBMRegressor(objective='quantile', alpha=0.10, **locked_params)
model_q50 = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **locked_params)
model_q90 = lgb.LGBMRegressor(objective='quantile', alpha=0.90, **locked_params)

callbacks = [
    lgb.early_stopping(stopping_rounds=100, verbose=True),
    lgb.log_evaluation(period=100)
]

print("\nTraining Q10 Model...")
model_q10.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks
)

print("\nTraining Q50 Model...")
model_q50.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks
)

print("\nTraining Q90 Model...")
model_q90.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks
)

os.makedirs('saved_models', exist_ok=True)
joblib.dump(model_q10, 'saved_models/lgbm_q10_final_arabian.pkl')
joblib.dump(model_q50, 'saved_models/lgbm_q50_final_arabian.pkl')
joblib.dump(model_q90, 'saved_models/lgbm_q90_final_arabian.pkl')

print("\nTraining Complete!")
