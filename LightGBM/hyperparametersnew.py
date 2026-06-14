import duckdb
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_pinball_loss 
file_path = r"C:\Users\parth\Documents\Chlorophyll\final_ml_dataset_cleanednewwo3sigma.parquet"
target_col = 'chlor_a_log10'
drop_cols = ['time', 'lat', 'lon', target_col]

print("Connecting to DuckDB for the Time-Spanning Scout Run...")
con = duckdb.connect()


print("Extracting a 0.5% sample from 1998-2022 for Training...")

query_train = f"""
    SELECT * FROM '{file_path}' 
    WHERE EXTRACT(YEAR FROM time) BETWEEN 1998 AND 2022
    USING SAMPLE 0.5 PERCENT
"""
df_train = con.execute(query_train).df()
X_train, y_train = df_train.drop(columns=drop_cols), df_train[target_col]

print("Extracting a 5% sample of 2023 for Validation...")
query_val = f"""
    SELECT * FROM '{file_path}' 
    WHERE EXTRACT(YEAR FROM time) IN (2023)
    USING SAMPLE 5 PERCENT
"""
df_val = con.execute(query_val).df()
X_val, y_val = df_val.drop(columns=drop_cols), df_val[target_col]

print(f"Loaded {len(X_train):,} training rows and {len(X_val):,} validation rows.")


def objective(trial):
    target_alpha = 0.50 

    params = {
        'objective': 'quantile', 
        'metric': 'rmse',          
        'boosting_type': 'gbdt',
        
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 800, 1000),
        'num_leaves': trial.suggest_int('num_leaves', 30, 90),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        
        'min_child_samples': trial.suggest_int('min_child_samples', 1000, 3000),
        
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),  # L1
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True), # L2
        
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        
        'n_jobs': -1,
        'verbose': -1
    }

    model = lgb.LGBMRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )
    
    preds = model.predict(X_val)
    error = mean_pinball_loss(y_val, preds, alpha=target_alpha)
    
    return error


print(f"\nCommencing Optuna Search (50 Trials)...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50) 


print("locked_params = {")

for key, value in study.best_trial.params.items():
    if isinstance(value, float):
        print(f"    '{key}': {value:.5f},")
    else:
        print(f"    '{key}': {value},")

# Add the static parameters that Optuna didn't tune
print("    'random_state': 42,")
print("    'max_bin': 63,")  # Keeping the memory-safety bin limit
print("    'n_jobs': -1,")
print("    'verbose': -1")
print("}")
print("="*50)

# # ---------------------------------------------------------
# # 5. PLOT HYPERPARAMETER IMPORTANCE
# # ---------------------------------------------------------
# print("\nGenerating Hyperparameter Importance Plot...")
# fig = optuna.visualization.matplotlib.plot_param_importances(study)
# plt.title("Hyperparameter Importance for RMSE Reduction")
# plt.tight_layout()
# plt.show()