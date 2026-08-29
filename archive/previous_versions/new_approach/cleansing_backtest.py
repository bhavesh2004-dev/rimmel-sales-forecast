import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_aggregate_sku
from src.anomaly_detector import detect_anomalies
from src.feature_engineering import build_features, get_feature_columns

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    # Avoid division by zero by setting minimum actual value to 1
    y_true_safe = np.maximum(y_true, 1)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

def run_cleansing_backtest():
    excel_path = os.path.join(os.path.dirname(__file__), '..', 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026.xlsx')
    output_dir = os.path.join(os.path.dirname(__file__), 'output_results')
    os.makedirs(output_dir, exist_ok=True)
    
    sku_code = 'RIM-MSC-E3DL-003'
    print(f"Running Data Cleansing Impact Backtest (XGBoost) for SKU: {sku_code}...")
    
    raw_df = load_and_aggregate_sku(excel_path, sku_code)
    
    # ----------------------------------------------------
    # DATASET A: RAW (Uncleansed Data - Unhandled Spikes & Stockouts)
    # ----------------------------------------------------
    raw_pipeline = raw_df.copy()
    raw_pipeline['target_sales'] = raw_pipeline['sold_qty'] # Raw sales as target
    raw_pipeline['rolling_mean_30'] = raw_pipeline['sold_qty'].rolling(window=30, min_periods=7).mean()
    raw_pipeline['is_spike'] = False
    raw_pipeline['is_stockout'] = False
    
    raw_feats = build_features(raw_pipeline)
    feature_cols = get_feature_columns()
    clean_raw = raw_feats.dropna(subset=feature_cols).copy().reset_index(drop=True)
    
    # ----------------------------------------------------
    # DATASET B: CLEANSED (Spikes Capped + Stockouts Imputed)
    # ----------------------------------------------------
    cleansed_pipeline = detect_anomalies(raw_df)
    
    # Impute stockouts (eod_stock == 0) with 30-day rolling mean demand
    stockout_mask = cleansed_pipeline['is_stockout']
    cleansed_pipeline['imputed_sales'] = np.where(
        stockout_mask,
        cleansed_pipeline['rolling_mean_30'],
        cleansed_pipeline['target_sales'] # target_sales already has spikes capped
    )
    cleansed_pipeline['target_sales'] = cleansed_pipeline['imputed_sales']
    
    cleansed_feats = build_features(cleansed_pipeline)
    clean_cleansed = cleansed_feats.dropna(subset=feature_cols).copy().reset_index(drop=True)
    
    # Backtest windows
    windows = [
        {'name': 'Window 1: May 2026', 'test_start': '2026-05-01', 'test_end': '2026-05-30'},
        {'name': 'Window 2: June 2026', 'test_start': '2026-06-01', 'test_end': '2026-06-30'},
        {'name': 'Window 3: July 2026', 'test_start': '2026-07-01', 'test_end': '2026-07-30'}
    ]
    
    comparison_results = []
    
    for win in windows:
        t_start = pd.to_datetime(win['test_start'])
        t_end = pd.to_datetime(win['test_end'])
        
        # --- Model 1: Trained on Raw Data ---
        train_raw_df = clean_raw[clean_raw['date'] < t_start]
        test_raw_df = clean_raw[(clean_raw['date'] >= t_start) & (clean_raw['date'] <= t_end)]
        
        model_raw = xgb.XGBRegressor(n_estimators=100, learning_rate=0.03, max_depth=5, random_state=42)
        model_raw.fit(train_raw_df[feature_cols], train_raw_df['target_sales'])
        pred_raw = np.maximum(0, model_raw.predict(test_raw_df[feature_cols]))
        
        rmse_raw = calculate_rmse(test_raw_df['sold_qty'], pred_raw)
        mape_raw = calculate_mape(test_raw_df['sold_qty'], pred_raw)
        mae_raw = mean_absolute_error(test_raw_df['sold_qty'], pred_raw)
        
        # --- Model 2: Trained on Cleansed Data ---
        train_clean_df = clean_cleansed[clean_cleansed['date'] < t_start]
        test_clean_df = clean_cleansed[(clean_cleansed['date'] >= t_start) & (clean_cleansed['date'] <= t_end)]
        
        model_cleansed = xgb.XGBRegressor(n_estimators=100, learning_rate=0.03, max_depth=5, random_state=42)
        model_cleansed.fit(train_clean_df[feature_cols], train_clean_df['target_sales'])
        pred_cleansed = np.maximum(0, model_cleansed.predict(test_clean_df[feature_cols]))
        
        rmse_cleansed = calculate_rmse(test_clean_df['sold_qty'], pred_cleansed)
        mape_cleansed = calculate_mape(test_clean_df['sold_qty'], pred_cleansed)
        mae_cleansed = mean_absolute_error(test_clean_df['sold_qty'], pred_cleansed)
        
        comparison_results.append({
            'Window': win['name'],
            'Raw RMSE': round(rmse_raw, 2),
            'Cleansed RMSE': round(rmse_cleansed, 2),
            'RMSE Reduction %': round((1 - rmse_cleansed / rmse_raw) * 100, 1),
            'Raw MAPE %': round(mape_raw, 2),
            'Cleansed MAPE %': round(mape_cleansed, 2),
            'MAPE Improvement %': round(mape_raw - mape_cleansed, 1),
            'Raw MAE': round(mae_raw, 2),
            'Cleansed MAE': round(mae_cleansed, 2)
        })
        
    res_df = pd.DataFrame(comparison_results)
    
    print("\n" + "="*85)
    print("BACKTEST COMPARISON: BEFORE VS AFTER DATA CLEANSING (XGBoost)")
    print("="*85)
    print(res_df.to_string(index=False))
    print("="*85)
    
    res_df.to_csv(os.path.join(output_dir, 'cleansing_backtest_comparison.csv'), index=False)
    
    # Plotting comparison bar chart
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    window_labels = ['May 2026', 'June 2026', 'July 2026']
    x = np.arange(len(window_labels))
    width = 0.35
    
    # RMSE Plot
    ax[0].bar(x - width/2, res_df['Raw RMSE'], width, label='Before Cleansing (Raw)', color='#e74c3c')
    ax[0].bar(x + width/2, res_df['Cleansed RMSE'], width, label='After Cleansing (Spikes Capped & Stockouts Imputed)', color='#2ecc71')
    ax[0].set_title('RMSE Error (Lower is Better)', fontweight='bold')
    ax[0].set_ylabel('RMSE')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(window_labels)
    ax[0].legend()
    
    # MAPE Plot
    ax[1].bar(x - width/2, res_df['Raw MAPE %'], width, label='Before Cleansing (Raw)', color='#e74c3c')
    ax[1].bar(x + width/2, res_df['Cleansed MAPE %'], width, label='After Cleansing (Cleansed)', color='#2ecc71')
    ax[1].set_title('MAPE Error % (Lower is Better)', fontweight='bold')
    ax[1].set_ylabel('MAPE %')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(window_labels)
    ax[1].legend()
    
    plt.suptitle(f'XGBoost Forecaster Performance: Before vs. After Data Cleansing ({sku_code})', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, 'cleansing_impact_comparison.png')
    plt.savefig(chart_path)
    plt.close()
    
    print(f"\nSaved CSV and comparison chart to: {output_dir}")

if __name__ == '__main__':
    run_cleansing_backtest()
