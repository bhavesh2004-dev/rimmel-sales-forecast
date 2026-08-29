"""
Multi-SKU Forecaster Module (Version 3.0)
Trains Ensemble / Croston models on Top 20 SKUs with Selling Price ($ USD) & Category features.
Computes backtest benchmarks and outputs August 2026 30-day predictions + 95% safety stock bands.
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .multi_sku_cleaner import prepare_top_20_cleaned_datasets
from .feature_engineering import build_features, get_feature_columns
from .model import SalesForecasterEnsemble
from .croston import CrostonForecaster

DEFAULT_NEW_DATASET = 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026 New.xlsx'

def calculate_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def calculate_mape(y_true, y_pred):
    y_true_safe = np.maximum(y_true, 1)
    return float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100)

def run_top_20_pipeline(excel_path: str = None, output_dir: str = None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, '..', '..')
    
    if excel_path is None or not os.path.exists(excel_path):
        excel_path = os.path.join(project_dir, DEFAULT_NEW_DATASET)
        
    if output_dir is None:
        output_dir = os.path.join(base_dir, '..', 'output_results')
        
    os.makedirs(output_dir, exist_ok=True)
    print("Step 1: Extracting and Cleansing Top 20 SKUs from New Dataset (Price & Category Features)...")
    
    cleaned_dict, data_quality_df = prepare_top_20_cleaned_datasets(excel_path)
    feature_cols = get_feature_columns()
    
    sku_master_summary = []
    august_forecast_list = []
    
    print("\nStep 2: Training Models with Price & Category Features across Top 20 SKUs...")
    
    for idx, (sku, df_sku) in enumerate(cleaned_dict.items(), 1):
        feature_df = build_features(df_sku)
        clean_df = feature_df.dropna(subset=feature_cols).copy().reset_index(drop=True)
        
        dq_row = data_quality_df[data_quality_df['order_item_sku'] == sku].iloc[0]
        zero_pct = dq_row['zero_sales_days_pct']
        stockout_cnt = dq_row['stockouts_imputed']
        category_name = dq_row['category']
        avg_price = dq_row['avg_selling_price_usd']
        
        is_intermittent = (zero_pct > 15.0 or stockout_cnt > 15)
        model_type = "Croston (Intermittent)" if is_intermittent else "LightGBM + RF Ensemble"
        
        # 1. Backtesting on July 2026 (2026-07-01 to 2026-07-31)
        july_train = clean_df[clean_df['date'] < '2026-07-01']
        july_test = clean_df[(clean_df['date'] >= '2026-07-01') & (clean_df['date'] <= '2026-07-31')]
        
        if is_intermittent:
            c_model = CrostonForecaster()
            c_model.fit(july_train['target_sales'])
            pred_july = c_model.predict(len(july_test))
            res_std = np.std(july_train['target_sales'])
        else:
            ens_model = SalesForecasterEnsemble()
            ens_model.fit(july_train[feature_cols], july_train['target_sales'])
            pred_july = np.maximum(0, ens_model.predict(july_test[feature_cols]))
            res_std = np.std(july_train['target_sales'] - ens_model.predict(july_train[feature_cols]))
            
        rmse_july = calculate_rmse(july_test['sold_qty'], pred_july)
        mape_july = calculate_mape(july_test['sold_qty'], pred_july)
        mae_july = mean_absolute_error(july_test['sold_qty'], pred_july)
        
        actual_july_total = float(july_test['sold_qty'].sum())
        pred_july_total = float(np.sum(pred_july))
        july_vol_acc = max(0.0, (1 - abs(actual_july_total - pred_july_total) / max(1, actual_july_total)) * 100)
        
        # 2. Predict August 2026 (2026-08-01 to 2026-08-30)
        august_dates = pd.date_range(start='2026-08-01', end='2026-08-30', freq='D', name='date')
        
        if is_intermittent:
            c_august = CrostonForecaster()
            c_august.fit(clean_df['target_sales'])
            august_preds = c_august.predict(len(august_dates))
            august_std = np.std(clean_df['target_sales'])
        else:
            ens_august = SalesForecasterEnsemble()
            ens_august.fit(clean_df[feature_cols], clean_df['target_sales'])
            
            august_df = pd.DataFrame({'date': august_dates})
            combined = pd.concat([clean_df[['date', 'target_sales', 'sold_qty', 'selling_price', 'category']], august_df], ignore_index=True)
            combined['target_sales'] = combined['target_sales'].fillna(clean_df['target_sales'].iloc[-7:].mean())
            combined['selling_price'] = combined['selling_price'].fillna(avg_price)
            combined['category'] = combined['category'].fillna(category_name)
            
            from .anomaly_detector import detect_anomalies
            combined_anom = detect_anomalies(combined)
            combined_feats = build_features(combined_anom)
            
            X_aug = combined_feats[combined_feats['date'].isin(august_dates)][feature_cols]
            august_preds = np.maximum(0, ens_august.predict(X_aug))
            august_std = np.std(clean_df['target_sales'] - ens_august.predict(clean_df[feature_cols]))
            
        august_df_res = pd.DataFrame({
            'order_item_sku': sku,
            'category': category_name,
            'selling_price_usd': avg_price,
            'date': august_dates,
            'predicted_sales': np.round(august_preds, 1),
            'lower_bound_95%': np.maximum(0, np.round(august_preds - 1.96 * august_std, 1)),
            'upper_bound_95%': np.round(august_preds + 1.96 * august_std, 1)
        })
        
        august_forecast_list.append(august_df_res)
        
        total_august_units = int(np.sum(august_preds))
        avg_august_daily = float(np.mean(august_preds))
        
        sku_master_summary.append({
            'SKU': sku,
            'Category': category_name,
            'Avg Price ($ USD)': avg_price,
            'Model Assigned': model_type,
            'Historical Volume': dq_row['total_historical_sales'],
            'Zero Days %': dq_row['zero_sales_days_pct'],
            'Stockouts Imputed': dq_row['stockouts_imputed'],
            'Spikes Capped': dq_row['spikes_capped'],
            'July Backtest RMSE': round(rmse_july, 2),
            'July Backtest MAPE %': round(mape_july, 1),
            'July Vol Accuracy %': round(july_vol_acc, 1),
            'August Total Forecast': total_august_units,
            'August Daily Avg': round(avg_august_daily, 1),
            'August Upper Safety Buffer': int(np.sum(august_df_res['upper_bound_95%']))
        })
        
        print(f"[{idx:02d}/20] SKU: {sku:<22} | Cat: {category_name[:15]:<15} | Price: ${avg_price:5.2f} | July Acc: {july_vol_acc:5.1f}% | Aug Forecast: {total_august_units:,} units")

    # Combine into Master DataFrames
    master_summary_df = pd.DataFrame(sku_master_summary)
    master_august_df = pd.concat(august_forecast_list, ignore_index=True)
    
    # Export CSVs & Push to SQL Database forecast_results table
    master_summary_path = os.path.join(output_dir, 'top_20_skus_summary.csv')
    master_august_path = os.path.join(output_dir, 'top_20_august_forecast.csv')
    
    master_summary_df.to_csv(master_summary_path, index=False)
    master_august_df.to_csv(master_august_path, index=False)
    
    sqlite_db_path = os.path.join(project_dir, 'new_approach', 'sales_forecasting.db')
    if os.path.exists(sqlite_db_path):
        from .sql_data_loader import save_forecast_to_sql
        try:
            save_forecast_to_sql(sqlite_db_path, master_august_df[['order_item_sku', 'date', 'predicted_sales', 'lower_bound_95%', 'upper_bound_95%']].rename(columns={'lower_bound_95%': 'lower_bound_95', 'upper_bound_95%': 'upper_bound_95'}))
        except Exception as e:
            print(f"SQL Forecast Persistence Notice: {e}")
            
    print("\n" + "="*85)
    print("VERSION 3.0 TOP 20 SKUs PIPELINE EXECUTION COMPLETED")
    print("="*85)
    print(f"Master Summary CSV Saved to   : {master_summary_path}")
    print(f"August Forecast CSV Saved to  : {master_august_path}")
    
    return master_summary_df, master_august_df, cleaned_dict
