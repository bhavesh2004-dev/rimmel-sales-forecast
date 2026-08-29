"""
Master Production Pipeline — Global Tweedie LightGBM (588 SKUs)
Executes:
1. Full Data Cleaning & Cold-Start Pre-processing
2. Global Feature Engineering with Category Lag Imputation
3. 3-Fold Walk-Forward Cross Validation (May, June, July 2026)
4. Final Model Training on Full Historical Dataset
5. August 2026 Daily Sales Forecast Generation (588 SKUs) with 95% Safety Stock
6. SQLite Database Persistence (forecast_results) & CSV Reports
"""
import os
import sys
import sqlite3
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.global_cleaner import prepare_global_dataset
from src.global_features import generate_global_features, get_global_feature_columns
from src.global_forecaster import train_tweedie_model, run_walk_forward_cv

def run_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(base_dir, '..', 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026 New.xlsx')
    output_dir = os.path.join(base_dir, 'output_results')
    db_path = os.path.join(base_dir, 'sales_forecasting.db')
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*85)
    print("STARTING GLOBAL PRODUCTION FORECASTING PIPELINE — FULL 588 SKU CATALOG")
    print("="*85)
    
    # 1. Clean Data
    df_clean = prepare_global_dataset(excel_path)
    
    # 2. Build Features
    df_featured = generate_global_features(df_clean)
    feature_cols = get_global_feature_columns()
    
    # 3. Walk-Forward Cross Validation
    cv_metrics_df = run_walk_forward_cv(df_featured, feature_cols)
    cv_metrics_df.to_csv(os.path.join(output_dir, 'global_588_cv_metrics.csv'), index=False)
    
    # 4. Train Final Model on All Data (Jan 1, 2025 to July 31, 2026)
    print("\nTraining Final Production Global Tweedie Model on Full Dataset (113,031 rows)...")
    train_data = df_featured.dropna(subset=feature_cols)
    final_model = train_tweedie_model(train_data[feature_cols], train_data['target_sales'])
    
    # 5. Generate 30-Day August 2026 Forecast Grid
    print("\nGenerating 30-Day August 2026 Forecasts for all 588 SKUs...")
    august_dates = pd.date_range(start='2026-08-01', end='2026-08-30', freq='D')
    all_skus = df_clean['sku'].unique()
    
    sku_cat_map = df_clean.groupby('sku')['category'].first().to_dict()
    sku_price_map = df_clean.groupby('sku')['selling_price'].last().to_dict()
    sku_hist_map = df_clean.groupby('sku')['history_days'].max().to_dict()

    august_rows = []
    
    # Prepare last available feature values for August recursive/multi-step forecast
    last_known = df_featured[df_featured['date'] == '2026-07-31'].set_index('sku')
    
    for sku in all_skus:
        sku_cat = sku_cat_map.get(sku, 'Cosmetics')
        sku_price = sku_price_map.get(sku, 5.00)
        hist_days = sku_hist_map.get(sku, 100)
        
        # Get baseline features from last known date
        if sku in last_known.index:
            base_row = last_known.loc[sku]
            if isinstance(base_row, pd.DataFrame):
                base_row = base_row.iloc[-1]
            lag_1 = float(base_row['target_sales'])
            rolling_7 = float(base_row['rolling_mean_7d'])
            rolling_14 = float(base_row['rolling_mean_14d'])
            rolling_30 = float(base_row['rolling_mean_30d'])
            rolling_std = float(base_row['rolling_std_7d'])
        else:
            lag_1, rolling_7, rolling_14, rolling_30, rolling_std = 0.5, 0.5, 0.5, 0.5, 0.2
            
        cat_encoded = float(base_row['category_encoded']) if sku in last_known.index else 1.0
        sku_encoded = float(base_row['sku_encoded']) if sku in last_known.index else 1.0
        
        for d in august_dates:
            day_of_week = d.dayofweek
            day_of_month = d.day
            month = d.month
            quarter = d.quarter
            is_weekend = 1 if day_of_week in [5, 6] else 0
            
            # Predict step
            feat_vector = pd.DataFrame([{
                'day_of_week': day_of_week,
                'day_of_month': day_of_month,
                'month': month,
                'quarter': quarter,
                'is_weekend': is_weekend,
                'category_encoded': cat_encoded,
                'sku_encoded': sku_encoded,
                'selling_price': sku_price,
                'rolling_mean_price_30d': sku_price,
                'price_discount_ratio': 1.0,
                'price_drop_pct': 0.0,
                'is_discounted': 0,
                'lag_1': lag_1,
                'lag_7': rolling_7,
                'lag_14': rolling_14,
                'lag_30': rolling_30,
                'rolling_mean_7d': rolling_7,
                'rolling_mean_14d': rolling_14,
                'rolling_mean_30d': rolling_30,
                'rolling_std_7d': rolling_std,
                'history_days': hist_days + (d - pd.to_datetime('2026-07-31')).days
            }])
            
            pred_qty = float(np.maximum(0, final_model.predict(feat_vector[feature_cols])[0]))
            
            # 95% Confidence Upper Safety Stock Limit
            std_err = max(0.5, rolling_std)
            upper_95 = pred_qty + 1.96 * std_err
            lower_95 = max(0, pred_qty - 1.96 * std_err)
            
            august_rows.append({
                'order_item_sku': sku,
                'category': sku_cat,
                'date': d.strftime('%Y-%m-%d'),
                'predicted_sales': round(pred_qty, 2),
                'lower_bound_95': round(lower_95, 2),
                'upper_bound_95': round(upper_95, 2),
                'model_type': 'Global LightGBM Tweedie'
            })
            
    df_august = pd.DataFrame(august_rows)
    august_csv = os.path.join(output_dir, 'global_588_august_forecast.csv')
    df_august.to_csv(august_csv, index=False)
    
    # 6. SKU Summary Report across all 588 SKUs
    print("\nGenerating SKU Summary Report across all 588 SKUs...")
    aug_summary = df_august.groupby(['order_item_sku', 'category']).agg({
        'predicted_sales': 'sum',
        'upper_bound_95': 'sum'
    }).reset_index()
    aug_summary.columns = ['SKU', 'Category', 'Total August Predicted Sales', 'August 95% Upper Safety Stock Buffer']
    
    # Add historical stats per SKU
    hist_stats = df_clean.groupby('sku').agg({
        'sold_qty': 'sum',
        'history_days': 'max',
        'selling_price': 'last'
    }).reset_index()
    hist_stats.columns = ['SKU', 'Historical Units Sold (19 Months)', 'Days of History', 'Last Selling Price ($)']
    
    sku_summary = pd.merge(aug_summary, hist_stats, on='SKU', how='left')
    sku_summary['Status'] = np.where(sku_summary['Days of History'] < 60, 'Cold-Start (<60 Days)', 'Established SKU')
    
    summary_csv = os.path.join(output_dir, 'global_588_skus_summary.csv')
    sku_summary.to_csv(summary_csv, index=False)
    
    # 7. Persist to SQLite forecast_results table
    print("\nPersisting forecasts into SQLite database (sales_forecasting.db)...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS forecast_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_item_sku VARCHAR(100),
        date VARCHAR(20),
        predicted_sales REAL,
        lower_bound_95 REAL,
        upper_bound_95 REAL,
        model_type VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Clear previous predictions and insert new
    cursor.execute("DELETE FROM forecast_results WHERE model_type = 'Global LightGBM Tweedie'")
    
    insert_data = df_august[['order_item_sku', 'date', 'predicted_sales', 'lower_bound_95', 'upper_bound_95', 'model_type']].values.tolist()
    cursor.executemany("""
    INSERT INTO forecast_results (order_item_sku, date, predicted_sales, lower_bound_95, upper_bound_95, model_type)
    VALUES (?, ?, ?, ?, ?, ?)
    """, insert_data)
    
    conn.commit()
    conn.close()
    
    print("\n" + "="*85)
    print("GLOBAL FORECASTING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*85)
    print(f"Total SKUs Processed and Forecasted  : {df_august['order_item_sku'].nunique()} / 588 SKUs")
    print(f"Total August 2026 Forecast Rows      : {len(df_august):,} daily prediction rows")
    print(f"Total August 2026 Projected Sales    : {df_august['predicted_sales'].sum():,.0f} units")
    print(f"Total August 95% Safety Stock Buffer : {df_august['upper_bound_95'].sum():,.0f} units")
    print(f"Master Summary Report Saved to        : {summary_csv}")
    print(f"Daily August Forecast CSV Saved to   : {august_csv}")
    print(f"SQLite DB Forecast Results Persisted : {db_path}")

if __name__ == '__main__':
    run_pipeline()
