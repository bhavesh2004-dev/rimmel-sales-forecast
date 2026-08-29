"""
Version 2 vs Version 3 Multi-Window Backtest Comparison Script
Evaluates May 2026, June 2026, and July 2026 30-day forecast accuracy across Top 20 SKUs:
- Version 2: Baseline Model (Without Price & Category)
- Version 3: Enhanced Model (With Selling Price $ USD Elasticity & Category Features)
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.multi_sku_cleaner import prepare_top_20_cleaned_datasets
from src.feature_engineering import build_features, get_feature_columns
from src.model import SalesForecasterEnsemble
from src.croston import CrostonForecaster

def calculate_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def calculate_mape(y_true, y_pred):
    y_true_safe = np.maximum(y_true, 1)
    return float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100)

def evaluate_window(clean_df, feature_cols, start_date, end_date, use_price_cat=True):
    train_df = clean_df[clean_df['date'] < start_date]
    test_df = clean_df[(clean_df['date'] >= start_date) & (clean_df['date'] <= end_date)]
    
    if len(test_df) == 0 or len(train_df) < 30:
        return 0.0, 0.0, 0.0, 0.0
        
    zero_pct = (train_df['sold_qty'] == 0).mean() * 100
    stockout_cnt = train_df['is_stockout'].sum() if 'is_stockout' in train_df.columns else 0
    is_intermittent = (zero_pct > 15.0 or stockout_cnt > 15)
    
    if is_intermittent:
        c_model = CrostonForecaster()
        c_model.fit(train_df['target_sales'])
        preds = c_model.predict(len(test_df))
    else:
        if not use_price_cat:
            v2_cols = [c for c in feature_cols if c not in ['selling_price', 'price_discount_ratio', 'price_drop_pct', 'is_discounted', 'category_freq_encoded']]
            cols_to_use = v2_cols
        else:
            cols_to_use = feature_cols
            
        ens_model = SalesForecasterEnsemble()
        ens_model.fit(train_df[cols_to_use], train_df['target_sales'])
        preds = np.maximum(0, ens_model.predict(test_df[cols_to_use]))
        
    actual_total = float(test_df['sold_qty'].sum())
    pred_total = float(np.sum(preds))
    
    rmse = calculate_rmse(test_df['sold_qty'], preds)
    mape = calculate_mape(test_df['sold_qty'], preds)
    mae = mean_absolute_error(test_df['sold_qty'], preds)
    vol_acc = max(0.0, (1 - abs(actual_total - pred_total) / max(1, actual_total)) * 100)
    
    return vol_acc, rmse, mape, mae

def run_comparison():
    excel_path = os.path.join(os.path.dirname(__file__), '..', 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026 New.xlsx')
    output_dir = os.path.join(os.path.dirname(__file__), 'output_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*85)
    print("RUNNING MULTI-WINDOW BACKTEST COMPARISON: VERSION 2 vs VERSION 3")
    print("="*85)
    print("Evaluating May 2026, June 2026, and July 2026 Across Top 20 SKUs...\n")
    
    cleaned_dict, data_quality_df = prepare_top_20_cleaned_datasets(excel_path)
    feature_cols = get_feature_columns()
    
    results = []
    
    windows = [
        ('May 2026', '2026-05-01', '2026-05-31'),
        ('June 2026', '2026-06-01', '2026-06-30'),
        ('July 2026', '2026-07-01', '2026-07-31')
    ]
    
    for sku, df_sku in cleaned_dict.items():
        feature_df = build_features(df_sku)
        clean_df = feature_df.dropna(subset=feature_cols).copy().reset_index(drop=True)
        category_name = df_sku['category'].iloc[0] if 'category' in df_sku.columns else 'Cosmetics'
        
        for w_name, s_date, e_date in windows:
            # Version 2 (Without Price/Category)
            v2_acc, v2_rmse, v2_mape, v2_mae = evaluate_window(clean_df, feature_cols, s_date, e_date, use_price_cat=False)
            
            # Version 3 (With Price & Category)
            v3_acc, v3_rmse, v3_mape, v3_mae = evaluate_window(clean_df, feature_cols, s_date, e_date, use_price_cat=True)
            
            acc_gain = v3_acc - v2_acc
            rmse_diff = v2_rmse - v3_rmse  # Positive means V3 reduced RMSE error!
            
            results.append({
                'SKU': sku,
                'Category': category_name,
                'Window': w_name,
                'V2 Accuracy %': round(v2_acc, 1),
                'V3 Accuracy %': round(v3_acc, 1),
                'Accuracy Gain %': round(acc_gain, 1),
                'V2 RMSE': round(v2_rmse, 2),
                'V3 RMSE': round(v3_rmse, 2),
                'RMSE Error Reduction': round(rmse_diff, 2)
            })
            
    res_df = pd.DataFrame(results)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'v2_vs_v3_backtest_comparison.csv')
    res_df.to_csv(csv_path, index=False)
    
    print("="*85)
    print("BACKTEST COMPARISON SUMMARY BY MONTH:")
    print("="*85)
    monthly_summary = res_df.groupby('Window').agg({
        'V2 Accuracy %': 'mean',
        'V3 Accuracy %': 'mean',
        'Accuracy Gain %': 'mean',
        'V2 RMSE': 'mean',
        'V3 RMSE': 'mean',
        'RMSE Error Reduction': 'mean'
    }).reset_index()
    
    print(monthly_summary.to_string(index=False))
    
    print("\n" + "="*85)
    print("OVERALL AVERAGE ACCURACY GAIN ACROSS ALL WINDOWS & TOP 20 SKUs:")
    print("="*85)
    overall_v2_acc = res_df['V2 Accuracy %'].mean()
    overall_v3_acc = res_df['V3 Accuracy %'].mean()
    overall_gain = overall_v3_acc - overall_v2_acc
    overall_v2_rmse = res_df['V2 RMSE'].mean()
    overall_v3_rmse = res_df['V3 RMSE'].mean()
    overall_rmse_red = overall_v2_rmse - overall_v3_rmse
    
    print(f"Version 2 Average Accuracy : {overall_v2_acc:.1f}% | Average RMSE Error: {overall_v2_rmse:.2f}")
    print(f"Version 3 Average Accuracy : {overall_v3_acc:.1f}% | Average RMSE Error: {overall_v3_rmse:.2f}")
    print(f"Net Accuracy Gain          : +{overall_gain:.1f}% points")
    print(f"Net RMSE Error Reduction   : -{overall_rmse_red:.2f} units lower error!")
    print(f"Detailed Comparison Saved to: {csv_path}")

if __name__ == '__main__':
    run_comparison()
