import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_aggregate_sku
from src.anomaly_detector import detect_anomalies
from src.feature_engineering import build_features, get_feature_columns
from src.model import SalesForecasterEnsemble

def run_range_forecast(train_start_str: str, train_end_str: str, predict_start_str: str, predict_end_str: str, sku_code: str = 'RIM-MSC-E3DL-003'):
    excel_path = os.path.join(os.path.dirname(__file__), '..', 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026.xlsx')
    output_dir = os.path.join(os.path.dirname(__file__), 'output_results')
    os.makedirs(output_dir, exist_ok=True)
    
    train_start = pd.to_datetime(train_start_str)
    train_end = pd.to_datetime(train_end_str)
    predict_start = pd.to_datetime(predict_start_str)
    predict_end = pd.to_datetime(predict_end_str)
    
    print("="*70)
    print(f"CUSTOM RANGE FORECASTER | SKU: {sku_code}")
    print(f"Training Data Range  : {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
    print(f"Prediction Data Range: {predict_start.strftime('%Y-%m-%d')} to {predict_end.strftime('%Y-%m-%d')}")
    print("="*70)
    
    # 1. Load raw SKU data
    raw_df = load_and_aggregate_sku(excel_path, sku_code)
    
    # Filter training data
    train_raw = raw_df[(raw_df['date'] >= train_start) & (raw_df['date'] <= train_end)].copy()
    if len(train_raw) == 0:
        raise ValueError("Training dataset is empty for the specified date range.")
        
    train_anomaly = detect_anomalies(train_raw)
    train_feats = build_features(train_anomaly)
    feature_cols = get_feature_columns()
    clean_train = train_feats.dropna(subset=feature_cols).copy().reset_index(drop=True)
    
    # 2. Fit model
    X_train = clean_train[feature_cols]
    y_train = clean_train['target_sales']
    
    forecaster = SalesForecasterEnsemble()
    forecaster.fit(X_train, y_train)
    
    pred_train = forecaster.predict(X_train)
    res_std = np.std(y_train - pred_train)
    
    # 3. Build prediction timeline
    predict_dates = pd.date_range(start=predict_start, end=predict_end, freq='D', name='date')
    predict_df = pd.DataFrame({'date': predict_dates})
    
    # Concatenate training history with prediction dates to build lag features
    combined = pd.concat([train_raw[['date', 'sold_qty']], predict_df], ignore_index=True)
    
    # Check if actual ground truth sales exist in raw_df for prediction range
    ground_truth = raw_df[(raw_df['date'] >= predict_start) & (raw_df['date'] <= predict_end)]
    has_ground_truth = len(ground_truth) == len(predict_dates)
    
    # Iterative prediction loop across prediction range
    preds = []
    combined_anomaly = detect_anomalies(combined)
    
    for d in predict_dates:
        feats = build_features(combined_anomaly)
        row_feat = feats[feats['date'] == d]
        
        if row_feat[feature_cols].isna().any().any():
            X_curr = row_feat[feature_cols].fillna(clean_train[feature_cols].mean())
        else:
            X_curr = row_feat[feature_cols]
            
        p = max(0, forecaster.predict(X_curr)[0])
        preds.append(p)
        
        # Update target_sales for future days in combined_anomaly
        combined_anomaly.loc[combined_anomaly['date'] == d, 'target_sales'] = p
        
    predict_df['predicted_sales'] = np.round(preds, 1)
    predict_df['lower_bound_95%'] = np.maximum(0, np.round(preds - 1.96 * res_std, 1))
    predict_df['upper_bound_95%'] = np.round(preds + 1.96 * res_std, 1)
    
    total_pred_units = int(np.sum(preds))
    print(f"\nTotal Predicted Volume ({len(predict_dates)} days): {total_pred_units:,} units")
    print(f"Average Daily Sales: {np.mean(preds):.1f} units/day")
    
    if has_ground_truth:
        actual_units = int(ground_truth['sold_qty'].sum())
        acc = (1 - abs(actual_units - total_pred_units) / actual_units) * 100
        mae = mean_absolute_error(ground_truth['sold_qty'], preds)
        print(f"Actual Volume in Range   : {actual_units:,} units")
        print(f"Range Accuracy           : {acc:.2f}%")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        predict_df['actual_sales'] = ground_truth['sold_qty'].values
    
    # Save CSV
    out_csv = os.path.join(output_dir, f"range_forecast_{predict_start_str}_to_{predict_end_str}.csv")
    predict_df.to_csv(out_csv, index=False)
    print(f"Exported prediction CSV to: {out_csv}")
    
    # Plot Chart
    plt.figure(figsize=(15, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot context training data (last 60 days of training)
    context_start = train_end - pd.Timedelta(days=60)
    context_df = train_raw[train_raw['date'] >= context_start]
    plt.plot(context_df['date'], context_df['sold_qty'], label='Training Sales History', color='#2c3e50', alpha=0.6, linewidth=1.5)
    
    if has_ground_truth:
        plt.plot(ground_truth['date'], ground_truth['sold_qty'], label='Actual Ground Truth Sales', color='#2c3e50', linewidth=2.0)
        
    plt.plot(predict_df['date'], predict_df['predicted_sales'], label='Predicted Sales Range', color='#e67e22', linewidth=2.5, marker='o', markersize=4)
    plt.fill_between(predict_df['date'], predict_df['lower_bound_95%'], predict_df['upper_bound_95%'], color='#f39c12', alpha=0.25, label='95% Uncertainty Band')
    
    title_str = f"Custom Range Forecast ({predict_start_str} to {predict_end_str}) | SKU: {sku_code} | Pred Total: {total_pred_units:,} Units"
    if has_ground_truth:
        title_str += f" | Acc: {acc:.1f}%"
        
    plt.title(title_str, fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Daily Units Sold', fontsize=11)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.legend(loc='upper left', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    
    out_plot = os.path.join(output_dir, f"range_forecast_{predict_start_str}_to_{predict_end_str}.png")
    plt.savefig(out_plot)
    plt.close()
    print(f"Exported forecast chart to: {out_plot}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Custom Range Sales Forecaster")
    parser.add_argument('--train_start', type=str, default='2025-01-01', help="Training start date (YYYY-MM-DD)")
    parser.add_argument('--train_end', type=str, default='2026-07-31', help="Training end date (YYYY-MM-DD)")
    parser.add_argument('--predict_start', type=str, default='2026-08-01', help="Prediction start date (YYYY-MM-DD)")
    parser.add_argument('--predict_end', type=str, default='2026-08-30', help="Prediction end date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    run_range_forecast(args.train_start, args.train_end, args.predict_start, args.predict_end)
