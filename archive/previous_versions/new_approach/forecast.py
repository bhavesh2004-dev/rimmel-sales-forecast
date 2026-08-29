import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_aggregate_sku
from src.anomaly_detector import detect_anomalies
from src.feature_engineering import build_features, get_feature_columns
from src.model import SalesForecasterEnsemble

def run_rolling_forecast(forecast_start_str: str = '2026-08-01', forecast_days: int = 30):
    excel_path = os.path.join(os.path.dirname(__file__), '..', 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026.xlsx')
    output_dir = os.path.join(os.path.dirname(__file__), 'output_results')
    os.makedirs(output_dir, exist_ok=True)
    
    sku_code = 'RIM-MSC-E3DL-003'
    forecast_start = pd.to_datetime(forecast_start_str)
    history_cutoff = forecast_start - pd.Timedelta(days=1)
    
    print(f"="*70)
    print(f"ROLLING FORECAST PIPELINE | SKU: {sku_code}")
    print(f"History Cutoff Date: {history_cutoff.strftime('%Y-%m-%d')}")
    print(f"Forecast Period: {forecast_start.strftime('%Y-%m-%d')} to {(forecast_start + pd.Timedelta(days=forecast_days-1)).strftime('%Y-%m-%d')}")
    print(f"="*70)
    
    # 1. Load data
    raw_df = load_and_aggregate_sku(excel_path, sku_code)
    
    # Filter history up to history_cutoff
    history_df = raw_df[raw_df['date'] <= history_cutoff].copy()
    
    if len(history_df) == 0:
        raise ValueError(f"No historical data available prior to {forecast_start_str}")
        
    anomaly_df = detect_anomalies(history_df)
    feature_df = build_features(anomaly_df)
    
    feature_cols = get_feature_columns()
    clean_data = feature_df.dropna(subset=feature_cols).copy().reset_index(drop=True)
    
    # 2. Train model on history up to history_cutoff
    X_train = clean_data[feature_cols]
    y_train = clean_data['target_sales']
    
    forecaster = SalesForecasterEnsemble()
    forecaster.fit(X_train, y_train)
    
    pred_train = forecaster.predict(X_train)
    residuals = y_train - pred_train
    res_std = np.std(residuals)
    
    # 3. Create Future Dates
    future_dates = pd.date_range(start=forecast_start, periods=forecast_days, freq='D', name='date')
    future_df = pd.DataFrame({'date': future_dates})
    
    combined = pd.concat([clean_data[['date', 'target_sales', 'sold_qty']], future_df], ignore_index=True)
    last_rolling_7 = clean_data['target_sales'].iloc[-7:].mean()
    combined['target_sales'] = combined['target_sales'].fillna(last_rolling_7)
    
    combined_anomaly = detect_anomalies(combined)
    combined_feats = build_features(combined_anomaly)
    
    future_feats = combined_feats[combined_feats['date'].isin(future_dates)].copy()
    X_future = future_feats[feature_cols]
    
    future_pred = forecaster.predict(X_future)
    future_feats['predicted_sales'] = np.round(future_pred, 1)
    future_feats['lower_bound_95%'] = np.maximum(0, np.round(future_pred - 1.96 * res_std, 1))
    future_feats['upper_bound_95%'] = np.round(future_pred + 1.96 * res_std, 1)
    
    total_units = int(np.sum(future_pred))
    avg_daily = float(np.mean(future_pred))
    
    latest_velocity = float(clean_data['trend_velocity'].iloc[-1])
    
    print(f"\nModel Training Complete.")
    print(f"Latest 7-Day Trend Velocity Index: {latest_velocity:.2f}x")
    print(f"Total Predicted Units ({forecast_days} days): {total_units:,} units")
    print(f"Average Daily Sales: {avg_daily:.1f} units/day")
    
    # Export CSV
    csv_file = f"forecast_{forecast_start_str}.csv"
    csv_path = os.path.join(output_dir, csv_file)
    display_cols = ['date', 'predicted_sales', 'lower_bound_95%', 'upper_bound_95%']
    future_feats[display_cols].to_csv(csv_path, index=False)
    print(f"Saved forecast to: {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run rolling 30-day forecast")
    parser.add_argument('--start_date', type=str, default='2026-08-01', help="Forecast start date (YYYY-MM-DD)")
    parser.add_argument('--days', type=int, default=30, help="Number of days to forecast")
    args = parser.parse_args()
    
    run_rolling_forecast(args.start_date, args.days)
