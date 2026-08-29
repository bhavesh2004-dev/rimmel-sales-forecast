import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_aggregate_sku
from src.anomaly_detector import detect_anomalies
from src.feature_engineering import build_features, get_feature_columns
from src.model import SalesForecasterEnsemble

def predict_august_2026():
    excel_path = os.path.join(os.path.dirname(__file__), '..', 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026.xlsx')
    output_dir = os.path.join(os.path.dirname(__file__), 'output_results')
    os.makedirs(output_dir, exist_ok=True)
    
    sku_code = 'RIM-MSC-E3DL-003'
    print(f"Loading data to generate August 2026 forecast for SKU: {sku_code}...")
    
    raw_df = load_and_aggregate_sku(excel_path, sku_code)
    anomaly_df = detect_anomalies(raw_df)
    feature_df = build_features(anomaly_df)
    
    feature_cols = get_feature_columns()
    clean_data = feature_df.dropna(subset=feature_cols).copy().reset_index(drop=True)
    
    # Train model on full history up to 2026-07-31
    X_train = clean_data[feature_cols]
    y_train = clean_data['target_sales']
    
    forecaster = SalesForecasterEnsemble()
    forecaster.fit(X_train, y_train)
    
    # Calculate Residual Std Dev for Uncertainty Band
    pred_train = forecaster.predict(X_train)
    residuals = y_train - pred_train
    res_std = np.std(residuals)
    
    # Create August 2026 Future Dates (2026-08-01 to 2026-08-30)
    august_dates = pd.date_range(start='2026-08-01', end='2026-08-30', freq='D', name='date')
    future_df = pd.DataFrame({'date': august_dates})
    
    # To compute features for August dates using direct anchor lags:
    # We combine historical clean_data with future_df
    combined = pd.concat([clean_data[['date', 'target_sales', 'sold_qty']], future_df], ignore_index=True)
    
    # For future target_sales during feature generation, forward fill with rolling mean 7 to avoid NaN
    last_rolling_7 = clean_data['target_sales'].iloc[-7:].mean()
    combined['target_sales'] = combined['target_sales'].fillna(last_rolling_7)
    
    # Re-apply anomaly detection and feature engineering on combined timeline
    combined_anomaly = detect_anomalies(combined)
    combined_feats = build_features(combined_anomaly)
    
    august_df = combined_feats[combined_feats['date'].isin(august_dates)].copy()
    X_august = august_df[feature_cols]
    
    august_pred = forecaster.predict(X_august)
    august_df['predicted_sales'] = np.round(august_pred, 1)
    august_df['lower_bound_95%'] = np.maximum(0, np.round(august_pred - 1.96 * res_std, 1))
    august_df['upper_bound_95%'] = np.round(august_pred + 1.96 * res_std, 1)
    
    total_august_units = int(np.sum(august_pred))
    avg_daily_units = float(np.mean(august_pred))
    
    print("\n" + "="*70)
    print(f"AUGUST 2026 SALES FORECAST SUMMARY FOR {sku_code}:")
    print("="*70)
    print(f"Total Predicted Units for August 2026 (30 days): {total_august_units:,} units")
    print(f"Average Predicted Sales per Day: {avg_daily_units:.1f} units/day")
    print("="*70)
    
    # Print Daily Forecast Table
    display_cols = ['date', 'predicted_sales', 'lower_bound_95%', 'upper_bound_95%']
    print("\nDaily August 2026 Forecast Breakdown:")
    print(august_df[display_cols].to_string(index=False))
    
    # Export CSV
    august_csv_path = os.path.join(output_dir, 'august_2026_forecast.csv')
    august_df[display_cols].to_csv(august_csv_path, index=False)
    
    # Generate Chart
    plt.figure(figsize=(15, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot last 60 days of history (June & July 2026)
    history_60d = clean_data[clean_data['date'] >= '2026-06-01']
    plt.plot(history_60d['date'], history_60d['sold_qty'], label='Historical Sales (June - July 2026)', color='#2c3e50', linewidth=1.8)
    
    # Plot August Forecast
    plt.plot(august_df['date'], august_df['predicted_sales'], label='August 2026 Forecast', color='#27ae60', linewidth=2.5, marker='o', markersize=4)
    plt.fill_between(august_df['date'], august_df['lower_bound_95%'], august_df['upper_bound_95%'], color='#2ecc71', alpha=0.25, label='95% Uncertainty Band')
    
    plt.title(f'August 2026 Future 30-Day Sales Forecast | SKU: {sku_code} | Total: {total_august_units:,} Units', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Daily Units Sold', fontsize=11)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.legend(loc='upper left', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, 'august_2026_forecast.png')
    plt.savefig(chart_path)
    plt.close()
    
    print(f"\nSaved August forecast CSV and chart to: {output_dir}")

if __name__ == '__main__':
    predict_august_2026()
