import os
import sys
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

def predict_week_iteratively(forecaster, history_df, target_dates, feature_cols, clean_data):
    combined = history_df[['date', 'target_sales', 'sold_qty']].copy()
    future_df = pd.DataFrame({'date': target_dates, 'target_sales': np.nan, 'sold_qty': np.nan})
    combined = pd.concat([combined, future_df], ignore_index=True)
    
    preds = []
    for d in target_dates:
        anom = detect_anomalies(combined)
        feats = build_features(anom)
        
        row_feat = feats[feats['date'] == d]
        if row_feat[feature_cols].isna().any().any():
            X_curr = row_feat[feature_cols].fillna(clean_data[feature_cols].mean())
        else:
            X_curr = row_feat[feature_cols]
            
        p = forecaster.predict(X_curr)[0]
        p = max(0, p)
        preds.append((d, p))
        
        combined.loc[combined['date'] == d, 'target_sales'] = p
        
    res_df = pd.DataFrame(preds, columns=['date', 'rolling_predicted_sales'])
    return res_df

def run_july_rolling_backtest():
    excel_path = os.path.join(os.path.dirname(__file__), '..', 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026.xlsx')
    output_dir = os.path.join(os.path.dirname(__file__), 'output_results')
    os.makedirs(output_dir, exist_ok=True)
    
    sku_code = 'RIM-MSC-E3DL-003'
    print(f"Loading data for July 2026 Weekly Rolling Backtest | SKU: {sku_code}...")
    
    full_raw_df = load_and_aggregate_sku(excel_path, sku_code)
    
    weekly_slices = [
        {'name': 'Week 1', 'start': '2026-07-01', 'end': '2026-07-07'},
        {'name': 'Week 2', 'start': '2026-07-08', 'end': '2026-07-14'},
        {'name': 'Week 3', 'start': '2026-07-15', 'end': '2026-07-21'},
        {'name': 'Week 4', 'start': '2026-07-22', 'end': '2026-07-31'}
    ]
    
    weekly_preds = []
    
    for slice_info in weekly_slices:
        start_date = pd.to_datetime(slice_info['start'])
        end_date = pd.to_datetime(slice_info['end'])
        cutoff_date = start_date - pd.Timedelta(days=1)
        
        history_df = full_raw_df[full_raw_df['date'] <= cutoff_date].copy()
        
        anomaly_df = detect_anomalies(history_df)
        feature_df = build_features(anomaly_df)
        feature_cols = get_feature_columns()
        clean_data = feature_df.dropna(subset=feature_cols).copy().reset_index(drop=True)
        
        X_train = clean_data[feature_cols]
        y_train = clean_data['target_sales']
        
        forecaster = SalesForecasterEnsemble()
        forecaster.fit(X_train, y_train)
        
        target_dates = pd.date_range(start=start_date, end=end_date, freq='D', name='date')
        
        week_res = predict_week_iteratively(forecaster, anomaly_df, target_dates, feature_cols, clean_data)
        weekly_preds.append(week_res)
        
        total_week_p = week_res['rolling_predicted_sales'].sum()
        latest_vel = clean_data['trend_velocity'].iloc[-1]
        print(f" {slice_info['name']} ({slice_info['start']} to {slice_info['end']}): Predicted {int(total_week_p)} units | Latest Trend Velocity: {latest_vel:.2f}x")
        
    combined_july_rolling = pd.concat(weekly_preds, ignore_index=True)
    
    # Merge with actual July sales
    july_actuals = full_raw_df[(full_raw_df['date'] >= '2026-07-01') & (full_raw_df['date'] <= '2026-07-31')].copy()
    july_eval = july_actuals.merge(combined_july_rolling, on='date', how='inner')
    
    # Run 1-shot forecast for July 2026
    history_june = full_raw_df[full_raw_df['date'] <= '2026-06-30'].copy()
    anomaly_june = detect_anomalies(history_june)
    feature_june = build_features(anomaly_june)
    feature_cols = get_feature_columns()
    clean_june = feature_june.dropna(subset=feature_cols).copy().reset_index(drop=True)
    
    model_1shot = SalesForecasterEnsemble()
    model_1shot.fit(clean_june[feature_cols], clean_june['target_sales'])
    
    july_dates = pd.date_range(start='2026-07-01', end='2026-07-31', freq='D', name='date')
    july_1shot_res = predict_week_iteratively(model_1shot, anomaly_june, july_dates, feature_cols, clean_june)
    july_eval['1shot_predicted_sales'] = july_1shot_res['rolling_predicted_sales']
    
    # Totals & Accuracy Calculations
    total_actual = july_eval['sold_qty'].sum()
    total_1shot = july_eval['1shot_predicted_sales'].sum()
    total_rolling = july_eval['rolling_predicted_sales'].sum()
    
    acc_1shot = (1 - abs(total_actual - total_1shot) / total_actual) * 100
    acc_rolling = (1 - abs(total_actual - total_rolling) / total_actual) * 100
    
    mae_1shot = mean_absolute_error(july_eval['sold_qty'], july_eval['1shot_predicted_sales'])
    mae_rolling = mean_absolute_error(july_eval['sold_qty'], july_eval['rolling_predicted_sales'])
    
    print("\n" + "="*70)
    print("JULY 2026 ACCURACY COMPARISON (1-SHOT VS WEEKLY ROLLING):")
    print("="*70)
    print(f"Total Actual July Sales       : {total_actual:,} units")
    print(f"1-Shot Forecast (July 1st)    : {int(total_1shot):,} units | Accuracy: {acc_1shot:.2f}% | MAE: {mae_1shot:.2f}")
    print(f"Weekly Rolling Forecast       : {int(total_rolling):,} units | Accuracy: {acc_rolling:.2f}% | MAE: {mae_rolling:.2f}")
    print("="*70)
    
    # Generate Comparison Plot
    plt.figure(figsize=(15, 7), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    plt.plot(july_eval['date'], july_eval['sold_qty'], label='Actual Daily Sales (July 2026)', color='#2c3e50', linewidth=2, marker='o', markersize=4)
    plt.plot(july_eval['date'], july_eval['1shot_predicted_sales'], label=f'Original 1-Shot Forecast ({acc_1shot:.1f}% Acc)', color='#e74c3c', linewidth=2, linestyle='--')
    plt.plot(july_eval['date'], july_eval['rolling_predicted_sales'], label=f'Weekly Rolling Forecast ({acc_rolling:.1f}% Acc)', color='#27ae60', linewidth=2.5, marker='s', markersize=4)
    
    plt.title(f'July 2026 Accuracy Improvement: 1-Shot ({acc_1shot:.1f}%) vs. Weekly Rolling ({acc_rolling:.1f}%)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Daily Units Sold', fontsize=11)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.legend(loc='upper left', framealpha=0.9, fontsize=11)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'july_weekly_rolling_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    
    july_eval.to_csv(os.path.join(output_dir, 'july_rolling_vs_1shot.csv'), index=False)
    print(f"\nSaved comparison chart to: {plot_path}")

if __name__ == '__main__':
    run_july_rolling_backtest()
